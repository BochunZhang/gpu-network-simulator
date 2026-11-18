// Fixed-size CUDA copy baseline. Run pairs sequentially on destination streams.
#include <cuda_runtime.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int kDevices = 4;
constexpr int kWarmup = 3;
constexpr int kIterations = 10;
constexpr size_t kMaxBytes = 64ULL * 1024 * 1024;
constexpr std::array<size_t, 2> kSizes = {1024ULL * 1024, kMaxBytes};

void check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
    }
}

std::string quote(const std::string& value) {
    std::ostringstream result;
    result << '"';
    for (unsigned char character : value) {
        if (character == '"' || character == '\\') result << '\\' << character;
        else if (character < 0x20) result << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(character);
        else result << character;
    }
    result << '"';
    return result.str();
}

std::string device_uuid(const cudaUUID_t& uuid) {
    std::ostringstream result;
    result << "GPU-" << std::hex << std::setfill('0');
    for (int i = 0; i < 16; ++i) {
        if (i == 4 || i == 6 || i == 8 || i == 10) result << '-';
        result << std::setw(2) << static_cast<unsigned int>(static_cast<unsigned char>(uuid.bytes[i]));
    }
    return result.str();
}

unsigned char pattern(size_t offset, int device) {
    uint32_t value = static_cast<uint32_t>(offset) ^ (0x9e3779b9U * static_cast<uint32_t>(device + 1));
    value ^= value >> 16;
    value *= 0x7feb352dU;
    value ^= value >> 15;
    value *= 0x846ca68bU;
    value ^= value >> 16;
    return static_cast<unsigned char>(value);
}

struct Device {
    explicit Device(int logical_index) : index(logical_index) {}
    int index;
    void* source = nullptr;
    void* destination = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cudaDeviceProp properties{};
    std::vector<int> enabled_peers;

    void allocate(const std::vector<unsigned char>& host) {
        check(cudaSetDevice(index), "cudaSetDevice");
        check(cudaGetDeviceProperties(&properties, index), "cudaGetDeviceProperties");
        check(cudaMalloc(&source, kMaxBytes), "cudaMalloc source");
        check(cudaMalloc(&destination, kMaxBytes), "cudaMalloc destination");
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
        check(cudaEventCreate(&start), "cudaEventCreate start");
        check(cudaEventCreate(&stop), "cudaEventCreate stop");
        check(cudaMemcpy(source, host.data(), kMaxBytes, cudaMemcpyHostToDevice), "initialize source");
        check(cudaDeviceSynchronize(), "source initialization completion");
    }

    ~Device() {
        cudaSetDevice(index);
        if (stream) cudaStreamSynchronize(stream);
        if (start) cudaEventDestroy(start);
        if (stop) cudaEventDestroy(stop);
        if (stream) cudaStreamDestroy(stream);
        if (source) cudaFree(source);
        if (destination) cudaFree(destination);
        for (int peer : enabled_peers) cudaDeviceDisablePeerAccess(peer);
    }
};

void copy(Device& source, Device& destination, size_t bytes) {
    if (source.index == destination.index) {
        check(cudaMemcpyAsync(destination.destination, source.source, bytes,
                              cudaMemcpyDeviceToDevice, destination.stream), "cudaMemcpyAsync D2D");
    } else {
        check(cudaMemcpyPeerAsync(destination.destination, destination.index, source.source,
                                  source.index, bytes, destination.stream), "cudaMemcpyPeerAsync");
    }
}

void write_context(const std::string& root, const std::vector<std::unique_ptr<Device>>& devices, int visible) {
    int runtime_version = 0;
    int driver_version = 0;
    check(cudaRuntimeGetVersion(&runtime_version), "cudaRuntimeGetVersion");
    check(cudaDriverGetVersion(&driver_version), "cudaDriverGetVersion");
    std::ofstream output(root + "/cuda-context.json");
    if (!output) throw std::runtime_error("cannot open cuda-context.json");
    const char* mask = std::getenv("CUDA_VISIBLE_DEVICES");
    output << "{\n  \"visible_device_count\": " << visible
           << ",\n  \"selected_device_count\": " << devices.size()
           << ",\n  \"cuda_runtime_version\": " << runtime_version
           << ",\n  \"cuda_driver_version\": " << driver_version
           << ",\n  \"compiled_cudart_version\": " << CUDART_VERSION
           << ",\n  \"cuda_visible_devices\": " << (mask ? quote(mask) : "null") << ",\n  \"devices\": [\n";
    for (size_t i = 0; i < devices.size(); ++i) {
        const auto& device = *devices[i];
        output << "    {\"logical_index\": " << device.index
               << ", \"uuid\": " << quote(device_uuid(device.properties.uuid))
               << ", \"name\": " << quote(device.properties.name)
               << ", \"compute_capability\": " << quote(std::to_string(device.properties.major) + "." + std::to_string(device.properties.minor))
               << ", \"total_memory_bytes\": " << device.properties.totalGlobalMem << "}"
               << (i + 1 == devices.size() ? "\n" : ",\n");
    }
    output << "  ]\n}\n";
    if (!output) throw std::runtime_error("failed to write cuda-context.json");
}

int run(const std::string& root) {
    int visible = 0;
    check(cudaGetDeviceCount(&visible), "cudaGetDeviceCount");
    if (visible < kDevices) throw std::runtime_error("task requires at least four CUDA-visible GPUs");
    std::vector<unsigned char> host(kMaxBytes);
    std::vector<std::unique_ptr<Device>> devices;
    for (int index = 0; index < kDevices; ++index) {
        for (size_t byte = 0; byte < host.size(); ++byte) host[byte] = pattern(byte, index);
        devices.emplace_back(new Device(index));
        devices.back()->allocate(host);
    }
    write_context(root, devices, visible);
    std::array<std::array<int, kDevices>, kDevices> accessible{};
    for (int destination = 0; destination < kDevices; ++destination) {
        check(cudaSetDevice(destination), "cudaSetDevice peer destination");
        for (int source = 0; source < kDevices; ++source) {
            if (source == destination) { accessible[source][destination] = 1; continue; }
            check(cudaDeviceCanAccessPeer(&accessible[source][destination], destination, source), "cudaDeviceCanAccessPeer");
            if (accessible[source][destination]) {
                cudaError_t enabled = cudaDeviceEnablePeerAccess(source, 0);
                if (enabled == cudaSuccess) devices[destination]->enabled_peers.push_back(source);
                else if (enabled == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
                else check(enabled, "cudaDeviceEnablePeerAccess");
            }
        }
    }
    std::ofstream output(root + "/transfer-matrix.csv");
    if (!output) throw std::runtime_error("cannot open transfer-matrix.csv");
    output << "kind,src,dst,bytes,iterations,warmup,peer_access,cuda_ms,wall_ms,gb_per_s,mismatched_bytes,src_uuid,dst_uuid\n";
    output << std::setprecision(12);
    bool success = true;
    for (size_t bytes : kSizes) {
        for (int source_index = 0; source_index < kDevices; ++source_index) {
            for (int destination_index = 0; destination_index < kDevices; ++destination_index) {
                auto& source = *devices[source_index];
                auto& destination = *devices[destination_index];
                const char* kind = source_index == destination_index ? "d2d" : "p2p";
                int peer = accessible[source_index][destination_index];
                double cuda_ms = 0.0;
                double wall_ms = 0.0;
                double bandwidth = 0.0;
                int64_t mismatches = -1;
                if (peer) {
                    check(cudaSetDevice(destination_index), "cudaSetDevice measured destination");
                    check(cudaMemsetAsync(destination.destination, 0, bytes, destination.stream), "clear destination");
                    for (int iteration = 0; iteration < kWarmup; ++iteration) copy(source, destination, bytes);
                    check(cudaStreamSynchronize(destination.stream), "warmup completion");
                    // Verification must depend on the measured batch, not on
                    // values left by an earlier warmup copy.
                    check(cudaMemsetAsync(destination.destination, 0, bytes, destination.stream), "clear measured destination");
                    check(cudaStreamSynchronize(destination.stream), "destination clear completion");
                    const auto begin = std::chrono::steady_clock::now();
                    check(cudaEventRecord(destination.start, destination.stream), "record start");
                    for (int iteration = 0; iteration < kIterations; ++iteration) copy(source, destination, bytes);
                    check(cudaEventRecord(destination.stop, destination.stream), "record stop");
                    check(cudaEventSynchronize(destination.stop), "timed copies completion");
                    const auto end = std::chrono::steady_clock::now();
                    float elapsed = 0.0F;
                    check(cudaEventElapsedTime(&elapsed, destination.start, destination.stop), "cudaEventElapsedTime");
                    cuda_ms = elapsed;
                    wall_ms = std::chrono::duration<double, std::milli>(end - begin).count();
                    if (!std::isfinite(cuda_ms) || cuda_ms <= 0.0 || !std::isfinite(wall_ms) || wall_ms <= 0.0) {
                        throw std::runtime_error("invalid completed-copy timing");
                    }
                    bandwidth = static_cast<double>(bytes) * kIterations / (cuda_ms * 1.0e6);
                    check(cudaMemcpy(host.data(), destination.destination, bytes, cudaMemcpyDeviceToHost), "verify destination D2H");
                    mismatches = 0;
                    for (size_t byte = 0; byte < bytes; ++byte) mismatches += host[byte] != pattern(byte, source_index);
                    if (mismatches) success = false;
                } else success = false;
                output << kind << ',' << source_index << ',' << destination_index << ',' << bytes << ','
                       << kIterations << ',' << kWarmup << ',' << peer << ',' << cuda_ms << ',' << wall_ms << ','
                       << bandwidth << ',' << mismatches << ',' << device_uuid(source.properties.uuid) << ','
                       << device_uuid(destination.properties.uuid) << '\n';
                output.flush();
                if (!output) throw std::runtime_error("failed to write transfer matrix");
            }
        }
    }
    return success ? 0 : 2;
}
}  // namespace

int main(int argc, char** argv) {
    try {
        const std::string prefix = "--output-dir=";
        if (argc != 2 || std::string(argv[1]).rfind(prefix, 0) != 0 || std::string(argv[1]).size() == prefix.size()) {
            throw std::runtime_error("usage: gpu_transfer --output-dir=/absolute/result/directory");
        }
        return run(std::string(argv[1]).substr(prefix.size()));
    } catch (const std::exception& error) {
        std::cerr << "GPU transfer benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
