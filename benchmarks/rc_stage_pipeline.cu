#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int owners = 512;
constexpr int owners_per_chunk = 64;
constexpr int chunks = owners / owners_per_chunk;
constexpr int threads = 128;
constexpr int tile = 96;
constexpr uint32_t mask = 0xffffffu;

void check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}
#define CUDA(call) check((call), #call)

struct Descriptor {
    // Packed BTH-like opcode/packet sequence and length/owner metadata.
    uint32_t opcode_psn, owner_length, arrival_ns;
};
struct Ready {
    uint32_t psn, length, owner, arrival_ns, last;
};
struct Output {
    uint32_t owner, psn, accepted, delivered_bytes, message_completions, finish_ns;
};
static_assert(sizeof(Output) == 6 * sizeof(uint32_t), "binary output has six u32 fields");

__host__ __device__ uint32_t initial_psn(int owner) { return (mask - 15 + owner % 7) & mask; }

__device__ Ready decode(Descriptor d) {
    return {d.opcode_psn & mask, d.owner_length & 0xffff, d.owner_length >> 16,
            d.arrival_ns, d.opcode_psn >> 24};
}

struct Queue {
    uint32_t expected, delivered = 0, completed = 0, finish = 0;

    __device__ Output commit(Ready r) {
        // All packets use the modeled directed 400 Gb/s link. Duplicates
        // consume wire time but do not advance the receive sequence or CQ.
        uint32_t service = (r.length + 80u + 49u) / 50u;
        finish = (finish > r.arrival_ns ? finish : r.arrival_ns) + service;
        bool accept = r.psn == expected;
        if (accept) {
            expected = (expected + 1) & mask;
            delivered += r.length;
            completed += r.last;
        }
        return {r.owner, r.psn, uint32_t(accept), delivered, completed, finish};
    }
};

template<bool fused, bool specialized>
__global__ void decode_tiles(const Descriptor* source, Ready* intermediate, Output* output,
                             int first_owner, int depth) {
    const int lane = threadIdx.x;
    const int local = blockIdx.x * depth;
    const int base = (first_owner + blockIdx.x) * depth;
    __shared__ Descriptor buffers[2][tile];
    __shared__ Ready ordered[tile];
    Queue queue{initial_psn(first_owner + blockIdx.x)};
    if (specialized && lane < 32) {
        for (int j = lane; j < tile && j < depth; j += 32)
            buffers[0][j] = source[base + j];
    }
    __syncthreads();
    for (int start = 0, step = 0; start < depth; start += tile, ++step) {
        int count = min(tile, depth - start);
        int slot = specialized ? step % 2 : 0;
        if constexpr (!specialized) {
            if (lane < count) buffers[0][lane] = source[base + start + lane];
            __syncthreads();
        }
        if (specialized && lane < 32) {
            // This warp fills the spare slot while the other three decode
            // the current slot. End-of-tile barrier publishes the next tile.
            for (int j = lane; j < tile && start + tile + j < depth; j += 32)
                buffers[1 - slot][j] = source[base + start + tile + j];
        } else {
            int j = specialized ? lane - 32 : lane;
            if (j < count) {
                Ready item = decode(buffers[slot][j]);
                if constexpr (fused) ordered[j] = item;
                else intermediate[local + start + j] = item;
            }
        }
        if constexpr (fused) {
            __syncthreads();
            if (lane == 32) {
#if RC_STAGE_UNROLL == 1
#pragma unroll 1
#elif RC_STAGE_UNROLL == 4
#pragma unroll 4
#elif RC_STAGE_UNROLL == 8
#pragma unroll 8
#endif
                for (int j = 0; j < count; ++j)
                    output[local + start + j] = queue.commit(ordered[j]);
            }
        }
        __syncthreads();
    }
}

__global__ void commit_queues(const Ready* input, Output* output, int first_owner, int depth) {
    // Match the fused kernel's owner-to-CTA mapping and commit lane.
    if (threadIdx.x != 32) return;
    const int owner = blockIdx.x;
    Queue queue{initial_psn(first_owner + owner)};
    for (int j = 0; j < depth; ++j) {
        int i = owner * depth + j;
        output[i] = queue.commit(input[i]);
    }
}

struct Slot {
    Ready* prepared = nullptr;
    Output* device = nullptr;
    Output* host = nullptr;
    cudaEvent_t produced{}, copied{};
};

void launch(const Descriptor* source, Slot& s, int first_owner, int depth, int variant,
            cudaStream_t stream) {
    const bool fused = variant & 1;
    const bool specialized = variant & 2;
    // Select the implementation on the host so each kernel has its own SASS.
    if (fused && specialized)
        decode_tiles<true,true><<<owners_per_chunk,threads,0,stream>>>(source, s.prepared, s.device, first_owner, depth);
    else if (fused)
        decode_tiles<true,false><<<owners_per_chunk,threads,0,stream>>>(source, s.prepared, s.device, first_owner, depth);
    else if (specialized)
        decode_tiles<false,true><<<owners_per_chunk,threads,0,stream>>>(source, s.prepared, s.device, first_owner, depth);
    else
        decode_tiles<false,false><<<owners_per_chunk,threads,0,stream>>>(source, s.prepared, s.device, first_owner, depth);
    CUDA(cudaGetLastError());
    if (!fused) {
        commit_queues<<<owners_per_chunk,threads,0,stream>>>(s.prepared, s.device, first_owner, depth);
        CUDA(cudaGetLastError());
    }
}

void fixture(int depth, std::vector<Descriptor>& input, std::vector<Output>& oracle) {
    for (int owner = 0; owner < owners; ++owner) {
        uint32_t unique = 0, bytes = 0, completed = 0, finish = 0;
        uint32_t psn = 0, length = 0, last = 0;
        for (int i = 0; i < depth; ++i) {
            bool duplicate = i % 13 == 12;
            if (!duplicate) {
                psn = (initial_psn(owner) + unique) & mask;
                length = 64 + (owner * 17 + unique * 113) % 4033;
                last = (unique % 4 == 3);
                ++unique;
                bytes += length;
                completed += last;
            }
            uint32_t arrival = uint32_t(i * (3 + owner % 37));
            input.push_back({uint32_t((last << 24) | psn), uint32_t(owner << 16) | length, arrival});
            uint32_t physical_bytes = length + 80;
            finish = std::max(finish, arrival) + physical_bytes / 50 + uint32_t(physical_bytes % 50 != 0);
            oracle.push_back({uint32_t(owner), psn, uint32_t(!duplicate), bytes, completed, finish});
        }
    }
}

double measure(const Descriptor* source, Slot (&slots)[2], cudaStream_t compute,
               cudaStream_t copy, int depth, int variant, const std::string& path) {
    bool asynchronous = variant & 4;
    size_t bytes = size_t(owners_per_chunk) * depth * sizeof(Output);
    auto start = std::chrono::steady_clock::now();
    std::ofstream out;
    out.exceptions(std::ios::failbit | std::ios::badbit);
    out.open(path, std::ios::binary | std::ios::trunc);
    auto flush = [&](int chunk) {
        auto& s = slots[asynchronous ? chunk % 2 : 0];
        CUDA(cudaEventSynchronize(s.copied));
        out.write(reinterpret_cast<const char*>(s.host), bytes);
    };
    for (int i = 0; i < chunks; ++i) {
        if (asynchronous && i >= 2) flush(i - 2);
        auto& s = slots[asynchronous ? i % 2 : 0];
        launch(source, s, i * owners_per_chunk, depth, variant, compute);
        CUDA(cudaEventRecord(s.produced, compute));
        CUDA(cudaStreamWaitEvent(copy, s.produced, 0));
        CUDA(cudaMemcpyAsync(s.host, s.device, bytes, cudaMemcpyDeviceToHost, copy));
        CUDA(cudaEventRecord(s.copied, copy));
        if (!asynchronous) flush(i);
    }
    if (asynchronous) {
        flush(chunks - 2);
        flush(chunks - 1);
    }
    out.close(); // End-to-end through filesystem close; no durable fsync.
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

void compare_file(const std::string& path, const std::vector<Output>& oracle) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot read " + path);
    std::vector<Output> block(4096);
    for (size_t at = 0; at < oracle.size(); at += block.size()) {
        size_t size = std::min(block.size(), oracle.size() - at) * sizeof(Output);
        in.read(reinterpret_cast<char*>(block.data()), size);
        if (size_t(in.gcount()) != size || std::memcmp(block.data(), oracle.data()+at, size))
            throw std::runtime_error("packet output differs in block starting " + std::to_string(at));
    }
    if (in.peek() != std::char_traits<char>::eof()) throw std::runtime_error("trailing packet outputs");
}

void run(const std::string& out_dir, int depth, std::ostream& csv, bool icache_probe = false) {
    std::vector<Descriptor> in;
    std::vector<Output> expected;
    fixture(depth, in, expected);
    Descriptor* device = nullptr;
    size_t n = size_t(owners_per_chunk) * depth;
    CUDA(cudaMalloc(&device, in.size()*sizeof(Descriptor)));
    CUDA(cudaMemcpy(device, in.data(), in.size()*sizeof(Descriptor), cudaMemcpyHostToDevice));
    Slot slots[2];
    cudaStream_t compute, copy;
    CUDA(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
    CUDA(cudaStreamCreateWithFlags(&copy, cudaStreamNonBlocking));
    for (Slot& s : slots) {
        CUDA(cudaMalloc(&s.prepared, n*sizeof(Ready)));
        CUDA(cudaMalloc(&s.device, n*sizeof(Output)));
        CUDA(cudaMallocHost(&s.host, n*sizeof(Output)));
        CUDA(cudaEventCreateWithFlags(&s.produced, cudaEventDisableTiming));
        CUDA(cudaEventCreateWithFlags(&s.copied, cudaEventDisableTiming));
    }
    std::string path = out_dir + "/records-" + std::to_string(depth) + ".bin";
    for (int repetition = -1; repetition < (icache_probe ? 1 : 5); ++repetition) {
        for (int j = 0; j < 8; ++j) {
            int v = repetition < 0 ? j : ((repetition % 2 ? 7-j : j) + repetition) % 8;
            if (icache_probe && v != 1) continue; // fused, cooperative, synchronous output
            double wall = measure(device, slots, compute, copy, depth, v, path);
            compare_file(path, expected);
            if (repetition >= 0)
                csv << depth << ',' << repetition << ',' << v << ',' << in.size() << ','
                    << std::setprecision(12) << wall << '\n';
        }
    }
    for (Slot& s : slots) {
        CUDA(cudaEventDestroy(s.produced)); CUDA(cudaEventDestroy(s.copied));
        CUDA(cudaFreeHost(s.host)); CUDA(cudaFree(s.device)); CUDA(cudaFree(s.prepared));
    }
    CUDA(cudaStreamDestroy(copy)); CUDA(cudaStreamDestroy(compute)); CUDA(cudaFree(device));
}
} // namespace

int main(int argc, char** argv) {
    try {
        bool icache_probe = argc == 3 && std::string(argv[2]) == "--icache";
        if (argc != 2 && !icache_probe) throw std::invalid_argument("usage: rc-stage-pipeline OUTDIR [--icache]");
        CUDA(cudaSetDevice(0));
        std::ofstream csv;
        csv.exceptions(std::ios::failbit | std::ios::badbit);
        csv.open(std::string(argv[1]) + "/samples.csv");
        csv << "depth,repeat,variant,records,wall_seconds\n";
        if (icache_probe) run(argv[1], 1024, csv, true);
        else for (int depth : {32, 1024}) run(argv[1], depth, csv);
        csv.close();
    } catch (const std::exception& err) {
        std::cerr << err.what() << '\n';
        return 1;
    }
}
