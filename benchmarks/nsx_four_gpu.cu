// Original fixed fixture for conservative cross-device event exchange.
#include <cuda_runtime.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int kOwners = 4;
constexpr int kPacketsPerSource = 48;
constexpr int kPackets = kOwners * kPacketsPerSource;
constexpr int kHops = 4;
constexpr int kEvents = kPackets * kHops;
constexpr int64_t kWindow = 10;
constexpr int kRoundLimit = 512;
using Clock = std::chrono::steady_clock;

struct Event {
    int64_t time;
    int packet;
    int hop;
    int owner;
    int bytes;
};

struct Commit {
    Event event;
    int64_t departure;
};

__host__ __device__ bool earlier(Event a, Event b) {
    if (a.time != b.time) return a.time < b.time;
    if (a.packet != b.packet) return a.packet < b.packet;
    return a.hop < b.hop;
}

__host__ __device__ int64_t serve(Event event, int64_t free_at) {
    return (event.time > free_at ? event.time : free_at) + 1 + event.bytes / 512;
}

__host__ __device__ Event forward(Event event, int64_t departure) {
    const int destination = event.hop == 1 ? event.owner : (event.owner + 1 + event.packet % 3) % kOwners;
    return {departure + kWindow + event.packet % 5, event.packet, event.hop + 1, destination, event.bytes};
}

Event seed(int owner, int index) {
    const int packet = owner * kPacketsPerSource + index;
    // Four simultaneous sources and clustered arrivals create ties and future work.
    return {6 * (index % 8), packet, 0, owner, 64 + 64 * (packet % 16)};
}

using Ledger = std::array<std::vector<Commit>, kOwners>;
using PairCounts = std::array<std::array<int, kOwners>, kOwners>;

struct Reference {
    Ledger ledger;
    PairCounts sent{};
    int equal_time_pairs = 0;
};

Reference serial_reference() {
    const auto reverse = [](Event a, Event b) { return earlier(b, a); };
    std::priority_queue<Event, std::vector<Event>, decltype(reverse)> queue(reverse);
    for (int owner = 0; owner < kOwners; ++owner) {
        for (int i = 0; i < kPacketsPerSource; ++i) queue.push(seed(owner, i));
    }
    std::array<int64_t, kOwners> free_at{};
    Reference result;
    while (!queue.empty()) {
        const Event event = queue.top();
        queue.pop();
        auto& owner = result.ledger[event.owner];
        if (!owner.empty() && owner.back().event.time == event.time) ++result.equal_time_pairs;
        const int64_t departure = serve(event, free_at[event.owner]);
        free_at[event.owner] = departure;
        owner.push_back({event, departure});
        if (event.hop + 1 < kHops) {
            const Event next = forward(event, departure);
            ++result.sent[event.owner][next.owner];
            queue.push(next);
        }
    }
    return result;
}

struct Counts {
    int pending;
    int committed;
    int sent[kOwners];
    int error;
};

struct State {
    Counts counts;
    int64_t free_at;
    Event pending[kPackets];
    Commit ledger[kEvents];
};

__global__ void process_window(State* state, const Event* input, Event* outgoing,
                               int owner, int received, int64_t begin) {
    // Each owner exclusively commits its output state. No host code orders events.
    Counts& counts = state->counts;
    for (int destination = 0; destination < kOwners; ++destination) counts.sent[destination] = 0;
    if (received > kPackets - counts.pending) {
        counts.error = 1;
        return;
    }
    for (int i = 0; i < received; ++i) {
        if (input[i].owner != owner || input[i].time < begin) {
            counts.error = 2;
            return;
        }
        state->pending[counts.pending++] = input[i];
    }
    for (;;) {
        int first = -1;
        for (int i = 0; i < counts.pending; ++i) {
            if (first < 0 || earlier(state->pending[i], state->pending[first])) first = i;
        }
        if (first < 0 || state->pending[first].time >= begin + kWindow) return;
        const Event event = state->pending[first];
        state->pending[first] = state->pending[--counts.pending];
        if (event.time < begin || counts.committed == kEvents) {
            counts.error = 3;
            return;
        }
        const int64_t departure = serve(event, state->free_at);
        state->free_at = departure;
        state->ledger[counts.committed++] = {event, departure};
        if (event.hop + 1 < kHops) {
            const Event next = forward(event, departure);
            const int slot = counts.sent[next.owner]++;
            if (next.time < begin + kWindow || slot >= kPackets) {
                counts.error = 4;
                return;
            }
            outgoing[next.owner * kPackets + slot] = next;
        }
    }
}

void check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}

struct DeviceOwner {
    explicit DeviceOwner(int index) : device(index) {}
    int device;
    State* state = nullptr;
    Event* input = nullptr;
    Event* outgoing = nullptr;
    cudaStream_t stream = nullptr;
    int received = 0;
    Counts counts{};

    void select() const { check(cudaSetDevice(device), "cudaSetDevice"); }

    void initialize(int owner) {
        select();
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
        check(cudaMalloc(&state, sizeof(State)), "cudaMalloc state");
        check(cudaMalloc(&input, kPackets * sizeof(Event)), "cudaMalloc input");
        check(cudaMalloc(&outgoing, kOwners * kPackets * sizeof(Event)), "cudaMalloc outgoing");
        State initial{};
        initial.counts.pending = kPacketsPerSource;
        for (int i = 0; i < kPacketsPerSource; ++i) initial.pending[i] = seed(owner, i);
        check(cudaMemcpyAsync(state, &initial, sizeof(initial), cudaMemcpyHostToDevice, stream), "upload owner state");
        sync();
    }

    void sync() const {
        select();
        check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    }

    ~DeviceOwner() {
        cudaSetDevice(device);
        if (stream) cudaStreamSynchronize(stream);
        if (outgoing) cudaFree(outgoing);
        if (input) cudaFree(input);
        if (state) cudaFree(state);
        if (stream) cudaStreamDestroy(stream);
    }
};

struct StagingBuffer {
    Event* data = nullptr;
    ~StagingBuffer() { if (data) cudaFreeHost(data); }
};

enum class Transport { local, staging, peer };

void transfer(DeviceOwner& from, DeviceOwner& to, const Event* send, int count,
              Transport mode, StagingBuffer& staging) {
    Event* receive = to.input + to.received;
    const size_t bytes = count * sizeof(Event);
    if (from.device == to.device) {
        to.select();
        check(cudaMemcpyAsync(receive, send, bytes, cudaMemcpyDeviceToDevice, to.stream), "local event batch");
    } else if (mode == Transport::staging) {
        from.select();
        check(cudaMemcpyAsync(staging.data, send, bytes, cudaMemcpyDeviceToHost, from.stream), "staged event read");
        from.sync();
        to.select();
        check(cudaMemcpyAsync(receive, staging.data, bytes, cudaMemcpyHostToDevice, to.stream), "staged event write");
        to.sync();  // The next pair may reuse the pinned host buffer.
    } else {
        to.select();
        check(cudaMemcpyPeerAsync(receive, to.device, send, from.device, bytes, to.stream), "peer event batch");
    }
    to.received += count;
}

std::string uuid_string(const cudaUUID_t& uuid) {
    std::ostringstream out;
    out << "GPU-" << std::hex << std::setfill('0');
    for (int i = 0; i < 16; ++i) {
        if (i == 4 || i == 6 || i == 8 || i == 10) out << '-';
        out << std::setw(2) << static_cast<unsigned>(static_cast<unsigned char>(uuid.bytes[i]));
    }
    return out.str();
}

void enable_peer_access() {
    for (int receiver = 0; receiver < kOwners; ++receiver) {
        check(cudaSetDevice(receiver), "select peer receiver");
        for (int sender = 0; sender < kOwners; ++sender) {
            if (sender == receiver) continue;
            int supported = 0;
            check(cudaDeviceCanAccessPeer(&supported, receiver, sender), "query directional peer access");
            if (!supported) throw std::runtime_error("peer access unavailable from GPU " + std::to_string(receiver)
                                                      + " to GPU " + std::to_string(sender));
            const cudaError_t status = cudaDeviceEnablePeerAccess(sender, 0);
            if (status == cudaErrorPeerAccessAlreadyEnabled) (void)cudaGetLastError();
            else check(status, "enable peer access");
        }
    }
}

struct Run {
    Ledger ledger;
    PairCounts exchanged{};
    int rounds = 0;
    int dropped = 0;
    int remaining = 0;
    int cross_batches = 0;
    int64_t cross_bytes = 0;
    double setup_seconds = 0;
    double loop_seconds = 0;
    double processing_and_control_seconds = 0;
    double exchange_seconds = 0;
};

double elapsed(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double>(end - start).count();
}

Run simulate(Transport mode, bool drop_one = false) {
    const auto start = Clock::now();
    StagingBuffer staging;
    std::array<std::unique_ptr<DeviceOwner>, kOwners> owners;
    for (int owner = 0; owner < kOwners; ++owner) {
        owners[owner] = std::make_unique<DeviceOwner>(mode == Transport::local ? 0 : owner);
        owners[owner]->initialize(owner);
    }
    if (mode == Transport::staging) {
        check(cudaMallocHost(reinterpret_cast<void**>(&staging.data), kPackets * sizeof(Event)), "allocate pinned staging");
    }

    Run result;
    const auto loop_start = Clock::now();
    result.setup_seconds = elapsed(start, loop_start);
    while (result.rounds < kRoundLimit) {
        const auto processing_start = Clock::now();
        const int64_t begin = result.rounds * kWindow;
        for (int owner = 0; owner < kOwners; ++owner) {
            DeviceOwner& state = *owners[owner];
            state.select();
            process_window<<<1, 1, 0, state.stream>>>(state.state, state.input, state.outgoing, owner,
                                                       state.received, begin);
            check(cudaGetLastError(), "launch process_window");
            state.received = 0;
        }
        result.remaining = 0;
        for (auto& owner : owners) {
            owner->sync();
            check(cudaMemcpy(&owner->counts, owner->state, sizeof(Counts), cudaMemcpyDeviceToHost), "read window counts");
            if (owner->counts.error) {
                throw std::runtime_error("device event invariant failed with code " + std::to_string(owner->counts.error));
            }
            result.remaining += owner->counts.pending;
        }

        const auto exchange_start = Clock::now();
        for (int src = 0; src < kOwners; ++src) {
            for (int dst = 0; dst < kOwners; ++dst) {
                DeviceOwner& source = *owners[src];
                DeviceOwner& destination = *owners[dst];
                int count = source.counts.sent[dst];
                if (drop_one && !result.dropped && source.device != destination.device && count > 0) {
                    --count;
                    ++result.dropped;
                }
                if (!count) continue;
                if (count > kPackets - destination.received) throw std::runtime_error("event inbox capacity exceeded");
                transfer(source, destination, source.outgoing + dst * kPackets, count, mode, staging);
                result.exchanged[src][dst] += count;
                result.remaining += count;
                if (source.device != destination.device) {
                    ++result.cross_batches;
                    result.cross_bytes += count * sizeof(Event);
                }
            }
        }
        for (auto& owner : owners) owner->sync();
        result.processing_and_control_seconds += elapsed(processing_start, exchange_start);
        result.exchange_seconds += elapsed(exchange_start, Clock::now());
        ++result.rounds;
        if (!result.remaining) break;
    }
    result.loop_seconds = elapsed(loop_start, Clock::now());
    for (int owner = 0; owner < kOwners; ++owner) {
        const DeviceOwner& state = *owners[owner];
        state.select();
        auto& ledger = result.ledger[owner];
        ledger.resize(state.counts.committed);
        if (!ledger.empty()) {
            const auto* device_ledger = reinterpret_cast<const char*>(state.state) + offsetof(State, ledger);
            check(cudaMemcpy(ledger.data(), device_ledger, ledger.size() * sizeof(Commit), cudaMemcpyDeviceToHost),
                  "read committed event ledger");
        }
    }
    return result;
}

int total(const Ledger& ledger) {
    int count = 0;
    for (const auto& owner : ledger) count += owner.size();
    return count;
}

bool matches(const Ledger& actual, const Ledger& expected) {
    for (int owner = 0; owner < kOwners; ++owner) {
        if (actual[owner].size() != expected[owner].size()) return false;
        for (size_t i = 0; i < actual[owner].size(); ++i) {
            const auto& a = actual[owner][i];
            const auto& b = expected[owner][i];
            if (earlier(a.event, b.event) || earlier(b.event, a.event) || a.event.owner != b.event.owner
                || a.event.bytes != b.event.bytes || a.departure != b.departure) return false;
        }
    }
    return true;
}

void write_ledger(const std::string& path, const Ledger& ledger) {
    std::ofstream out(path);
    out.exceptions(std::ios::badbit | std::ios::failbit);
    out << "owner,arrival_ns,packet,hop,bytes,departure_ns\n";
    for (const auto& group : ledger) {
        for (const auto& commit : group) {
            const auto event = commit.event;
            out << event.owner << ',' << event.time << ',' << event.packet << ',' << event.hop
                << ',' << event.bytes << ',' << commit.departure << '\n';
        }
    }
    out.close();
}

void write_run(std::ostream& out, const Run& run, const Reference& reference) {
    out << std::boolalpha << "{\"drained\":" << (run.remaining == 0)
        << ",\"remaining_events\":" << run.remaining << ",\"rounds\":" << run.rounds
        << ",\"committed_events\":" << total(run.ledger) << ",\"injected_drops\":" << run.dropped
        << ",\"ledger_match\":" << matches(run.ledger, reference.ledger)
        << ",\"exchange_counts_match\":" << (run.exchanged == reference.sent)
        << ",\"cross_batches\":" << run.cross_batches << ",\"cross_bytes\":" << run.cross_bytes
        << ",\"setup_seconds\":" << run.setup_seconds << ",\"loop_seconds\":" << run.loop_seconds
        << ",\"processing_and_control_seconds\":" << run.processing_and_control_seconds
        << ",\"exchange_seconds\":" << run.exchange_seconds << ",\"events_by_source_and_destination\":[";
    for (int src = 0; src < kOwners; ++src) {
        out << (src ? ",[" : "[");
        for (int dst = 0; dst < kOwners; ++dst) {
            if (dst) out << ',';
            out << run.exchanged[src][dst];
        }
        out << ']';
    }
    out << "]}";
}

Run record_run(const std::string& directory, const std::string& name, const Reference& reference,
               Transport mode, bool drop_one = false) {
    Run run = simulate(mode, drop_one);
    write_ledger(directory + "/" + name + ".csv", run.ledger);
    std::cout << name << ' ';
    write_run(std::cout, run, reference);
    std::cout << std::endl;
    return run;
}
}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("usage: nsx-four-gpu OUTPUT_DIR");
        const std::string output = argv[1];
        int visible = 0;
        check(cudaGetDeviceCount(&visible), "cudaGetDeviceCount");
        if (visible != kOwners) throw std::runtime_error("fixture requires exactly four selected CUDA devices");
        std::array<std::string, kOwners> uuids;
        for (int i = 0; i < kOwners; ++i) {
            cudaDeviceProp properties{};
            check(cudaGetDeviceProperties(&properties, i), "cudaGetDeviceProperties");
            uuids[i] = uuid_string(properties.uuid);
        }

        const Reference reference = serial_reference();
        write_ledger(output + "/serial.csv", reference.ledger);
        const Run single = record_run(output, "single-gpu", reference, Transport::local);
        const Run staged = record_run(output, "four-gpu-staging", reference, Transport::staging);
        const Run dropped = record_run(output, "negative-drop", reference, Transport::staging, true);
        enable_peer_access();
        const Run peer = record_run(output, "four-gpu-p2p", reference, Transport::peer);

        std::ofstream report(output + "/event-exchange.json");
        report.exceptions(std::ios::badbit | std::ios::failbit);
        report << std::setprecision(12) << "{\n\"gpu_uuids\":[";
        for (int i = 0; i < kOwners; ++i) report << (i ? ",\"" : "\"") << uuids[i] << '"';
        report << "],\n\"window_ns\":" << kWindow << ",\"packet_count\":" << kPackets
               << ",\"events_per_packet\":" << kHops << ",\"event_record_bytes\":" << sizeof(Event)
               << ",\"reference_committed_events\":" << total(reference.ledger)
               << ",\"reference_equal_time_pairs\":" << reference.equal_time_pairs << ",\n\"runs\":{\n\"single-gpu\":";
        write_run(report, single, reference);
        report << ",\n\"four-gpu-staging\":";
        write_run(report, staged, reference);
        report << ",\n\"four-gpu-p2p\":";
        write_run(report, peer, reference);
        report << ",\n\"negative-drop\":";
        write_run(report, dropped, reference);
        report << "\n}}\n";
        report.close();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
