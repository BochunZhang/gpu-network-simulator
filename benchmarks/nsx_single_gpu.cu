#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef NSX_CPU_ONLY
#define DEVICE_CODE
#else
#include <cuda_runtime.h>
#define DEVICE_CODE __host__ __device__
#endif

namespace {
constexpr int kSources = 4;
constexpr int kBranches = 2;
constexpr int kOwners = 2 * kBranches;
constexpr int kPacketsPerSource = 2;
constexpr int kWaves = 3;
constexpr int kCapacity = kSources * kPacketsPerSource;
#ifndef NSX_CPU_ONLY
constexpr int kInfinity = 1 << 30;
#endif

enum class Mode { global, local, inclusive_probe };

struct Packet {
    int time_ns;
    int source;
    int sequence;
    int bytes;
};

struct Injection {
    int wave;
    Packet packet;
};

constexpr Injection kTrace[kSources][kPacketsPerSource] = {
    {{1, {3, 0, 0, 200}}, {2, {5, 0, 1, 201}}},
    {{0, {5, 1, 0, 210}}, {0, {6, 1, 1, 211}}},
    {{0, {2, 2, 0, 220}}, {0, {4, 2, 1, 221}}},
    {{0, {2, 3, 0, 230}}, {0, {5, 3, 1, 231}}},
};

struct Queue {
    Packet packets[kCapacity];
    int next;
    int end;
};

struct State {
    Injection trace[kSources][kPacketsPerSource];
    Queue inputs[kSources];
    Queue links[kBranches];
    Queue ledger[kOwners];
    int input_met[kSources];
    int link_met[kBranches];
    int progress[kWaves][kOwners];
    int branch0_frontier[kWaves];
    int source0_before[kWaves];
    int source1_after[kWaves];
    bool overflow[kOwners];
};

DEVICE_CODE int link_delay_ns(int branch) {
    return branch == 0 ? 2 : 7;
}

DEVICE_CODE bool earlier(const Packet& a, const Packet& b) {
    if (a.time_ns != b.time_ns) return a.time_ns < b.time_ns;
    if (a.source != b.source) return a.source < b.source;
    return a.sequence < b.sequence;
}

State cpu_reference() {
    struct Event {
        int owner;
        Packet packet;
    };
    // Reference sees the complete input and selects the globally earliest event.
    // Each forwarding commit independently creates its dependent sink event.
    std::vector<Event> pending;
    for (const auto& input : kTrace) {
        for (const auto& entry : input) {
            pending.push_back({entry.packet.source / 2, entry.packet});
        }
    }
    State state{};
    while (!pending.empty()) {
        const auto first = std::min_element(pending.begin(), pending.end(), [](const Event& a, const Event& b) {
            return earlier(a.packet, b.packet);
        });
        const auto [owner, packet] = *first;
        pending.erase(first);
        auto& log = state.ledger[owner];
        log.packets[log.end++] = packet;
        if (owner < kBranches) {
            auto arrival = packet;
            arrival.time_ns += link_delay_ns(owner);
            pending.push_back({owner + kBranches, arrival});
        }
    }
    return state;
}

#ifndef NSX_CPU_ONLY
void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

__device__ void append(Queue& queue, Packet packet, bool& overflow) {
    if (queue.end == kCapacity) {
        overflow = true;
        return;
    }
    queue.packets[queue.end++] = packet;
}

__global__ void publish_sources(State* state, int wave) {
    const int source = threadIdx.x;
    Queue& queue = state->inputs[source];
    const auto* trace = state->trace[source];
    while (queue.end < kPacketsPerSource && trace[queue.end].wave <= wave) {
        queue.packets[queue.end] = trace[queue.end].packet;
        ++queue.end;
    }
    // The promise describes the earliest future publication, excluding all
    // packets already in the queue. It is published after writing the prefix.
    state->input_met[source] =
        queue.end == kPacketsPerSource ? kInfinity : trace[queue.end].packet.time_ns;
}

__global__ void forward_packets(State* state, int wave, Mode mode) {
    const int branch = threadIdx.x;
    const int begin = mode == Mode::global ? 0 : branch * 2;
    const int end = mode == Mode::global ? kSources : begin + 2;
    int frontier = kInfinity;
    for (int i = begin; i < end; ++i) {
        if (state->input_met[i] < frontier) frontier = state->input_met[i];
    }
    Queue& left = state->inputs[branch * 2];
    Queue& right = state->inputs[branch * 2 + 1];
    if (branch == 0) {
        state->branch0_frontier[wave] = frontier;
        state->source0_before[wave] = left.end - left.next;
    }
    while (left.next < left.end || right.next < right.end) {
        const bool choose_left = right.next == right.end ||
            (left.next < left.end && earlier(left.packets[left.next], right.packets[right.next]));
        Queue& selected = choose_left ? left : right;
        const Packet packet = selected.packets[selected.next];
        // A producer can still publish another packet at the frontier.
        if (packet.time_ns > frontier ||
                (packet.time_ns == frontier && mode != Mode::inclusive_probe)) {
            break;
        }
        ++selected.next;
        append(state->ledger[branch], packet, state->overflow[branch]);
        Packet arrival = packet;
        arrival.time_ns += link_delay_ns(branch);
        append(state->links[branch], arrival, state->overflow[branch]);
    }
    state->link_met[branch] = frontier == kInfinity ? kInfinity : frontier + link_delay_ns(branch);
    state->progress[wave][branch] = state->ledger[branch].end;
    if (branch == 0) state->source1_after[wave] = right.end - right.next;
}

__global__ void receive_packets(State* state, int wave) {
    const int branch = threadIdx.x;
    const int owner = branch + kBranches;
    Queue& link = state->links[branch];
    while (link.next < link.end && link.packets[link.next].time_ns < state->link_met[branch]) {
        append(state->ledger[owner], link.packets[link.next++], state->overflow[owner]);
    }
    state->progress[wave][owner] = state->ledger[owner].end;
}

State run_cuda(Mode mode, State* device) {
    State state{};
    for (int source = 0; source < kSources; ++source) {
        std::copy_n(kTrace[source], kPacketsPerSource, state.trace[source]);
    }
    check_cuda(cudaMemcpy(device, &state, sizeof(state), cudaMemcpyHostToDevice), "initialize state");
    for (int wave = 0; wave < kWaves; ++wave) {
        publish_sources<<<1, kSources>>>(device, wave);
        check_cuda(cudaGetLastError(), "publish_sources launch");
        forward_packets<<<1, kBranches>>>(device, wave, mode);
        check_cuda(cudaGetLastError(), "forward_packets launch");
        receive_packets<<<1, kBranches>>>(device, wave);
        check_cuda(cudaGetLastError(), "receive_packets launch");
    }
    check_cuda(cudaDeviceSynchronize(), "execute CUDA waves");
    check_cuda(cudaMemcpy(&state, device, sizeof(state), cudaMemcpyDeviceToHost), "read terminal state");
    return state;
}
#endif

void write_ledger(const State& state, const std::string& path) {
    std::ofstream out(path);
    out.exceptions(std::ios::badbit | std::ios::failbit);
    out << "owner,index,time_ns,source,sequence,bytes\n";
    for (int owner = 0; owner < kOwners; ++owner) {
        const auto& log = state.ledger[owner];
        for (int index = 0; index < log.end; ++index) {
            const auto& packet = log.packets[index];
            out << owner << ',' << index << ',' << packet.time_ns << ',' << packet.source
                << ',' << packet.sequence << ',' << packet.bytes << '\n';
        }
    }
    out.close();
}

#ifndef NSX_CPU_ONLY
void write_progress(const State& state, const std::string& path) {
    std::ofstream out(path);
    out.exceptions(std::ios::badbit | std::ios::failbit);
    out << "wave,forward0,forward1,sink0,sink1,frontier0_ns,source0_before,source1_after\n";
    for (int wave = 0; wave < kWaves; ++wave) {
        out << wave;
        for (const int count : state.progress[wave]) out << ',' << count;
        out << ',' << state.branch0_frontier[wave] << ',' << state.source0_before[wave]
            << ',' << state.source1_after[wave] << '\n';
    }
    out.close();
}

void record_terminal(const State& state, const char* variant) {
    int sources_closed = 0;
    int links_closed = 0;
    int pending_inputs = 0;
    int pending_links = 0;
    int overflows = 0;
    for (int i = 0; i < kSources; ++i) {
        const auto& queue = state.inputs[i];
        sources_closed += state.input_met[i] == kInfinity && queue.end == kPacketsPerSource;
        pending_inputs += queue.end - queue.next;
    }
    for (int i = 0; i < kBranches; ++i) {
        links_closed += state.link_met[i] == kInfinity;
        pending_links += state.links[i].end - state.links[i].next;
    }
    for (const bool overflow : state.overflow) overflows += overflow;
    std::cout << "record=nsx_met variant=" << variant << " sources_closed=" << sources_closed
              << " links_closed=" << links_closed << " pending_inputs=" << pending_inputs
              << " pending_links=" << pending_links << " overflows=" << overflows << '\n';
}
#endif
}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("usage: nsx-single-gpu OUTPUT_DIRECTORY");
        const std::string output_dir = argv[1];
        write_ledger(cpu_reference(), output_dir + "/cpu.csv");
#ifndef NSX_CPU_ONLY
        int device_count = 0;
        check_cuda(cudaGetDeviceCount(&device_count), "enumerate CUDA-visible devices");
        if (device_count != 1) throw std::runtime_error("task expects exactly one CUDA-visible device");
        State* device = nullptr;
        check_cuda(cudaMalloc(&device, sizeof(State)), "allocate state");
        const Mode modes[] = {Mode::global, Mode::local, Mode::inclusive_probe};
        const char* variants[] = {"global", "local", "inclusive-probe"};
        for (int i = 0; i < 3; ++i) {
            const State state = run_cuda(modes[i], device);
            const std::string prefix = output_dir + "/" + variants[i];
            write_ledger(state, prefix + ".csv");
            write_progress(state, prefix + "-progress.csv");
            record_terminal(state, variants[i]);
        }
        check_cuda(cudaFree(device), "release state");
#endif
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "NSX-inspired single-GPU fixture: " << error.what() << '\n';
        return 1;
    }
}
