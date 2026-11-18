// Strict-frontier scheduling of staged, collective-shaped fan-in streams on one GPU.
// Each independent owner serializes two ingress streams onto one FIFO output port.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
#ifdef __CUDACC__
#include <cuda_runtime.h>
#define HD __host__ __device__
#else
#define HD
#endif

namespace {
using U64 = std::uint64_t;
constexpr int waves = 12;
#ifdef __CUDACC__
constexpr int lanes = 32;
#endif
constexpr unsigned infinity = ~0u;
struct Frame { unsigned time, bytes; };
struct Source { int offset, length; };
struct Publication { int end[2]; unsigned met; };
struct State { int pos[2]; U64 finish, bytes, hash; };
struct Mark { int worker, events; };

void check(bool ok, const char* detail) {
    if (!ok) throw std::runtime_error(detail);
}
HD void commit(State& s, Frame frame, int input, unsigned index) {
    U64 start = s.finish < frame.time ? frame.time : s.finish;
    s.finish = start + (frame.bytes + 49) / 50; // 400 Gb/s
    s.bytes += frame.bytes;
    U64 identity = (static_cast<U64>(index) << 1) | static_cast<unsigned>(input);
    s.hash = (s.hash ^ identity) * 1099511628211ULL;
    s.hash = (s.hash ^ s.finish) * 1099511628211ULL;
}

// Each input FIFO is ordered. Equal-time events sort by input index then sequence.
// No new inbound packet can arrive before the source's published MET. Departures
// affect this owner only and must retain the previous packet's link finish time.
HD int consume(const Frame* frames, const Source* inputs, const Publication& p,
               unsigned frontier, bool inclusive, State& s) {
    int before = s.pos[0] + s.pos[1];
    for (;;) {
        unsigned t[2];
        for (int x = 0; x < 2; ++x) {
            t[x] = s.pos[x] < p.end[x] ? frames[inputs[x].offset + s.pos[x]].time : infinity;
        }
        int x = t[1] < t[0] ? 1 : 0;
        if (t[x] == infinity || t[x] > frontier || (!inclusive && t[x] == frontier)) break;
        int index = s.pos[x]++;
        commit(s, frames[inputs[x].offset + index], x, index);
    }
    return s.pos[0] + s.pos[1] - before;
}
bool equal(const State& a, const State& b) {
    return a.pos[0] == b.pos[0] && a.pos[1] == b.pos[1] && a.hash == b.hash &&
           a.bytes == b.bytes && a.finish == b.finish;
}

struct Fixture {
    int owners;
    std::vector<Frame> frames;
    std::vector<Source> sources;
    std::vector<Publication> publications;
    std::vector<State> expected;
    std::vector<unsigned> global_met;
    Fixture(int count, bool multi)
        : owners(count), sources(2 * count), publications(waves * count),
          expected(count), global_met(waves, infinity) {
        int extra = count * 8;
        for (int owner = 0; owner < count; ++owner) {
            // Identical total event count, concentrated on 64 owners or one owner.
            int hot = multi ? (owner < 64 ? extra / 64 : 0) : (owner == 0 ? extra : 0);
            std::vector<std::tuple<unsigned,int,int>> full;
            for (int x = 0; x < 2; ++x) {
                auto& source = sources[2 * owner + x];
                source = {static_cast<int>(frames.size()), 24 + hot};
                for (int seq = 0; seq < source.length; ++seq) {
                    unsigned arrival = static_cast<unsigned>((owner % 127) * 37 + (seq / 2) * 16);
                    unsigned bytes = static_cast<unsigned>(64 * (1 + (owner + seq + 3 * x) % 16));
                    frames.push_back({arrival, bytes});
                    full.emplace_back(arrival, x, seq);
                }
            }
            // Independent reference: all arrivals are known, sorted by the full key.
            std::sort(full.begin(), full.end());
            State& ref = expected[owner];
            for (const auto& [arrival,x,seq] : full) {
                const auto& frame = frames[sources[2 * owner + x].offset + seq];
                U64 start = std::max<U64>(arrival, ref.finish);
                ref.finish = start + (frame.bytes + 49) / 50;
                ref.hash = (ref.hash ^ (2ULL * seq + x)) * 1099511628211ULL;
                ref.hash = (ref.hash ^ ref.finish) * 1099511628211ULL;
                ref.bytes += frame.bytes;
                ++ref.pos[x];
            }
            for (int wave = 0; wave < waves; ++wave) {
                auto& p = publications[wave * count + owner];
                p.met = infinity;
                for (int x = 0; x < 2; ++x) {
                    auto src = sources[2 * owner + x];
                    int visible = (wave + 1) * src.length / waves;
                    // Input 0 of owner 0 publishes no packet in the first round;
                    // input 1 reveals a packet tied with its undisclosed queue head.
                    if (owner == 0 && x == 0 && wave == 0) visible = 0;
                    p.end[x] = visible;
                    if (visible < src.length)
                        p.met = std::min(p.met, frames[src.offset + visible].time);
                }
                global_met[wave] = std::min(global_met[wave], p.met);
            }
        }
    }
    int prefix(int owner, int wave, bool local) const {
        const auto& p = publications[wave * owners + owner];
        unsigned bound = local ? p.met : global_met[wave];
        int result = 0;
        for (int x = 0; x < 2; ++x) {
            auto begin = frames.begin() + sources[2 * owner + x].offset;
            auto end = std::lower_bound(begin, begin + p.end[x], bound,
                                        [](Frame f, unsigned time) { return f.time < time; });
            result += static_cast<int>(end - begin);
        }
        return result;
    }
};

void verify(const Fixture& f, const std::vector<State>& state, bool probe) {
    int mismatches = 0;
    for (int i = 0; i < f.owners; ++i) {
        if (!equal(state[i], f.expected[i])) ++mismatches;
    }
    if (probe) check(mismatches && !equal(state[0], f.expected[0]), "unsafe equal-MET control escaped oracle");
    else check(!mismatches, "per-owner digest/finish/bytes/order diverged");
}

void host_tests() {
    for (bool multi : {false,true}) {
        Fixture f(256, multi);
        for (int mode = 0; mode < 3; ++mode) {
            bool local = mode != 0, probe = mode == 2;
            std::vector<State> s(f.owners);
            for (int wave = 0; wave < waves; ++wave) {
                for (int i = 0; i < f.owners; ++i) {
                    const auto& p = f.publications[wave * f.owners + i];
                    consume(f.frames.data(), f.sources.data() + i * 2, p,
                            local ? p.met : f.global_met[wave], probe, s[i]);
                    if (!probe) check(s[i].pos[0] + s[i].pos[1] == f.prefix(i, wave, local),
                                      "CPU published prefix mismatch");
                }
            }
            verify(f, s, probe);
        }
    }
}

#ifdef __CUDACC__
void cuda_check(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
template<class T> struct Buffer {
    T* p = nullptr;
    explicit Buffer(size_t n) { cuda_check(cudaMalloc(&p, n * sizeof(T))); }
    ~Buffer() { cudaFree(p); }
    void put(const std::vector<T>& values) {
        cuda_check(cudaMemcpy(p, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
    void get(std::vector<T>& values) {
        cuda_check(cudaMemcpy(values.data(), p, values.size() * sizeof(T), cudaMemcpyDeviceToHost));
    }
};

__global__ void reduce_met(const Publication* publications, int count, unsigned* met) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned bound = i < count ? publications[i].met : infinity;
    for (int delta = 16; delta; delta /= 2)
        bound = min(bound, __shfl_down_sync(0xffffffffu, bound, delta));
    if (!(threadIdx.x % lanes)) atomicMin(met, bound);
}

// Compact tiles with any causally ready owner. Discovery and atomic queue
// construction are charged to the dynamic variant's measured resident loop.
template<bool local>
__global__ void discover(const Frame* frames, const Source* sources, const Publication* pubs,
                         const unsigned* global, int count, const State* states,
                         int* tiles, int* num_tiles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    bool ready = false;
    if (i < count) {
        Publication p = pubs[i];
        unsigned frontier = local ? p.met : *global;
        for (int x = 0; x < 2; ++x) {
            int pos = states[i].pos[x];
            if (pos < p.end[x] && frames[sources[2*i+x].offset + pos].time < frontier) ready = true;
        }
    }
    if (__any_sync(0xffffffffu, ready) && threadIdx.x % lanes == 0)
        tiles[atomicAdd(num_tiles, 1)] = i;
}

// One persistent warp-sized CTA services a tile of 32 owners, one lane per owner.
// Static mapping cycles over tiles. Dynamic CTAs claim the next unprocessed tile.
template<bool local, bool dynamic, bool diagnostic>
__global__ void process(const Frame* frames, const Source* sources, const Publication* pubs,
                        const unsigned* global, int count, int* cursor, const int* tiles, const int* num_tiles, State* states,
                        Mark* marks, bool probe) {
    int first = blockIdx.x * lanes;
    for (;;) {
        if (dynamic) {
            int claim = threadIdx.x == 0 ? atomicAdd(cursor, 1) : 0;
            claim = __shfl_sync(0xffffffffu, claim, 0);
            if (claim >= *num_tiles) return;
            first = tiles[claim];
        }
        if (first >= count) return;
        int i = first + threadIdx.x;
        if (i < count) {
            Publication p = pubs[i];
            int done = consume(frames, sources + i * 2, p, local ? p.met : *global, probe, states[i]);
            if (diagnostic) marks[i] = {static_cast<int>(blockIdx.x), done};
        }
        first += gridDim.x * lanes;
    }
}

struct Device {
    Buffer<Frame> frames;
    Buffer<Source> sources;
    Buffer<Publication> pubs;
    Buffer<State> state;
    Buffer<Mark> marks;
    Buffer<unsigned> global;
    Buffer<int> cursor, tiles, num_tiles;
    explicit Device(const Fixture& f)
        : frames(f.frames.size()), sources(f.sources.size()), pubs(f.publications.size()),
          state(f.owners), marks(f.owners * waves), global(1), cursor(1), tiles((f.owners + lanes - 1)/lanes), num_tiles(1) {
        frames.put(f.frames); sources.put(f.sources); pubs.put(f.publications);
    }
};
template<bool local, bool dynamic, bool diagnostic>
double run(const Fixture& f, Device& d, int workers, int variant, bool probe, std::ofstream& progress) {
    cuda_check(cudaMemset(d.state.p, 0, f.owners * sizeof(State)));
    if (diagnostic) cuda_check(cudaMemset(d.marks.p, 0, waves * f.owners * sizeof(Mark)));
    cuda_check(cudaDeviceSynchronize());
    auto begin = std::chrono::steady_clock::now();
    for (int wave = 0; wave < waves; ++wave) {
        auto pubs = d.pubs.p + wave * f.owners;
        if (!local) {
            cuda_check(cudaMemsetAsync(d.global.p, 0xff, sizeof(unsigned)));
            reduce_met<<<(f.owners + 127) / 128, 128>>>(pubs, f.owners, d.global.p);
            cuda_check(cudaGetLastError());
        }
        if (dynamic) {
            cuda_check(cudaMemsetAsync(d.cursor.p, 0, sizeof(int)));
            cuda_check(cudaMemsetAsync(d.num_tiles.p, 0, sizeof(int)));
            discover<local><<<(f.owners + 127)/128,128>>>(d.frames.p, d.sources.p, pubs, d.global.p,
                f.owners, d.state.p, d.tiles.p, d.num_tiles.p);
            cuda_check(cudaGetLastError());
        }
        process<local,dynamic,diagnostic><<<workers,lanes>>>(d.frames.p, d.sources.p, pubs,
            d.global.p, f.owners, d.cursor.p, d.tiles.p, d.num_tiles.p, d.state.p, d.marks.p + wave * f.owners, probe);
        cuda_check(cudaGetLastError());
    }
    cuda_check(cudaDeviceSynchronize());
    auto end = std::chrono::steady_clock::now();

    std::vector<State> state(f.owners);
    d.state.get(state);
    verify(f, state, probe);
    if (diagnostic && !probe) {
        std::vector<Mark> marks(waves * f.owners);
        std::vector<int> prefix(f.owners, 0);
        d.marks.get(marks);
        for (int w = 0; w < waves; ++w) {
            int total = 0, peak = 0, active = 0;
            std::vector<int> work(workers, 0);
            for (int i = 0; i < f.owners; ++i) {
                auto m = marks[w * f.owners + i];
                check(m.worker >= 0 && m.worker < workers && m.events >= 0, "bad CTA sample");
                prefix[i] += m.events;
                check(prefix[i] == f.prefix(i, w, local), "per-owner safe prefix mismatch");
                work[m.worker] += m.events;
                peak = std::max(peak, m.events);
                active += m.events > 0;
                total += m.events;
            }
            progress << variant << ',' << w << ',' << total << ',' << peak << ',' << active << ','
                     << *std::max_element(work.begin(), work.end()) << ','
                     << std::count_if(work.begin(), work.end(), [](int n){return n > 0;}) << '\n';
        }
    }
    return std::chrono::duration<double,std::micro>(end-begin).count();
}

template<bool diagnostic>
double variant_run(int v, const Fixture& f, Device& d, int workers, std::ofstream& stream) {
    switch (v) {
    case 0: return run<false,false,diagnostic>(f,d,workers,v,false,stream);
    case 1: return run<false,true,diagnostic>(f,d,workers,v,false,stream);
    case 2: return run<true,false,diagnostic>(f,d,workers,v,false,stream);
    default: return run<true,true,diagnostic>(f,d,workers,v,false,stream);
    }
}

void sweep(const std::string& output) {
    cudaDeviceProp info{};
    cuda_check(cudaGetDeviceProperties(&info, 0));
    int workers = std::min(512, 2 * info.multiProcessorCount);
    for (bool multi : {false,true}) {
        std::string label = multi ? "multi-hot" : "single-hot";
        Fixture f(32768, multi);
        Device d(f);
        std::ofstream progress, timings;
        progress.exceptions(std::ios::badbit | std::ios::failbit);
        timings.exceptions(std::ios::badbit | std::ios::failbit);
        progress.open(output + "/" + label + "-progress.csv");
        timings.open(output + "/" + label + "-timings.csv");
        progress << "variant,wave,events,max_owner,active_owners,max_cta,active_ctas\n";
        timings << "repetition,order,variant,owners,workers,events,resident_loop_us\n";
        run<true,false,false>(f,d,workers,2,true,progress);
        for (int v = 0; v < 4; ++v) {
            variant_run<true>(v,f,d,workers,progress);
            variant_run<false>(v,f,d,workers,progress); // one ordinary warmup per variant
        }
        for (int rep = 0; rep < 5; ++rep) {
            for (int order = 0; order < 4; ++order) {
                int v = (rep + order) % 4;
                double us = variant_run<false>(v,f,d,workers,progress);
                timings << rep << ',' << order << ',' << v << ',' << f.owners << ',' << workers
                        << ',' << f.frames.size() << ',' << us << '\n';
            }
        }
        progress.close();
        timings.close();
    }
}
#endif
} // namespace

int main(int argc, char** argv) {
    try {
        check(argc == 2, "usage: training-owner-sweep --self-test|OUTPUT_DIR");
        if (std::string(argv[1]) == "--self-test") host_tests();
#ifdef __CUDACC__
        else sweep(argv[1]);
#else
        else throw std::runtime_error("CUDA required for sweep");
#endif
        std::cout << "owner sweep gates passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
