// Synthetic collective and CQE-dependency fixtures for the single-device RC mechanism.
#include "training_roce_engine.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <queue>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {
using training_roce::Engine;
using training_roce::Peer;
using training_roce::Stats;
using U64 = std::uint64_t;
constexpr U64 never = std::numeric_limits<U64>::max();

void require(bool pass, const std::string& message) {
    if (!pass) throw std::runtime_error(message);
}

struct Options {
    std::string backend = "cpu", workload = "closed-loop", fault = "none", trace, profile;
    int trace_limit = 8192;
    unsigned ranks = 4;
    U64 bytes = 8192;
};

U64 number(const std::string& value) {
    U64 result = 0;
    const auto end = value.data() + value.size();
    auto parsed = std::from_chars(value.data(), end, result);
    require(parsed.ec == std::errc{} && parsed.ptr == end, "invalid integer: " + value);
    return result;
}

Options options(int argc, char** argv) {
    Options opts;
    std::set<std::string> seen;
    for (int i = 1; i < argc; i += 2) {
        std::string key = argv[i];
        require(i + 1 < argc && seen.insert(key).second, "missing or repeated option: " + key);
        std::string value = argv[i + 1];
        if (key == "--backend") opts.backend = value;
        else if (key == "--workload") opts.workload = value;
        else if (key == "--fault") opts.fault = value;
        else if (key == "--trace") opts.trace = value;
        else if (key == "--profile") opts.profile = value;
        else if (key == "--trace-limit") {
            U64 limit = number(value);
            require(limit > 0 && limit <= std::numeric_limits<int>::max(), "invalid trace limit");
            opts.trace_limit = static_cast<int>(limit);
        }
        else if (key == "--bytes") opts.bytes = number(value);
        else if (key == "--ranks") {
            U64 ranks = number(value);
            require(ranks >= 2 && ranks <= 1024, "--ranks must be in [2,1024]");
            opts.ranks = static_cast<unsigned>(ranks);
        } else throw std::runtime_error("unknown option: " + key);
    }
    require(opts.backend == "cpu" || opts.backend == "cuda" || opts.backend == "compare",
            "--backend must be cpu, cuda or compare");
    require(opts.workload == "closed-loop" || opts.workload == "ring" ||
            opts.workload == "all-to-all", "--workload must be closed-loop, ring or all-to-all");
    require(opts.fault == "none" || opts.fault == "drop-data" || opts.fault == "drop-ack" ||
            opts.fault == "rnr", "--fault must be none, drop-data, drop-ack or rnr");
    require(opts.bytes > 0 && opts.bytes <= (U64{1} << 30), "--bytes must be in [1,2^30]");
    require(opts.workload != "closed-loop" || opts.ranks == 4, "closed-loop requires 4 ranks");
    require(!seen.count("--trace-limit") || !opts.profile.empty(), "--trace-limit requires --profile");
    require(opts.profile.empty() || opts.trace != opts.profile, "--trace and --profile need distinct paths");
    return opts;
}

enum class Kind { post, recv, send };
const char* name(Kind kind) {
    if (kind == Kind::post) return "post";
    return kind == Kind::recv ? "recv" : "send";
}

struct Dependency { unsigned id; Kind kind; };
struct Op {
    int src, dst;
    // In the ring this identifies the originating rank of the circulating chunk.
    int origin;
    U64 compute_ns;
    std::vector<Dependency> parents;
};

std::vector<Op> make_workload(const Options& opts) {
    if (opts.workload == "closed-loop") {
        return {{0, 1, -1, 0, {}}, {2, 3, -1, 0, {}},
                {1, 0, -1, 7, {{1, Kind::recv}}},
                {0, 1, -1, 31, {{1, Kind::send}}},
                {2, 3, -1, 31, {{2, Kind::send}}}};
    }
    const int n = static_cast<int>(opts.ranks);
    std::vector<Op> ops;
    ops.reserve(n * (n - 1));
    for (int round = 0; round < n - 1; ++round) {
        for (int rank = 0; rank < n; ++rank) {
            if (opts.workload == "all-to-all") {
                ops.push_back({rank, (rank + round + 1) % n, rank, 0, {}});
                continue;
            }
            Op op{rank, (rank + 1) % n, (rank + n - round) % n, 0, {}};
            if (round > 0) {
                unsigned prior_base = (round - 1) * n + 1;
                op.parents = {{prior_base + rank, Kind::send},
                              {prior_base + (rank + n - 1) % n, Kind::recv}};
            }
            ops.push_back(std::move(op));
        }
    }
    return ops;
}

using Event = std::tuple<Kind, unsigned, int, int, U64, U64>; // kind,id,src,dst,bytes,time_ns
struct Run {
    std::vector<Event> ledger;
    Stats stats{};
    std::vector<training_roce::Trace> profile;
    std::vector<Peer> profile_peers;
    bool profile_truncated = false;
    U64 training_end = 0, drain_end = 0, execution_us = 0;
    U64 digest = 14695981039346656037ULL, events = 0;
};
struct Times { U64 post = never, recv = never, send = never; };

// Pair 2->3 never shares a port with 0<->1. Its initial WR therefore has
// an independent closed-form reference for every packet and tail size.
// The two-hop store-and-forward star charges 64 B overhead at 50 B/ns;
// stop-and-wait sends the next segment only after both ACK hops finish.
void check_disjoint_pair(const Options& opts, const std::vector<Times>& times) {
    constexpr U64 mtu = 4096;
    auto trip_to_receiver = [](U64 bytes) { return 2 * (500 + (64 + bytes + 49) / 50); };
    const U64 ack_trip = trip_to_receiver(0);
    const U64 full_segments_before_last = (opts.bytes - 1) / mtu;
    const U64 tail = opts.bytes - full_segments_before_last * mtu;
    U64 recv = full_segments_before_last * (trip_to_receiver(mtu) + ack_trip)
               + trip_to_receiver(tail);
    require(times[1].recv == recv && times[1].send == recv + ack_trip,
            "independent port pair differs from two-hop serialization reference");
    U64 send = recv + ack_trip;
    if (opts.fault == "rnr") {
        U64 delay = trip_to_receiver(std::min(opts.bytes, mtu)) + ack_trip + 4000;
        recv += delay;
        send += delay;
    } else if (opts.fault == "drop-data" || opts.fault == "drop-ack") {
        // A lost last ACK prevents sender completion but does not undo receiver delivery.
        if (opts.fault == "drop-data" || full_segments_before_last > 0) recv += 100000;
        send += 100000;
    }
    require(times[0].recv == recv && times[0].send == send,
            "first WR differs from analytical fault/timeout reference");
}

Run run(const Options& opts, bool cuda) {
    const auto start = std::chrono::steady_clock::now();
    const auto ops = make_workload(opts);
    const unsigned count = static_cast<unsigned>(ops.size());
    std::set<std::pair<int,int>> pairs;
    for (const auto& op : ops) pairs.emplace(op.src, op.dst);
    std::vector<Peer> peers;
    for (const auto& pair : pairs) peers.push_back({pair.first, pair.second, 0});
    training_roce::Config config;
    config.ranks = static_cast<int>(opts.ranks);
    config.cuda = cuda;
    config.drop_first_data = opts.fault == "drop-data";
    config.drop_first_ack = opts.fault == "drop-ack";
    config.rnr_first = opts.fault == "rnr";
    if (!opts.profile.empty() && (cuda || opts.backend == "cpu")) config.trace_limit = opts.trace_limit;
    Engine engine(config, peers);

    // Frontend deadlines are a conservative bound: advance returns before a later
    // compute completion if an earlier local CQE exposes new work.
    using Release = std::pair<U64, unsigned>;
    std::priority_queue<Release, std::vector<Release>, std::greater<Release>> ready;
    std::vector<unsigned> pending(count);
    std::vector<std::array<std::vector<unsigned>,2>> children(count);
    std::vector<Times> stamps(count);
    for (unsigned i = 0; i < count; ++i) {
        const auto& op = ops[i];
        pending[i] = static_cast<unsigned>(op.parents.size());
        if (!pending[i]) ready.emplace(op.compute_ns, i);
        for (const auto& dep : op.parents) {
            require(dep.id > 0 && dep.id <= i, "dependency must name an earlier operation");
            const auto& parent = ops[dep.id - 1];
            int owner = dep.kind == Kind::recv ? parent.dst : parent.src;
            require(owner == op.src, "remote CQE cannot unblock this rank");
            children[dep.id - 1][dep.kind == Kind::recv].push_back(i);
        }
    }
    // Track logical chunk provenance without assigning fake payload values.
    std::vector<std::vector<bool>> seen(opts.ranks, std::vector<bool>(opts.ranks, false));
    for (unsigned i = 0; i < opts.ranks; ++i) seen[i][i] = true;
    Run result;
    const bool capture = opts.backend == "compare" || !opts.trace.empty();
    if (capture) result.ledger.reserve(3ULL * count);
    auto record = [&](Kind kind, unsigned i) {
        Event event{kind, i + 1, ops[i].src, ops[i].dst, opts.bytes, engine.now()};
        if (capture) result.ledger.push_back(event);
        auto mix = [&](U64 value) {
            for (unsigned byte = 0; byte < 8; ++byte) {
                result.digest ^= value & 255;
                result.digest *= 1099511628211ULL;
                value >>= 8;
            }
        };
        std::apply([&](auto... field) { (mix(static_cast<U64>(field)), ...); }, event);
        ++result.events;
    };

    unsigned completions = 0;
    while (completions < 2 * count) {
        while (!ready.empty() && ready.top().first <= engine.now()) {
            auto [time, i] = ready.top();
            ready.pop();
            require(time == engine.now() && stamps[i].post == never, "lost frontend release boundary");
            const Op& op = ops[i];
            require(op.origin < 0 || seen[op.src][op.origin], "chunk unavailable at forwarding rank");
            engine.post(i + 1, op.src, op.dst, 0, opts.bytes);
            stamps[i].post = time;
            record(Kind::post, i);
        }
        U64 horizon = ready.empty() ? never : ready.top().first;
        auto batch = engine.advance(horizon);
        require(!batch.empty() || (horizon != never && engine.now() == horizon),
                "missing CQE: no network event or future frontend release");
        for (const auto& cqe : batch) {
            require(cqe.id >= 1 && cqe.id <= count, "unknown CQE identity");
            unsigned i = static_cast<unsigned>(cqe.id - 1);
            auto& times = stamps[i];
            const auto& op = ops[i];
            require(cqe.src == op.src && cqe.dst == op.dst && cqe.bytes == opts.bytes &&
                    cqe.time_ns == engine.now() && times.post < cqe.time_ns, "unexpected CQE fields");
            U64& completion_time = cqe.receive ? times.recv : times.send;
            require(completion_time == never, "duplicate CQE");
            if (cqe.receive && op.origin >= 0) {
                require(!seen[op.dst][op.origin], "received a repeated chunk");
                seen[op.dst][op.origin] = true;
            }
            completion_time = cqe.time_ns;
            record(cqe.receive ? Kind::recv : Kind::send, i);
            ++completions;
            for (unsigned child : children[i][cqe.receive]) {
                require(pending[child] > 0, "released a dependency twice");
                if (--pending[child] == 0) ready.emplace(cqe.time_ns + ops[child].compute_ns, child);
            }
        }
    }
    result.training_end = engine.now();
    for (unsigned i = 0; i < count; ++i) {
        const auto& t = stamps[i];
        require(t.post < t.recv && t.recv < t.send, "send completed before receiver acceptance or ACK");
        U64 parent_time = 0;
        for (const auto& dep : ops[i].parents) {
            const auto& parent = stamps[dep.id - 1];
            parent_time = std::max(parent_time, dep.kind == Kind::recv ? parent.recv : parent.send);
        }
        require(t.post == parent_time + ops[i].compute_ns, "incorrect dependency release time");
    }
    if (opts.workload == "closed-loop") {
        check_disjoint_pair(opts, stamps);
    } else {
        for (const auto& row : seen)
            require(std::all_of(row.begin(), row.end(), [](bool bit) { return bit; }),
                    "collective missing a source chunk");
    }
    while (!engine.idle()) {
        require(engine.advance(never).empty(), "unexpected completion during drain");
    }
    engine.close();
    result.drain_end = engine.now();
    result.stats = engine.stats();
    if (config.trace_limit) {
        result.profile = engine.trace();
        result.profile_peers = peers;
        result.profile_truncated = engine.trace_truncated();
    }
    const auto& s = result.stats;
    require(s.posts == count && s.cqes == count && s.recv_cqes == count &&
            s.qps_created == peers.size() && s.qps_closed == s.qps_created &&
            s.dreq_packets == s.qps_created && s.drep_packets == s.qps_created &&
            s.payload_bytes == count * opts.bytes, "failed work/byte/QP conservation");
    require(s.dropped_ack == (opts.fault == "drop-ack") &&
            s.dropped_data == (opts.fault == "drop-data") && s.rnr_packets == (opts.fault == "rnr") &&
            (opts.fault == "none" || s.retransmits > 0),
            "packet fault/retry differs from fixture");
    result.execution_us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - start).count();
    return result;
}

auto counters(const Stats& s) {
    return std::tie(s.qps_created, s.qps_closed, s.posts, s.cqes, s.recv_cqes, s.data_packets,
                    s.ack_packets, s.rnr_packets, s.retransmits, s.wire_bytes, s.payload_bytes,
                    s.dropped_data, s.dropped_ack, s.duplicates, s.dreq_packets, s.drep_packets);
}

void compare(const Run& cpu, const Run& cuda) {
    auto diff = std::mismatch(cpu.ledger.begin(), cpu.ledger.end(), cuda.ledger.begin(), cuda.ledger.end());
    if (diff.first != cpu.ledger.end() || diff.second != cuda.ledger.end())
        throw std::runtime_error("CPU/CUDA ledger divergence at row " +
                                 std::to_string(diff.first - cpu.ledger.begin()));
    require(counters(cpu.stats) == counters(cuda.stats), "CPU/CUDA counters differ");
    require(cpu.training_end == cuda.training_end && cpu.drain_end == cuda.drain_end,
            "CPU/CUDA completion or drain differs");
}

void write_trace(const std::string& path, const std::vector<Event>& ledger) {
    std::ofstream out;
    out.exceptions(std::ios::badbit | std::ios::failbit);
    out.open(path);
    out << "seq\tkind\tid\tsrc\tdst\tbytes\ttime_ns\n";
    unsigned seq = 0;
    for (const auto& [kind,id,src,dst,bytes,time] : ledger)
        out << seq++ << '\t' << name(kind) << '\t' << id << '\t' << src << '\t' << dst << '\t'
            << bytes << '\t' << time << '\n';
    out.close();
}

void write_profile(const std::string& path, const Run& run) {
    std::ofstream out;
    out.exceptions(std::ios::badbit | std::ios::failbit);
    out.open(path);
    out << "time_ns\tdevice_tick_begin\tdevice_tick_end\tsm\tcta\tqp\tsrc\tdst\tlane"
           "\tbatch_width\tstage\tframe\top_id\n";
    static constexpr const char* stages[] = {"endpoint", "timeout", "rnr-timer", "egress"};
    static constexpr const char* frames[] = {"ack", "rnr", "drep", "dreq", "data"};
    for (const auto& row : run.profile) {
        const auto& peer = run.profile_peers.at(row.qp);
        require(row.stage >= 0 && row.stage < 4, "unknown profile stage");
        require(row.frame >= 0 && row.frame < 5, "unknown frame type");
        out << row.time_ns << '\t' << row.device_tick_begin << '\t' << row.device_tick_end << '\t'
            << row.sm << '\t' << row.cta << '\t' << row.qp << '\t' << peer.src << '\t' << peer.dst
            << '\t' << peer.lane << '\t' << row.batch_width << '\t' << stages[row.stage] << '\t' << frames[row.frame]
            << '\t' << row.op_id << '\n';
    }
    out.close();
}

void print_summary(const Options& o, const Run& run) {
    const auto& s = run.stats;
    std::cout << "{\"passed\":true,\"workload\":\"" << o.workload << "\",\"backend\":\"" << o.backend
              << "\",\"fault\":\"" << o.fault << "\",\"ranks\":" << o.ranks
              << ",\"bytes_per_wr\":" << o.bytes << ",\"operations\":" << s.posts
              << ",\"trace_events\":" << run.events
              << ",\"training_end_ns\":" << run.training_end << ",\"drain_end_ns\":" << run.drain_end
              << ",\"ledger_digest\":\"" << std::hex << std::setw(16) << std::setfill('0') << run.digest << std::dec
              << "\",\"matches_cpu\":" << (o.backend == "compare" ? "true" : "null")
              << ",\"profile_rows\":" << run.profile.size()
              << ",\"profile_truncated\":" << (run.profile_truncated ? "true" : "false")
              << ",\"execution_us\":" << run.execution_us
              << ",\"stats\":{\"qps_created\":" << s.qps_created << ",\"qps_closed\":" << s.qps_closed
              << ",\"posts\":" << s.posts << ",\"cqes\":" << s.cqes << ",\"recv_cqes\":" << s.recv_cqes
              << ",\"data_packets\":" << s.data_packets << ",\"ack_packets\":" << s.ack_packets
              << ",\"rnr_packets\":" << s.rnr_packets << ",\"retransmits\":" << s.retransmits
              << ",\"wire_bytes\":" << s.wire_bytes << ",\"payload_bytes\":" << s.payload_bytes
              << ",\"window_steps\":" << s.window_steps << ",\"dropped_data\":" << s.dropped_data
              << ",\"dropped_ack\":" << s.dropped_ack << ",\"duplicates\":" << s.duplicates
              << ",\"dreq_packets\":" << s.dreq_packets << ",\"drep_packets\":" << s.drep_packets << "}}\n";
}
} // namespace

int main(int argc, char** argv) {
    try {
        Options opts = options(argc, argv);
        Run result;
        if (opts.backend == "compare") {
            Run cpu = run(opts, false);
            result = run(opts, true);
            compare(cpu, result);
        } else {
            result = run(opts, opts.backend == "cuda");
        }
        if (!opts.trace.empty()) write_trace(opts.trace, result.ledger);
        if (!opts.profile.empty()) write_profile(opts.profile, result);
        print_summary(opts, result);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "training RoCE fixture failed: " << e.what() << '\n';
        return 1;
    }
}
