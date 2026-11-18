#pragma once

#include <cstdint>
#include <memory>
#include <vector>

namespace training_roce {

struct Peer { int src, dst, lane; };

struct Config {
    int ranks = 2;
    bool cuda = false;
    bool drop_first_data = false;
    bool drop_first_ack = false;
    bool rnr_first = false;
    int trace_limit = 0;
};

struct Cqe {
    std::uint64_t id;
    int src, dst;
    std::uint64_t bytes, time_ns;
    bool receive;
};

struct Trace {
    std::uint64_t time_ns, device_tick_begin, device_tick_end;
    unsigned sm, cta;
    int qp, batch_width, stage, frame;
    std::uint64_t op_id;
};

struct Stats {
    std::uint64_t qps_created = 0, qps_closed = 0, posts = 0, cqes = 0, recv_cqes = 0;
    std::uint64_t data_packets = 0, ack_packets = 0, rnr_packets = 0, retransmits = 0;
    std::uint64_t wire_bytes = 0, payload_bytes = 0, window_steps = 0;
    std::uint64_t dropped_data = 0, dropped_ack = 0, duplicates = 0;
    std::uint64_t dreq_packets = 0, drep_packets = 0;
};

// Persistent directed QP send streams with a preposted receiver. All QPs are
// configured before time zero; post() never allocates a QP. advance() returns
// all local completions at the earliest CQE time, or stops at its horizon.
// UINT64_MAX drains to the next CQE (or to idle).
class Engine {
public:
    Engine(Config config, std::vector<Peer> peers);
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    void post(std::uint64_t id, int src, int dst, int lane, std::uint64_t bytes);
    std::vector<Cqe> advance(std::uint64_t horizon_ns);
    std::uint64_t now() const;
    bool idle() const;
    void close();
    Stats stats() const;
    std::vector<Trace> trace() const;
    bool trace_truncated() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace training_roce
