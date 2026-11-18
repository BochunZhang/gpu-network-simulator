// A deterministic stop-and-wait RC packet model for single-device scheduling
// experiments. Each directed QP uses the source uplink and destination downlink
// of a shared star switch. Both directions cost serialization plus propagation.
#include "training_roce_engine.h"

#include <algorithm>
#include <deque>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#define HD __host__ __device__
#else
#define HD
#endif

namespace training_roce {
namespace {

using U64 = std::uint64_t;
constexpr U64 infinity = ~U64{0};
constexpr U64 mtu = 4096, propagation_ns = 500, timeout_ns = 100000, rnr_ns = 4000;
constexpr unsigned max_retries = 7;
constexpr int batch_size = 32;

// Timers follow incoming frames. Egress arbitration follows both and visits
// control frames before DATA for requests at the same time. No charge is
// inserted for same-owner state transitions.
enum Stage : unsigned { endpoint, timeout, retry, egress };
enum Frame : unsigned { ack, rnr, drep, dreq, data };

struct Event {
    U64 time, serial_psn;
    int qp;
    Stage stage;
    Frame frame;
    unsigned hop, attempt, segment;
    U64 wr_id;
};

HD bool earlier(const Event& a, const Event& b) {
    if (a.time != b.time) return a.time < b.time;
    if (a.stage != b.stage) return a.stage < b.stage;
    if (a.frame != b.frame) return a.frame < b.frame;
    if (a.qp != b.qp) return a.qp < b.qp;
    if (a.serial_psn != b.serial_psn) return a.serial_psn < b.serial_psn;
    if (a.attempt != b.attempt) return a.attempt < b.attempt;
    return a.hop < b.hop;
}

struct Request { U64 id, bytes; bool faults; };
struct Qp {
    Peer peer;
    Request wr;
    U64 remaining, serial_psn, expected;
    unsigned attempt;
    bool busy, closed, used_data_drop, used_ack_drop, used_rnr;
    Stats stats;
};

struct Model {
    int ranks, count;
    bool drop_data, drop_ack, inject_rnr;
    int error, size, capacity, completed, completion_capacity;
    U64 now, steps;
    Event* heap;
    Qp* qp;
    U64* link_free;
    Cqe* cqe;
    Trace* trace;
    int trace_size, trace_limit;
    bool truncated;
};

HD void insert(Model& m, Event e) {
    if (m.size >= m.capacity) { m.error = 2; return; }
    int pos = m.size++;
    while (pos) {
        int parent = (pos - 1) / 2;
        if (!earlier(e, m.heap[parent])) break;
        m.heap[pos] = m.heap[parent];
        pos = parent;
    }
    m.heap[pos] = e;
}

HD Event pop(Model& m) {
    Event head = m.heap[0], tail = m.heap[--m.size];
    if (!m.size) return head;
    int pos = 0;
    while (pos < m.size / 2) {
        int child = pos * 2 + 1;
        if (child + 1 < m.size && earlier(m.heap[child + 1], m.heap[child])) ++child;
        if (!earlier(m.heap[child], tail)) break;
        m.heap[pos] = m.heap[child];
        pos = child;
    }
    m.heap[pos] = tail;
    return head;
}

HD bool stale(const Model& m, const Event& e) {
    if (e.stage != timeout && e.stage != retry) return false;
    const Qp& q = m.qp[e.qp];
    return !q.busy || q.serial_psn != e.serial_psn || q.attempt != e.attempt;
}

HD void purge(Model& m) {
    while (m.size && stale(m, m.heap[0])) pop(m);
}

HD unsigned segment_bytes(const Qp& q) {
    return static_cast<unsigned>(q.remaining < mtu ? q.remaining : mtu);
}

HD Event packet(U64 time, int qp, Frame f, U64 psn, unsigned attempt, unsigned bytes, U64 wr_id = 0) {
    return {time, psn, qp, egress, f, 0, attempt, bytes, wr_id};
}

// Each parallel transition only writes its owning QP or egress port. Follow-up
// events and CQEs are merged in event-key order after the batch has completed.
struct Output {
    Event next[2];
    int count;
    bool has_cqe;
    Cqe cqe;
    int error;
};

HD void transmit(const Model& m, int i, U64 t, Output& o) {
    Qp& q = m.qp[i];
    if (t >= infinity - timeout_ns) { o.error = 1; return; }
    o.next[o.count++] = packet(t, i, data, q.serial_psn, q.attempt, segment_bytes(q), q.wr.id);
    o.next[o.count++] = {t + timeout_ns, q.serial_psn, i, timeout, data, 0, q.attempt, 0, q.wr.id};
    ++q.stats.data_packets;
}

HD void complete(Qp& q, U64 t, bool receive, Output& o) {
    o.cqe = {q.wr.id, q.peer.src, q.peer.dst, q.wr.bytes, t, receive};
    o.has_cqe = true;
    if (receive) ++q.stats.recv_cqes;
    else ++q.stats.cqes;
}

HD int port(const Model& m, Event e) {
    const Peer& p = m.qp[e.qp].peer;
    bool forward = e.frame == data || e.frame == dreq;
    int src = forward ? p.src : p.dst, dst = forward ? p.dst : p.src;
    return e.hop == 0 ? src : m.ranks + dst;
}

HD int owner(const Model& m, const Event& e) {
    return e.stage == egress ? m.count + port(m, e) : e.qp;
}

HD void transition(Model& m, const Event& e, Output& out) {
    Qp& q = m.qp[e.qp];
    if (e.stage == egress) {
        // Wire charge includes a fixed 64-byte model header per frame. It is a
        // declared microbenchmark profile, not the CPU SimAI wire encoder.
        U64 bytes = 64u + (e.frame == data ? e.segment : 0u);
        int link = port(m, e);
        U64 start = m.link_free[link] < e.time ? e.time : m.link_free[link];
        U64 serialization = (bytes + 49) / 50; // 400 Gbit/s, integer ns
        if (start >= infinity - serialization - propagation_ns) { out.error = 1; return; }
        m.link_free[link] = start + serialization;
        Event arrival = e;
        arrival.time = m.link_free[link] + propagation_ns;
        ++arrival.hop;
        if (arrival.hop == 2) arrival.stage = endpoint;
        out.next[out.count++] = arrival;
        return;
    }
    if (e.stage == timeout || e.stage == retry) {
        if (stale(m, e)) return;
        if (q.attempt >= max_retries) { out.error = 3; return; }
        ++q.attempt;
        ++q.stats.retransmits;
        transmit(m, e.qp, e.time, out);
        return;
    }
    if (e.frame == dreq) {
        ++q.stats.drep_packets;
        out.next[out.count++] = packet(e.time, e.qp, drep, 0, 0, 0);
        return;
    }
    if (e.frame == drep) {
        q.closed = true;
        ++q.stats.qps_closed;
        return;
    }
    if (e.frame == data) {
        if (m.drop_data && q.wr.faults && !q.used_data_drop) {
            q.used_data_drop = true;
            ++q.stats.dropped_data;
            return;
        }
        if (e.serial_psn > q.expected) { out.error = 4; return; }
        if (e.serial_psn == q.expected && m.inject_rnr && q.wr.faults && !q.used_rnr) {
            q.used_rnr = true;
            ++q.stats.rnr_packets;
            out.next[out.count++] = packet(e.time, e.qp, rnr, e.serial_psn, e.attempt, 0, e.wr_id);
            return;
        }
        if (e.serial_psn == q.expected) {
            ++q.expected;
            if (q.remaining == e.segment) complete(q, e.time, true, out);
        } else {
            ++q.stats.duplicates;
        }
        ++q.stats.ack_packets;
        out.next[out.count++] = packet(e.time, e.qp, ack, e.serial_psn, e.attempt, 0, e.wr_id);
        return;
    }
    if (e.frame == ack && m.drop_ack && q.wr.faults && !q.used_ack_drop) {
        q.used_ack_drop = true;
        ++q.stats.dropped_ack;
        return;
    }
    if (!q.busy || e.serial_psn != q.serial_psn) return;
    if (e.frame == rnr) {
        if (e.attempt != q.attempt) return;
        if (e.time >= infinity - rnr_ns) { out.error = 1; return; }
        out.next[out.count++] = {e.time + rnr_ns, e.serial_psn, e.qp, retry, data, 0, q.attempt, 0, e.wr_id};
        return;
    }
    q.remaining -= segment_bytes(q);
    ++q.serial_psn;
    q.attempt = 0;
    if (q.remaining) transmit(m, e.qp, e.time, out);
    else {
        q.busy = false;
        complete(q, e.time, false, out);
    }
}

struct Batch {
    Event event[batch_size];
    int owner[batch_size];
    Output output[batch_size];
    int size;
};

HD void prepare(Model& m, U64 horizon, Batch& b) {
    b.size = 0;
    purge(m);
    if (m.error || !m.size || m.capacity - m.size < 2 * batch_size) return;
    Event first = m.heap[0];
    if (first.time > horizon || (first.time == horizon && first.stage == egress)) return;
    if (m.completed && (first.time != m.now || first.stage != endpoint)) return;

    // Stop at a conflict to preserve the total arbitration order. Same-phase
    // events from distinct owners can safely commit in parallel.
    while (b.size < batch_size && m.size) {
        const Event& e = m.heap[0];
        if (e.time != first.time || e.stage != first.stage) break;
        int current_owner = owner(m, e);
        for (int i = 0; i < b.size; ++i) {
            if (b.owner[i] == current_owner) return;
        }
        b.owner[b.size] = current_owner;
        b.event[b.size++] = pop(m);
    }
}

HD void run_one(Model& m, Batch& b, int i) {
    const Event& e = b.event[i];
    Output& o = b.output[i];
    o.count = 0;
    o.has_cqe = false;
    o.error = 0;
    const bool record = i < m.trace_limit - m.trace_size;
    U64 begin = 0, end = 0;
    unsigned sm = 0, cta = 0;
#ifdef __CUDA_ARCH__
    if (record) {
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(begin));
        asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
        cta = blockIdx.x;
    }
#endif
    transition(m, e, o);
    if (record) {
#ifdef __CUDA_ARCH__
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(end));
#endif
        m.trace[m.trace_size + i] = {e.time, begin, end, sm, cta, e.qp, b.size, static_cast<int>(e.stage), static_cast<int>(e.frame), e.wr_id};
    }
}

HD void commit(Model& m, Batch& b) {
    m.now = b.event[0].time;
    ++m.steps;
    if (m.trace_limit) {
        int room = m.trace_limit - m.trace_size;
        if (b.size > room) m.truncated = true;
        m.trace_size += b.size < room ? b.size : room;
    }
    for (int i = 0; i < b.size; ++i) {
        const Event& e = b.event[i];
        const Output& o = b.output[i];
        if (o.error) { m.error = o.error; return; }
        if (e.stage == egress)
            m.qp[e.qp].stats.wire_bytes += 64u + (e.frame == data ? e.segment : 0u);
        for (int j = 0; j < o.count; ++j) insert(m, o.next[j]);
        if (o.has_cqe) {
            if (m.completed >= m.completion_capacity) { m.error = 5; return; }
            m.cqe[m.completed++] = o.cqe;
        }
    }
}

void run_cpu(Model& m, U64 horizon) {
    Batch b{};
    while (true) {
        prepare(m, horizon, b);
        if (!b.size) return;
        for (int i = 0; i < b.size; ++i) run_one(m, b, i);
        commit(m, b);
    }
}

#ifdef __CUDACC__
__global__ void run_cuda(Model* m, U64 horizon) {
    __shared__ Batch b;
    while (true) {
        if (threadIdx.x == 0) prepare(*m, horizon, b);
        __syncthreads();
        if (!b.size) return;
        if (static_cast<int>(threadIdx.x) < b.size) run_one(*m, b, threadIdx.x);
        __syncthreads();
        if (threadIdx.x == 0) commit(*m, b);
        __syncthreads();
    }
}

void check(cudaError_t err, const char* what) {
    if (err != cudaSuccess)
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
}
#endif

template<class T>
class Buffer {
public:
    Buffer() = default;
    ~Buffer() { release(); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    void allocate(int n, bool gpu) {
        if (n < 1) throw std::invalid_argument("invalid engine buffer size");
#ifdef __CUDACC__
        if (gpu) {
            check(cudaMallocManaged(&ptr, sizeof(T) * n), "cudaMallocManaged");
            managed = true;
            std::uninitialized_value_construct_n(ptr, n);
            return;
        }
#else
        if (gpu) throw std::runtime_error("CUDA requested in CPU-only build");
#endif
        ptr = new T[n]{};
    }
    void release() {
#ifdef __CUDACC__
        if (managed) { cudaFree(ptr); ptr = nullptr; managed = false; return; }
#endif
        delete[] ptr;
        ptr = nullptr;
    }
    void swap(Buffer& b) {
        std::swap(ptr, b.ptr);
#ifdef __CUDACC__
        std::swap(managed, b.managed);
#endif
    }
    T* ptr = nullptr;
private:
#ifdef __CUDACC__
    bool managed = false;
#endif
};

} // namespace

struct Engine::Impl {
    Config config;
    Buffer<Model> state;
    Buffer<Qp> qps;
    Buffer<U64> links;
    Buffer<Cqe> completions;
    Buffer<Event> events;
    Buffer<Trace> traces;
    std::map<std::tuple<int,int,int>, int> ids;
    std::set<U64> work_ids;
    std::vector<std::deque<Request>> waiting;
    bool closed = false;
    U64 posts = 0;

    Impl(Config cfg, const std::vector<Peer>& peers) : config(cfg) {
        if (cfg.ranks < 2 || cfg.ranks > 1000000 || cfg.trace_limit < 0 || peers.size() > 10000000)
            throw std::invalid_argument("invalid rank, QP or trace count");
        state.allocate(1, cfg.cuda);
        Model& m = *state.ptr;
        m.ranks = cfg.ranks;
        m.count = static_cast<int>(peers.size());
        m.capacity = std::max(256, 2 * m.count + 2 * batch_size);
        m.completion_capacity = std::max(1, 2 * m.count);
        m.drop_data = cfg.drop_first_data;
        m.drop_ack = cfg.drop_first_ack;
        m.inject_rnr = cfg.rnr_first;
        qps.allocate(std::max(1, m.count), cfg.cuda);
        links.allocate(2 * cfg.ranks, cfg.cuda);
        completions.allocate(m.completion_capacity, cfg.cuda);
        events.allocate(m.capacity, cfg.cuda);
        traces.allocate(std::max(1, cfg.trace_limit), cfg.cuda);
        m.qp = qps.ptr; m.link_free = links.ptr; m.heap = events.ptr;
        m.cqe = completions.ptr; m.trace = traces.ptr; m.trace_limit = cfg.trace_limit;
        waiting.resize(peers.size());
        for (int i = 0; i < m.count; ++i) {
            Peer p = peers[i];
            if (p.src < 0 || p.dst < 0 || p.src >= cfg.ranks || p.dst >= cfg.ranks ||
                p.src == p.dst || p.lane < 0 || !ids.emplace(std::make_tuple(p.src,p.dst,p.lane),i).second)
                throw std::invalid_argument("invalid or duplicate configured QP");
            m.qp[i].peer = p;
            m.qp[i].stats.qps_created = 1;
        }
    }

    void reserve(int additional) {
        Model& m = *state.ptr;
        if (m.capacity - m.size >= additional) return;
        if (m.capacity > std::numeric_limits<int>::max() / 2 ||
            additional > std::numeric_limits<int>::max() - m.size)
            throw std::overflow_error("event heap too large");
        int capacity = std::max(m.capacity * 2, m.size + additional);
        Buffer<Event> replacement;
        replacement.allocate(capacity, config.cuda);
        std::copy_n(m.heap, m.size, replacement.ptr);
        events.swap(replacement);
        m.heap = events.ptr;
        m.capacity = capacity;
    }

    void start(int id, Request w) {
        reserve(2);
        Model& m = *state.ptr;
        Qp& q = m.qp[id];
        q.busy = true;
        q.wr = w;
        q.remaining = w.bytes;
        q.attempt = 0;
        Output o{};
        transmit(m, id, m.now, o);
        if (o.error) throw std::overflow_error("cannot schedule initial RC timer");
        for (int j = 0; j < o.count; ++j) insert(m, o.next[j]);
    }

    std::vector<Cqe> advance(U64 horizon) {
        Model& m = *state.ptr;
        if (horizon < m.now) throw std::invalid_argument("frontier moved backwards");
        m.completed = 0;
        do {
            reserve(2 * batch_size);
#ifdef __CUDACC__
            if (config.cuda) {
                run_cuda<<<1,batch_size>>>(&m, horizon);
                check(cudaGetLastError(), "RC kernel launch");
                check(cudaDeviceSynchronize(), "RC kernel execution");
            } else
#endif
            run_cpu(m, horizon);
            if (m.error) throw std::runtime_error("RC engine failed, code " + std::to_string(m.error));
        } while (m.capacity - m.size < 2 * batch_size);
        std::vector<Cqe> result(m.cqe, m.cqe + m.completed);
        if (result.empty() && horizon != infinity) m.now = horizon;
        for (const Cqe& c : result) {
            if (c.receive) continue;
            // Request ownership is indexed by the persistent peer/lane. The ID
            // can be arbitrary so lookup does not depend on submission order.
            int i = active_wr.at(c.id);
            active_wr.erase(c.id);
            if (!waiting[i].empty()) {
                Request w = waiting[i].front();
                waiting[i].pop_front();
                start(i,w);
            }
        }
        return result;
    }
    std::map<U64,int> active_wr;
};

Engine::Engine(Config c, std::vector<Peer> peers)
    : impl_(new Impl(c, peers)) {}
Engine::~Engine() = default;

void Engine::post(U64 id, int src, int dst, int lane, U64 bytes) {
    Impl& s = *impl_;
    if (s.closed) throw std::logic_error("posting after close");
    auto found = s.ids.find(std::make_tuple(src,dst,lane));
    if (found == s.ids.end()) throw std::invalid_argument("unconfigured training QP");
    if (!s.work_ids.insert(id).second) throw std::invalid_argument("duplicate work request ID");
    int i = found->second;
    Qp& q = s.qps.ptr[i];
    Request w{id,bytes,s.posts == 0};
    s.active_wr.emplace(id,i);
    ++s.posts;
    ++q.stats.posts;
    q.stats.payload_bytes += bytes;
    if (q.busy) s.waiting[i].push_back(w);
    else s.start(i,w);
}

std::vector<Cqe> Engine::advance(U64 horizon) {
    if (impl_->closed) throw std::logic_error("advance after close");
    return impl_->advance(horizon);
}

U64 Engine::now() const { return impl_->state.ptr->now; }

bool Engine::idle() const {
    auto& m = *impl_->state.ptr;
    purge(m);
    return !m.size && impl_->active_wr.empty();
}

void Engine::close() {
    if (impl_->closed) return;
    if (!idle()) throw std::logic_error("close before network drain");
    Model& m = *impl_->state.ptr;
    impl_->reserve(m.count);
    for (int i = 0; i < m.count; ++i) {
        ++m.qp[i].stats.dreq_packets;
        insert(m, packet(m.now,i,dreq,0,0,0));
    }
    impl_->advance(infinity);
    for (int i = 0; i < m.count; ++i)
        if (!m.qp[i].closed) throw std::logic_error("missing QP close reply");
    impl_->closed = true;
}

Stats Engine::stats() const {
    Stats sum{};
    Model& m = *impl_->state.ptr;
    for (int i = 0; i < m.count; ++i) {
        const Stats& s = m.qp[i].stats;
        sum.posts += s.posts; sum.cqes += s.cqes; sum.recv_cqes += s.recv_cqes;
        sum.data_packets += s.data_packets; sum.ack_packets += s.ack_packets;
        sum.rnr_packets += s.rnr_packets; sum.retransmits += s.retransmits;
        sum.qps_created += s.qps_created; sum.qps_closed += s.qps_closed;
        sum.payload_bytes += s.payload_bytes; sum.wire_bytes += s.wire_bytes;
        sum.dropped_data += s.dropped_data; sum.dropped_ack += s.dropped_ack;
        sum.duplicates += s.duplicates; sum.dreq_packets += s.dreq_packets;
        sum.drep_packets += s.drep_packets;
    }
    sum.window_steps = m.steps;
    return sum;
}

std::vector<Trace> Engine::trace() const {
    const auto& m = *impl_->state.ptr;
    return {m.trace,m.trace+m.trace_size};
}

bool Engine::trace_truncated() const { return impl_->state.ptr->truncated; }
} // namespace training_roce
