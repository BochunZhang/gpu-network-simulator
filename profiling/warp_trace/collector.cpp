// Host-side hooks for a single, sequential FT48 replay loop. CUDA work must
// have completed on its stream before its end hook.
#include <iket/iket_profiling_session.h>
#include <dlfcn.h>
#include <spawn.h>
#include <sys/wait.h>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

extern char **environ;
namespace {
namespace fs = std::filesystem;
using namespace iket;
struct Session {
    void* library = nullptr;
    IketProfilingSessionHandle handle = nullptr;
    fs::path output;
    std::string phase;
    uint64_t first = 0, count = 0, batch = 1;
    bool in_launch = false;

    template<class F> F symbol(const char* name) {
        auto f = reinterpret_cast<F>(dlsym(library, name));
        if (!f) throw std::runtime_error(name);
        return f;
    }
    static const char* env(const char* name) {
        const char* value = std::getenv(name);
        if (!value || !*value) throw std::runtime_error(std::string("missing ") + name);
        return value;
    }
    void init() {
        if (library) return;
        output = fs::absolute(env("FT48_WARP_OUTPUT"));
        fs::create_directories(output);
        if (fs::exists(output / "pending.iket.json") || fs::exists(output / "index.sqlite"))
            throw std::runtime_error("trace output already contains a capture");
        if (const char* s = std::getenv("FT48_WARP_BATCH")) {
            size_t used = 0;
            batch = std::stoull(s, &used);
            if (!batch || *s == '-' || s[used]) throw std::runtime_error("invalid FT48_WARP_BATCH");
        }
        library = dlopen(env("FT48_IKET_LIBRARY"), RTLD_NOW | RTLD_LOCAL);
        if (!library) throw std::runtime_error(dlerror());
    }
    static void check(uint32_t status, const char* name) {
        if (status) throw std::runtime_error(std::string(name) + " returned " + std::to_string(status));
    }
#define IKET(method, ...) check(symbol<decltype(&IketProfilingSession##method)>("IketProfilingSession" #method)(__VA_ARGS__), #method)
    void begin(const char* kind, uint64_t id) {
        init();
        if (in_launch) throw std::runtime_error("overlapping begin hooks");
        if (handle && phase != kind) flush();
        if (!handle) {
            phase = kind;
            first = id;
            count = 0;
            IketProfilingSessionOptions options{};
            options.struct_size = sizeof(options);
            IKET(OptionsInit, &options);
            options.flags |= IKET_PROFILING_SESSION_WARP_LIFETIME_ONLY;
            IKET(Create, &handle);
            IKET(SetOptions, handle, &options);
            IKET(Start, handle);
        }
        if (id != first + count) throw std::runtime_error("nonconsecutive replay in a window");
        in_launch = true;
    }
    void end(const char* kind, uint64_t id) {
        if (!in_launch || phase != kind || first + count != id)
            throw std::runtime_error("end does not match active replay");
        in_launch = false;
        ++count;
        if (count == batch || phase == "direct") flush();
    }
    void flush() {
        if (in_launch) throw std::runtime_error("flush during a replay");
        if (!handle) return;
        IKET(Stop, handle);
        auto raw = output / "pending.iket.json";
        IKET(SaveTrace, handle, raw.c_str());
        IKET(Destroy, handle);
        handle = nullptr;
        const char* python = std::getenv("FT48_WARP_PYTHON");
        std::vector<std::string> args = {
            python && *python ? python : "python3", env("FT48_WARP_DECODER"), "ingest",
            output.string(), env("FT48_WARP_SIMULATOR"), phase, std::to_string(first), std::to_string(count)
        };
        std::vector<char*> argv;
        for (auto& arg : args) argv.push_back(arg.data());
        argv.push_back(nullptr);
        pid_t pid;
        if (posix_spawnp(&pid, argv[0], nullptr, nullptr, argv.data(), environ))
            throw std::runtime_error("could not launch warp decoder");
        int status;
        pid_t waited;
        do { waited = waitpid(pid, &status, 0); } while (waited < 0 && errno == EINTR);
        if (waited < 0 || !WIFEXITED(status) || WEXITSTATUS(status))
            throw std::runtime_error("warp decoder failed; pending.iket.json retained");
        fs::remove(raw);
    }
};
Session session;
template<class F> void hook(F f) {
    const char* output = std::getenv("FT48_WARP_OUTPUT");
    if (!output || !*output) return;
    try { f(); }
    catch (const std::exception& e) {
        std::fprintf(stderr, "FT48 warp capture: %s\n", e.what());
        std::fflush(stderr);
        std::_Exit(2);
    }
}
}
extern "C" void ft48_warp_begin(uint64_t id) { hook([&] { session.begin("graph", id); }); }
extern "C" void ft48_warp_end(uint64_t id) { hook([&] { session.end("graph", id); }); }
extern "C" void ft48_warp_begin_direct(uint64_t id) { hook([&] { session.begin("direct", id); }); }
extern "C" void ft48_warp_end_direct(uint64_t id) { hook([&] { session.end("direct", id); }); }
extern "C" void ft48_warp_finish() { hook([] { session.flush(); }); }
