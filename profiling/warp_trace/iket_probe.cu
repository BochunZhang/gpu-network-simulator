#include <cuda_runtime.h>
#include <iket/iket_profiling_session.h>
#include <dlfcn.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" __global__ void iket_probe_kernel(int* output) {
    if (threadIdx.x == 0) *output = 42;
}

namespace {
using namespace iket;

void stage(const char* name, const char* status) {
    std::fprintf(stderr, "stage=%s status=%s\n", name, status);
}

void check_cuda(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return;
    std::fprintf(stderr, "%s returned %d: %s\n", operation, status, cudaGetErrorString(status));
    std::exit(1);
}

template<class Function, class... Args>
void call_iket(void* library, const char* name, Args... args) {
    auto function = reinterpret_cast<Function>(dlsym(library, name));
    if (!function) {
        std::fprintf(stderr, "%s: %s\n", name, dlerror());
        std::exit(1);
    }
    stage(name, "begin");
    const auto status = function(args...);
    if (status != IKET_PROFILING_SESSION_SUCCESS) {
        std::fprintf(stderr, "%s returned %u\n", name, status);
        std::exit(1);
    }
    stage(name, "ok");
}

#define IKET(method, ...) \
    call_iket<decltype(&IketProfilingSession##method)>(library, "IketProfilingSession" #method, __VA_ARGS__)

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s {direct|graph-late|graph-early} TRACE_PATH\n", argv[0]);
        return 2;
    }
    const bool direct = std::strcmp(argv[1], "direct") == 0;
    const bool early = std::strcmp(argv[1], "graph-early") == 0;
    if (!direct && !early && std::strcmp(argv[1], "graph-late") != 0) {
        std::fprintf(stderr, "unknown mode: %s\n", argv[1]);
        return 2;
    }
    const char* library_path = std::getenv("FT48_IKET_LIBRARY");
    if (!library_path || !*library_path) {
        std::fprintf(stderr, "FT48_IKET_LIBRARY is required\n");
        return 2;
    }

    std::fprintf(stderr, "mode=%s\n", argv[1]);
    stage("cuda_context", "begin");
    check_cuda(cudaSetDevice(0), "cudaSetDevice");
    int driver_version = 0, runtime_version = 0;
    cudaDeviceProp properties{};
    check_cuda(cudaDriverGetVersion(&driver_version), "cudaDriverGetVersion");
    check_cuda(cudaRuntimeGetVersion(&runtime_version), "cudaRuntimeGetVersion");
    check_cuda(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties");
    std::fprintf(stderr, "device=0 gpu=%s sm=%d.%d driver_api=%d runtime=%d build_runtime=%d\n",
                 properties.name, properties.major, properties.minor, driver_version,
                 runtime_version, CUDART_VERSION);
    stage("cuda_context", "ok");

    cudaStream_t stream = nullptr;
    int* device_output = nullptr;
    check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
    check_cuda(cudaMalloc(&device_output, sizeof(*device_output)), "cudaMalloc");
    check_cuda(cudaMemsetAsync(device_output, 0, sizeof(*device_output), stream), "cudaMemsetAsync");
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize before capture");

    void* arguments[] = {&device_output};
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    if (!direct) {
        stage("graph_create", "begin");
        check_cuda(cudaGraphCreate(&graph, 0), "cudaGraphCreate");
        cudaGraphNode_t node = nullptr;
        cudaKernelNodeParams params{};
        params.func = reinterpret_cast<void*>(iket_probe_kernel);
        params.gridDim = dim3(1);
        params.blockDim = dim3(32);
        params.kernelParams = arguments;
        check_cuda(cudaGraphAddKernelNode(&node, graph, nullptr, 0, &params), "cudaGraphAddKernelNode");
        stage("graph_create", "ok");
    }

    const auto instantiate = [&] {
        stage("graph_instantiate", "begin");
        check_cuda(cudaGraphInstantiate(&executable, graph, 0), "cudaGraphInstantiate");
        stage("graph_instantiate", "ok");
    };
    if (!direct && !early) instantiate();

    stage("iket_dlopen", "begin");
    void* library = dlopen(library_path, RTLD_NOW | RTLD_LOCAL);
    if (!library) {
        std::fprintf(stderr, "dlopen: %s\n", dlerror());
        return 1;
    }
    stage("iket_dlopen", "ok");
    std::uint32_t iket_major = 0, iket_minor = 0;
    IKET(GetApiVersion, &iket_major, &iket_minor);
    std::fprintf(stderr, "iket_api=%u.%u\n", iket_major, iket_minor);
    IketProfilingSessionOptions options{};
    options.struct_size = sizeof(options);
    IKET(OptionsInit, &options);
    options.flags |= IKET_PROFILING_SESSION_WARP_LIFETIME_ONLY;
    options.log_level = IKET_PROFILING_SESSION_LOG_INFO;
    IketProfilingSessionHandle handle = nullptr;
    IKET(Create, &handle);
    IKET(SetOptions, handle, &options);
    IKET(Start, handle);
    if (early) instantiate();

    stage("launch", "begin");
    if (direct) {
        check_cuda(cudaLaunchKernel(reinterpret_cast<const void*>(iket_probe_kernel),
                                    dim3(1), dim3(32), arguments, 0, stream), "cudaLaunchKernel");
    } else {
        check_cuda(cudaGraphLaunch(executable, stream), "cudaGraphLaunch");
    }
    stage("launch", "ok");
    stage("synchronize", "begin");
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize after launch");
    stage("synchronize", "ok");
    int output = 0;
    check_cuda(cudaMemcpy(&output, device_output, sizeof(output), cudaMemcpyDeviceToHost), "cudaMemcpy");
    if (output != 42) {
        std::fprintf(stderr, "kernel sentinel mismatch: expected 42, got %d\n", output);
        return 1;
    }
    stage("kernel_sentinel", "ok");
    IKET(Stop, handle);
    IKET(SaveTrace, handle, argv[2]);
    IKET(Destroy, handle);

    if (!direct) {
        check_cuda(cudaGraphExecDestroy(executable), "cudaGraphExecDestroy");
        check_cuda(cudaGraphDestroy(graph), "cudaGraphDestroy");
    }
    check_cuda(cudaFree(device_output), "cudaFree");
    check_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy");
    return 0;
}
