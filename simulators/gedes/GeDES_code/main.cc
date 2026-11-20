#include <cuda_runtime.h>
#include "gpu_queue.cuh"
#include <vector>
#include "thread_pool.h"
// #include "test.cuh"
#include "conf.h"
#include "topology.h"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <exception>
#ifdef FT48_WARP_TRACE
#include <cstdint>
#include <cstdlib>
extern "C" bool ft48_warp_begin(std::uint64_t replay);
extern "C" bool ft48_warp_end(std::uint64_t replay);
extern "C" void ft48_warp_finish();
#endif

enum TEST
{
    PROTOCOL1,
    PROTOCOL2
};

// __global__ void print_kernel(GPUQueue<int64_t> *queue)
// {
//     int tid = threadIdx.x;
//     printf("Thread %d: %d\n", tid, TEST::PROTOCOL1);

//     __syncthreads();
// }

// __global__ void test_array(int64_t* arr, int size) {
//     int tid = blockDim.x * blockIdx.x + threadIdx.x;

//     if (tid == size - 1)
//         printf("%d\n", arr[tid]);
// }

#if TEST_FRAME_MOVE
#include "packet_manager_test_conf.cuh"
#endif

int main(int argc, char *argv[])
{

    VDES::Topology topology;

    /**
     * Stage1: Set up the topology.
     */

    /**
     * @breif: The basic config parameters for the topology.
     */
    // int node_num = 8;
    // int nic_num_per_node = 2;
    // int ingress_capacity = 1000;
    // int egress_capacity = 1000;
    // int sw_num = 18;
    // int port_num_per_switch = 4;
    // int switch_ingress_capacity = 1000;
    // int switch_egress_capacity = 1000;
    int ft_k = 16;
    uint32_t ip_group_size = 48;
    uint32_t ip_base_addr = 0xc0a80000;
    uint32_t ip_mask = 0xffffff00;
    int tx_rate = 100;
    int popogation_delay = 1000;

    int64_t timeslot_start = 0;
    int64_t timeslot_end = 1000;

    int expected_packet_num = 10000;
    double deviate = 0.5;

    int look_ahead_timeslot = 6;
    int packet_pool_size = 8000000; // Default packet pool size

    int flow_time_range = 1000000;

    bool audit_detail = false;
    bool audit_early_stop = false;
    std::string output_file_name;

    // Parse command line arguments
    if (argc > 1)
    {
        for (int i = 1; i < argc; ++i)
        {
            std::string arg = argv[i];

            // Parse ft_k parameter
            if (arg.find("--ft_k=") == 0)
            {
                ft_k = std::stoi(arg.substr(7));
            }
            // Parse expected_packet_num parameter
            else if (arg.find("--packets=") == 0)
            {
                expected_packet_num = std::stoi(arg.substr(10));
            }
            // Also support --average_flow_size parameter for compatibility
            else if (arg.find("--average_flow_size=") == 0)
            {
                expected_packet_num = std::stoi(arg.substr(20));
            }
            // Parse output_file_name parameter
            else if (arg.find("--output=") == 0)
            {
                output_file_name = arg.substr(9);
            }
            // Parse packet_pool_size parameter
            else if (arg.find("--packet_pool_size=") == 0)
            {
                packet_pool_size = std::stoi(arg.substr(19));
            }
            else if (arg.find("--flow_time_range=") == 0)
            {
                flow_time_range = std::stoi(arg.substr(18));
            }
            else if (arg == "--audit_early_stop=1")
            {
                audit_early_stop = true;
            }
            else if (arg == "--audit_detail=1")
            {
                audit_detail = true;
            }
            else
            {
                std::cout << "Unknown argument: " << arg << std::endl;
                std::cout << "Usage: " << argv[0] << " [--ft_k=value] [--packets=value] [--average_flow_size=value] [--output=filename] [--packet_pool_size=value]" << std::endl;
                return 1;
            }
        }
    }

    std::cout << "Building Network" << std::endl;
    using AuditClock = std::chrono::steady_clock;
    const auto audit_init_start = AuditClock::now();
    VDES::SetFlowTimeRange(flow_time_range);
    VDES::create_thread_pool(VDES::THREAD_POOL_SIZE);
    VDES::SetInitialMemoryPoolSize(packet_pool_size);
    VDES::InitializeMemoryPools();
    topology.SetFattreeProperties(ft_k, ip_group_size, ip_base_addr, ip_mask);
    topology.SetChannelParams(tx_rate, popogation_delay);
    topology.SetTimeslotInfo(timeslot_start, timeslot_end);
    topology.SetLookAheadTimeslot(look_ahead_timeslot);
    topology.SetTransmissionPlanParams(expected_packet_num, deviate, 41U);
    topology.BuildTopology();
#ifdef FT48_WARP_TRACE
    const char *trace_output = std::getenv("FT48_WARP_OUTPUT");
    const bool trace_replays = trace_output && *trace_output;
#else
    constexpr bool trace_replays = false;
#endif
#if ENABLE_GPU_MEM_POOL
    topology.BuildGraphOnlyGPU(trace_replays);
#elif ENABLE_HUGE_GRAPH
    topology.BuildHugeGraph();
#endif
    cudaError_t audit_init_error = cudaDeviceSynchronize();
    const auto audit_init_end = AuditClock::now();
    if (audit_init_error != cudaSuccess) {
        std::cerr << "Cuda error: " << cudaGetErrorString(audit_init_error) << std::endl;
        return 1;
    }
    int audit_rounds = 0;
    const int audit_max_rounds = 1000000;
    double audit_finish_check_seconds = 0;
    double audit_run_graph_seconds = 0;
    auto audit_finished = [&]() {
        if (!audit_detail) return topology.IsFinished();
        const auto begin = AuditClock::now();
        const bool result = topology.IsFinished();
        audit_finish_check_seconds += std::chrono::duration<double>(AuditClock::now() - begin).count();
        return result;
    };
    std::cout << "Simulation Start" << std::endl;
    const auto audit_loop_start = AuditClock::now();
    for (int i = 0; i < audit_max_rounds && (!audit_early_stop || i < 1) && !audit_finished(); ++i) {
#ifdef FT48_WARP_TRACE
        // A fresh executable belongs to the IKET session that records its replays.
        if (trace_replays && ft48_warp_begin(i)) topology.InstantiateGraph();
#endif
#if ENABLE_HUGE_GRAPH
        if (audit_detail) {
            const auto begin = AuditClock::now();
            topology.RunGraph();
            audit_run_graph_seconds += std::chrono::duration<double>(AuditClock::now() - begin).count();
        } else {
            topology.RunGraph();
        }
#endif
#ifdef FT48_WARP_TRACE
        if (trace_replays && ft48_warp_end(i)) topology.DestroyGraphExecutable();
#endif
        ++audit_rounds;
        if (i % 20000 == 0) {
            std::cout << "Simulation timeline " << i / 1000000.0 << "s" << std::endl;
        }
    }
#ifdef FT48_WARP_TRACE
    if (trace_replays) {
        ft48_warp_finish();
        topology.DestroyGraphExecutable();
    }
#endif
    const auto audit_loop_end = AuditClock::now();
    const auto audit_sync_start = AuditClock::now();
    cudaError_t audit_sync_error = cudaDeviceSynchronize();
    const auto audit_sync_end = AuditClock::now();
    const bool audit_upstream_finished = topology.IsFinished();
    const double audit_loop_seconds = std::chrono::duration<double>(audit_loop_end - audit_loop_start).count();
    const double audit_sync_seconds = std::chrono::duration<double>(audit_sync_end - audit_sync_start).count();
    std::cout << "Simulation Duration: " << audit_loop_seconds + audit_sync_seconds << "s" << std::endl;
    cudaError_t audit_last_error = cudaGetLastError();
    if (audit_sync_error != cudaSuccess || audit_last_error != cudaSuccess) {
        std::cerr << "Cuda error: " << cudaGetErrorString(audit_sync_error != cudaSuccess ? audit_sync_error : audit_last_error) << std::endl;
        return 1;
    }
    VDES::CompletionCounts audit_counts{};
    const auto audit_readback_error = topology.ReadCompletionCounts(audit_counts);
    if (audit_readback_error != cudaSuccess) {
        std::cerr << "GeDES completion readback failed: " << cudaGetErrorString(audit_readback_error) << std::endl;
        return 1;
    }
    const auto audit_output_start = AuditClock::now();
    topology.RecordFlowResults(output_file_name);
    const auto audit_output_end = AuditClock::now();
    std::cout << std::setprecision(17)
              << "record=gedes_run rounds=" << audit_rounds
              << " max_rounds=" << audit_max_rounds
              << " upstream_finished=" << (audit_upstream_finished ? 1 : 0)
              << " completed_nodes=" << audit_counts.completed_nodes
              << " total_nodes=" << audit_counts.total_nodes
              << " completed_channel_blocks=" << audit_counts.completed_channel_blocks
              << " total_channel_blocks=" << audit_counts.total_channel_blocks
              << " detail=" << (audit_detail ? 1 : 0)
              << " init_seconds=" << std::chrono::duration<double>(audit_init_end - audit_init_start).count()
              << " loop_seconds=" << audit_loop_seconds
              << " tail_sync_seconds=" << audit_sync_seconds
              << " output_seconds=" << std::chrono::duration<double>(audit_output_end - audit_output_start).count()
              << " finish_check_seconds=" << (audit_detail ? audit_finish_check_seconds : -1.0)
              << " run_graph_seconds=" << (audit_detail ? audit_run_graph_seconds : -1.0)
              << std::endl;
    const auto drain_start = AuditClock::now();
    try {
        topology.WriteDrainAudit(output_file_name);
    } catch (const std::exception &error) {
        std::cerr << "GeDES terminal snapshot failed: " << error.what() << std::endl;
        return 1;
    }
    std::cout << "record=gedes_drain seconds="
              << std::chrono::duration<double>(AuditClock::now() - drain_start).count()
              << std::endl;
    return 0;
}

// int main()
// {
//     // GPUQueue<int64_t>* queue = create_gpu_queue<int64_t>(10, 10);

//     // int64_t* arr = new int64_t[10];
//     // for (int i = 0; i < 10; i++)
//     // {
//     //     arr[i] = i;
//     // }

//     // GPUQueue<int64_t> queue_cpu;
//     // cudaMemcpy(&queue_cpu, queue, sizeof(GPUQueue<int64_t>), cudaMemcpyDeviceToHost);
//     // int64_t** queues_cpu = new int64_t * [1];
//     // cudaMemcpy(queues_cpu, queue_cpu.queues, sizeof(int64_t*) * 1, cudaMemcpyDeviceToHost);
//     // cudaMemcpy(queues_cpu[0], arr, 10 * sizeof(int64_t), cudaMemcpyHostToDevice);

//     // launch kernel
//     // print_kernel << <1, 10 >> > (queue);

//     // test trans delay
//     // test_trans_delay();

//     // initialize memory pools
//     VDES::InitializeMemoryPools();
// #ifdef TEST_P2P
//     // launch thread pool
//     VDES::global_pool = new VDES::ThreadPool(VDES::THREAD_POOL_SIZE);

//     // test p2p channel controller
//     test_p2p_channel_controller();

//     // int size = 50000;
//     // dim3 gridD((size + 1023) / 1024);
//     // int64_t *arr_h = (int64_t*)malloc(size * sizeof(int64_t)), *arr_d;
//     // for (int i = 0; i < size; i++) {
//     //     arr_h[i] = i;
//     // }
//     // cudaMalloc((void**)&arr_d, size * sizeof(int64_t));
//     // cudaMemcpy(arr_d, arr_h, size * sizeof(int64_t), cudaMemcpyHostToDevice);
//     // test_array<<<gridD, 1024>>>(arr_d, size);

//     cudaDeviceSynchronize();

//     cudaError_t err = cudaGetLastError();
//     if (err != cudaSuccess)
//     {
//         printf("Cuda error: %s\n", cudaGetErrorString(err));
//         return 1;
//     }

//     // stop thread pool
//     VDES::global_pool->stop();

//     delete VDES::global_pool;
//     printf("Test p2p channel controller done.\n");
// #endif

// #ifdef TEST_FRAME_DECAPSULATION
//     // test frame decapsulation
//     // function_test();
//     // cudaError_t err = cudaGetLastError();
//     // if (err != cudaSuccess)
//     // {
//     //     printf("Cuda error: %s\n", cudaGetErrorString(err));
//     //     return 1;
//     // }

//     // launch thread pool
//     VDES::global_pool = new VDES::ThreadPool(VDES::THREAD_POOL_SIZE);

//     // test decapsulation
//     test_decapsulation_kernel();

//     cudaDeviceSynchronize();

//     cudaError_t err = cudaGetLastError();
//     if (err != cudaSuccess)
//     {
//         printf("Cuda error: %s\n", cudaGetErrorString(err));
//         return 1;
//     }

//     // stop thread pool
//     VDES::global_pool->stop();

//     delete VDES::global_pool;

//     printf("Test frame decapsulation done.\n");
// #endif

// #ifdef TEST_IPV4_CONTROLLER
//     // test ipv4 function
//     // ipv4_function_test();

//     // launch thread pool
//     VDES::global_pool = new VDES::ThreadPool(VDES::THREAD_POOL_SIZE);

//     // test ipv4 controller
//     test_ipv4_controller();

//     cudaDeviceSynchronize();

//     cudaError_t err = cudaGetLastError();
//     if (err != cudaSuccess)
//     {
//         printf("Cuda error: %s\n", cudaGetErrorString(err));
//         return 1;
//     }

//     // stop thread pool
//     VDES::global_pool->stop();

//     delete VDES::global_pool;
//     printf("Test ipv4 controller done.\n");
// #endif

// #ifdef TEST_TCP_CONTROLLER
//     // test tcp function
//     // test_tcp_functions();

//     // launch thread pool
//     // VDES::global_pool = new VDES::ThreadPool(VDES::THREAD_POOL_SIZE);
//     VDES::create_thread_pool(VDES::THREAD_POOL_SIZE);

//     // test tcp controller
//     test_tcp_controller();

//     cudaDeviceSynchronize();

//     cudaError_t err = cudaGetLastError();
//     if (err != cudaSuccess)
//     {
//         printf("Cuda error: %s\n", cudaGetErrorString(err));
//         return 1;
//     }

//     // stop thread pool
//     VDES::global_pool->stop();

//     delete VDES::global_pool;
//     printf("Test tcp controller done.\n");
// #endif

// #ifdef TEST_UDP_CONTROLLER
//     // test udp function
//     // test_udp_functions();

//     // launch thread pool
//     // VDES::global_pool = new VDES::ThreadPool(VDES::THREAD_POOL_SIZE);
//     VDES::create_thread_pool(VDES::THREAD_POOL_SIZE);

//     // test tcp controller
//     test_udp_receiver();
//     // test_udp_sender();

//     cudaDeviceSynchronize();

//     cudaError_t err = cudaGetLastError();
//     if (err != cudaSuccess)
//     {
//         printf("Cuda error: %s\n", cudaGetErrorString(err));
//         return 1;
//     }

//     // stop thread pool
//     VDES::global_pool->stop();

//     delete VDES::global_pool;
//     printf("Test udp controller done.\n");
// #endif

// #ifdef TEST_FRAME_ENCAP_CONTROLLER
//     // test frame encap function
//     // test_frame_encap_functions();

//     VDES::create_thread_pool(VDES::THREAD_POOL_SIZE);

//     // test tcp controller
//     test_frame_encapsulation_controller();
//     // test_udp_sender();

//     cudaDeviceSynchronize();

//     cudaError_t err = cudaGetLastError();
//     if (err != cudaSuccess)
//     {
//         printf("Cuda error: %s\n", cudaGetErrorString(err));
//         return 1;
//     }

//     // stop thread pool
//     VDES::global_pool->stop();

//     delete VDES::global_pool;
//     printf("Test frame encap controller done.\n");
// #endif

// #if TEST_IPV4_DECAPSULATION_CONTROLLER
//     // test ipv4 decapsulation function
//     // test_ipv4_decap_functions();

//     VDES::create_thread_pool(VDES::THREAD_POOL_SIZE);

//     // test ipv4 decapsulation controller
//     test_ipv4_decap_controller();

//     cudaError_t err = cudaGetLastError();
//     if (err != cudaSuccess)
//     {
//         printf("Cuda error: %s\n", cudaGetErrorString(err));
//         return 1;
//     }

//     // stop thread pool
//     VDES::global_pool->stop();

//     delete VDES::global_pool;
//     printf("Test ipv4 decapsulation controller done.\n");
// #endif

// #if TEST_IPV4_ENCAPSULATION_CONTROLLER
//     // test ipv4 encapsulation function
//     // test_ipv4_encap_functions();

//     VDES::create_thread_pool(VDES::THREAD_POOL_SIZE);

//     // test ipv4 encapsulation controller
//     test_ipv4_encap_controller();

//     cudaError_t err = cudaGetLastError();
//     if (err != cudaSuccess)
//     {
//         printf("Cuda error: %s\n", cudaGetErrorString(err));
//         return 1;
//     }

//     // stop thread pool
//     VDES::global_pool->stop();

//     delete VDES::global_pool;
//     printf("Test ipv4 encapsulation controller done.\n");
// #endif

// #if TEST_SWITCH_CONTROLLER
//     // test switch function
//     // test_switch_function();

//     VDES::create_thread_pool(VDES::THREAD_POOL_SIZE);

//     // test switch controller
//     test_switch_controller();

//     cudaError_t err = cudaGetLastError();
//     if (err != cudaSuccess)
//     {
//         printf("Cuda error: %s\n", cudaGetErrorString(err));
//         return 1;
//     }

//     // stop thread pool
//     VDES::global_pool->stop();

//     delete VDES::global_pool;
//     printf("Test switch controller done.\n");
// #endif

// #ifdef TEST_FRAME_MOVE
//     VDES::create_thread_pool(VDES::THREAD_POOL_SIZE);

//     VDES::GPUQueueManager manager;
//     manager.enable_global_cache_strategy();
//     VDES::test_time_strategy_basic_move_out();
//     // VDES::test_time_strategy_basic();
//     // VDES::test_time_strategy_frame();
//     // VDES::test_time_strategy_ipv4();

//     cudaDeviceSynchronize();

//     // stop thread pool
//     VDES::global_pool->stop();

//     delete VDES::global_pool;
//     printf("Test frame move done.\n");
// #endif

//     return 0;
// }
