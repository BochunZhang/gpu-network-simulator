# GPU 包级网络仿真研究路线

此文保留研究方向与候选实验，供方案评审时参考。正式任务以评审通过后发布的清单为准。[任务发布规则](../README.md#任务与子任务)

第一条主线是**在保证包级语义的情况下缩短热点 LP 的串行路径，并把独立的事件工作组织成适合 GPU 的批次**。先做一个同模型的 CPU 标量版本和 CUDA 版本，通过正确性对比来定位真实瓶颈。LP（逻辑进程）是拥有一组仿真状态及其事件顺序的单元。把一个 LP 分配给一个 lane 只是最初的映射方式。

## 从同模型基线判断需不需要拆 LP

v0 支持整数仿真时间、固定速率和传播时延、FIFO、确定性路由、包级序列化、有限队列和丢包。针对每个共享状态的所有者定义可重现的事件顺序。同时间戳依模型规定的事件优先级和稳定序号定序。事件处理可生成新的本地和远端事件，任何批量方案都必须把这些影响纳入安全边界。小用例保存逐次状态转移和输出做差分，大用例按 owner 记录有序提交摘要、首个分歧、包和字节守恒、丢包、FCT、队列峰值和最终排空状态。

| 首批实验 | 输入和对照 | 决策依据 |
| --- | --- | --- |
| 正确性 | 单链路串行化、两路汇聚、同时间戳到达与释放、有限队列溢出。用已知答案与 CPU 标量对照 | 到达与离开事件、丢包、FIFO 顺序和时间完全匹配 |
| 单卡基线 | Fat-tree `k=4/8/16`，均匀矩阵与 incast。相同种子、网络配置和包工作量，比较 CPU 标量、同模型 CPU 并行基线和单卡 CUDA。记录预热后的多次运行分布 | 同时记墙钟、初始化/传输、事件处理、队列维护/归并、屏障、日志和每秒已提交事件 |
| 根因诊断 | 按轮与 LP 记录无待处理根、因果边界阻塞、ready LP 数、每 LP 批长、最重 LP 和实际提交数。单独采集 Nsight 时间线与内存/分支计数器 | 有 ready 工作但分配不均时调度优先。工作集中在一个 owner 时考虑拆分。窗口小则检验 lookahead。DRAM/访存等待突出则先改布局 |
| 动手点 | 比较 lane/LP、warp/热点 LP、按事件类型分桶，并记录分桶和归并成本。对单出口与共享交换机分别改动 | 以正确性一致后端到端节省的时间作为选型依据 |

按 LP 队首和工作分布区分无事件、因果阻塞、调度不均和 owner 内串行依赖。对每个阶段内不可拆分的 LP，可用 `总 owner 工作量 / 最重 owner 工作量` 估计忽略调度成本时的并行上界。跨阶段的分母是各阶段最重 owner 工作量之和。结合每轮活跃宽度、每个 warp 的有效 lane 数、可发射 warp、寄存器溢出和 L2/DRAM 事务判断瓶颈。计数器采集单独运行，速度比较使用普通执行数据。[NVIDIA Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)

如果热点是交换机，优先把路由查表、报文整理和可证明独立的端口操作移到并行阶段，给共享 MMU、阈值决策、PFC 及端口队列分别建立明确的状态所有者。端口进入共享仲裁时按相同事件键提交，同一时刻的入队、释放和控制消息也遵循参考顺序。如果热点是一个固定速率、无抢占且批内包集合可判定的出口，可以研究用 scan 计算 `finish[i] = max(finish[i-1], arrival[i]) + service[i]`。每个包表示成 `max(x + service, arrival + service)`，函数复合满足结合律。验证同输出时间后再考虑有限队列、PFC 或服务速率变化下的边界和回退。

## Warp 调度和同步

第一版为每个活跃 owner 顺序处理安全事件，建立 `lane/LP` 基线。第二版在已批准执行的 LP 队首里按协议阶段和状态所有者分桶，使用确定性的 compact/dispatch。只并行处理无冲突的状态访问。同一个 LP 的后继由前驱提交后重新判定。对于一个热 LP，可以让 warp 并行预处理数据和做归约，由 owner 按参考事件顺序提交。

CUDA 的 warp 指令效率和仿真因果顺序由各自的条件决定。控制流一致性来自同类工作分派。warp 内 lane 交换数据时，在共同控制点确定参与掩码，各参与 lane 用相同掩码执行相应的同步或 shuffle。在有共享内存生产者和消费者时使用适当的 `__syncwarp(mask)` 或 CTA barrier。跨 block 的步骤先通过 kernel/stream 依赖建立边界。为不满 warp、分支和同时间戳冲突做差分用例，并对照不同 block 大小。[CUDA warp 同步](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html)

## 跨 GPU 分区和 NVLink

先做同机两卡。每个 NIC 或完整交换机及其共享 MMU 由一张 GPU 持有。按预采样事件成本平衡分区，按跨分区消息数、字节数和边界传播延迟来评估切边。比较按模拟拓扑的 pod 划分和基于工作量与边权的划分。物理 GPU 拓扑决定放置及搬运方式，仿真拓扑决定事件时间和协议。

初版使用保守时间窗。根据所有入边对最早可能到达时刻的有效下界，执行严格早于安全上界的事件。批量交换已提交的带时间戳消息，确认复制完成并在目的 GPU 按稳定事件键合并后才计算下一轮边界。仿真结束需要同时确认任务源、队列与在途消息均排空。零延迟依赖先放到同一分区或建立专门的有序协调。先用 host staging 验证协议，再测同一划分下的异步 P2P。用 `nvidia-smi topo -m`、P2P 能力查询和实测带宽确认传输路径。记录强/弱扩展、切边比例、每窗消息批大小、搬运与同步、最长分区、端到端和已提交事件速率，然后推进 4 卡。[CUDA 设备 P2P 查询](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__DEVICE.html) · [CUDA 对等复制](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__MEMORY.html)

NVLS 有两种独立用法。作为训练模拟目标时，采样相应 collective 的延迟、带宽和资源占用，建立模型并用留出的配置验证。作为模拟器宿主能力时，先采用常规归约与批量 P2P，跨卡归约占比达到可见水平后，再测设备与算子均支持的实现及端到端收益。时间戳消息交换使用有序协议协调。[NCCL NVLS 支持](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-nvls-enable)

## HBM、搬运和日志

| GPU 资源 | 首批候选与量化方式 |
| --- | --- |
| HBM / L2 | 先用按字段连续排列的事件和状态、紧凑 ID、预分配池与有界队列，记录峰值 bytes/在途包、bytes/QP、容量上限和 DRAM 事务。比较热路由或 QP 状态的 L2 策略与默认布局，并对照显存容量和 L2 命中。 [CUDA L2 控制](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/l2-cache-control.html) |
| Shared memory | CTA 内暂存批次元数据、合并缓冲或热状态。测 bank 冲突、每 CTA 的资源占用、L2/DRAM 流量和吞吐。 |
| TMA | 设备支持时，对连续且对齐的事件或状态 tile 做 global/shared 批量复制。与常规合并加载对比，包括填充、屏障和组织 tile 的开销。 [CUDA 异步数据复制](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html) |
| Copy engine | 使用少量页锁定缓冲和非默认 streams 双缓冲日志与输入，用 CUDA event 管理缓冲复用。测设备的可重叠能力和时间线上的实际重叠，也测日志与 P2P 争用。 [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#asynchronous-and-overlapping-transfers-with-computation) |

HBM 留给在途数据、协议和调度状态、有限 staging。默认输出固定大小统计、异常及已提交摘要，调试用例开启全日志。全日志按 owner 和逻辑序号以二进制块异步搬到 host，host 压缩落盘。队列见高水位时在安全边界暂停提交并 flush。显式标记 overflow 和日志完整性，记录日志模式、停顿以及搬运时间。可选采样用于诊断，精确对照时保留所需的完整提交证据。

## 训练负载接入和研究假设

先接入一个可离线生成的微型 workload。每条记录含 rank/GPU、communicator、step 与 op ID、依赖、字节数和计算时长来源。演进顺序为固定消息矩阵 → Dense 的 AllReduce/ReduceScatter/AllGather → MoE AllToAll。CPU 预处理 DAG、拓扑与 collective 算法参数，输入压成连续数组，大任务使用可复用模板和分块输入。GPU 在依赖满足时生成实际报文和控制事件，以网络完成事件驱动下一步，保持计算与通信重叠。独立的持久 QP RoCEv2 路径以训练前创建 QP、训练中复用、排空后释放为边界。逐级增加 ACK/重传、共享有限 MMU/PFC、拥塞控制。每项加入后用匹配的 CPU 模型重跑。需要用实测训练 trace 时先核对计算时长、算法和通道配置，再与独立实测对照估计仿真误差。

额外值得验证的系统问题是**协议可验证的混合批执行**。在普通数据时段批量执行规则简单的包，在 PFC、超时或共享资源冲突发生的位置切回逐事件处理。每次批执行需证明其覆盖的事件前缀安全，并在发现边界前禁止提前发布外部结果。用均匀流量、incast、MoE 突发分别测有效批长、回退率、整理成本和端到端吞吐，找出其收益区间。另测很多独立小实验的批吞吐，判断 GPU 更适合降低单次仿真延迟还是加速参数扫描。
