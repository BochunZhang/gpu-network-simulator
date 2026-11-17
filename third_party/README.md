# 第三方参考与复现

| 项目 | 引用 | 首轮验证 |
| --- | --- | --- |
| GeDES | EuroSys 2026。[论文](https://doi.org/10.1145/3767295.3769327) · [作者源码 `1376c63`](https://github.com/mobinets/GeDES/tree/1376c638a0ae66116430f953da137d0c536675ad) | 从固定源码构建，运行作者最小评测参数 FatTree-8，检查有效的 TCP 完成时间。 |
| Days AGO | 当前 Days 执行器。[架构](https://days.sh/docs/architecture/overview) · [作者源码 `048d487`](https://github.com/iQua/days/tree/048d48743119257fd8d4b2e5f0fb2228153fd955) · [旧版论文](https://days.sh/papers/days.pdf) | 运行同一 TCP fixture，对照 Scalar、CPU 和 CUDA 的完整状态指纹。 |
| NSX | NAIC 2025。[作者论文](https://www.cs.rice.edu/~eugeneng/papers/NAIC25.pdf) | 未找到公开的作者实现。参考其 GPU 上的模块划分以及跨 GPU 事件队列设计。 |
| HeteroSim | The Web Conference 2026。[论文](https://dl.acm.org/doi/10.1145/3774904.3792254) | 未找到公开的作者实现。参考其训练图、microbatch 调度、计算与通信重叠，以及异构链路上的 collective 建模。 |

GeDES 仓库公开了源码和评测脚本，仓库根目录未见许可证。Days 采用 AGPL-3.0。两套代码都从上游拉取到独立的实验目录，并以表中完整提交固定源代码。 [GeDES 仓库](https://github.com/mobinets/GeDES) · [Days 许可证](https://github.com/iQua/days/blob/main/LICENSE)

## GeDES

作者 README 给出的开发环境为 Ubuntu 22.04、CUDA 11.8+、NVIDIA 驱动 575+ 和 CMake。用临时源码副本修正 `GeDES_code/CMakeLists.txt` 中的源码路径，按实测 compute capability 为 CUDA 构建选择架构，并记录实际 CMake、C++ 编译器、`nvcc` 和驱动版本。构建得到的 `GeDES` 可执行文件使用以下作者评测参数运行。 [构建说明](https://github.com/mobinets/GeDES/blob/main/README.md) · [作者评测参数](https://github.com/mobinets/GeDES/blob/main/script.py)

```bash
./GeDES --ft_k=8 --packet_pool_size=1000000 --average_flow_size=10000 \
  --flow_time_range=1000000 --output=flows.csv
```

保留 CUDA 日志、程序报告的运行时间和原始 CSV。通过条件为构建成功、运行退出码为零、程序报告正的运行时间，且 CSV 中至少一条流的 `flow_size > 0`、`completed_timestamp > start_timestamp`、`flow_completion_time == completed_timestamp - start_timestamp`。CSV 会输出未完成的流，且仿真循环有最大轮数。这个门限验证实际出现了正的流完成时间。论文的性能复现需要额外检查负载完成量、配置和对照。 [主程序](https://github.com/mobinets/GeDES/blob/1376c638a0ae66116430f953da137d0c536675ad/GeDES_code/main.cc) · [CSV 写入实现](https://github.com/mobinets/GeDES/blob/main/GeDES_code/code/c%2B%2B/src/transport/tcp_controller.cc)

## Days AGO

在固定版本根目录执行。CUDA 构建需要 Rust/Cargo 和 CUDA 13 `nvcc`。该版本 CUDA 后端允许 `sm_86`、`sm_89` 和 `sm_121`。机器若使用 H100（`sm_90`）或 GB200（`sm_100`），需要排独立移植任务，记录架构补丁并重新验证 Scalar 与 CUDA 的输出。 [固定版 CUDA 初始化](https://github.com/iQua/days/blob/048d48743119257fd8d4b2e5f0fb2228153fd955/executor/src/cuda.rs#L5731) · [NVIDIA GPU 架构列表](https://developer.nvidia.com/cuda/gpus)

```bash
cargo build --locked --release --features cuda --bin days
fixture=configs/benchmarks/tcp/fattree_k4_tcp_cubic_f16_smoke.toml
target/release/days --engine scalar "$fixture"
target/release/days --engine cpu --workers 2 --repetitions 1 "$fixture"
target/release/days --engine cuda "$fixture"
```

三个结果的 `record=days_result` 须携带相同的 `result_bytes` 和 `result_fnv1a64`。保留三个原始结果、fixture 的 SHA-256 和版本信息。通过后再逐项引入 RoCE/PFC、collective 和训练 trace，并检查每项的模型边界。首轮 TCP 的一致性只覆盖该 fixture 与 Days 自身串行实现的对照。 [CLI](https://days.sh/docs/reference/cli/days) · [训练 trace 适配及模型偏差](https://days.sh/docs/configuration/workloads)
