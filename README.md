# GPU Network Simulator

GPU 包级离散事件仿真实验仓库。远程 GPU 设备定期拉取公开代码和任务，将状态与结果保存在本机，由用户手动下载结果包。[研究路线](docs/research-roadmap.md)给出模型、指标和实验顺序。[第三方参考](third_party/README.md)记录作者入口、固定提交和通过条件。

## 在 GPU 设备上启动

需要 POSIX 主机（CUDA 实验用 Linux）、Python 3.9+ 和 Git。以普通用户运行，在剩余空间充足的数据盘目录下克隆。执行器通过匿名 Git 读取公开仓库。GeDES 需要可用的 NVIDIA 驱动、`nvcc`、C++ 编译器和 CMake。Days AGO 另需 Rust/Cargo 和兼容 CUDA 13 的 `nvcc`。

```bash
git clone https://github.com/BochunZhang/gpu-network-simulator.git
cd gpu-network-simulator
# 首台设备使用 gpu01，先执行一个任务，核对硬件盘点结果。
python3 tools/run_experiments.py --worker-id gpu01 --once
# 持续检查并执行分配给 gpu01 的任务。
python3 tools/run_experiments.py --worker-id gpu01
```

执行器约每 10 秒抓取 `main` 并检查新任务，实验运行期间也会检查。网络请求超过 10 秒时，下一轮顺延。任务状态写入本地 `status/gpu01/`，运行日志和结果写入本地 `results/`。这些目录由 Git 忽略，结果包由用户手动下载。

**tmux 是可选工具。** 保持终端打开时直接运行上述命令。需要关闭 SSH 窗口时，可以用 `nohup` 启动，稍后查看日志。

```bash
mkdir -p results
nohup python3 -u tools/run_experiments.py --worker-id gpu01 > results/runner.log 2>&1 < /dev/null &
tail -f results/runner.log
```

也可自行放进 `tmux` 或系统服务。`Ctrl-C` 停止前台轮询器，已经启动的实验会在后台继续。重启轮询器后会从本地结果更新状态。

`/tmp/gpu-network-simulator.lock` 按主机串行化本项目 worker，包含不同 checkout。锁文件由运行中的进程持有。跨容器运行时需要共享同一锁目录和 inode。GPU smoke 默认选 `nvidia-smi` 看到的首张物理卡。若要选卡，启动前设置 `GPU_NET_SIM_GPU_UUID=GPU-…`，可用 `nvidia-smi -L` 查询完整 UUID。已有的 `CUDA_VISIBLE_DEVICES` 需要取消或设为相同 UUID。

可将状态和结果改到数据盘上的固定位置。首次执行、持续运行和恢复时都使用相同的 worker ID、状态目录与结果目录。

```bash
python3 tools/run_experiments.py --worker-id gpu01 \
  --status-dir /your/data/disk/gpu-network-simulator/status \
  --results-dir /your/data/disk/gpu-network-simulator/results
```

## 从本地开发端分配任务

每个任务占一个 `tasks/000001.json` 格式的文件。新实验使用下一个编号。同一个实验修正配置或实现时保留 `id`，递增整数 `version`，把代码与清单放在同一次提交里推到 `main`。远程设备会执行新版本绑定的代码，并保留旧版状态和结果。

```json
{"id": 1, "version": 2, "worker": "gpu01", "kind": "gpu_inventory", "params": {},
 "description": "Record the GPU and toolchain"}
```

`worker` 指定唯一执行设备，与启动参数 `--worker-id` 一致。首台设备使用 `gpu01`，增加设备时使用 `gpu02` 等不同 ID，并在本地开发端提前分配任务。每个 ID 只对应一台物理设备。设备仅能读取仓库，因此两台设备复用相同 ID 时无法相互发现。同一任务各版本默认固定分配给同一设备。需要迁移时，先人工停止原设备的轮询器，确认实验进程已经退出并收集结果，再修改 `worker` 并递增 `version`。

每台设备按递增编号执行属于自己的任务。不同设备可并行执行各自的任务。同一设备上，旧版本处于 `running` 时，新版本等待。任务清单必须包含 `worker`。配置发生变化时递增 `version`，相同版本的配置变更会报错。

| 编号 | 当前版本 | 设备 | 类型与通过条件 |
| --- | --- | --- | --- |
| 000001 | 2 | gpu01 | `gpu_inventory` 列出至少一张 NVIDIA GPU，记录显存、计算能力、驱动、拓扑和工具链。 |
| 000002 | 2 | gpu01 | `gedes_smoke` 固定源码构建成功，作者 k8 命令实际执行并给出有效流完成时间。 |
| 000003 | 2 | gpu01 | `days_cuda_smoke` 同一 TCP fixture 三种后端均处理数据，完整状态长度和指纹一致。固定版支持 SM 8.6/8.9/12.1。 |

每次执行绑定任务 ID、版本、配置摘要和 Git 提交。任务类型与参数由对应提交的实验处理器验证，新增实验处理器会随拉取的代码生效。升级调度器本身时，在 GPU 设备上更新 checkout 后重启 worker。请按设备的 GPU 架构和工具链分配实验。失败版本在修正环境或代码后通过递增版本重跑。

## 查看状态与下载结果

默认本地状态路径为 `status/gpu01/000001/v2.json`。状态包括 `running`、`succeeded`、`failed` 和 `abandoned`。执行器重启时依据本地状态和结果恢复，完成的版本会跳过。

```bash
cat status/gpu01/000001/v2.json
cat results/000001/v2/result.json
```

若崩溃使状态停在 `running`，先停止原设备上的轮询器并确认实验进程已经退出，再在原设备上将该版本标记为 `abandoned`。使用自定义目录时传入相同的目录参数。

```bash
python3 tools/run_experiments.py --worker-id gpu01 --abandon 1 2
```

每个版本的日志和输出保存在 `results/000001/v2/`，供用户手动下载的摘要包为同级的 `results/000001/v2.tar.gz`。压缩包包含 `result.json` 执行记录和白名单内的小型摘要、CSV 文件。构建日志和大 trace 留在设备上，临时源码和编译目录会清理。失败结果也应保留，分析时可另取 `worker.log`、`configure.log` 或 `build.log`。将压缩包下载后发送给分析端即可。

本地验证命令适用于具备 Git 和 Python 的机器。

```bash
python3 -B -m unittest discover -s tests -v
```
