# GPU Network Simulator

单独的 GPU 包级离散事件仿真实验仓库。首批任务采集硬件和工具链、验证公开实现能否在目标机器上运行。[研究路线](docs/research-roadmap.md)给出自研仿真器的模型、指标和实验顺序；[第三方参考](third_party/README.md)记录作者入口、固定提交和通过条件。

## 在 GPU 机器上启动

需要 POSIX 主机（CUDA 实验用 Linux）、Python 3.9+ 和 Git。仓库公开可读，执行 worker 的账号需要本仓库的 **Contents 写权限**，用于领取任务和提交状态。同时需要能访问论文代码的公开仓库。以普通用户运行，先配置非交互 Git 认证。GeDES 需要可用的 NVIDIA 驱动、`nvcc`、C++ 编译器和 CMake。Days AGO 另需 Rust/Cargo 和兼容 CUDA 13 的 `nvcc`。使用权限充足且有余量的数据盘。

```bash
git clone https://github.com/BochunZhang/gpu-network-simulator.git
cd gpu-network-simulator
# 先只领一个任务，核对设备和结果目录；新队列的第一项是硬件盘点。
python3 tools/run_experiments.py --once
# 此后保持前台运行，可在 tmux 里手动启动；默认在仓库 results/ 写结果。
python3 tools/run_experiments.py
# 也可把结果改到容量充足的数据盘：
# python3 tools/run_experiments.py --results-dir /your/data/disk/gpu-network-simulator/results
```

进程约每 10 秒抓取 `main` 和 `experiment-status` 并查看新任务，包括实验运行期间；网络请求超过 10 秒时，下一轮顺延。`/tmp/gpu-network-simulator.lock` 按主机串行化所有本项目 worker，包含不同 checkout。GPU smoke 默认选 `nvidia-smi` 看到的首张物理卡。若机器共享，启动前用 `GPU_NET_SIM_GPU_UUID=GPU-…` 明确选择一张卡，使用 `nvidia-smi -L` 查完整 UUID；已有的 `CUDA_VISIBLE_DEVICES` 需要取消或设为相同 UUID。锁只协调本项目 worker；跨容器运行请共享同一锁目录和 inode。

## 从开发机提交并查看任务

每个任务占一个 `tasks/000001.json` 格式的文件。新实验使用下一个编号；同一个实验修正配置或实现时保留 id，增加整数 `version`，把代码与清单放在同一次提交里推到 `main`。每个 worker 按递增编号领取当前可执行版本。不同任务可在不同主机并行；同一任务的新版本会等旧版本退出。

```json
{"id": 1, "version": 2, "kind": "gpu_inventory", "params": {},
 "description": "Repeat inventory after updating the driver"}
```

首批固定任务：

| 编号 | 类型 | 通过条件 |
| --- | --- | --- |
| 000001 | `gpu_inventory` | 列出至少一张 NVIDIA GPU，记录显存、计算能力、驱动、拓扑和工具链。 |
| 000002 | `gedes_smoke` | 固定源码构建成功，作者 k8 命令实际执行并给出有效流完成时间。 |
| 000003 | `days_cuda_smoke` | 同一 TCP fixture 三种后端均实际处理数据，完整状态长度和指纹一致。固定版仅支持 SM 8.6/8.9/12.1。 |

同一次运行绑定 **任务 id、version、任务配置摘要和 Git 提交**。任务清单只支持执行器认识的固定类型和参数；实验代码取自该提交。新增任务类型或升级调度器时，在 GPU 主机审核更新后重启 worker。拥有写权限的合作者可以修改实验代码，请把 `main` 限定给信任的合作者。

初版没有硬件路由。混用不同架构的机器时，只在满足当期任务要求的机器上启动 worker。若已领取的版本因环境不匹配失败，修正环境或代码后递增 version。无效任务清单或未知类型会阻塞后续领取，修正并推送后继续。

状态位于独立的 `experiment-status` 分支，例：`status/000001/v1.json`。首次领取会创建分支。领取和完成通过 Git 推送时的旧版本比较串行化，提交成功后才开始执行；本地结果写完但状态尚未成功推送时，worker 会继续重试。

同一个队列的所有机器必须使用同一仓库和状态分支。不要重写或删除状态分支。如确实另开独立队列，为它另设 `--status-branch` 和 `--results-dir`；执行器遇到无法匹配到远程认领的本地终态会停止领取，以免重复执行。

```bash
git fetch origin experiment-status
git ls-tree -r --name-only FETCH_HEAD status/
git show FETCH_HEAD:status/000001/v1.json
```

终态为 `succeeded`、`failed` 或 `abandoned`。`Ctrl-C` 停止轮询，当前实验在后台继续；完成后用同一结果目录重启 worker，补发状态。若崩溃或推送中断使任务停在 `running` 且没有本地 `result.json`，停止认领主机的轮询器，确认该主机上的实验进程已经退出，再在该主机运行 `python3 tools/run_experiments.py --abandon 1 1`（自定义目录时照常传入）。需要重跑时，在开发机递增 version。不按时间自动释放领取，以免慢实验被另一台机器重复执行。

## 下载结果

每个版本将运行日志和输出留在 `results/000001/v1/`，要带回分析的是同级的 `results/000001/v1.tar.gz`。压缩包包含 `result.json` 的执行记录和白名单内的小型摘要、CSV 文件；构建日志和大 trace 留在机器上，临时源码和编译目录会清理。任务失败后也应保留结果目录，必要时另取 `worker.log`、`configure.log` 或 `build.log`。把包带回本地即可分析；Git 状态分支只同步状态和版本等少量元数据。

本地验证（不需要 GPU）：

```bash
python3 -B -m unittest discover -s tests -v
```
