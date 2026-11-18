# GPU Network Simulator

GPU 包级离散事件仿真实验仓库。多台 GB200 设备从公开仓库拉取任务，在共享 CPFS 上抢占执行，状态与结果写入 CPFS，由用户手动下载结果包。[研究路线](docs/research-roadmap.md)与[第三方参考](third_party/README.md)记录研究计划和源码版本。

## 启动执行器

在所有设备都能访问的同一 CPFS 目录下克隆一次。需要 Git、Bash 和 Python 3.9+。首批实验需要可用的 NVIDIA 驱动和 `nvidia-smi`，GeDES 还需要 CUDA Toolkit、C++ 编译器和 CMake。

```bash
git clone https://github.com/BochunZhang/gpu-network-simulator.git
cd gpu-network-simulator
bash tools/runner.sh --once
```

`--once` 最多执行一个可领取任务。持续运行时，各台设备在同一个共享仓库目录执行同一条命令。

```bash
bash tools/runner.sh
```

入口自动准备 Python 环境，随后约每 10 秒拉取 `main` 并检查任务。实验运行期间继续检查更新。每个任务使用领取时的 Git 提交快照，后续拉取不会改变正在执行的实验。GPU 设备只通过网络读取公开代码和依赖，结果保存在共享文件系统。

终端保持打开时可直接前台运行。需要断开 SSH 后继续执行时可以使用 `nohup`，tmux 可选。

```bash
nohup bash tools/runner.sh > /tmp/gpu-network-runner.log 2>&1 < /dev/null &
tail -f /tmp/gpu-network-runner.log
```

`/tmp/gpu-network-simulator.lock` 在每台主机上串行化本项目实验。每台四卡 GB200 设备默认一次执行一个任务，实验处理器决定使用满足任务需求的可见 GPU。`CUDA_VISIBLE_DEVICES` 可限制可见设备，`GPU_NET_SIM_GPU_UUID` 可在单卡 smoke 中指定可见池内的一张卡。`--worker-id` 仅用于记录执行者，默认使用主机名。

## Python 环境

`tools/runner.sh` 从任意工作目录定位仓库，在以下目录维护运行环境。

- `.python/<系统>-<架构>/` 保存所选 Python 运行时和元数据。存在 `uv` 时优先安装受管理的 Python 3.12。下载不可用时使用已安装的兼容 Python，并保存显式引用。
- `.venv/<系统>-<架构>-py<版本>-<运行时摘要>/` 保存虚拟环境。不同系统、架构和 Python 运行时分别使用目录。

共享目录的首次初始化通过原子目录锁协调。可通过 `GPU_NET_SIM_PYTHON=/usr/bin/python3 bash tools/runner.sh` 显式使用系统 Python。`--help` 显示执行器参数。

## 任务清单

每个任务使用独立的 JSON 文件，文件名固定为 `tasks/000.json`、`tasks/001.json`，编号范围为 `000` 到 `999`。编号在 JSON 中使用字符串保留前导零。

```json
{
  "id": "000",
  "name": "GB200 four-GPU hardware inventory",
  "version": 1,
  "kind": "gpu_inventory",
  "params": {},
  "resources": {"gpus": 4},
  "enabled": true,
  "description": "Record GPUs, memory, NVLink topology and the toolchain."
}
```

| 字段 | 含义 |
| --- | --- |
| `id` | 必填，三位编号，与文件名一致。 |
| `name` | 必填，任务名称。 |
| `version` | 必填，正整数。修正代码或配置时递增，同一次提交包含代码和清单。 |
| `kind` | 必填，由该提交中的 `tools/experiment_handlers.py` 识别的实验类型。 |
| `params` | 必填，该处理器的参数对象。当前 smoke 使用空对象。 |
| `resources.gpus` | 可选，所需最少可见 GPU 数量，默认 0。设备满足数量后才能领取。 |
| `enabled` | 可选，默认 `true`。设为 `false` 暂停领取。 |
| `description` | 可选，实验说明与通过条件。 |

任务清单共享给所有设备。各设备按编号尝试领取可运行任务，已经被占用的任务会跳过，继续寻找其他任务。同一任务的新版本等旧版本释放任务锁后才执行。实验代码和处理器随拉取的提交更新，新增处理器会自动生效。升级调度器本身时，待当前实验完成后停止各台轮询器，更新共享 checkout，再重启入口。

| 编号 | 名称 | GPU 需求 | 当前安排 |
| --- | --- | --- | --- |
| `000` | GB200 四卡硬件与工具链盘点 | 4 | 启用，记录物理和可见 GPU、显存、驱动及 NVLink 拓扑。 |
| `001` | GeDES Fat-tree k8 CUDA smoke | 1 | 启用，按实测计算能力编译，验证有效流完成。 |
| `002` | Days AGO Scalar/CPU/CUDA 对照 | 1 | 暂停，待固定源码增加 GB200 `sm_100` 支持后启用并递增版本。 |

旧版六位编号的状态和结果原样保留，当前调度器使用三位编号清单。

## CPFS 锁与状态

运行时目录位于同一个共享仓库下，各台设备必须使用同一套目录。

```text
.lock/000.task.lock/      同一任务所有版本共用的执行锁
.lock/000.v1.lock/        任务 000 版本 1 的永久领取记录
status/000/v1.json        共享状态
results/000/v1/           执行记录、日志和输出
results/000/v1.tar.gz     手动下载的摘要包
```

领取通过原子 `mkdir` 竞争。完成后先保存终态，再释放任务执行锁，保留版本领取记录。版本锁中保存主机、进程、领取标识及版本信息。重复启动会跳过已领取的版本。同一共享 checkout 的 Git 拉取和源码导出通过短期目录锁串行化，实验从独立快照执行。

CPFS 挂载需要提供跨客户端互斥的目录创建与可见的元数据更新。CPFS 的 NFS 协议服务不支持 NLM 和原子追加，因此共享协调使用目录与独立文件，每个实验单独记录日志。当前测试在本地文件系统验证了竞争与恢复，真实 CPFS 跨主机验证需要在用户设备上执行。[CPFS 限制](https://www.alibabacloud.com/help/en/cpfs/cpfsonecs/product-overview/limits-1)

状态包括 `running`、`succeeded`、`failed` 和 `abandoned`。崩溃遗留的锁保留给人工恢复。先在记录的原主机上停止轮询器并确认实验进程已经退出，再执行以下命令。需要重跑时，在开发端递增任务版本。

```bash
bash tools/runner.sh --abandon 000 1
```

如果原主机的 Git 拉取进程崩溃并留下 Git 协调锁，确认该进程已退出后，在原主机执行 `bash tools/runner.sh --recover-git-lock`。环境初始化崩溃遗留的锁需要根据其中的主机和进程记录人工检查后恢复。若崩溃发生在目录创建后、`owner.json` 写入前，该目录会保持阻塞。此时先停止可能持有该锁的轮询器及子进程，确认已全部退出，再人工移走该孤立目录并保留恢复记录。正常运行时保留 `.lock/`、`status/` 和 `results/`，供后续启动检查。

## 下载结果

```bash
cat status/000/v1.json
cat results/000/v1/result.json
```

把 `results/000/v1.tar.gz` 从 CPFS 下载并发送给分析端。摘要包包含执行版本、源码提交、小型统计与 CSV。构建日志和大 trace 保留在结果目录，临时源码与构建目录会清理。失败结果也应保留，分析时可另取 `worker.log`、`configure.log` 和 `build.log`。

本地检查可运行以下命令，GPU 性能与 CPFS 跨机行为仍以目标设备上的实验为准。

```bash
bash tools/runner.sh --help
python3 -B -m unittest discover -s tests -v
```
