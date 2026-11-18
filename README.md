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

## 当前程序依赖

| 任务或固定实现 | 执行依赖 |
| --- | --- |
| `000` 系统盘点 | Python 标准库、`nvidia-smi` 与被探测的工具 |
| `001` GeDES | C++17、CMake、CUDA Toolkit 和 NVIDIA 驱动 |
| `002` Days AGO CUDA | Rust/Cargo、CUDA 13 与 `cudarc`，另需完成固定版本的 SM100 适配 |
| `003` 四卡传输 | C++17、NVCC 和 CUDA Runtime |

当前这些程序无需安装 PyTorch 或 NCCL。Days 中的 collective/NCCL 名称用于描述模拟通信，宿主执行依赖以构建清单为准。后续真实训练或 NCCL collective 校准任务会单独声明相应依赖。

## Python 环境

`tools/runner.sh` 从任意工作目录定位仓库，在以下目录维护运行环境。

- `.python/<系统>-<架构>/` 保存所选 Python 运行时和元数据。优先复用设备上可用的 Python 3.9+ 基础解释器，并保存显式引用。已有解释器不可用时，通过 `uv` 获取受管理的 Python 3.12，下载时使用系统信任的证书。
- `.venv/<系统>-<架构>-py<版本>-<运行时摘要>-b2/` 保存虚拟环境。创建时使用基础解释器的真实路径。不同系统、架构和 Python 运行时分别使用目录，旧版环境保留。

共享目录的首次初始化通过原子目录锁协调。可通过 `GPU_NET_SIM_PYTHON=/usr/bin/python3 bash tools/runner.sh` 显式使用系统 Python。`--help` 显示执行器参数。

## 任务清单

每个任务使用独立的 JSON 文件，文件名固定为 `tasks/000.json`、`tasks/001.json`，编号范围为 `000` 到 `999`。编号在 JSON 中使用字符串保留前导零。

```json
{
  "id": "000",
  "name": "GB200 four-GPU hardware inventory",
  "slug": "system-info",
  "version": 2,
  "command": "bash tools/runner.sh --task 000 --once",
  "kind": "gpu_inventory",
  "params": {},
  "resources": {"gpus": 4},
  "enabled": true,
  "execution": {"mode": "full"},
  "description": "Collect fresh cases and publish an artifact manifest."
}
```

| 字段 | 含义 |
| --- | --- |
| `id` | 必填，三位编号，与文件名一致。 |
| `name` | 必填，任务完整名称。 |
| `slug` | 必填，简短的英文名称，如 `system-info`，用于状态和结果目录。 |
| `command` | 必填，复现实验的执行器入口命令，写入运行状态。 |
| `version` | 必填，正整数。修正代码或配置时递增，同一次提交包含代码和清单。 |
| `kind` | 必填，由该提交中的 `tools/experiment_handlers.py` 识别的实验类型。 |
| `params` | 必填，该处理器的参数对象。当前 smoke 使用空对象。 |
| `resources.gpus` | 可选，所需最少可见 GPU 数量，默认 0。设备满足数量后才能领取。 |
| `enabled` | 可选，默认 `true`。设为 `false` 暂停领取。 |
| `description` | 可选，实验说明与通过条件。 |
| `execution` | 可选，`full` 或带基线的 `incremental`。省略时按完整运行处理。 |

运行状态包含任务名称、简称、入口命令、源码提交和执行主机。处理器将实际调用的程序、完整参数、工作目录、退出码与时间写入 `commands.json`，并带入结果记录。新增 `slug` 和 `command` 保持现有实验版本，旧的 `000/v1` 状态和结果继续读取。

任务清单共享给所有设备。各设备按编号尝试领取可运行任务，已经被占用的任务会跳过，继续寻找其他任务。同一任务的新版本等旧版本释放任务锁后才执行。实验代码和处理器随拉取的提交更新，新增处理器会自动生效。升级调度器本身时，待当前实验完成后停止各台轮询器，更新共享 checkout，再重启入口。

| 编号 | 当前版本 | 名称 | GPU 需求 | 当前安排 |
| --- | --- | --- | --- | --- |
| `000` | 3 | GB200 系统盘点与增量验证 | 4 | 引用已完成 v2 的清单摘要，刷新实时探测，复用兼容的派生报告。 |
| `001` | 3 | GeDES 完成状态与主机阶段诊断 | 1 | 修复流启动时间队列容量，按原参数重跑并记录退出时的完成计数。 |
| `002` | 1 | Days AGO Scalar/CPU/CUDA 对照 | 1 | 暂停，待固定源码增加 GB200 `sm_100` 支持并准备 Rust/Cargo。 |
| `003` | 1 | 四卡 CUDA copy/P2P 基线 | 4 | 检查 4 个本卡与 12 个有向跨卡组合，在两种大小下测量并逐字节验证。 |
| `004` | 1 | GeDES 排空审计 | 1 | 核对 k4/k8 的流、TCP 字节与队列状态，验证提前停止会被检出。 |
| `005` | 1 | NSX 单卡局部时间承诺 | 1 | 对照 CPU 串行事件记录，检查输入队列暂空、时间边界和同时间戳次序。 |
| `006` | 1 | NSX 四卡事件交换 | 4 | 同一网络比较单卡、四卡主机中转和四卡 CUDA P2P，验证顺序与排空。 |


旧版六位编号的状态和结果原样保留，当前调度器使用三位编号清单。

## 同编号增量执行

任务 000 已完成 v2 的完整采集并产生 `artifacts-manifest.json`。当前 v3 保留编号与简称，引用已归档 v2 产物清单的 SHA-256，验证增量执行。以下是清单中的执行配置片段，其他必填字段继续保留。

```json
{
  "id": "000",
  "slug": "system-info",
  "version": 3,
  "execution": {
    "mode": "incremental",
    "base": {
      "id": "000",
      "version": 2,
      "artifacts_manifest_sha256": "3eb996f919b1545a53e1d970a5b0d02fc55a60dd266e09bb450f76f684e3e0a4"
    },
    "reuse": "verified-compatible"
  }
}
```

上面的 SHA-256 来自已完成 v2 的结果记录及归档确认状态。更换基线时，使用对应产物清单的实际摘要。基线需要终态 `result.json` 与已封存的产物清单，异常退出留下的未封存检查点保留供排查。先完成基线，再发布引用它的新版本。当前队列每个编号只有一份最新清单，基线尚未执行时保留其清单供设备领取。

runner 在永久领取前检查基线，从 CPFS 的 `results/` 或 `results-archive/` 校验并冻结所需文件。缺少基线时写入 `status/000-system-info/v3.waiting.json`，不消耗该版本的永久领取记录，继续查找其他任务。基线在归档移动中的位置变化由共享锁协调。

每个 case 的输入、相关实现、依赖产物和必要环境共同决定复用键。只有通过校验且文件完整的兼容 case 才能复用。失败、新增、依赖变化或产物损坏的 case 重新执行。基线整体失败时，独立通过且兼容的 case 仍可复用。`execution.force_rerun` 可以列出本次必须执行的 case，`execution.targets` 可以选择目标及其依赖。

新版本写入独立的 `results/000-system-info/v3/`。其产物清单记录每个 case 的 `executed`/`reused` 来源、执行结果、文件哈希与耗时，结果记录中的 `case_counts` 汇总执行、复用、失败和跳过数量。每个 case 完成后原子保存清单，失败后的修正继续沿用任务 000 的新版本。所有目标 case 通过后，新版本才记为成功。

当前 `gpu_inventory` 有 `hardware-info`、`topology`、`nvlink-check`、`toolchain` 和 `inventory-report` 五个 case。前四项在本次设备上重新采集，确定性的 `inventory-report` 可在输入和实现一致时复用。其他旧版处理器继续支持完整运行，增加 case 接口后可使用增量模式。当前增量机制覆盖已提交 case 的复用和重跑，仿真内部的中途恢复需要模型实现完整状态快照。

## 当前诊断任务

`001 v2` 在单张 GB200 上构建成功，k4 在 4,786 轮满足上游完成条件。三个 k8 case 各输出 128 条有效 FCT 记录，均在一百万轮上限处退出，且上游完成条件仍为假。两次普通运行的仿真循环为 78.67 和 78.79 秒，均对应触顶的运行。详细 case 的 `RunGraph` 包含图启动和 stream 同步。

`001 v3`（执行提交 `4ce8139`）保留固定的上游源码、种子 41、k4/k8 参数和轮次上限。临时源码副本按实际生成的启动时间数分配队列，避免 k8 将 128 项复制到容量为 100 的分配中。退出后读取上游 timer 的 TCP 节点和信道块完成计数。

同一张 GB200 上的 v3 复测通过全部 5 个 case。k4 在 4,786 轮完成，节点与信道块计数分别为 16/16 和 3/3。三个 k8 均在 7,001 轮完成，计数为 128/128 和 24/24。各 case 的 FCT CSV 与 v2 逐字节一致。两次 k8 普通计时的仿真循环均约为 1.646 秒。v2 的 78.67 和 78.79 秒包含触顶前的额外轮次，耗时差反映了退出条件恢复。

每个 run 保存实际迭代轮数、是否触及原有一百万轮上限、`upstream_finished`、有效 FCT 行数和报告中首条流的输入摘要。上游完成判定为假会标记该 case 失败，并保留其诊断输出。后续重复和详细 case 独立依赖构建，继续收集证据。CSV 仅覆盖每连接第一条流，后续需要进一步验证完整协议排空。旧 v1 的生成器使用 `random_device`，因此这批结果按新的固定输入条件解释。

`003 v1` 从当前 CUDA 可见设备中选取前四张，保持既有可见性设置。每张卡分配两个 64 MiB 缓冲区，分别用 1 MiB 和 64 MiB 数据测试本卡 D2D 与所有有向 P2P 组合。每个组合预热三次、计时复制十次，使用目标设备 stream 的 CUDA event 计时，再进行完整数据校验。结果有 32 行，单位为 payload 的十进制 GB/s，同时记录 CUDA 版本、逻辑 GPU UUID、peer 能力和拓扑。物理链路归因需要额外计数器验证。

`003 v1` 在一台四卡 GB200 上通过全部 32 项数据校验。按 CUDA event 计时，各有向 P2P 对的有效载荷吞吐在 1 MiB 时为 85.76–118.94 GB/s，在 64 MiB 时为 664.67–686.38 GB/s。每次测量只运行一对拷贝。

`004 v1` 在单张 GB200 上通过排空审计。k4 和 k8 分别完成 16 条和 128 条流，各连接的计划与确认字节一致，12 组受审计队列均为空，FCT CSV 与 `001 v3` 逐字节一致。k8 的提前停止对照在一轮后仍有 128 条待处理流。此时信道队列为空，独立的流与 TCP 状态检查仍判定尚未排空。

`005 v1` 和 `006 v1` 是独立的 NSX 机制实验。`005` 比较单卡全局与局部时间承诺，要求逐模块事件顺序与串行参考一致，并检测在时间边界错误提交的反例。`006` 固定拓扑和输入，把同一网络分到 4 卡，分别验证主机中转和 CUDA P2P 的 12 个有向跨卡路径，以单卡和 CPU 的事件记录校验顺序、完整性及排空。性能记录覆盖该固定用例。[NSX 论文](https://www.cs.rice.edu/~eugeneng/papers/NAIC25.pdf)

`005 v1` 在单张 GB200 上通过。两个正确调度均与 CPU 参考的 16 条事件逐项一致。在第一轮，局部承诺让独立分支提交了 4 条转发事件，全局承诺为 2 条。错误地处理时间戳等于承诺边界的反例打乱了同时间戳顺序，被事件校验检出。

`006 v1` 在四张 GB200 上通过。串行 CPU、单卡、四卡主机中转及四卡 CUDA P2P 的 768 条事件记录逐字节一致。两种四卡路径各交换 384 条跨卡事件，覆盖全部 12 个有向 GPU 对，并在 28 个窗口内排空。故意丢弃一条跨卡事件的对照最终少了 3 条提交记录，虽然队列为空，仍被事件校验检出。单次诊断的循环耗时为单卡 3.243 ms、四卡主机中转 6.569 ms、四卡 P2P 3.550 ms。主机中转串行复用一个缓冲区，P2P 在窗末同步批次，该结果仅比较本用例中这两种传输实现。

已收结果的确认保留在 `status/`。远端轮询时先处理确认，再领取最新任务版本。实际结果保存在本地与 CPFS，Git 同步任务清单、代码和状态元数据。

## CPFS 锁与状态

运行时目录位于同一个共享仓库下，各台设备必须使用同一套目录。

```text
.lock/000.task.lock/      同一任务所有版本共用的执行锁
.lock/000.v1.lock/        任务 000 版本 1 的永久领取记录
status/000-system-info/v2.json        共享状态
results/000-system-info/v2/           执行记录、日志和输出
results/000-system-info/000-system-info.v2.tar.gz     手动下载的摘要包
```

领取通过原子 `mkdir` 竞争。完成后先保存终态，再释放任务执行锁，保留版本领取记录。版本锁中保存主机、进程、领取标识及版本信息。重复启动会跳过已领取的版本。同一共享 checkout 的 Git 拉取和源码导出通过短期目录锁串行化，实验从独立快照执行。

CPFS 挂载需要提供跨客户端互斥的目录创建与可见的元数据更新。CPFS 的 NFS 协议服务不支持 NLM 和原子追加，因此共享协调使用目录与独立文件，每个实验单独记录日志。当前测试在本地文件系统验证了竞争与恢复，真实 CPFS 跨主机验证需要在用户设备上执行。[CPFS 限制](https://www.alibabacloud.com/help/en/cpfs/cpfsonecs/product-overview/limits-1)

状态包括 `running`、`succeeded`、`failed` 和 `abandoned`。崩溃遗留的锁保留给人工恢复。先在记录的原主机上停止轮询器并确认实验进程已经退出，再执行以下命令。需要重跑时，在开发端递增任务版本。

```bash
bash tools/runner.sh --abandon 000 2
```

如果原主机的 Git 拉取进程崩溃并留下 Git 协调锁，确认该进程已退出后，在原主机执行 `bash tools/runner.sh --recover-git-lock`。环境初始化崩溃遗留的锁需要根据其中的主机和进程记录人工检查后恢复。若崩溃发生在目录创建后、`owner.json` 写入前，该目录会保持阻塞。此时先停止可能持有该锁的轮询器及子进程，确认已全部退出，再人工移走该孤立目录并保留恢复记录。正常运行时保留 `.lock/`、`status/` 和 `results/`，供后续启动检查。

## 结果回收与归档

```bash
cat status/000-system-info/v2.json
cat results/000-system-info/v2/result.json
```

压缩包统一使用 `{三位编号}-{任务简称}.v{版本}.tar.gz`，例如 `000-system-info.v2.tar.gz`、`001-gedes-smoke.v1.tar.gz`。文件保存在对应任务目录，下载后单独查看文件名也能识别任务和版本。运行状态、结果记录和归档确认包含 `archive_name`。既有 `vN.tar.gz` 结果包继续兼容读取，迁移通过重命名保留原始字节及校验和。

GPU 设备完成后，`results/000-system-info/v2/` 和同级 `000-system-info.v2.tar.gz` 表示待下载数据。把原始 `000-system-info.v2.tar.gz` 下载到本地分析仓库的 `results-remote/`，保留压缩包原始字节。文件可放进子目录，归档工具通过包内执行记录确定任务身份。

```bash
python3 tools/archive_results.py
```

归档工具校验包内记录和文件，再写入本地 `results/000-system-info/v2/`，保留同级原始 `000-system-info.v2.tar.gz`，生成 `status/000-system-info/v2.archived.json`。成功归档后清理已消费的输入包，重复导入同一个包会校验并跳过。输入目录支持完整的 `.tar.gz` 结果包，松散目录保持原样供人工检查。

分析端随后把 `v2.archived.json` 提交并推送到 `main`。确认记录包含执行状态、简称、命令、领取标识、源码提交和原始压缩包的 SHA-256。原始结果、`results-remote/` 与 `results-archive/` 保留在本地，Git 跟踪归档确认记录。

CPFS 执行器下一次拉取后，核对确认记录与本机压缩包和执行身份，随后把远程端数据移到以下路径。

```text
results-archive/000-system-info/v2/
results-archive/000-system-info/000-system-info.v2.tar.gz
```

此后 CPFS 的 `results/` 只保留尚未确认归档的数据。实验的 `state` 保持 `succeeded` 或 `failed` 等原结果，`delivery_state` 记录 `awaiting_download` 或 `archived`。失败实验同样可以回收归档。移动中断时可继续处理，遇到不同内容的目标文件会停止并保留数据。崩溃恢复后继续使用同一组结果目录和归档目录。

某个版本的压缩包缺失、改名未被识别或校验冲突时，runner 会保留该版本数据，并写入 `status/000-system-info/v2.archive-error.json` 等诊断记录，列出原因、预期校验和和搜索路径，继续处理其他归档确认及新任务。补齐或修正文件后，后续轮询重新尝试。文件只改名时 SHA-256 保持一致，原始压缩包按确认记录校验。

任务处理器通过拉取的提交更新，runner 控制器修复需要在共享 checkout 更新后重启进程。遇到旧命名兼容问题时，先执行 `git pull --ff-only`，再重启 `bash tools/runner.sh`；运行中的旧控制器不会因 Git 拉取而替换自身代码。

旧版包缺少简称和命令时，工具按实验类型补全简称，并把生成的入口命令标记为复现命令。原始压缩包保持原样，确保已经下载的旧版包也能通过 SHA-256 确认。

构建日志和大 trace 保留在远程结果目录，随版本目录一起移入 `results-archive/`。摘要包包含执行记录、小型统计与 CSV，分析时可另取 `worker.log`、`configure.log` 和 `build.log`。

本地检查可运行以下命令，GPU 性能与 CPFS 跨机行为仍以目标设备上的实验为准。

```bash
bash tools/runner.sh --help
python3 -B -m unittest discover -s tests -v
```

## 单卡训练网络机制实验

每项任务选择一张当前可见的物理 GPU。公开用例可以独立构建；完整训练 DAG 在项目的训练后端上另外比较。

| 任务 | 执行与输出 |
| --- | --- |
| 007 | 分段 SEND、丢 DATA/ACK、RNR、小型 Ring 和全对全交换。逐项比较单 CTA CUDA 与标量参考的提交、接收 CQE、发送 CQE、字节和排空；保存每例 CSV。 |
| 008 | 对相同 GeDES k8 二进制先正常运行，再分别用 Nsight Systems 采集第 2000 和 6900 轮起的 16 轮。三次均须保留输入哈希、128 条流、7001 轮及排空门限。返回的 timeline.json 可在 Perfetto 打开；完整 .nsys-rep 和 SQLite 文件保存在 CPFS 原始目录。 |
| 009 | 短窗 RC 设备内 trace，将 globaltimer、SM、CTA、阶段、QP 和 WR 与虚拟时刻关联，核对插桩前后相同的提交记录。主机同步与设备执行可能重叠。 |
| 010 | 比较安全时间边界与 CTA 任务分派的组合，用独立顺序参考检验热点队列的输出。 |
| 011 | 对相同模拟包比较拆分/融合 kernel、协作加载/专职 producer warp、同步/双缓冲输出。比较完整输出字节，记录五轮配对时长、编译资源与可用的机器指令报告。 |
| 012 | 在相同融合 kernel 上改变循环展开程度，比较静态 SASS、寄存器/溢出、正常计时与设备开放的指令计数器。 |

合成 RC 用例设定 400 Gbps、每跳 500 ns、一台交换机、每帧固定 64 字节模型开销、4096 字节分段与每 QP 停止等待。训练式后继用端点自己的完成事件解锁。ring 用转发块来源和邻接依赖检查消息顺序，全对全提交固定 peer 交换。真实 SimAI 拓扑、管理网络、PAT 和训练步的验收使用相应训练入口。

从仓库根目录可运行标量检查：

```sh
c++ -std=c++17 -O2 -x c++ benchmarks/training_roce_engine.cu benchmarks/training_roce_main.cpp -o /tmp/training-roce
/tmp/training-roce --backend cpu --workload closed-loop --ranks 4 --bytes 8193 --fault drop-ack
```

GB200 任务使用 `bash tools/runner.sh --task 007 --once`，其余编号同理。
