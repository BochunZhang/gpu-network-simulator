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
| `system-000-init-environment-00` 系统盘点 | Python 标准库、`nvidia-smi` 与被探测的工具 |
| `gedes-000-completion-audit-00` GeDES | C++17、CMake、CUDA Toolkit 和 NVIDIA 驱动 |
| `days-000-cuda-comparison-00` Days AGO CUDA | Rust/Cargo、CUDA 13 与 `cudarc`，另需完成固定版本的 SM100 适配 |
| `system-001-gpu-transfer-00` 四卡传输 | C++17、NVCC 和 CUDA Runtime |

当前这些程序无需安装 PyTorch 或 NCCL。Days 中的 collective/NCCL 名称用于描述模拟通信，宿主执行依赖以构建清单为准。后续真实训练或 NCCL collective 校准任务会单独声明相应依赖。

## Python 环境

`tools/runner.sh` 从任意工作目录定位仓库，在以下目录维护运行环境。

- `.python/<系统>-<架构>/` 保存所选 Python 运行时和元数据。优先复用设备上可用的 Python 3.9+ 基础解释器，并保存显式引用。已有解释器不可用时，通过 `uv` 获取受管理的 Python 3.12，下载时使用系统信任的证书。
- `.venv/<系统>-<架构>-py<版本>-<运行时摘要>-b2/` 保存虚拟环境。创建时使用基础解释器的真实路径。不同系统、架构和 Python 运行时分别使用目录，旧版环境保留。

共享目录的首次初始化通过原子目录锁协调。可通过 `GPU_NET_SIM_PYTHON=/usr/bin/python3 bash tools/runner.sh` 显式使用系统 Python。`--help` 显示执行器参数。

## 任务与子任务

每个研究项目按实验顺序独立编号。父任务的标识是 `{simulator或system}-{三位序号}-{简称}`，清单保存在 `tasks/{父任务标识}.json`。例如 `system-000-init-environment` 属于系统盘点，`gedes-000-completion-audit` 属于 GeDES，两者的实验序号均为 `000`。每个父任务内按发布顺序给子任务分配两位编号 `00` 至 `99`。

| 操作 | 清单改动 | runner 行为 |
| --- | --- | --- |
| 首次发布 | 写入父任务和子任务的 `status`，格式为 UTC `YYYY-MM-DDTHH:MM:SSZ` | 领取满足 GPU 数量要求的子任务，记录两个发布时间。父任务时间固定。 |
| 补充实验 | 保留子任务 `00` 的拓扑 A、B、C，追加 `01`，在其参数中仅指定拓扑 D | 已领取的 `00` 保留原始记录。新发布的 `01` 有自己的领取锁、状态和结果。 |
| 调试后重跑 | 在同一个提交中修复代码或参数，并把该子任务的 `status` 改为更晚的时间 | 用新提交运行一次，旧执行记录和结果包继续保留。 |
| 暂停领取 | 将子任务 `enabled` 设为 `false` | 暂停该子任务的新领取。恢复后沿用其原有发布时间和领取记录。 |
| 废弃子任务 | 将子任务的 `status` 改为 `"ignore"`，保留两位编号和定义 | 停止新领取，让已领取的实验结束。runner 校验并迁移待归档的结果，在 CPFS 标记废弃。 |
| 废弃父任务 | 将父任务的 `status` 改为 `"ignore"`，保留全部子任务 | 对所有子任务及其历史执行应用废弃规则。废弃的标识永久保留。 |

例如，下面的示意清单先发布硬件探测，再单独补测工具链。每个子任务都有执行处理器 `kind` 和参数对象 `params`。`name` 默认沿用父任务名，`resources.gpus` 表示至少需要几张可见 GPU。`execution` 可选，默认完整执行。

```json
{
  "id": "system-002-inventory-refresh",
  "name": "GB200 inventory refresh",
  "status": "2026-11-18T11:00:00Z",
  "subtasks": {
    "00": {
      "status": "2026-11-18T11:00:00Z",
      "kind": "gpu_inventory",
      "params": {},
      "resources": {"gpus": 4},
      "execution": {"mode": "full", "targets": ["hardware-info"]}
    },
    "01": {
      "status": "2026-11-18T12:00:00Z",
      "kind": "gpu_inventory",
      "params": {},
      "resources": {"gpus": 4},
      "execution": {"mode": "full", "targets": ["toolchain"]}
    }
  }
}
```

在真实清单上可选择整个父任务，或直接选择子任务。两种命令加上 `--once` 都最多领取一个实验。

```bash
bash tools/runner.sh --task system-000-init-environment --once
bash tools/runner.sh --task system-000-init-environment-00 --once
```

运行状态保存执行身份、源码提交和执行主机。子任务的执行命令自动生成。处理器将实际调用的程序、参数、工作目录、退出码与时间写入 `commands.json`。新增处理器可随任务提交发布。**更新 runner 控制器时**，先让旧进程中的实验完成，停止各台旧轮询器，在共享 checkout 执行 `git pull --ff-only` 并重启 `bash tools/runner.sh`。

### 现有任务目录

每项旧任务迁入对应父任务的 `00`。表中标明它在迁移时的旧执行身份，可用于查找原锁、状态和压缩包。迁移清单的 `legacy` 固定此映射。首次迁移沿用原来的领取与结果，后续发布时间使用分类子任务标识。

| 父任务 | 旧执行身份 | GPU | 迁移时领取状态 | 实验目的 |
| --- | --- | ---: | --- | --- |
| [days-000-cuda-comparison](tasks/days-000-cuda-comparison.json) | `002 v1` | 1 | 暂停 | 对照 Days Scalar、CPU 和 CUDA 实现，等待 GB200 SM100 适配。 |
| [gedes-000-completion-audit](tasks/gedes-000-completion-audit.json) | `001 v3` | 1 | 开放 | 修复流启动队列，核对 k4/k8 的 TCP 节点和信道完成计数。 |
| [gedes-001-drain](tasks/gedes-001-drain.json) | `004 v1` | 1 | 开放 | 核对 k4/k8 流、TCP 字节与队列排空，验证提前停止检测。 |
| [gedes-002-timeline](tasks/gedes-002-timeline.json) | `008 v1` | 1 | 开放 | 采集固定 GeDES k8 的两个 GPU 时间窗口。 |
| [nsx-000-single-gpu](tasks/nsx-000-single-gpu.json) | `005 v1` | 1 | 开放 | 比较单卡全局与局部时间边界，验证同刻顺序和因果性。 |
| [nsx-001-four-gpu](tasks/nsx-001-four-gpu.json) | `006 v1` | 4 | 开放 | 比较单卡和四卡中转、P2P 路径，验证事件顺序、完整性与排空。 |
| [roce-000-training-checks](tasks/roce-000-training-checks.json) | `007 v1` | 1 | 暂停 | 核对单卡 RC 收发及 CQE 依赖。 |
| [roce-001-training-profile](tasks/roce-001-training-profile.json) | `009 v1` | 1 | 开放 | 采集单卡 RC kernel 内部事件并与标量对照核验。 |
| [roce-002-owner-sweep](tasks/roce-002-owner-sweep.json) | `010 v1` | 1 | 暂停 | 比较因果边界、CTA 任务分派与热点效应。 |
| [roce-003-stage-pipeline](tasks/roce-003-stage-pipeline.json) | `011 v1` | 1 | 暂停 | 比较 RC 处理各阶段和结果输出的组合。 |
| [roce-004-instruction-profile](tasks/roce-004-instruction-profile.json) | `012 v1` | 1 | 暂停 | 比较 RC kernel 的指令工作集、寄存器与正常计时。 |
| [system-000-init-environment](tasks/system-000-init-environment.json) | `000 v3` | 4 | 开放 | 刷新 GB200 实时探测，用 v2 基线验证派生报告复用。 |
| [system-001-gpu-transfer](tasks/system-001-gpu-transfer.json) | `003 v1` | 4 | 开放 | 测量四卡本地及有向跨卡复制的两种大小，逐字节验证结果。 |

## 子任务的增量执行

同一父任务内，新发布的子任务可用已完成的较早执行作为增量基线，引用执行记录中的 `id`、`version` 和产物清单的实际 SHA-256。以下是已迁移的系统盘点子任务 `00` 使用的 `execution` 配置片段，基线为旧编号 `000 v2`。

```json
{
  "execution": {
    "mode": "incremental",
    "base": {
      "id": "000",
      "version": 2,
      "artifacts_manifest_sha256": "<基线产物清单的 SHA-256>"
    },
    "reuse": "verified-compatible"
  }
}
```

更换基线时，使用对应产物清单的实际摘要。基线需要终态 `result.json` 与已封存的产物清单，异常退出留下的未封存检查点保留供排查。先完成基线，再发布引用它的新版本。追加子任务时保留基线子任务的清单与历史结果。

runner 在永久领取前检查基线，从 CPFS 的 `results/` 或 `results-archive/` 校验并冻结所需文件。缺少基线时写入 `status/000-system-info/v3.waiting.json`，不消耗该版本的永久领取记录，继续查找其他任务。基线在归档移动中的位置变化由共享锁协调。

每个 case 的输入、相关实现、依赖产物和必要环境共同决定复用键。只有通过校验且文件完整的兼容 case 才能复用。失败、新增、依赖变化或产物损坏的 case 重新执行。基线整体失败时，独立通过且兼容的 case 仍可复用。`execution.force_rerun` 可以列出本次必须执行的 case，`execution.targets` 可以选择目标及其依赖。

新版本写入独立的 `results/000-system-info/v3/`。其产物清单记录每个 case 的 `executed`/`reused` 来源、执行结果、文件哈希与耗时，结果记录中的 `case_counts` 汇总执行、复用、失败和跳过数量。每个 case 完成后原子保存清单，修正后的重新发布使用该子任务的下一次发布时间。所有目标 case 通过后，新版本才记为成功。

当前 `gpu_inventory` 有 `hardware-info`、`topology`、`nvlink-check`、`toolchain` 和 `inventory-report` 五个 case。前四项在本次设备上重新采集，确定性的 `inventory-report` 可在输入和实现一致时复用。其他旧版处理器继续支持完整运行，增加 case 接口后可使用增量模式。当前增量机制覆盖已提交 case 的复用和重跑，仿真内部的中途恢复需要模型实现完整状态快照。

## 已迁移子任务的诊断口径

本节使用上述映射表中的旧执行身份，以便与相应历史结果直接核对。

`001 v3`（执行提交 `4ce8139`）保留固定的上游源码、种子 41、k4/k8 参数和轮次上限。临时源码副本按实际生成的启动时间数分配队列，避免 k8 将 128 项复制到容量为 100 的分配中。退出后读取上游 timer 的 TCP 节点和信道块完成计数。

每个 run 保存实际迭代轮数、是否触及原有一百万轮上限、`upstream_finished`、有效 FCT 行数和报告中首条流的输入摘要。上游完成判定为假会标记该 case 失败，并保留其诊断输出。后续重复和详细 case 独立依赖构建，继续收集证据。CSV 仅覆盖每连接第一条流，后续需要进一步验证完整协议排空。

`003 v1` 从当前 CUDA 可见设备中选取前四张，保持既有可见性设置。每张卡分配两个 64 MiB 缓冲区，分别用 1 MiB 和 64 MiB 数据测试本卡 D2D 与所有有向 P2P 组合。每个组合预热三次、计时复制十次，使用目标设备 stream 的 CUDA event 计时，再进行完整数据校验。结果有 32 行，单位为 payload 的十进制 GB/s，同时记录 CUDA 版本、逻辑 GPU UUID、peer 能力和拓扑。物理链路归因需要额外计数器验证。

`005 v1` 和 `006 v1` 是独立的 NSX 机制实验。`005` 比较单卡全局与局部时间承诺，要求逐模块事件顺序与串行参考一致，并检测在时间边界错误提交的反例。`006` 固定拓扑和输入，把同一网络分到 4 卡，分别验证主机中转和 CUDA P2P 的 12 个有向跨卡路径，以单卡和 CPU 的事件记录校验顺序、完整性及排空。性能记录覆盖该固定用例。[NSX 论文](https://www.cs.rice.edu/~eugeneng/papers/NAIC25.pdf)

后续回收的归档确认写入 `status/`。远端轮询时先处理确认，再领取最新任务版本。实际结果保存在本地与 CPFS，Git 同步任务清单、代码和状态元数据。

## CPFS 锁与状态

运行时目录位于同一个共享仓库下，各台设备使用同一套目录。新发布的子任务用完整子任务标识作为目录名，用 UTC 发布时间的 Unix 秒作为运行记录中的整数 `version`。以下是上文示意子任务 `01` 的路径。

```text
.lock/system-002-inventory-refresh-01.task.lock/
.lock/system-002-inventory-refresh-01.v1795003200.lock/
status/system-002-inventory-refresh-01/v1795003200.json
results/system-002-inventory-refresh-01/v1795003200/
results/system-002-inventory-refresh-01/system-002-inventory-refresh-01.v1795003200.tar.gz
```

领取通过原子 `mkdir` 竞争。完成后先保存终态，再释放任务执行锁，保留版本领取记录。版本锁中保存主机、进程、领取标识及版本信息。重复启动会跳过已领取的版本。同一共享 checkout 的 Git 拉取和源码导出通过短期目录锁串行化，实验从独立快照执行。

CPFS 挂载需要提供跨客户端互斥的目录创建与可见的元数据更新。CPFS 的 NFS 协议服务不支持 NLM 和原子追加，因此共享协调使用目录与独立文件，每个实验单独记录日志。当前测试在本地文件系统验证了竞争与恢复，真实 CPFS 跨主机验证需要在用户设备上执行。[CPFS 限制](https://www.alibabacloud.com/help/en/cpfs/cpfsonecs/product-overview/limits-1)

状态包括 `running`、`succeeded`、`failed` 和 `abandoned`。崩溃遗留的锁保留给人工恢复。先在记录的原主机上停止轮询器并确认实验进程已经退出，再执行以下命令。恢复命令使用领取记录中的执行 ID 与整数版本。需要重跑时，在开发端更新子任务发布时间。下例恢复旧执行身份 `000 v2`。

```bash
bash tools/runner.sh --abandon 000 2
```

如果原主机的 Git 拉取进程崩溃并留下 Git 协调锁，确认该进程已退出后，在原主机执行 `bash tools/runner.sh --recover-git-lock`。环境初始化崩溃遗留的锁需要根据其中的主机和进程记录人工检查后恢复。若崩溃发生在目录创建后、`owner.json` 写入前，该目录会保持阻塞。此时先停止可能持有该锁的轮询器及子进程，确认已全部退出，再人工移走该孤立目录并保留恢复记录。正常运行时保留 `.lock/`、`status/` 和 `results/`，供后续启动检查。

## 结果回收与归档

```bash
cat status/000-system-info/v2.json
cat results/000-system-info/v2/result.json
```

新包名为 `{父任务标识}-{两位子任务编号}.v{发布时间的Unix秒}.tar.gz`。下文以兼容迁移的原始结果包 `000-system-info.v2.tar.gz` 为例。文件保存在对应任务目录，下载后单独查看文件名也能识别任务和版本。运行状态、结果记录和归档确认包含 `archive_name`。既有 `vN.tar.gz` 结果包继续兼容读取，迁移通过重命名保留原始字节及校验和。

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

废弃的子任务由 runner 校验结果身份、原包内容和 SHA-256，直接向 CPFS 的 `results-archive/` 归档。已有下载归档回执的包沿用其校验值。归档后，执行记录的 `disposition` 为 `ignored`。仍在运行的实验完成后再处理，归档失败时保留原数据并在下次轮询重试。实验的 `state` 保持 `succeeded` 或 `failed` 等原结果，`delivery_state` 记录 `awaiting_download` 或 `archived`。失败实验同样可以回收归档。移动中断时可继续处理，遇到不同内容的目标文件会停止并保留数据。崩溃恢复后继续使用同一组结果目录和归档目录。

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

本轮先开放 GeDES 时间线和 RC 训练剖析两个子任务采集 profiler。其余训练实验等首批时间线回收分析后按需启用。每项任务使用一张物理 GPU。研究问题、阶段顺序与验收条件见 [plan.md](plan.md)。

| 任务 | 执行与输出 |
| --- | --- |
| `roce-000-training-checks-00` | 分段 SEND、丢 DATA/ACK、RNR、小型 Ring 和全对全交换。逐项比较单 CTA CUDA 与标量参考的提交、接收 CQE、发送 CQE、字节和排空；保存每例 CSV。 |
| `gedes-002-timeline-00` | 对相同 GeDES k8 二进制先正常运行，再分别用 Nsight Systems 采集第 2000 和 6900 轮起的 16 轮。三次均须保留输入哈希、128 条流、7001 轮及排空门限。返回的 timeline.json 可在 Perfetto 打开；完整 .nsys-rep 和 SQLite 文件保存在 CPFS 原始目录。 |
| `roce-001-training-profile-00` | 短窗 RC 设备内 trace，将 globaltimer、SM、CTA、阶段、QP 和 WR 与虚拟时刻关联，核对插桩前后相同的提交记录。主机同步与设备执行可能重叠。 |
| `roce-002-owner-sweep-00` | 比较安全时间边界与 CTA 任务分派的组合，用独立顺序参考检验热点队列的输出。 |
| `roce-003-stage-pipeline-00` | 对相同模拟包比较拆分/融合 kernel、协作加载/专职 producer warp、同步/双缓冲输出。比较完整输出字节，记录五轮配对时长、编译资源与可用的机器指令报告。 |
| `roce-004-instruction-profile-00` | 在相同融合 kernel 上改变循环展开程度，比较静态 SASS、寄存器/溢出、正常计时与设备开放的指令计数器。 |

合成 RC 用例设定 400 Gbps、每跳 500 ns、一台交换机、每帧固定 64 字节模型开销、4096 字节分段与每 QP 停止等待。训练式后继用端点自己的完成事件解锁。ring 用转发块来源和邻接依赖检查消息顺序，全对全提交固定 peer 交换。真实 SimAI 拓扑、管理网络、PAT 和训练步的验收使用相应训练入口。

从仓库根目录可运行标量检查：

```sh
c++ -std=c++17 -O2 -x c++ benchmarks/training_roce_engine.cu benchmarks/training_roce_main.cpp -o /tmp/training-roce
/tmp/training-roce --backend cpu --workload closed-loop --ranks 4 --bytes 8193 --fault drop-ack
```

任务随代码推送到本仓库，远程开发机上的 runner 自动拉取并运行。用户下载原始压缩包到 `results-remote/` 后，由助手校验、分析并提交归档确认。
