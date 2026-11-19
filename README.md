# GPU Network Simulator

多台 GB200 设备从本仓库拉取经项目负责人 review 的 GPU 包级离散事件仿真实验，在共享 CPFS 上领取任务、保存状态和结果。原始结果包由用户手动下载。

2026-11-18 建立的新队列从零项任务开始。此前 13 项实验已单独备份，旧编号、结果和运行状态仅供回查。后续如要重做其中一个实验，先 review 其目的、配置和验收条件，再按新队列的分类顺序发布。[历史实验索引](history/pre-review-20261118/README.md) · [研究路线草案](docs/research-roadmap.md) · [第三方参考](third_party/README.md)

## FT48 模拟器源码与 warp 测量

FT48 处理器从领取提交中的 [Days AGO](simulators/days-ago/SOURCE.md) 和 [GeDES](simulators/gedes/PROVENANCE.md) 构建。两项 `02` 子任务在单张 GB200 上采集完整 FT48 运行的 warp 进入与退出、CTA 归属及 SM 位置。Days 按 graph replay 和末尾的 gather 回收，GeDES 每 32 次 replay 回收一批。两项都另存 CUDA Graph 的依赖图。

下载结果包后，打开 `warp/preview.perfetto.json` 查看第一轮的部分 CTA，或按 replay 和 kernel 选择完整数据中的其他区间。包内的 `warp/lifetimes.bin` 和 `warp/index.sqlite` 分别保存全部记录和索引；若重放次数与已有 FT48 数据相同，单是二进制记录约为 Days 0.812 GiB、GeDES 51.38 GiB（未压缩）。[记录格式与查看命令](profiling/warp_trace/README.md)

## 启动执行器

在所有设备都能访问的同一 CPFS 目录下克隆一次。需要 Git、Bash 和 Python 3.9+。GPU 实验按所选处理器准备相应驱动和编译工具。

```bash
git clone https://github.com/BochunZhang/gpu-network-simulator.git
cd gpu-network-simulator
bash tools/runner.sh --once
```

`--once` 最多执行一个可领取任务。持续运行时，各台设备在同一个共享仓库目录执行同一条命令。

```bash
bash tools/runner.sh
```

入口自动准备 Python 环境，由 `tools/runner_loop.py` 持有主机设备锁。runner 约每 10 秒检查任务和控制器更新。控制器及其依赖从 `origin/main` 的同一 Git 提交导出，保存在 `.lock/controller-cache/{工具树哈希}/tools/`。控制器在当前任务完成、结果包封存且状态已核对之后、下次领取之前切换。只改任务清单或回执的提交可直接读取，控制器继续使用当前工具树。共享 checkout 无需随轮询切换。

`--source-branch` 选择任务分支，控制器始终来自 `origin/main`。每个任务固定领取时的实验源码和控制器快照，从启动到打包使用同一份执行代码。共享 Git、锁、状态、结果和 Nsight Systems 安装路径始终以实际 CPFS 仓库目录为根。GPU 设备只通过网络读取公开代码和依赖，结果保存在共享文件系统。

从旧启动入口迁移时，先让各台设备的在途实验封存，停止旧轮询器，在共享 checkout 执行一次 `git pull --ff-only`，然后逐台重启 `bash tools/runner.sh`。首次启动新版入口后，后续控制器更新由 runner 处理。

任务进入终态后，runner 输出一行 `complete <任务标识> <版本> state=<状态> archive_name=<包名>`。成功和已记录的失败均会输出；包未生成时 `archive_name=-`。

终端保持打开时可直接前台运行。需要断开 SSH 后继续执行时可以使用 `nohup`，tmux 可选。

```bash
nohup bash tools/runner.sh > /tmp/gpu-network-runner.log 2>&1 < /dev/null &
tail -f /tmp/gpu-network-runner.log
```

`/tmp/gpu-network-simulator.lock` 在每台主机上串行化本项目实验。每台四卡 GB200 设备默认一次执行一个任务，实验处理器决定使用满足任务需求的可见 GPU。`CUDA_VISIBLE_DEVICES` 可限制可见设备，`GPU_NET_SIM_GPU_UUID` 可在单卡 smoke 中指定可见池内的一张卡。`--worker-id` 仅用于记录执行者，默认使用主机名。

## 仓库内处理器的依赖

| 处理器 | 执行依赖 |
| --- | --- |
| 系统盘点 | Python 标准库、`nvidia-smi` 与被探测的工具 |
| GeDES | C++17、CMake、CUDA Toolkit 和 NVIDIA 驱动 |
| Days AGO CUDA | Rust/Cargo、支持 SM100 的 CUDA 13 与 `cudarc`。FatTree48 处理器在固定源码上应用 SM100 适配 |
| 四卡传输 | C++17、NVCC 和 CUDA Runtime |

当前这些程序无需安装 PyTorch 或 NCCL。Days 中的 collective/NCCL 名称用于描述模拟通信，宿主执行依赖以构建清单为准。后续真实训练或 NCCL collective 校准任务会单独声明相应依赖。

Days AGO 在构建前检测 `rustc` 和 `cargo`，缺少可用工具链时自动用当前账户执行固定源码中的 `tools/install_rust.sh`。安装与版本探测记录随实验回包。也可以提前以 runner 账户执行 `bash tools/install_rust.sh`。

## Python 环境

`tools/runner.sh` 从任意工作目录定位仓库，在以下目录维护运行环境。

- `.python/<系统>-<架构>/` 保存所选 Python 运行时和元数据。优先复用设备上可用的 Python 3.9+ 基础解释器，并保存显式引用。已有解释器不可用时，通过 `uv` 获取受管理的 Python 3.12，下载时使用系统信任的证书。
- `.venv/<系统>-<架构>-py<版本>-<运行时摘要>-b2/` 保存虚拟环境。创建时使用基础解释器的真实路径。不同系统、架构和 Python 运行时分别使用目录，旧版环境保留。

共享目录的首次初始化通过原子目录锁协调。可通过 `GPU_NET_SIM_PYTHON=/usr/bin/python3 bash tools/runner.sh` 显式使用系统 Python。`--help` 显示执行器参数。

## 任务与子任务

每个研究项目按实验顺序独立编号。父任务的标识是 `{simulator或system}-{三位序号}-{简称}`，清单保存在 `tasks/{父任务标识}.json`。分类可包含连字符，例如 `days-ago`。系统、Days AGO 和 GeDES 各自的第一个实验都可使用序号 `000`。每个父任务内按发布顺序给子任务分配两位编号 `00` 至 `99`。

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
  "id": "system-000-inventory-refresh",
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

使用 `--task` 可选择父任务或具体子任务。`--once` 最多领取一个实验。下面两条命令使用同一份示意清单。

```bash
bash tools/runner.sh --task system-000-inventory-refresh --once
bash tools/runner.sh --task system-000-inventory-refresh-00 --once
```

运行状态保存执行身份、实验源码提交和执行主机。结果记录用 `runner_controller_commit` 和 `runner_controller_tree` 标识控制器快照，`runner_sha256` 记录该次领取、执行和打包使用的控制器脚本校验和。子任务的执行命令自动生成。处理器将实际调用的程序、参数、工作目录、退出码与时间写入 `commands.json`。新增处理器可随任务提交发布。

## 子任务的增量执行

同一父任务内，新子任务可用已完成的较早执行作为增量基线，引用其执行记录中的 `id`、`version` 和产物清单的实际 SHA-256。以下示例引用上文示意清单中的子任务 `00`。

```json
{
  "execution": {
    "mode": "incremental",
    "base": {
      "id": "system-000-inventory-refresh-00",
      "version": 1794999600,
      "artifacts_manifest_sha256": "<基线产物清单的 SHA-256>"
    },
    "reuse": "verified-compatible"
  }
}
```

更换基线时，使用对应产物清单的实际摘要。基线需要终态 `result.json` 与已封存的产物清单，异常退出留下的未封存检查点保留供排查。先完成基线，再发布引用它的新版本。追加子任务时保留基线子任务的清单与历史结果。

runner 在永久领取前检查基线，从 CPFS 的 `results/` 或 `results-archive/` 校验并冻结所需文件。缺少基线时写入 `status/{子任务标识}/{version_label}.waiting.json`，不消耗该版本的永久领取记录，继续查找其他任务。基线在归档移动中的位置变化由共享锁协调。

每个 case 的输入、相关实现、依赖产物和必要环境共同决定复用键。只有通过校验且文件完整的兼容 case 才能复用。失败、新增、依赖变化或产物损坏的 case 重新执行。基线整体失败时，独立通过且兼容的 case 仍可复用。`execution.force_rerun` 可以列出本次必须执行的 case，`execution.targets` 可以选择目标及其依赖。

新版本写入独立的 `results/{子任务标识}/{version_label}/`。其产物清单记录每个 case 的 `executed`/`reused` 来源、执行结果、文件哈希与耗时，结果记录中的 `case_counts` 汇总执行、复用、失败和跳过数量。每个 case 完成后原子保存清单，修正后的重新发布使用该子任务的下一次发布时间。所有目标 case 通过后，新版本才记为成功。

当前 `gpu_inventory` 有 `hardware-info`、`topology`、`nvlink-check`、`toolchain` 和 `inventory-report` 五个 case。前四项在本次设备上重新采集，确定性的 `inventory-report` 可在输入和实现一致时复用。其他旧版处理器继续支持完整运行，增加 case 接口后可使用增量模式。当前增量机制覆盖已提交 case 的复用和重跑，仿真内部的中途恢复需要模型实现完整状态快照。

## CPFS 锁与状态

运行时目录位于同一个共享仓库下，各台设备使用同一套目录。每个子任务首次发布从 `v1` 开始，以新时间戳发布时序号加一。序号按 Git 发布记录计算，暂停或恢复领取沿用该次发布的序号。UTC 发布时间按 `yymmdd-hhmmss` 附在序号后，形成 `version_label`。领取锁和增量配置继续使用 UTC 发布时间的 Unix 秒，对应记录中的整数 `version`。以下是上文示意子任务 `01` 首次发布的路径。

```text
.lock/system-000-inventory-refresh-01.task.lock/
.lock/system-000-inventory-refresh-01.v1795003200.lock/
status/system-000-inventory-refresh-01/v1-261118-120000.json
results/system-000-inventory-refresh-01/v1-261118-120000/
results/system-000-inventory-refresh-01/system-000-inventory-refresh-01.v1-261118-120000.tar.gz
```

已按旧格式领取的执行沿用原始路径和包名。浅克隆可用 `git fetch --unshallow origin` 补齐用于计算序号的发布历史。

领取通过原子 `mkdir` 竞争。完成后先保存终态，再释放任务执行锁，保留版本领取记录。版本锁中保存主机、进程、领取标识及版本信息。重复启动会跳过已领取的版本。同一共享 checkout 的 Git 拉取和源码导出通过短期目录锁串行化，实验从独立快照执行。

CPFS 挂载需要提供跨客户端互斥的目录创建与可见的元数据更新。CPFS 的 NFS 协议服务不支持 NLM 和原子追加，因此共享协调使用目录与独立文件，每个实验单独记录日志。当前测试在本地文件系统验证了竞争与恢复，真实 CPFS 跨主机验证需要在用户设备上执行。[CPFS 限制](https://www.alibabacloud.com/help/en/cpfs/cpfsonecs/product-overview/limits-1)

状态包括 `running`、`succeeded`、`failed` 和 `abandoned`。崩溃遗留的锁保留给人工恢复。先在记录的原主机上停止轮询器并确认实验进程已经退出，再执行以下命令。恢复命令接受领取记录中的版本标签或整数版本。需要重跑时，在开发端更新子任务发布时间。下例中的参数取自上文示意记录。

```bash
bash tools/runner.sh --abandon system-000-inventory-refresh-01 v1-261118-120000
```

如果原主机的 Git 拉取进程崩溃并留下 Git 协调锁，确认该进程已退出后，在原主机执行 `bash tools/runner.sh --recover-git-lock`。环境初始化崩溃遗留的锁需要根据其中的主机和进程记录人工检查后恢复。若崩溃发生在目录创建后、`owner.json` 写入前，该目录会保持阻塞。此时先停止可能持有该锁的轮询器及子进程，确认已全部退出，再人工移走该孤立目录并保留恢复记录。正常运行时保留 `.lock/`、`status/` 和 `results/`，供后续启动检查。

## 结果回收与归档

设备完成后，CPFS 的 `results/{子任务标识}/{version_label}/` 保存执行记录与原始输出。压缩包位于同级，完整文件名由状态记录中的 `archive_name` 给出，格式为 `{子任务标识}.{version_label}.tar.gz`。

用户将原始压缩包下载到本地仓库的 `results-remote/` 后，运行归档工具。

```bash
python3 tools/archive_results.py
```

归档工具核验身份、包内容和 SHA-256，在本地 `results/` 保留原包及提取内容，并生成 `status/{子任务标识}/{version_label}.archived.json`。重复导入同一个包会校验并跳过。成功消费的输入包会从 `results-remote/` 移除，冲突或损坏的输入保留供排查。

分析端将回执提交并推送到 `main`。CPFS 执行器核对领取标识、源码提交、任务摘要、执行状态和包的 SHA-256，再将数据目录及原包移到 `results-archive/{子任务标识}/`。

废弃的子任务由 runner 校验结果身份、原包内容和 SHA-256，直接向 CPFS 的 `results-archive/` 归档。已有下载归档回执的包沿用其校验值。归档后，执行记录的 `disposition` 为 `ignored`。仍在运行的实验完成后再处理，归档失败时保留原数据并在下次轮询重试。实验的 `state` 保持 `succeeded` 或 `failed` 等原结果，`delivery_state` 记录 `awaiting_download` 或 `archived`。失败实验同样可以回收归档。移动中断时可继续处理，遇到不同内容的目标文件会停止并保留数据。崩溃恢复后继续使用同一组结果目录和归档目录。

某个版本的压缩包缺失、改名未被识别或校验冲突时，runner 会保留该版本数据，并写入 `status/{子任务标识}/{version_label}.archive-error.json` 等诊断记录，列出原因、预期校验和和搜索路径，继续处理其他归档确认及新任务。补齐或修正文件后，后续轮询重新尝试。文件只改名时 SHA-256 保持一致，原始压缩包按确认记录校验。

旧版包缺少简称和命令时，工具按实验类型补全简称，并把生成的入口命令标记为复现命令。原始压缩包保持原样，确保已经下载的旧版包也能通过 SHA-256 确认。

结果包包含执行记录、小型统计和 CSV，也包含已生成的原始 Nsight Systems `.nsys-rep` 报告。每个执行版本的原始报告合计最多 8 GiB，与小文件配额分别计算。超出配额的报告保留在远程目录，并在 `result.json` 的 `archive_omitted` 中列出。仿真验收失败时，已经生成的报告也会纳入打包。归档后可在本地对应的 `results/{子任务标识}/{version_label}/` 目录中打开报告。

构建日志和派生 SQLite 等大文件保留在远程结果目录，随版本目录一起移入 `results-archive/`。分析时可另取 `worker.log`、`configure.log` 和 `build.log`。

本地检查可运行以下命令，GPU 性能与 CPFS 跨机行为仍以目标设备上的实验为准。

```bash
bash tools/runner.sh --help
python3 -B -m unittest discover -s tests -v
```
