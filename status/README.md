CPFS 运行状态写入 `status/{三位编号}-{简称}/v{version}.json`，例如 `status/000-system-info/v1.json`。状态中包含任务名称、简称、命令、版本、源码提交、执行主机和结果。

本地分析端导入 `results-remote/` 的原始结果包后，生成同目录下的 `v1.archived.json` 并提交到 Git。CPFS 执行器拉取该确认记录，验证执行身份和压缩包 SHA-256，再把远程数据从 `results/` 移到 `results-archive/`。

`vN.json` 是共享文件系统上的运行状态，`vN.archived.json` 是 Git 跟踪的归档确认。执行状态和归档状态分别记录，任务锁与版本记录保留。具体步骤见根目录 README。

摘要包名称为 `{id}-{slug}.v{version}.tar.gz`，例如 `000-system-info.v2.tar.gz`。状态和归档确认用 `archive_name` 保存该名称。

已回收并发布归档确认的版本如下。远端 runner 拉取确认后，会校验原包并将对应数据移入 `results-archive/`。

| 任务版本 | 执行结果 | 归档确认 |
| --- | --- | --- |
| `000-system-info v2` | succeeded，5 个 case 全部通过 | [v2.archived.json](000-system-info/v2.archived.json) |
| `001-gedes-smoke v1` | succeeded，GeDES k8 smoke 通过 | [v1.archived.json](001-gedes-smoke/v1.archived.json) |

`000-system-info v3` 是基于已完成 v2 的下一版增量任务，使用新版本领取记录，保留 v2 的完成标记。

本批后续版本为 `000 v3`（增量验证）、`001 v2`（GeDES 完成状态与主机阶段诊断）及 `003 v1`（四卡 CUDA copy/P2P 基线）。已完成版本的归档确认继续保留，最新执行进度以 CPFS 中的状态为准。
