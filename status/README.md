CPFS 运行状态写入 `status/{三位编号}-{简称}/v{version}.json`，例如 `status/000-system-info/v1.json`。状态中包含任务名称、简称、命令、版本、源码提交、执行主机和结果。

本地分析端导入 `results-remote/` 的原始结果包后，生成同目录下的 `v1.archived.json` 并提交到 Git。CPFS 执行器拉取该确认记录，验证执行身份和压缩包 SHA-256，再把远程数据从 `results/` 移到 `results-archive/`。

`vN.json` 是共享文件系统上的运行状态，`vN.archived.json` 是 Git 跟踪的归档确认。执行状态和归档状态分别记录，任务锁与版本记录保留。具体步骤见根目录 README。
