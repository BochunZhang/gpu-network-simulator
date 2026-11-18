CPFS 在 `status/{父任务标识}-{两位子任务编号}/{version_label}.json` 记录每次执行的领取、源码提交、执行主机、结果和归档状态。版本标签形如 `v1-261118-120000`，包含该子任务的发布序号和 UTC 发布时间。按旧格式领取的执行沿用原路径。此前实验的回执作为历史资料单独备份。[历史实验索引](../history/pre-review-20261118/README.md)

分析端导入 `results-remote/` 的原始结果包后，生成同目录下的 `{version_label}.archived.json` 并提交到 Git。CPFS 执行器拉取回执，验证执行身份和压缩包 SHA-256，再将数据从 `results/` 移到 `results-archive/`。状态和回执的 `archive_name` 字段保存完整包名。

发布 `ignore` 后，runner 将废弃标记写入 CPFS 的 `ignored.json`，验证并归档已结束的实验，记录 `disposition: ignored`，保留执行结果和永久领取记录。归档失败时保留数据与 `{version_label}.archive-error.json` 供重试。操作步骤见[仓库说明](../README.md)。
