CPFS 在 `status/{父任务标识}-{两位子任务编号}/v{发布时间的Unix秒}.json` 记录每次执行的领取、源码提交、执行主机、结果和归档状态。首次迁移的子任务沿用原来的三位编号目录和整数版本，例如 `status/000-system-info/v2.json`，对应 `system-000-init-environment-00` 的历史执行。

分析端导入 `results-remote/` 的原始结果包后，生成同目录下的 `v{version}.archived.json` 并提交到 Git。CPFS 执行器拉取回执，验证执行身份和压缩包 SHA-256，再将数据从 `results/` 移到 `results-archive/`。状态和回执的 `archive_name` 字段保存完整包名。

发布 `ignore` 后，runner 将废弃标记写入 CPFS 的 `ignored.json`，验证并归档已结束的实验，记录 `disposition: ignored`，保留执行结果和永久领取记录。归档失败时保留数据与 `v{version}.archive-error.json` 供重试。操作步骤见[仓库说明](../README.md)。
