任务状态由执行器写入本地 `status/{worker}/{id}/v{version}.json`。例如 `status/gpu01/000001/v2.json`。

运行状态目录由 Git 忽略。远程设备只读取公开仓库，状态和结果通过用户手动下载的结果包带回分析端。多设备任务在清单的 `worker` 字段中预先分配，每个 worker ID 对应一台设备。

重启时保留并使用相同的 worker ID、状态目录和结果目录。具体启动、恢复和下载方式见根目录 README。
