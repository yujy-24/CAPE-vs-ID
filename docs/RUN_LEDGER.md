# 跨机器运行记录

每次训练至少记录两次事件：**启动时**提交 `started`（运行 ID、机器、代码版本、输入版本、命令和目标日志路径）；**结束时**提交 `completed`、`failed` 或 `interrupted`，附检查点校验、原始训练日志和评估结果。结束状态必须以进程退出及产物检查为依据；评估可追加 `evaluated` 事件。不同机器重建的输入要记录哈希，不能只凭数据集名称或窗口数量判断一致。

| 时间（UTC+8） | 运行 ID | 事件 | 可核查内容 |
|---|---|---|---|
| 2026-11-25 17:02:27 | `id_seed43_20261125_remote` | `started` | [配置](../results/id_seed43_20261125/config.jsonl)，源码 `a46f93be98064ac39d8b3cd3f0ebb72ced17d9c4`，RTX 4090，`train.py --arm id --seed 43 --tag _seed43` |
| 2026-11-25 17:08:44 | `id_seed43_20261125_remote` | `completed` | 20,000 步；[训练日志](../results/id_seed43_20261125/train.log)、[逐步指标](../results/id_seed43_20261125/metrics.jsonl)、[检查点校验](../results/id_seed43_20261125/checksums.sha256) |
| 2026-11-25 17:09:12 | `id_seed43_20261125_remote` | `evaluated` | [验证日志](../results/id_seed43_20261125/eval.log)：9,950 对，R² −0.0404，MAE 0.2033；权重一致性待确认 |

下一次启动训练时，先在此表追加并提交 `started` 行；跑完或中断后再提交最终事件及原始产物。运行详情与已知限制见 [ID seed 43 记录](../results/id_seed43_20261125/README.md)。
