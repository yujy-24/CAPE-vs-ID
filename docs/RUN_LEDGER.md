# 跨机器运行记录

每次训练至少记录两次事件：**启动时**提交 `started`（运行 ID、机器、代码版本、输入版本、命令和目标日志路径）；**结束时**提交 `completed`、`failed` 或 `interrupted`，附检查点校验、原始训练日志和评估结果。结束状态必须以进程退出及产物检查为依据；评估可追加 `evaluated` 事件。不同机器重建的输入要记录哈希，不能只凭数据集名称或窗口数量判断一致。

| 时间（UTC+8） | 运行 ID | 事件 | 可核查内容 |
|---|---|---|---|
| 2026-11-25 17:02:27 | `id_seed43_20261125_remote` | `started` | [配置](../results/id_seed43_20261125/config.jsonl)，源码 `a46f93be98064ac39d8b3cd3f0ebb72ced17d9c4`，RTX 4090，`train.py --arm id --seed 43 --tag _seed43` |
| 2026-11-25 17:08:44 | `id_seed43_20261125_remote` | `completed` | 20,000 步；[训练日志](../results/id_seed43_20261125/train.log)、[逐步指标](../results/id_seed43_20261125/metrics.jsonl)、[检查点校验](../results/id_seed43_20261125/checksums.sha256) |
| 2026-11-25 17:09:12 | `id_seed43_20261125_remote` | `evaluated` | [验证日志](../results/id_seed43_20261125/eval.log)：9,950 对，R² −0.0404，MAE 0.2033；随后核验两机权重一致 |
| 2026-11-25 17:39:19 | `id_seed43_20261125_remote` | `verified` | 本机核对（见下方哈希注记）：encoder 两机逐字节一致 ⇒ 跨种子比较有效；manifest 字节差异仅涉 cape 辅助列，不影响 id 臂 |
| 2026-11-25 19:05:50 | `id_shuf_20261125_remote` | `started` | [开始时间](../results/id_shuf_20261125/started_at.txt)、[训练配置](../results/id_shuf_20261125/config.jsonl)；源码 `a46f93be98064ac39d8b3cd3f0ebb72ced17d9c4` |
| 2026-11-25 19:12:14 | `id_shuf_20261125_remote` | `completed` | `train_exit=0`，20,000 步；[状态](../results/id_shuf_20261125/status.txt)、[训练日志](../results/id_shuf_20261125/train.log)、[逐步指标](../results/id_shuf_20261125/metrics.jsonl)、[检查点校验](../results/id_shuf_20261125/checksums.sha256) |
| 2026-11-25 19:13:54 | `id_shuf_20261125_remote` | `evaluated` | [评估日志](../results/id_shuf_20261125/eval.log)：9,950 对，R² −0.1182，MAE 0.2167 |
| 2026-11-25 19:47:08 | `id_l1_20261125_remote` | `started` | [开始时间](../results/id_l1_20261125/started_at.txt)、[配置](../results/id_l1_20261125/config.jsonl)；源码 `a46f93be98064ac39d8b3cd3f0ebb72ced17d9c4` |
| 2026-11-25 19:53:32 | `id_l1_20261125_remote` | `completed` | `train_exit=0`，20,000 步；[状态](../results/id_l1_20261125/status.txt)、[训练日志](../results/id_l1_20261125/train.log)、[逐步指标](../results/id_l1_20261125/metrics.jsonl)、[检查点校验](../results/id_l1_20261125/checksums.sha256) |
| 2026-11-25 19:53:52 | `id_l1_20261125_remote` | `evaluated` | [评估日志](../results/id_l1_20261125/eval.log)：9,950 对，R² −0.1804，MAE 0.1997 |

| 2026-11-26 01:09:13 | `cape_noswap_20261126_local` | `interrupted` | step ~5100/20000 被 Claude Code 内存压力回收器终止（系统级空闲期内存紧张，非训练错误；第二次发生，前次为 cache 构建）。终检点未写出，重启须从头。待用户释放内存后重启 |

**哈希核对注记（2026-11-25 17:39，本机 WSL）**：
- **encoder 一致**：本机 `/home/yjy/LaWAM/weights/dinov3-vitb16-pretrain-lvd1689m/model.safetensors`（342,662,192 B）SHA-256 = `9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b`，与远端 ModelScope 文件完全相同。`encoder_sha256_matches_hugging_face_main: false` 仅表示与 HF 当前 main 文件字节不同，差异原因尚未验证，**不是两机差异**——seed 42（本机）与 seed 43（远端）特征出自同一份权重，`comparison_status` 可从 `provisional_encoder_checksum_mismatch` 升级为有效。
- **manifest 字节不同、实质等价**：本机 `cache_v3/manifest.parquet` SHA-256 = `d9cffd6a…` ≠ 远端 `358078aa…`；窗口数一致（7,993/995，dropped 0），差异位于 t1/t2/rand 辅助列与序列化元数据（两机 numpy/pyarrow 版本不同的 seeded 抽样漂移）。id 臂训练与探针**不读这些列**（cape 配对专用），故 id 两种子对比不受影响；若日后跨机跑 **cape** 臂，须先统一 manifest 哈希。

| 2026-11-26 00:27:22 | `cape_noswap_20261126_local` | `started` | `train.py --arm cape --no-swap --tag _noswap`（源码 HEAD 见下一提交），本机 RTX 4070，日志 `cache_v3/ckpt/cape_noswap_train.log`（不入 git） |

下一次启动训练时，先在此表追加并提交 `started` 行；跑完或中断后再提交最终事件及原始产物。运行详情与已知限制见 [ID seed 43 记录](../results/id_seed43_20261125/README.md)。
