# ID seed 43 — 2026-11-25

这次运行在单张 RTX 4090、60 GiB 容器内存的 Linux 实例上完成。仓库版本为 `a46f93be98064ac39d8b3cd3f0ebb72ced17d9c4`。17:02:27（UTC+8）开始训练，17:08:44 完成 20,000 步，17:09:12 完成留出任务评估。只运行了 ID 臂。

| 指标 | 结果 |
|---|---:|
| 训练／验证窗口 | 7,993／995；丢弃 0 |
| 验证动作对 | 9,950 |
| ID 原生预测头汇总 R² | −0.0404 |
| ID 原生预测头 MAE | 0.2033 |

原始产物：`config.jsonl` 为训练参数，`metrics.jsonl` 为逐步训练与验证损失，`train.log` 和 `eval.log` 为终端输出，`cache_build_summary.log` 为缓存构建摘要，`checksums.sha256` 记录检查点、manifest 和实际 DINOv3 权重的 SHA-256；完整机器可读摘要见 `run.json`。

**比较状态：待确认编码器一致性。**该实例通过 ModelScope 取得的 DINOv3 文件为 342,662,192 字节，仓库骨干前向输出形状符合预期，但其 SHA-256 与 [Hugging Face 官方同名权重](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m) 的 `main` 版本不一致。此处如实保留结果，不能仅凭窗口数量相同就将它与 seed 42 当作严格同权重的两次随机种子实验。需与 seed 42 实际使用的权重哈希或特征缓存核对后，再纳入跨种子汇总。

检查点 `final.pt` 为 80,401,406 字节，保存在本次实例的 `/root/shared-nvme/CAPE-vs-ID/cache_v3/ckpt/id_seed43/final.pt`；仓库的 `.gitignore` 排除 `*.pt` 和 `cache_v3/`，因此这里仅提交原始数值日志与校验信息。服务器的 `branch_*.npy` 旁路文件因共享盘容量限制放在实例系统盘；释放实例后，若再运行当前加载器，须从共享盘 `cache_v3/*_shard_*.npz` 重新提取旁路文件。
