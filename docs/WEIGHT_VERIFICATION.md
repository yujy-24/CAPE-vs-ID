# DINOv3 权重一致性检查

2026-11-25 的云端 ID seed 43 使用了 ModelScope 的 `facebook/dinov3-vitb16-pretrain-lvd1689m` 文件。该文件大小为 342,662,192 字节，仓库骨干前向输出符合预期。**seed 42 原机器的文件 SHA-256 与云端文件完全相同**，已由合作者记录并再次独立核对，见 [`RUN_LEDGER.md`](RUN_LEDGER.md)。因此，先前“两个 seed 使用不同权重”的疑虑已解决；无需为此重跑 ID seed 43。

两机共同使用的文件与 [Hugging Face 官方同名模型当前 `main` 版本](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m) 的 LFS SHA-256 不同。ModelScope 此条目由用户上传；目前未取得 HF 当前文件内容，**不能断言是 HF 后续更新，也不能断言权重数值不同**。这是与官方当前文件的关系问题，不影响已确认的两机文件一致性。

**文件哈希不同尚不能证明模型参数不同。**Safetensors 的头部信息和张量保存顺序也会改变整个文件的 SHA-256。仓库的 [`tools/fingerprint_dino.py`](../tools/fingerprint_dino.py) 按张量名称排序，对名称、数据类型、形状与原始数值字节计算规范化 SHA-256，以区分文件包装差异和参数差异；同时记录 `config.json`、`preprocessor_config.json` 的哈希及图像均值、标准差。

如需进一步核对预处理配置，可在 **seed 42 原训练机器**运行：

```bash
python tools/fingerprint_dino.py ~/LaWAM/weights/dinov3-vitb16-pretrain-lvd1689m > seed42_encoder_fingerprint.json
sha256sum /实际路径/cache_v2/manifest.parquet
```

将其与 [`results/id_seed43_20261125/encoder_fingerprint.json`](../results/id_seed43_20261125/encoder_fingerprint.json) 比较。权重文件已确认相同；还可核对 `config_sha256`、`preprocessor_sha256`，因为仓库的 `backbone.py` 会读取预处理均值和标准差。合作者的 manifest 核对指出字节差异位于 CAPE 使用的辅助列；ID 的动作与帧窗口相同，详见运行台账。

若研究必须声称使用了 **HF 当前 `main` 权重**，需由有访问权限的账号下载该文件，对它运行同一个指纹工具：若文件 SHA 不同、但 `weights_tensor_sha256` 相同且模型及预处理配置相同，则差异仅在文件包装，编码器参数相同；若张量哈希不同，则两次现有运行都使用了另一个参数版本。要测量版本变化对结果的影响，必须用 HF 当前权重重新提取特征并重新训练、评估相应实验臂；只替换权重文件而继续使用旧 `cache_v3` 无效，因为缓存里已经固化了旧编码器输出。
