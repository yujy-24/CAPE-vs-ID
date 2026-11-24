# ID vs CAPE —— 逆动力学 vs 目标收敛对比（本机 4070）

预注册设计见 `E:\world_model\指导与笔记\IDvsCAPE_对比实验设计.md`，
过程记录见 `E:\world_model\指导与笔记\IDvsCAPE_实验记录.md`（跑一步记一步）。

## 运行方式（全部在 WSL + lawam 环境）

```bash
# Windows 侧调用（当前会话用）：
wsl.exe -d Ubuntu-24.04 -e bash -lc 'cd /mnt/e/world_model/id_vs_cape && bash run_all.sh smoke'

# WSL 内部直接：
cd /mnt/e/world_model/id_vs_cape
bash run_all.sh smoke      # env check + manifest + 模块自检
bash run_all.sh cache      # 特征缓存（~1-2h，结束后人工看 cache/align_check/*.jpg）
bash run_all.sh train      # 三臂训练（各自 ~1-2h @ 8GB）
bash run_all.sh eval       # 全套指标
```

缓存默认落在 `E:\world_model\id_vs_cape\cache\`（经 /mnt/e，跨 9p 写慢）。
若要提速/省盘：`export IDVC_CACHE=~/.idvc_cache`（ext4 侧，约 5.6 GB，注意 WSL 磁盘配额）。

## 文件一览

| 文件 | 作用 |
|---|---|
| `common.py` | 路径、常量（H=10, 8000/1000 窗口, STEPS=20000, BS=8）、工具 |
| `00_check_env.py` | CUDA/数据集/对账/顺序解码/DINOv3 全绿检查 |
| `build_manifest.py` | 任务级 32/8 划分 + 窗口采样（t1/t2/随机负例） |
| `backbone.py` | 冻结 DINOv3 ViT-B/16（fp16，patch 196×768 + CLS） |
| `build_cache.py` | 每视频文件一遍顺序解码 → 特征分片；`--dump-frames` 出对齐自检图（前 k 窗，偏静止） |
| `pick_align_windows.py` | 对齐闸门加强版：按窗内 gripper 指令摆幅选 top-k 窗出网格图（满摆幅=真事件） |
| `windows.py` | WindowDataset（三臂共用）+ ID 配对采样 |
| `modules.py` | ContextEncoder / ActionQueryDecoder / ProjHead / StateBridge / IDNet |
| `losses.py` | InfoNCE / ID L1-L2 / forward cosine |
| `train.py` | 单臂训练（`--arm cape|id|fwd`，`--no-swap` = H3 消融） |
| `probe_r2.py` | 层1：动作可恢复性 R²（ridge 探针；id 臂 = IDNet 直评） |
| `eval_retrieval.py` | 层2：未来状态检索 Hit@1/5/10（各臂原生打分 + 混淆类型） |
| `diag_pairwise_cos.py` | 层3 前置诊断：空间塌缩度（时序/跨场景 cos、swap-wins 率） |

## 关键坑（已踩过，2026-11-24 实测）

- **视频是 AV1**（AV01）：全数据集仅 4 个 mp4（单文件最多 101k 帧）。pip 版 cv2 **无 AV1 解码器**；
  torchvision 整文件读会 **OOM（~20GB）**。唯一可用后端 = **PyAV（libdav1d）顺序流式解码**。
- build_cache 用"按集数区间过界即提特征"的流式结构，RGB 峰值 ≈ 单集；`/tmp` 是 tmpfs（VM 重启即清），
  长任务日志写 E 盘。
- **DINOv3 输出 201 token = CLS + 4 register + 196 patch** → patch 必须取 `out[:, -196:]`。
- parquet v3.0 `action` 列是 fixed_size_list：逐行 `np.stack`，不能整列 astype。
- pandas 行访问 `r.T` 撞 Series 转置属性 → 一律 `r["T"]` 括号式。
- 视频是多 episode 共享 mp4，episode 帧号 → 文件内偏移 = `round(from_timestamp * 20)`。
- 8GB VRAM：backbone 只在 cache 阶段用（bs=32 fp16）；训练阶段只吃缓存特征。
- patch 缓存（t1/t2 ×196×768 fp16）约 5.4 GB；pooled 仅 0.2 GB。

## 当前状态

- [x] 代码脚手架全部就位
- [x] `run_all.sh smoke` 全绿
- [x] cache 建立（7993/991 窗，dropped 0）+ 人工核对 align_check 图 —— **闸门：动作-画面对齐，2026-11-24 通过**（5 个满摆幅窗偏差 0–2 帧，见实验记录·三）
- [x] 三臂首跑 20k 步 + 主表 v1 —— **cape 塌缩（u 动作盲，cos 1.0000）机理已定位，见实验记录·五**；fwd 垫底复现 42 号结论
- [ ] 预登记 sweep（cape τ×2 / no-swap、id l2 / shuffle）→ 主表 v2 → 回填设计文档判读规则
