# ID vs CAPE —— 逆动力学 vs 目标收敛对比（本机 4070）

预注册设计见 `E:\world_model\指导与笔记\IDvsCAPE_对比实验设计.md`（§11 = v2 修正案），
过程记录见 `E:\world_model\指导与笔记\IDvsCAPE_实验记录.md`（跑一步记一步）。
v1 代码+结果快照：`E:\world_model\id_vs_cape_v1_snapshot\`（降级为预实验，两表并报）。

## 运行方式（全部在 WSL + lawam 环境）

```bash
# Windows 侧调用（当前会话用）：
wsl.exe -d Ubuntu-24.04 -e bash -lc 'cd /mnt/e/world_model/id_vs_cape && bash run_all.sh smoke'

# WSL 内部直接：
cd /mnt/e/world_model/id_vs_cape
bash run_all.sh smoke      # env check + manifest v2 + 模块自检
bash run_all.sh cache      # 特征缓存 v2（~1-2h；对齐闸门沿用 v1 已验，无需再看图）
bash run_all.sh train      # 三臂 × 2 种子（matched steps/seed，各臂按自己论文超参）
bash run_all.sh controls   # id within-task shuffle / L1 对照
bash run_all.sh eval       # 全套指标电池
```

缓存落在 `E:\world_model\id_vs_cape\cache_v2\`（~17GB；v1 cache 仍在 `cache\`）。
**长任务必须用 Claude Code 后台任务跑**：WSL 内 `nohup ... &` 随 wsl.exe 会话退出即死。

## v2 协议要点（对论文的忠实实现，防 v1 塌缩重演）

- 分支按 CAPE：t2=T−U[5,12]、t1=t2−U[5,12]（不同剩余动作长度汇于同一 o_T；v1 间隔仅 3 帧=坍缩根源）
- 状态编码 = 同一 ctx/dec/g 通路 + 可学习**空动作查询**（预测与状态同空间，论文 App.A）
- **同 episode 组批**（8 集×4 窗）：负例=组内他窗键（同背景异时刻）+组内换动作+跨批随机 —— 论文防塌缩核心
- τ=0.07 固定；lr/wd=1e-4、warmup 2k cosine、20k 步；反事实敏感度监控每 2k 步
- id：L2 默认、within-task shuffle、有效 batch 16（累积×2）、2 种子
- 探针 = 42 号 MLP（1536→256→128→7、3k 步、bs512、3 种子、patch-mean 特征）
- 检索 = 论文式同任务全池（~250 键）、horizon=t2−t1、排除自身 t1

## 文件一览

| 文件 | 作用 |
|---|---|
| `common.py` | v2 常量（分支区间、PAST=24、τ=0.07、组批 8×4、DIM/IMG）、工具 |
| `00_check_env.py` | CUDA/数据集/对账/顺序解码/DINOv3 全绿检查 |
| `build_manifest.py` | 任务级 32/8 划分 + CAPE 式分支采样 |
| `backbone.py` | 冻结 DINOv3 ViT-B/16（返回 patch/CLS/patch-mean 三元组） |
| `build_cache.py` | 29 槽 pooled+pmean + probe 分片 npy + 完整性闸门 |
| `windows.py` | WindowDataset（组批映射/变长动作/probe 分片读取）+ ID 配对 |
| `modules.py` | ContextEncoder / ActionQueryDecoder（空动作查询）/ ProjHead / FwdModel / IDNet |
| `losses.py` | InfoNCE / ID L1-L2 / forward cosine |
| `train.py` | 单臂训练（cape 组批；id L2+shuffle+累积；fwd）+ 反事实监控 |
| `probe_r2.py` | 层2：MLP 探针（主）/ ridge（辅）/ id_direct / y-shuffle 地板 |
| `eval_retrieval.py` | 层3：同任务全池检索 Hit@1/5+MRR（frozen=零动作基线） |
| `diag_pairwise_cos.py` | 空间诊断（A 原始特征；B 分支忠实版 + o_T 键一致性 + swap-wins） |
| `pick_align_windows.py` | 对齐闸门加强版（v1 已用，v2 沿用结论） |

## 关键坑（已踩过，2026-11-24 实测）

- **视频是 AV1**（AV01）：全数据集仅 4 个 mp4（单文件最多 101k 帧）。pip 版 cv2 **无 AV1 解码器**；
  torchvision 整文件读会 **OOM（~20GB）**。唯一可用后端 = **PyAV（libdav1d）顺序流式解码**。
- build_cache 用"按集数区间过界即提特征"的流式结构，RGB 峰值 ≈ 单集；`/tmp` 是 tmpfs（VM 重启即清），
  长任务日志写 E 盘。
- **DINOv3 输出 201 token = CLS + 4 register + 196 patch** → patch 必须取 `out[:, -196:]`。
- parquet v3.0 `action` 列是 fixed_size_list：逐行 `np.stack`，不能整列 astype。
- pandas `df.T` 撞 DataFrame 转置属性 → 一律 `df["T"]` 括号式（v2 又踩一次）。
- 视频是多 episode 共享 mp4，episode 帧号 → 文件内偏移 = `round(from_timestamp * 20)`。
- 8GB VRAM：backbone 只在 cache 阶段用（bs=32 fp16）；训练阶段只吃缓存特征。
- **probe patch 不能一次性 np.stack**（~23GB 瞬时 RAM）→ 每 256 窗分片落盘，读取端 mmap 迭代。
- WSL 内 `nohup ... &` 随 wsl.exe 会话退出被杀 → 长任务用 Claude Code 后台任务。

## 当前状态

- [x] v1 全流程 + 主表 v1（cape 塌缩机理定位，实验记录·三–六）→ 快照降级为预实验
- [x] 论文协议审计 + v2 修正案预登记（设计文档 §11）
- [x] v2 代码全套（11 文件过编译+自检；DIM/IMG 漏定义、df.T 两处即修）
- [x] manifest v2（8988 窗；ctx 间隔中位 9 帧=0.45s）
- [x] cache_v2 构建（三次启动：VM 崩溃→OOM 定位 patch 攒积 13.6GB→逐集释放修复→**dropped 0**，17GB，~15 分钟）
- [x] **v2.0 主力（seed 42）三臂完成 + 主表**（实验记录·十）：坍缩逃逸成立（cos_random 1.0→0.02）；检索层爆冷（cape Hit@1=0.000=随机）→ 根因=正对缺目标状态（附录 A "zero-horizon training samples"）→ **v2.1 修正案预登记（§12）**：多正对 InfoNCE，cache 增存 o_T patch
- [x] **v2.1 判决：修正无效**（实验记录·十一）：检索仍随机（Hit@1 0.008/MRR 0.028）、探针 −0.335 分毫不动、goal_cos 全程不爬——**logsumexp 正对集饱和**（u1↔u2 的 e^12 压 s_T 的 e^−0.6，权重比 10⁻⁶，目标正对零梯度）。回读原文：零视野样本应是**独立第三分支**（式 3 成对形式，每对独立分子）→ **v2.2 修正案预登记（§13）**：三分支逐对对称 InfoNCE + s_T 入负键池；代码已落地+冒烟通过，重训待指令（含预登记终止判据：goal_cos<0.2 或检索仍随机 ⇒ 停止 Cape 实现迭代，主线按 ID 探针对比收口）
- [ ] 剩余队列（被内存压力中止，待重启）：id_seed43、cape/fwd_seed43（cape 用 v2.2 协议）、id_shuf、id_l1
- [ ] eval 全电池终版 → 主表 v2.2 + H1/H2/H3 判读
