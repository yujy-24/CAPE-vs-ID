# ID vs CAPE 对比实验——过程与结果记录

> 配套设计文档：`IDvsCAPE_对比实验设计.md`（预登记口径，判读规则以它为准）。
> 记录规则：每完成一步记一节；所有数字写明口径（split/seed/步数）；失败和绕路照记。

---

## 一、2026-11-24 立项与环境核查

**立项**：4090 容器暂不可用，转本机 4070 Laptop（8GB）。CAPE vs 逆动作正面比较 = 已查证空白（两篇互不引用、各自只处决过对方远亲）。

**环境事实（已核实）**：
- WSL2 Ubuntu-24.04，lawam env：python3.10 / torch 2.6.0+cu124 / transformers 5.2.0 / timm 1.0.22 / einops 0.8.1；CUDA 可用 = **RTX 4070 Laptop GPU（8GB）**。
- 数据：`~/LaWAM/dataset/libero_merged_no_noops_20hz`（LeRobot **v3.0**，1693 集 / 273,465 帧 / 40 任务 / fps 20 / action[7] / state[8]；episode 长度 75~505，中位 140）。
  - episodes parquet 给每集：`videos/observation.images.image/{chunk,file}_index` + `from/to_timestamp`（ep0: 0→10.7s，与 214 帧/20fps 精确吻合）；data parquet 全量动作。
  - 无 lerobot 包（LaWAM 是 vendored 读取器）⇒ 自写 parquet+mp4 读取。
- 骨干：`~/LaWAM/weights/dinov3-vitb16-pretrain-lvd1689m`（HF 快照，safetensors + preprocessor_config）。
- **坑**：torchvision `read_video` 带时间区间截取返回 0 帧（pts 对不上）；全量读整个 mp4 极慢（>2min 超时）。⇒ 解码器做成 cv2 优先、torchvision 全段读兜底，由 `00_check_env.py` 判定。cv2/pyav 是否在 lawam env 里**待核查**。

**代码骨架（已搭）**：`E:\world_model\id_vs_cape\`，13 文件（README 有用法）。设计要点：manifest 一次生成三臂共用（数据 exposure 严格对齐）；特征缓存后训练不吃 backbone 显存。

**待办（下一步）**：
1. `00_check_env.py` 跑全绿（视频解码后端、DINOv3 加载、帧数-长度对账）；
2. `build_manifest.py` → 核对任务级划分 32/8；
3. `build_cache.py` → **人工抽查 5 个窗口的帧-动作对齐**（坑 1）；
4. frozen 基线数字 → ③④训练。

## 二、2026-11-24 代码自检 + 三大坑修正 + 验证链 1-2 关通过

**验证链结果**：
1. `modules.py` 自检 ✓：CAPE(ctx2+dec2+g) **25.2M** / IDNet(4L)+head **20.1M**——容量基本匹配（cape 略大 25%，主表脚注注明）。
2. `00_check_env.py` **ALL GREEN**（exit 0）：4070 8GB / 1693 集 ts-帧数对账 diff=0 / action[7] / fps=20 精确。
3. `build_manifest.py` ✓：40 任务 → **32 train / 8 val**（seed 42）；**8984 窗 = 7993/991**（略低于 8000/1000 目标：每集首窗口因随机负例"只取更早集"规则被跳过，符合设计）。

**今天踩出并修掉的坑（全部已改代码）**：
- **视频是 AV1**（fourcc AV01，全数据集仅 **4 个 mp4**，单文件最多 101,469 帧）：pip 版 cv2 无 AV1 软解，之前"cv2 seek 失败"实为编解码器缺失。→ 全面改 **PyAV 12.3.0（libdav1d）**顺序流式解码，~2000fps。
- **torchvision 整文件读 = OOM kill（exit 137）**：10 万帧 ×256×256×3 ≈ 19.7GB。这解释了之前 read_video "无输出超时"之谜（缓冲 stdout 随进程一起被杀）。
- 由此发现 build_cache 原设计**一次持有整个文件的 RGB ≈ 6GB 也会 OOM** → 重写为流式解码 + 按集数区间"过界即提特征、即释放"；特征字典 ~200MB、patch 峰值 ~250MB。
- **DINOv3 带 register tokens**：输出 201 = CLS+4reg+196patch；`out[:,1:]` 会把 reg 混进 patch（backbone.py 的 assert 拦住了）→ 改 `out[:, -196:]`。
- parquet v3.0 的 `action` 列是 fixed_size_list，不能整列 `.astype` → 逐行 `np.stack`。
- torch 2.6 的 `nn.ModuleList(dict(...))` 不接受 dict → 拆 DecBlock 模块类。
- `/tmp` 是 tmpfs，WSL VM 空闲重启即清空（后台任务日志丢失）→ 长任务日志一律写 E 盘项目目录（`cache_build.log`）。
- eval_retrieval 初版三处 bug 自查修掉：placeholder 列错位、混合设备高级索引、ID 打分用 t2 造成 gap>1 分布外。

**预注册修正案（训练前登记，见设计文档 §6 表）**：ID 臂检索打分 z_ctx 由 t2 改为 **o_{T-1}**、目标动作为 a_{T-1}——真候选变成 1 步转移（分布内），时间近邻 2~3 步。检索池按文档对齐为 10 候选（真未来+2 时间近邻+6 随机+1 换动作）。

**待办更新**：
1. ~~env 全绿、manifest 32/8~~ ✓
2. build_cache（后台进行中，日志 `cache_build.log`）→ **人工抽查 align_check jpg：帧-动作对齐**（闸门）
3. frozen 基线（探针 R² + 零动作 Hit@K + diag_pairwise_cos）→ ③④训练

## 三、2026-11-24 特征缓存完成 + 对齐闸门通过

**缓存构建（build_cache.py）**：
- 首次后台跑被会话内存压力杀掉（exit 137），经用户授权重启一次后完成（exit 0）。日志 `cache_build.log`。
- 结果：**7993 train / 991 val 全保留，dropped 0**（完整性闸门零触发，与 manifest 8984 精确对账）→ 16 train + 2 val 分片（SHARD=512）。
- 每窗内容抽验：ctx_patch [2,196,768] fp16（t1/t2）+ pooled [15,768]（窗口 H+1=11 帧 + T+1..T+3 未来 + 随机负例）+ act_win [10,7] f32 + meta。

**对齐闸门（坑 1）——两轮才有效**：
- 第一轮（`--dump-frames` 默认前 5 窗 ep6_T{20,54,56,70,112}）：抽查 T70/T112 = 静止保持段，画面全程无变化、夹爪指令平直 —— **平凡一致、零检验力**。教训：按 (episode,T) 排序的前几窗恰好全落在保持段，选择偏差使闸门形同虚设。
- 第二轮（新写 `pick_align_windows.py`）：对全部 train 窗按**窗口内 gripper 指令摆幅**打分，取 top-5 且 episode 去重 → ep1411_T54 / ep1414_T79 / ep1614_T121 / ep1594_T99 / ep1574_T98，全部 swing=1.00（完整开↔闭事件）。VLM 逐帧核验（画面夹爪状态 + 指令曲线阶跃位置，±2 帧容差）：

| 窗口 | 指令阶跃位置 | 画面可见闭合帧 | 偏差 |
|---|---|---|---|
| ep1411_T54 | t=45→46 | t=47 | 1–2 帧 |
| ep1414_T79 | t=71→72 | t=72 | ≤1 帧 |
| ep1614_T121 | t=116→117/118 | t=117 | ≤1 帧 |
| ep1594_T99 | t=93→94 | t=94 | ≤1 帧 |
| ep1574_T98 | t=91→92 | t=91–92 | 0 帧 |

- **结论：5/5 通过**。偏差 0–2 帧 = 0–100ms @20fps，符合"指令先行于执行"的物理延迟量级。闸门 **通过**，帧-动作对齐可信。
- 脚注：各窗 VLM 对指令极性读法不一（1=开 vs 0=开，合并数据集各子集可能不同），不影响对齐结论（只看阶跃↔事件时刻）。ep1411 窗内无可见抬起，属"抓取未离地"段，但阶跃与闭合仍对齐。

**待办更新**：
1. ~~cache + 对齐闸门~~ ✓（本轮）
2. frozen 基线三件套：`probe_r2.py --space frozen` / `eval_retrieval.py --arm frozen` / `diag_pairwise_cos.py`（原始 CLS 部分）
3. 三臂训练（先 `--steps 200` 冒烟再全量）→ 全套指标 → 回填判读规则

## 四、2026-11-24 frozen 基线 + 探针协议修正 + 三臂冒烟通过

**frozen 基线三件套（val=991 窗，任务级 8 个未见任务；池 9 候选，chance Hit@1=1/9≈0.111）**：
- **层1 探针 R²**（ridge, α∈{0.1,1,10,100}，concat CLS 对 1536-d）：任务移位 val **mean −0.290**（逐维 −0.13/−0.67/−0.44/−0.50/−0.07/−0.38/+0.15）。
  - **关键对照**：train 内部随机 90/10 划分 → **mean R² 0.482**（逐维 0.27~0.63）。⇒ frozen CLS 对里有近一半线性可回收的动作信息，**是任务级移位把它打成负数**（val 任务动作分布整体偏移：如 dim1 均值 0.101→0.040、dim3 −0.0004→0.0090）。这不是 bug，是"监督源在分布外是否保真"的基线画像。
- **层2 零动作检索**：cos(o_t1, cand)：Hit@1/5/10 = **0.513 / 0.991 / 1.000**。top-1 类型：true 508（51%）、tn+1 169、tn+2 89、批内随机终帧 b1-b5 合计 225、manifest rand 0。⇒ 纯时序相近性已拿下一半；三臂要打的零动作线 = **0.513**。（Hit@10 对 9 候选池恒为 1，仅 cape 池=10 时有效——设计使然，主看 Hit@1/5。）
- **层3 前置塌缩诊断（原始 CLS）**：cos(o_T,o_{T+d})=0.977/0.965/0.957（d=1/2/3）；跨集随机 0.729；同任务随机子集 0.835（n=209）；跨窗同任务 0.852。⇒ 原始空间时序-随机间隙 ~0.25，任务间可分；塌缩风险主要看训练后臂空间（B 部分）。

**探针协议两处修正（训练前登记，任何臂出分之前）**：
1. **bug**：配对数误取 pooled 全长 15→14 对，而动作只有 H=10 → 特征/标签错位（111902 vs 79930，matmul 直接炸）。cape/fwd 空间同样会炸，在 frozen 上先拦住。修：`n = H`。
2. **修正案（预登记）**：ridge 前按 **train 统计 z-score** 特征。理由：原始 CLS 范数大、X^T X 病态，α 扫不回来（实测标准化前 −0.230 / 后 −0.290，结论不变但数值条件稳妥）。42 号协议惯例。

**训练链三处小修（冒烟暴露，全是接口错位）**：`id_pairs_from_window` 收 ndarray 崩 → 入口 `as_tensor`；collate 反向按 ndarray 写 → `as_tensor`；cape collate 多解包 branch 3 元组 → 修。**三臂 200 步冒烟全过**（id 0.1min / cape 0.2 / fwd 0.1；参数量 20.1M / 25.2M+0.6M 对账无误）。

**评测链冒烟（smoke 权重跑全套，rc=0）**：数字垃圾属预期（Hit@1≈随机、cape swap 候选 top-1 647/991）。一个**先记后验**的观察：未训练 cape 的 swap-wins 已 = 0.052、cos(u1,u2)=0.956 vs cos(u1,uswap)=0.249 —— 同窗双支路动作高度相关、跨窗动作差异大，随机网络即可分开 ⇒ **B 部分诊断的分辨力要看训练后**（若训练把 cos(u1,u2) 推高同时 uswap 压低才说明学到"目标收敛"；初始态如此不奇怪）。

**待办更新**：
1. ~~frozen 基线~~ ✓（本轮）
2. 三臂全量训练（STEPS=20000 / BS=8 / LR=3e-4 / SEED=42，matched）——后台进行中，日志 `train_full.log`
3. 全套指标（probe 4 空间 / 检索 4 臂 / diag A+B）→ τ 扫描 {0.05,0.1,0.2}、id l1/l2 → H3 消融 → 回填判读规则

## 五、2026-11-24 三臂首跑主表 v1 + cape 塌缩机理定位

**训练**（matched 20k/8/3e-4/42；耗时 id 5.4min / cape 28.3 / fwd 11.7，19:08–19:58）。id train loss 0.06、val 平台 ~0.19——任务移位缺口在损失层已显形。

**主表 v1**（val 991 窗任务级移位；池 9 候选 chance Hit@1=0.111）：

| 臂 | 层1 探针 R² | 层2 Hit@1 / Hit@5 | 层3 诊断 |
|---|---|---|---|
| frozen | −0.290 | 0.513 / 0.991 | 时序 cos 0.977 / 随机 0.729 |
| fwd | −0.348 | 0.082 / 0.598 | — |
| **cape** | −0.128 | **0.000** / 0.454 | B: cos(u1,u2)=**1.0000**、cos(u1,uswap)=**1.0000**、swap-wins 0.129 |
| id | −0.163 | 0.100 / 0.674 | top-1 类型均匀散开 |

**H1 首跑判读**：探针层序 cape(−0.128) > id(−0.163) > frozen(−0.290) > fwd(−0.348)，方向符合 H1 但全部近零——任务移位下无臂"显著优于"。检索层全面反转：**三臂全输零动作基线 0.513**，cape 归零。fwd 垫底与 42 号"forward 对照无效"一致 ✓。

**cape 塌缩 = 真实训练结果，非 bug（证据链）**：
1. val InfoNCE 自 step 2000 起 =0.0（正对余弦完美）而 train loss 卡 **0.69315=ln2**——正对与 swap 负例完全打平的死地板；
2. `cos(u1, u(随机动作)) = 1.0000`：喂噪声动作 u 逐位不变 ⇒ 解码器**完全忽略动作流**（初始化时 cos(u1,uswap)=0.249 是动作敏感的，训练主动摧毁）；
3. `||u||=1.00`（ProjHead 内部 L2 归一化）⇒ 排除范数作弊，训练/检索打分实为余弦，全链无归一化 bug；manifest 核实 t1<t2 全成立（均隔 3.0 帧）。
- **机理**：同集近帧 ctx 高度冗余（驾驶线 L2 cos 0.9955 同病）——达成 u1≈u2 的最廉价路径是把动作贡献压零；swap 负例在 τ=0.1 拉不住。Hit@1=0.000 的机械成因：swap 候选与查询共享 ctx 且 u 动作盲 ⇒ cos=1 恒第一（991/991）；Hit@5=0.454 ≈ 4/9 条件随机。
- **预登记协议内在张力（记录在案）**：训练把 g(bridge(未来帧)) 当负例推离 u1，与检索层"召回真未来"方向直接对抗；sweep 后定评测修正案（9 候选池主口径等）。
- **科学含义**：这正是 v6/L2 病灶在受控监督设定下的复现——**goal-convergent 目标在冗余上下文上靠上下文相似性而非动作信息达成目标**，直接支持"监督源区分"论点（CAPE 的机器不自动注入动作相关性；其原文用 DROID 视觉多样性高、支路间隔长，可能因此未踩坑=二期差异点）。

**id 检索 ≈chance（0.100）**：IDNet 在移位任务上预测整体偏移，排序被分布外偏置主导（真候选无优先）——与 42 号"小固定子集探针值会骗人/迁移敏感"的告诫一致。

**预登记 sweep 已启动**（后台 ~95min，`sweep_train.log`）：cape τ∈{0.05, 0.2}（裁决温度能否救回动作敏感）、cape --no-swap（H3，预期塌得更彻底）、id --form l2、id --shuffle-labels（H3 错位对照，新加旗标：批内 roll 动作标签）。

**待办**：sweep 出分 → 主表 v2 + H3 判读 → cape 评测口径修正案登记 → 决策层（二期）。

## 六、2026-11-24 sweep 被内存压力中止（1/5 完成）+ τ=0.05 判读

**事件**：sweep 后台任务被 Claude Code 系统内存压力中止（20:41）。完成 **cape τ=0.05** 一个，余 τ=0.2 / no-swap / id-l2 / id-shuffle 未开始，ckpt 均在（id/cape/fwd 正式+smoke、cape_tau005）。

**τ=0.05 结果（塌缩对温度不敏感）**：
- 训练签名与 τ=0.1 完全相同且更快：loss 从 step 2000 起钉死 0.6931（ln2），val InfoNCE=0.0；
- 检索 Hit@1=**0.000**（swap 991/991 霸榜）、Hit@5=0.441；B 诊断 cos(u1,u2)=cos(u1,uswap)=**1.0000**、swap-wins 0.118；
- 探针 R²=−0.225（比 τ=0.1 的 −0.128 还差）。
- **判读：降低温度不能救回动作敏感——冗余 ctx 下 goal-convergent 的退化解是结构性的，不是超参失误**。τ=0.2 大概率同结局（待补跑确认）。

**待办**：①用户指示后重启 sweep 余项（τ02 / noswap / id_l2 / id_shuffle，约 65 分钟）；②主表 v2 + H3 判读；③cape 评测口径修正案（9 候选池主口径）——注意即使换成 9 池，u 与 g(bridge(未来)) 的训练对抗关系仍在，Hit@ 数字只能作"塌缩佐证"不能作"检索能力"读。

## 七、2026-11-24 深夜 用户指令触发协议复审：v1 判为"偏离原文的预实验"，v2 重建预登记

**用户指令**："仔细阅读 cape 和 id 的论文，务必按照论文中的策略来，包括防坍缩这类问题，以防再走弯路。"

**两篇原文全读**（CAPE 解密副本 p1-19 含附录；42 号全文协议节+附录 G/M）。关键发现：
- **塌缩的根因在我的实现偏离，不全是数据本性**——v1 三处结构性偏离（详见设计文档 §11 审计表）：状态编码走了独立 StateBridge（论文=同一解码器+可学习空动作查询，预测/状态同空间）；批结构=随机跨任务（论文=同 episode 16 样本组，硬负例同背景只差动作——**这是论文的防塌缩核心**，附录 D.2 自己记录了去掉负例即塌缩，与我们的 ln2 地板现象同款）；支路间隔 0.15s（论文 clip 内支路隔数秒、剩余动作长度不同）。
- 检索池难度不对（论文 ~420 同任务候选、zero-action 6%；v1 9-10 混合候选、zero-action 51% ⇒ 任务不要求动作信息）。
- 42 号：探针应为 MLP（3k 步 bs512 3 seeds、patch-mean 特征）而非 ridge；shuffle 对照应**同任务内置换**而非批内 roll；**其附录 M 冻结探针全任务覆盖 0.449→−0.006 = 我们 frozen −0.29 的同类现象**（绝对值作地板、主张靠 Δ）。
- τ=0.07（不调）、id L2 默认（L1 差 <0.05）、warmup 2k / eff batch 16 / 2 seeds 等细节一并校正。

**处置**：v1 代码+日志快照至 `id_vs_cape_v1_snapshot\`（v1 结果可复现、降级为预实验，两表都报告）；v2 修正案已预登记于设计文档 §11（于任何 v2 数字前）；sweep 余项（τ02 等）**作废不再跑**（旧协议无意义）。开始 v2 实现：manifest/cache（29 槽 pooled+patch-mean）→ 空动作编码模块 → 同集组批训练 → MLP 探针/同任务池检索 → 反事实敏感度监测。

## 八、2026-11-24 v2 代码全套落地 + manifest v2 生成 + cache_v2 构建启动

**代码冻结**（11 个文件，全部过编译+模块自检：CAPE 25.2M / IDNet 20.1M / state enc 单位范数）：
- `common.py`：v2 常量（t2=T−U[5,12]、t1=t2−U[5,12]、PAST=24、τ=0.07、lr/wd=1e-4、warmup 2k、组批 8 集×4 窗=32、ID 累积×2=有效 16）；`cache_v2/` 路径。**修复**：v2 重写时漏掉 DIM/IMG 两常量导致 import 崩（补回 768/224）。
- `build_manifest.py`：分支采样按论文（不同剩余长度汇于同一 o_T）。**修复**：`df.T` 被 pandas 当转置 → `df["T"]`。
- `build_cache.py`：29 槽 pooled+pmean；probe 窗口 patch（val 全量 + train 前 2500，manifest 顺序稳定）**改分片落盘**（每 256 窗一个 npy part）——原设计最后一次性 `np.stack` 需瞬时 ~23GB RAM，WSL 只有 14G 必炸（前两次后台被杀的元凶之一）。
- `modules.py`：空动作查询 `forward_state`（同一 ctx/dec/g 通路 → 预测与状态同空间）；位置表扩到 H_MAX_ACT=24；新增 `FwdModel`（同干、输出头 768、含 `encode_state`）。
- `train.py`：cape 组批损失（负例=组内他窗 t1'/t2' 空动作键 + 组内换动作结果 + 跨批随机 2 键；τ=0.07）；**反事实坍缩监控**每 2k 步（真/同任务外窗/随机动作的 u 敏感度 + swap-wins）；id L2 默认、within-task shuffle（整 H 块动作同任务捐赠，破坏帧↔动作对应）、梯度累积；fwd 用 FwdModel 预测终态 CLS。
- `probe_r2.py`：42 号 MLP 探针为主（1536→256→128→7、3k 步、bs512、AdamW 3e-4、3 种子、per-dim+pooled R²）；ridge（训练统计 z-score）为辅；`id_direct` = IDNet 原生头直读；`--y-shuffle` 同任务置换地板对照；臂特征缓存 npy 复用。
- `eval_retrieval.py`：论文式检索——query=(ctx_t1, acts[t1..t2)) 预测 o_{t2}（horizon 5-12 恰为 branch-2 长度分布=分布内泛化）；池=同任务全部 val 窗的 t1/t2 状态编码（~250 键）；排除自身 t1 键；frozen=零动作 CLS 余弦基线；id=1 步 IDNet 打分（anchor t2−1 特权，登记为诊断量非公平预测）；Hit@1/5+MRR（micro+macro）。
- `diag_pairwise_cos.py`：A 部分 CLS+pmean 双特征；B 部分分支忠实版（u1/u2 异 ctx 异动作汇 o_T、u_b 对 enc(o_T) 键一致性、组内换动作 swap-wins、反事实敏感度）。
- `windows.py`：probe 分片读取助手（probe_parts/probe_window_frame）；`run_all.sh`：train 2 种子 + controls（shuffle/L1）+ eval 全电池。

**manifest v2 ✓**：8988 窗（train 7993 / val 995，32/8 任务），ctx 间隔 5–12 帧（中位 9=0.45s；**v1 仅 3 帧——坍缩根源**），branch-1 剩余中位 17 帧（0.85s）。

**cache_v2 构建启动**（后台任务 byd1o9rw9，日志 `cache_v2/build_cache.log`，预计 1–2h，~17GB）。坑两次：WSL 里 `nohup ... &` 随 wsl.exe 会话退出即死（两次假启动），改用 Claude Code 后台任务保持进程树存活后确认：pid 存活、GPU 1.7GB/50%。

**下一步**：cache 完成过完整性闸门 → 三臂×2 种子训练 + controls → eval 全电池 → 主表 v2（v1 表降级预实验，两表并报）。

## 九、2026-11-24 cache_v2 三次启动始末（VM 崩溃→OOM 定位→修复→零丢弃）+ 冒烟七修 + 正式训练启动

**三次启动**：①WSL 内 `nohup &` 随 wsl.exe 会话退出即死（假启动，且 `| tee` 把退出码吞成 0）；②改 Claude Code 后台任务后 **WSL VM 整体崩溃**（journal unclean shutdown、dxg 报错），python 随之被杀；③再启动 **EXIT=137 OOM**——`echo EXIT=$?` 在管道后又把码抹平，从任务输出文件里读到真实 137。

**OOM 定位（diag_mem.py）**：视频原生 256×256；逐集 range 不交错（peak_concurrent_eps=1，RGB 流仅 ~120MB/集，无辜）；**真凶=patch 攒积**——assemble 原在文件尾执行，file-000 单文件要持有 30260 帧 patch（probe 前 2500 训练窗按 episode 排序聚在早期集）≈ 9.1GB，全程 45212 帧 = 13.6GB > 13GB 空闲。**修复=episode 一 flush 立即 assemble+释放其 patch**（feats/pmeans 全程仅 ~0.5GB，随机负例跨文件引用需要全局保留，实测无碍）。修后 file-000 用时 ~4 分钟，**全程 ~15 分钟完成：7993/995 窗 dropped 0，16+2 分片，14 个 probe 分片（2500/995 行），共 17GB**——远快于 v1（当时没修峰，慢在换页）。

**冒烟七修**（30 步×三臂 + 评测全链路）：①id pmean fp16 × 参数 fp32 → id_pairs_from_window 显式 float32；②branch() 返回 numpy、pad_actions 按 v1 期望 tensor → 改 torch.as_tensor(dtype=float32) 双兼容；③pad_actions 返回单值但全部调用点按 (tensor, lengths) 解包 → 统一单值（train/eval_retrieval/diag 共 8 处）；④组内负例数不齐（2-3 窗的集）→ 不足的锚点用批内随机键垫到同一 K；⑤probe_xy 拿同一个 ds 映射两个 split 的 probe_meta（train 侧查到 val 窗 (2,41)）→ 按 split 各查各的（数据核对：manifest/shards/probe_meta 三方完全一致，纯读取 bug）；⑥numpy Generator 无 randint → integers；⑦Xva numpy 无 .to → from_numpy。

**冒烟评测两项设计确认**（数字无意义仅验机器）：池难度到位——frozen 零动作 Hit@1 从 v1 的 0.513（9-10 混合候选）降到 **0.162**（同任务 ~250 键，池大小 132-370），接近论文 DROID 6.07% 的量级，任务终于要求动作信息；诊断 B 部分全指标可算（u_b 对 enc(o_T) 键一致性、组内换动作、反事实敏感度）。

**正式训练启动**（后台 b08dux3zy）：id/cape/fwd × seed 42/43 + id_shuffle + id_l1 共 8 个串行任务，`set -e` 单臂失败即停，逐臂日志 `cache_v2/train_{arm}{tag}.log`，预计 6-8h。cape 每 2000 步打印反事实敏感度（cos_foreign/cos_random）+ swap-wins——v2 协议是否逃出 v1 坍缩的第一现场指标。


