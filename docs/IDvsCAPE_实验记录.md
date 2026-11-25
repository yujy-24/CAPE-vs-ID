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

## 十、2026-11-25 v2.0 主力结果（seed 42）：坍缩逃逸成立、检索层爆冷、根因定位→v2.1 修正案

**进程**：训练队列被 Claude Code 内存压力于 id_seed43 半途中止（按约定不自动重启）。**seed 42 三臂在中止前已全部完成**（id 7.6min / cape 155.4min / fwd 12.5min）；剩余：id_seed43（半途无 ckpt）、cape/fwd_seed43、id_shuf、id_l1 待用户指示。

**坍缩逃逸（v2.0 cape，训练监控终段）**：

| 监控量 | v1 坍缩签名 | v2.0 |
|---|---|---|
| cos(u真, u随机动作) | 1.0000 | **0.02–0.08** |
| cos(u真, u外窗动作) | 1.0000 | 0.05–0.07 |
| swap-wins | 0.118 | **≈0** |
| val InfoNCE | 0.0 | ~0.05 |

**v2.0 主表（seed 42，held-out 8 任务）**：

| 臂 | 探针 R²(pooled, 3 seeds) | 检索 Hit@1 / Hit@5 / MRR | 池~250 同任务键 |
|---|---|---|---|
| frozen | −0.712 ± 0.069 | **0.162 / 0.389 / 0.278** | 零动作基线 |
| frozen y-shuffle 地板 | −0.533 ± 0.084 | — | |
| id_direct（原生头） | −0.009（MAE 0.198） | 0.002 / 0.032 / 0.034 | 1 步特权 anchor |
| cape | −0.335 ± 0.039 | 0.000 / 0.012 / **0.021（=随机排序期望 0.022）** | |
| fwd | −0.369 ± 0.056 | 0.040 / 0.156 / 0.116 | |

诊断 B（训练后 cape）：cos(u1,u2)=0.863 ✓ 同果收敛；cos(u2,uswap)=0.118 ✓ 动作敏感；**cos(u1,enc(o_T))=−0.047 —— u 与状态编码空间正交**。

**判读（v2.0 口径下）**：①防坍缩机制移植成功——同集组批硬负例把 v1 的结构性坍缩彻底治好，且不依赖温度（v1 已示 τ 不敏感）；②探针层全负 + id_direct≈0：任务级移位下任何臂都无浅层可迁移动作信息（42 号 App M 同型现象，绝对值作地板）；③检索层三臂仍全输零动作基线，cape 恰为随机——非坍缩所致，而是**几何缺口**。

**根因（对照原文）**：CAPE 附录 A 空动作编码三用途之第三 = "zero-horizon training samples"（目标状态以零动作支路进**正对集**）；v2.0 只用 u1↔u2 正对、状态键全放负例侧 ⇒ u 被推离状态流形（测得 −0.05）⇒ 检索随机。检索协议本身与附录 C.2 逐条一致。**v2.1 修正案已预登记（设计文档 §12，先于任何 v2.1 数字）**：多正对 InfoNCE（正对={u2, enc(o_T)}）+ cache 增存全窗 o_T patch（+2.7GB；id/fwd 不受影响无需重训）；残留偏离（论文的 T±1..3 邻近时刻负例未实现，以组内他窗键近似）如实入局限节。

**v2.1 执行**：cache 重建 + cape_v21 重训（后台 bfrusxg3f，监控新增 goal_cos 读数：u→enc(o_T) 余弦应从 −0.05 起爬升）。



## 十一、2026-11-25 v2.1 判决：修正无效——logsumexp 正对集饱和（数学性死锁）；v2.2 预登记

**v2.1 结果（cape_v21，178.5 min，seed 42）**：

| 指标 | v2.0 | v2.1 | 判读 |
|---|---|---|---|
| 检索 Hit@1 | 0.000 | 0.008 | 仍≈随机（chance≈0.004） |
| 检索 MRR | 0.021 | 0.0284 | 仍≈随机（期望 0.022） |
| 探针 R² | −0.335 | −0.335 | **分毫未动** |
| cos(u1,enc(o_T)) | −0.047 | −0.037 | 仍正交 |
| 训练 goal_cos | —（无此项） | −0.12→−0.06 全程不爬 | 监控实锤 |
| cos(u1,u2) / swap-wins | 0.863 / ≈0 | 0.872 / 0.020 | 坍缩逃逸保持 |

**根因（数学，非超参）**：多正对 logsumexp 分子里 u1↔u2 项 cos 0.87 @ τ=0.07 → e^12.4≈2×10⁵，s_T 项 cos −0.04 → e^−0.6≈0.55；softmax 权重比≈10⁻⁶ ⇒ **目标状态正对的梯度≈0**。loss 早已被 u1↔u2 满足，s_T 永不被拉。探针 R² 分毫不动即是旁证：v2.1 网络与 v2.0 实质等价。错在把附录 A "zero-horizon training samples" 读成"塞进正对集"。

**回读原文（附录 A + 正文式 3）**：式 (3) 是**成对**形式（正对 (u1,u2)，各自独立分子+对称项）；"zero-horizon training samples where the action sequence length is zero" 是**批次中的独立训练样本**（第三条汇于同一 o_T 的分支），与其他分支**成对**出入式 (3)——每对一个独立分子，梯度不饱和。另：批次里其他窗口的 s_T 亦应在负键池中（训练样本全体互为负例）。

**v2.2 修正案（先于任何 v2.2 数字预登记，设计文档 §13）**：
- 损失 = 三分支 {u1, u2, s_T} 逐对对称 InfoNCE 取均值：(u1,u2)+(u1,s_T)+(u2,s_T)，每对独立分子（式 3 原形）；
- 负键池从 {k1,k2} 扩为 {k1, k2, s_T}（他窗），换动作+跨批随机不变；
- 监控增 nce_goal（(u1,s_T)与(u2,s_T) 两项均值，应显著下降）+ goal_cos（应爬升过 0）；
- id/fwd 不受影响无需重训；cache 无变化无需重建。
- 预登记判据：若 20k 步后 goal_cos 仍 <0.2 或检索仍随机 → 判"本数据规模/池结构下 CAPE 检索优势不可复现"，按预登记主线（42 号 ID 探针对比）收口，不再迭代 Cape 实现细节。

**执行**：v2.2 代码落地（train.py cape_loss 重写，冒烟 250 步 2.0min 通过，loss 8.45→1.58 有限）；**重训（约 160min）待用户指令**（按 GPU 占用约定）。v2.1 结果与 v2.2 代码已推 git。

**重训前原文复核（11-25，用户指令"再次根据原文确认机制"）——五项全部核对原文原文文字**：

1. **式 (3) 成对形式**（正文第 3.2 节逐字）：分子只有 exp(s(u⁽¹⁾,u⁽²⁾)/τ) 单正对 + 对称项（"with a symmetric term obtained by swapping the two branches"）。**全文不存在多正对 logsumexp**——v2.1 的共享分子形式系我的发明（§12 已如实登记为读法错误）。
2. **零视野样本=独立训练样本**（附录 A 逐字）："We use this empty-action state encoding for candidate-key encoding in retrieval, goal-state encoding in planning, and **zero-horizon training samples where the action sequence length is zero**"——空动作查询为可学习参数（"learnable empty-action query"），样本走同一通路。Table 5（训练超参全表：AdamW/1e-4/cosine/batch 2048/每组 16 样本/τ=0.07/H=5）**未给零视野样本出现比率**；正文偏移 "choose two offsets k1 and k2" 亦无范围。⇒ 配对比率是原文未定细节；v2.2 取确定性三分支实现（每窗三对全训），两种读法（第三分支 vs 偶发替换某分支）的公共核心一致：目标状态以独立样本身份、**成对、满梯度**参与——此核心由 1+2 联合锁定，登记为"残留歧义"而非偏离。
3. **负集构成**（正文逐字）："temporal neighbors from nearby but distinct future time steps, outcomes produced by swapped or mismatched action sequences, and randomly sampled states from other trajectories"——三分结构与 v2.2 一致（他窗 s_T 入池=批次训练样本全体互为负例）；T±1..3 精确邻近负例仍以组内他窗键近似（残留偏离，局限节报告）。
4. **检索协议 C.1**（附录逐字）：query 由"action query conditioned on the visual context token"解码、candidate keys 用空动作查询、余弦相似度、同任务全帧池（论文 ~420 候选 vs 本实现 ~250）——与 eval_retrieval 一致。
5. **消融表交叉佐证**：Action-Visual Alignment（"directly aligning the predictive future state with future observation"，即纯 (u,enc(o_T)) 对齐）检索 Hit@5 达全模型 84~91%（80.19 vs 84.19 @h=5 等）——**(u, enc(o_T)) 对齐通道就是论文检索性能的来源**；全模型=分支对+零视野对的完整目标函数。v2.0/v2.1 恰好完全没有该通道（cos −0.04）⇒ 随机检索与论文机制自洽。v2.2 逐对式将其接回=全模型读法。

**结论：v2.2 机制与原文锁定一致，可执行；唯一残留歧义（零视野配对比率未指定）已登记。重训仍待用户指令。**

## 十二、2026-11-25 官方开源代码对齐调查 + v2.3 移植落地

**开源代码调查（用户指令"查开源代码，直接套用或参考复刻"）**：
- **CAPE 官方仓库已放出**：`github.com/chen2151643/CAPE`（论文页/arXiv/HF papers 均无索引，GitHub API 搜索命中——"每次调查开源代码就报错"的 ToolSearch 1210 毒已绕过，全程 webReader+curl）。已 clone 至 `E:\world_model\external_repos\CAPE`。数据管线绑死 DROID（双外参相机/rot6d/euler/lerobot 包），**不能直接套用，机制层逐行参考**。
- **42 号仓库确认公开**：`k1seul/corl2026-action-relevant-latents`（MIT，CoRL 2026）。ID aux loss = `concat(f_t,f_{t+1})→MLP(512,256)→L2` on patch-mean——与我们的 id 臂机制一致（IDNet 4 层 20.1M 为四臂容量对齐设计，非机制偏离）。已 clone 留档。

**官方代码 vs v2.2 四个裁决性差异**（详见设计文档 §14）：①零视野配对=数据侧随机 50%（第二支路即 anchor），非三分支全对；②分支=anchor 前 1..5 步 × stride4，非 5..12 帧连续；③负池=整批 keys+全批换动作（~1023+1024 vs 我们组内3+跨批2），差两个量级；④last-token 取 valid_len−1 非 padding 末位（v2.2 实现瑕疵，已修）。另发现官方配置与论文两处矛盾（vislayer0 / droid_12000 vs 论文 3 层 / 8000 条），取论文口径并登记。

**v2.3 实现（§14 预登记后动工）**：
- `common.py`：PROTOCOL="v3"（cache_v3）、STRIDE_V23=4/HV23=5（0.2s/步，均在 PAST=24 内 ⇒ manifest 不变、id/fwd 免重训）。
- `build_cache.py`：patch 槽位增存 branch_patch 6 槽 [T−20..T]；ctx_patch/goal_patch 原样保留（脚本过渡兼容）；完整性闸门覆盖新槽。cache_v3 重建后台运行中（`cache_v3/build_cache.log`）。
- `modules.py`：decoder 按 lengths 在**最后有效位** gather（官方 valid_len−1），pad 位替换为空查询 token——单元验证 b) 通过：padding 不再泄漏进终态 token（v2.2 的隐性偏差修正）；`forward_state` 不变。
- `train.py`：cape_loss 重写为官方 `compute_contrastive_loss` 单卡全局版逐行移植——每窗一对（query=k~U{1..5}，key=50% anchor/50% 异支路）、负池=整批 keys+全批 derangement 换动作、单正对+对称 CE、warmup 5%（官方 train.py:236）。监控增 state_frac，保留 goal_cos/swapwins/反事实。
- `eval_retrieval.py` 重写：query k~U{1..5}（seed 1234 四臂同 query）、池=每窗{分支帧,anchor}双键、排除自身分支键；frozen/fwd/id 不重训同协议重评。`diag_pairwise_cos.py` B 部分支路同步 v2.3。`test_v23_units.py` 新增（4 项全绿）。
- 冒烟与重训：cache_v3 完成后先 30 步冒烟；160min 重训按 GPU 约定待用户指令。

**v2.3 落地收尾（同日晚）**：
- cache_v3 构建两段完成（首跑被 Claude Code 内存压力回收器终止于 13/16 片，非构建错误；整跑重Build）：**train 16+val 2 分片、7993/995 窗口、dropped 0、BUILD_EXIT=0**，窗口集与 cache_v2 完全一致（manifest 未动）。
- **内存坑**：branch_patch 留在 npz 内被 WindowDataset 物化 ⇒ ~24GB 常驻爆 15GiB WSL VM（冒烟首跑 OSError Errno 12）。修复：新增 `extract_branch_npy.py` 抽出每 split 单文件 `branch_{split}.npy`，WindowDataset 改 `mmap_mode='r'` 挂载（调用点零改动，常驻回到 v2 的 ~9.3GB）。run_all.sh cache 阶段已挂抽取步骤；.gitignore 补 cache_v3/。
- **30 步冒烟（tag _smoke，0.3min）+ no_grad 监控验证（`smoke_diag_v23.py`）**：train loss 2.06 / val 1.78（随机基线 ln95≈4.55）；**state_frac 0.454≈0.5**（官方 50% anchor 配对机制确认，160 样本噪声内）；**swapwins 0.137≪0.5**（swap 负样本被分离）；cf 排序正确：own 0.811 > foreign 0.713 ≈ random 0.749。v2.3 机制端到端验证通过，全量重训（20k 步，~160min）待用户 GPU 指令。

**v2.3 全量重训进行中（2026-11-25 晚启动，中期进度供参考）**：
- 运行：`train.py --arm cape` 20k 步（seed 42，官方超参），~1.45 步/s，GPU 99%/4.7GB，日志 `cache_v3/ckpt/cape_train.log`（不入 git）。
- 中期监控（step 4000→6000）：val 0.797→0.644；**goal_cos 0.842→0.831**（v2.0/v2.1 该通道为 −0.04 的死区，现已接通，远高于 0.2 终止线）；**swapwins 0.033→0.038**（swap 负样本强分离）；cf foreign/random 0.43/0.43→0.31/0.27（own≫wrong 排序正确）；state_frac 批内波动属 50% anchor 抽样噪声。
- 判读（中期、非结论）：预登记两终止条件（goal_cos<0.2 / 检索随机）均远离；主表判定仍以训后 eval_retrieval（四臂同协议）为准。训完补全条目 + v2.3 主表。

**分工与缓存口径确认（2026-11-25，致合作者）**：
- 分工：id_seed43 → id_shuf → id_l1 由合作者认领（命令即 `python train.py --arm id --seed 43 --tag _seed43`，从头跑，代码只存 final.pt）；我方负责 v2.3 cape 全量重训（进行中）+ 训后评测主表。任何一方动队列前先 `git fetch` 对表，避免重复。
- **原计划的 seed 42（cache_v2 训）与 seed 43（cache_v3 训）同口径，仅指采样及代码协议**：①cache_v3 用未改动的 manifest 重建，窗口集与 v2 完全一致（7993/995、dropped 0，构建日志）；②id/fwd 臂训练不读 branch_patch（id 用 pmean/act_win，fwd 用 pooled/ctx），v2→v3 对它们只是存储布局变化；③seed 42 的 id/fwd final.pt 已复制到 `cache_v3/ckpt/{id,fwd}/`。跨机器重建时仍须核对 DINO 权重或特征哈希，详见第十三节。
- 若在本机跑：4070 正被 cape 重训占用（预计 ~2h 后空出），请错峰；若在自有环境跑：缓存不入 git，需先 `build_cache.py` + `extract_branch_npy.py`（README 已注明）。

## 十三、2026-11-25 ID seed 43 云端训练与评估

**事件记录**：17:02:27（UTC+8）启动 `train.py --arm id --seed 43 --tag _seed43`；17:08:44 训练 20,000 步完成并写出 `final.pt`；17:09:12 完成 `probe_r2.py --space id_direct` 评估。开始、完成及评估三项事件也写入 [`RUN_LEDGER.md`](RUN_LEDGER.md)。本次仅执行 ID 臂，源码提交为 `a46f93be98064ac39d8b3cd3f0ebb72ced17d9c4`。

**环境与输入**：单张 RTX 4090（运行时报告 23,028 MiB 显存）、60 GiB 容器内存；Python 3.12.3、PyTorch `2.7.0a0+7c8ec84dab.nv25.03`。新实例共享盘为空，故从公开 `jialei02/libero_merged_no_noops_20hz` 数据集和 DINOv3 ViT-B/16 权重重新构建 `cache_v3`；manifest 为训练 7,993／验证 995 窗口，缓存构建丢弃 0。缓存分片保存在共享盘，`branch_*.npy` 旁路文件因共享盘容量限制放在实例系统盘，并以符号链接接入加载器。

**训练及留出任务结果**：L2 逆动力学，batch 8、累积 2、20,000 步，训练耗时约 6.3 分钟；最后记录的验证损失为 step 18,000 的 `0.10544`。ID 原生预测头在 9,950 个验证动作对上得到汇总 R² `-0.0404`、MAE `0.2033`，逐维 R² 为 `[-0.1514, -0.1930, 0.1279, -0.2113, -0.0887, -0.0390, -0.0532]`。原始配置、逐步指标、训练日志、评估日志、缓存摘要及校验和均在 [`results/id_seed43_20261125/`](../results/id_seed43_20261125/)；二进制检查点保存在服务器共享盘，未加入 Git。

**跨种子权重核验（已解决）**：本次 ModelScope 权重的文件 SHA-256 与 seed 42 原机器使用的 DINOv3 权重完全相同；见 [`RUN_LEDGER.md`](RUN_LEDGER.md) 的完整哈希核对。两机不存在权重文件差异，seed 43 可以用于同权重的 ID 跨种子比较。两机共同使用的文件与 Hugging Face 当前 `main` 文件的字节哈希不同，但原因尚未验证；这属于“是否等同当前官方版本”的独立问题，不能反推 seed 42 与 seed 43 的权重不同。后续队列仍为 `id_shuf → id_l1`。
## 十四、2026-11-25 v2.3 全量重训完成 + 检索主表：**终止判据未触发，官方机制移植成功**

**训练**：`train.py --arm cape` 20k 步 134.9min，TRAIN_EXIT=0。终态监控：goal_cos 0.851（v2.0/v2.1 死区 −0.04 → 通道接通）、swapwins 0.000、cf foreign/random 0.240/0.161；val loss 6k 处 0.644 见底后缓升至 0.911（80 epoch 曝光=官方口径，按预登记取 20k 终模，不早停）。

**检索主表（eval_retrieval v2.3 协议，995 val 查询，池 132–370 键/任务，随机 Hit@1≈0.004）**：

| 臂 | Hit@1 | Hit@5 | MRR | macro Hit@1 | macro Hit@5 |
|---|---|---|---|---|---|
| **cape (v2.3)** | **0.1920** | **0.4492** | **0.3192** | **0.2101** | **0.4833** |
| frozen | 0.1246 | 0.3317 | 0.2302 | 0.1102 | 0.3049 |
| fwd | 0.0482 | 0.1698 | 0.1176 | 0.0507 | 0.1858 |
| id* | 0.0080 | 0.0412 | 0.0388 | 0.0095 | 0.0521 |

*id 检索为预登记的"1 步打分、anchor T−1 不可知"诊断性指标，非公平预报（见记录·五）；其主指标在二层探针。

**判读**：
1. **检索显著高于随机且 cape > frozen（最强零训练基线）+6.7pt Hit@1 / +11.8pt Hit@5 / +8.9pt MRR**——v2.0/v2.1 连续两版随机的检索层，在逐行对齐官方代码（50% anchor 配对、整批负池+换动作、valid_len−1 gather）后被救活。**§13/§14 预登记的"检索随机 ⇒ 终止"判据未触发，CAPE 实现迭代无需收口。**
2. 机制归因清晰：v2.x 死因=正对缺目标状态+负池过窄（组内 3+2），v2.3 把 (u, enc(o_T)) 对齐通道与大批硬负样本同时接回——与 CAPE 论文附录 A 消融（Action-Visual Alignment 贡献检索性能 84–91%）一致。
3. 四臂排序 cape > frozen > fwd > id：前向预测（fwd）在此协议下反而低于零动作 frozen——预测性状态编码的"预报价值"不等于"检索价值"，支持两层假设中"监督源区分"的核心论点（fwd 拿后果不拿对齐）。
（探针五连与 diag A/B 进行中，随后补录+主表终版。）
**与十三节的交叉核对（已解决，详见 RUN_LEDGER 哈希注记）**：两机 DINOv3 权重 SHA-256 逐字节一致（`9a21ac3d…`）——与 HF 当前 `main` 文件字节不同的原因尚未验证，但并非两机差异；seed 43 的 R² −0.0404 可用于同权重的 ID 跨种子比较。manifest 字节差异仅涉 cape 辅助列（t1/t2/rand），id 臂不读，不影响此项对比。

**探针五连 + diag A/B（v2.3 终版补录，seed 42）**：

二层探针（9,950 val 对，25,000 训练对，1 步动作 **MLP 读出**（42 号协议 fresh probe [2D→256→128→7]，ridge 为次要对照未启用)）：

| 空间 | R² pooled | MAE | 逐维最优 |
|---|---|---|---|
| frozen（y-shuffle 地板） | −0.533 | 0.2573 | — |
| frozen | −0.712 | 0.2677 | −0.152 |
| **id_direct** | **−0.009** | **0.1984** | dim2 **+0.161** |
| cape | −0.486 | 0.2496 | −0.079 |
| fwd | −0.861 | 0.2773 | −0.404 |

**仪器警示（如实登记）**：全部空间 R²≤0，且 y-shuffle 地板（−0.533）高于真实 frozen（−0.712）——MLP 探针（即已是非线性仪器）在本设定下没有可用的绝对刻度，判读只取排序。跨种子一致性：id seed42 −0.009/MAE 0.198 ≈ seed43（远端）−0.040/0.203 ✓。

diag（600 val 窗）：A 部 CLS 时近 cos 0.96–0.99、跨集 0.73（LIBERO 帧相似几何如旧）；B 部 cape 空间：同结果两支路 cos(u1,u2) 0.853、cos(u,enc(o_T)) 0.41（p95 0.79，训练监控 0.85 为训练对口径）、foreign-acts 0.218、random-acts 0.123、swap-wins 0.072——排序 own≫foreign≫random 干净成立。

## 十四（终）、v2.3 主表判读（对照 §2 预登记假设）

- **H1（复现层）半成立**：检索层 cape 显著优于两基线（Hit@1 0.192 vs frozen 0.125/fwd 0.048，+6.7pt ≈ 6σ；Hit@5/MRR 同向）✓；探针层 id 相对最优（−0.009 vs frozen −0.712/fwd −0.861）但**绝对值 ≈0，未达"动作可恢复"的正值主张** ✗——42 号主张在此数据/规模未复现为正值，只复现为相对排序。
- **H2（分层层）按排序口径成立**："各有侧重"的图案出现：id 赢探针排序、cape 赢检索——两种监督源塑造的几何确实分化。附加注记：fwd 检索（0.048）反而低于零动作 frozen（0.125），"预报价值≠检索价值"，与"监督源区分"核心论点一致。
- **H3（机制层）尚未收口**：id-shuffle 对照已完成，结果见第十五节；**cape-noswap（去换动作负例）尚未入队——建议补入**，它是"检索增益来自 swap 硬负例"归因的最后一块。
- 终止判据终版结论：**未触发**。CAPE 实现迭代以 v2.3 收口（官方代码对齐后机制成立），后续迭代只做消融与多种子，不再改协议。

**剩余队列**：cape-noswap（建议）→ 双种子检索并表（seed43 cape 可选）。ID 的 `id_shuf` 与 `id_l1` 控制均已完成，详见第十五、十六节。

## 十五、2026-11-25 ID shuffled-label control 完成

**运行记录**：19:05:50（UTC+8）开始，19:12:14 训练结束（`train_exit=0`），19:13:54 评估日志落盘。命令为 `train.py --arm id --shuffle-labels --form l2 --seed 42 --tag _shuf`，20,000 步；训练 7,993 窗、验证 995 窗。源码提交 `a46f93be98064ac39d8b3cd3f0ebb72ced17d9c4`。本次只运行 ID 控制臂，没有启动 CAPE 或 fwd。原始起止时间、状态、配置、逐步指标和终端输出已提交至 [`results/id_shuf_20261125/`](../results/id_shuf_20261125/)。

**结果**：留出验证集 9,950 个动作对上，ID 原生头汇总 R² `-0.1182`、MAE `0.2167`，逐维 R² `[-0.0815, -0.1073, -0.2538, -0.0464, 0.0119, -0.1145, -0.0353]`。相同 seed、相同评估口径下，正常 ID 为 R² `-0.0087`、MAE `0.1984`；打乱标签后两个指标均变差，支持训练时的帧—动作对应关系有作用。两个绝对 R² 都为负，不能据此声称跨任务动作已被准确恢复。`y_shuffle: false` 表示评估仍与真实动作比较，打乱只发生在控制组训练阶段。

**控制实现限制**：当前 `train.py` 在同任务窗口内做随机置换，未强制去除固定点，也未强制跨 episode；本结果应按该已实现的 shuffle 对照解释，不应说所有帧—动作对应关系都被彻底破坏。H3 的 CAPE noswap 消融仍未完成。

**检查点**：`/root/shared-nvme/CAPE-vs-ID/cache_v3/ckpt/id_shuf/final.pt`；Git 仅保存其 SHA-256 和日志，二进制权重未加入仓库。

## 十六、2026-11-25 ID L1 损失对照完成

**运行记录**：19:47:08（UTC+8）开始，19:53:32 训练结束（`train_exit=0`），19:53:52 评估日志落盘。命令为 `train.py --arm id --form l1 --seed 42 --tag _l1`，20,000 步；训练 7,993 窗、验证 995 窗。源码提交 `a46f93be98064ac39d8b3cd3f0ebb72ced17d9c4`。原始起止时间、配置、209 条逐步指标、训练与评估日志及检查点 SHA-256 均在 [`results/id_l1_20261125/`](../results/id_l1_20261125/)。

**同 seed 损失形式对比**：留出验证集 9,950 个动作对上，L1 的 ID 原生头汇总 R² `-0.1804`、MAE `0.1997`，逐维 R² `[-0.1251, -0.1581, 0.0665, -0.0495, 0.0128, -0.1206, -0.4287]`。正常 L2 seed 42 为 R² `-0.0087`、MAE `0.1984`。L1 的 MAE 仅高 `0.0013`，但平方误差口径的 R² 明显更差；本次运行没有证据显示 L1 优于 L2。训练时 L1、L2 损失量纲不同，不能直接比较损失数值；两个绝对 R² 均非正值，也不能据此声称留出任务上的动作已被良好恢复。

**检查点**：`/root/shared-nvme/CAPE-vs-ID/cache_v3/ckpt/id_l1/final.pt`；Git 保存路径、校验和与日志，未提交二进制权重。ID 分支已完成预登记队列中的第二种子、打乱标签与 L1 对照。

## 十五、2026-11-25 晚 ID 对照组齐套（远端 4090，合作者执行）+ H3-id 判读

**对照组主表（probe id_direct，9,950 val 对，统一协议）**：

| 运行 | R² pooled | MAE | 作用 |
|---|---|---|---|
| id L2 seed42（本机） | −0.009 | 0.198 | 主力 |
| id L2 seed43（远端） | −0.040 | 0.203 | 种子对照 ✓ 两种子一致 |
| **id_shuf**（seed42，within-task 错位） | −0.118 | 0.217 | **H3-id 消融** |
| id_l1（seed42） | −0.180 | 0.200 | 损失形式对照 |

**H3-id 判读（如实口径）**：方向一致（错位训练使 R² 下降 0.109、MAE 升 0.019），但**主效应本身 ≈0（−0.009），没有正值可供摧毁**——42 号论文"正值 → 错位归零"的对照模式在本设定下无从观测；且 Δ≈0.11 在探针跨运行的散布（σ≈0.03–0.09）内仅 ~1.3σ，**判为"无法判定"而非"成立"**。这是 H1-id 未复现（正值主张缺席）的直接下游：仪器无量程，消融无对比度。id_l1 较 L2 低 0.17 ⇒ L2 默认形式确认。

**IDvsCAPE 项目收口状态**：四臂主表（十四终）+ 三对照（本节）齐备；encoder/manifest 跨机一致性已双向核验（RUN_LEDGER）。唯一未做的预登记消融为 **cape-noswap（H3-cape 半边）**——鉴于 H3-id 已因量程问题无法判定，cape-noswap 的预期对比度需先评估：若 noswap 检索降幅 ≥5pt 则值得跑（135min），否则该半边同样记"无法判定"。待议。

## 十六、2026-11-26 cape-noswap 消融启动（H3-cape 半边）

预登记见设计文档 §14 补充。实现：`--no-swap` 移除换动作负例（官方 use_action_swap=False 口径），swapwins 降级为 no_grad 监控。冒烟 30 步 0.2min 通过。全量 20k 步 seed 42 本机后台运行（~135min）。

## 十七、2026-11-26 检索分层分析（探索性，noswap 训练期间；脚本 analyze_retrieval.py）

按查询视野 k 与任务分层（协议同 eval_retrieval v2.3，seed 1234 同查询）：

| 视野 | cape Hit@1 | frozen Hit@1 |
|---|---|---|
| 0.2s（k=1） | 0.238 | **0.386** |
| 0.4s | 0.225 | 0.099 |
| 0.6s | 0.159 | 0.067 |
| 0.8s | 0.155 | 0.036 |
| 1.0s（k=5） | **0.182** | 0.023 |

**机制判读**：frozen 随视野单调崩塌（8 倍跌幅、75.9% 错误落在 1s 内近邻键）= 外观匹配的天花板；cape 全视野平坦——**增益精确集中在 frozen 死掉的长视野区**（k=5 处 8 倍优势），动作条件预测桥接了外观无法跨越的时间缺口，CAPE 论文主张的直接演示。任务维度互补（task 38：frozen 0.042 vs cape 0.442；task 2 反向 0.211 vs 0.059）。cape 残余错误更"远"（错键中位 0.95s vs 0.55s）——剩余混淆非近邻帧混淆。注：探索性分析（非预登记对比），用于机制解释与写作素材。

**十六（终）、noswap 消融结果（2026-11-26，88.9min，TRAIN_EXIT=0，r2 重跑）**：

| 模型 | Hit@1 | Hit@5 | MRR | macro Hit@1 |
|---|---|---|---|---|
| cape v2.3 主力 | 0.1920 | 0.4492 | 0.3192 | 0.2101 |
| **cape noswap** | **0.2533** | **0.5407** | **0.3896** | **0.2880** |

**H3-cape 判读：不成立（反向显著）**。预登记阈值预期"降幅≥5pt=swap 关键成分"；实测 **+6.1pt**（成对同查询协议，≈3σ），方向相反且超阈值。预登记未规定反向情形的处理，如实记录：swap 换动作负例在本设定**不是检索增益来源，反而是负贡献**。

**机制假说（有诊断佐证）**：swap 负例要求把 (ctx, foreign_acts) 的预测推离 anchor 键，但 LIBERO 动作多模态——相似上下文可由不同动作序列达成等价结局，该目标惩罚实际等价的结局对 ⇒ 几何被扭曲。佐证：noswap 的 val 对齐更干净（cos(u,enc(o_T)) **0.547** vs 主力 0.406），own−foreign/random 边距相当（0.19/0.29 vs 0.19/0.28），且 swapwins 监控（无监督）训练中仍达 0.000–0.067——动作可分性由正对任务自发涌现，无需 swap 监督。分层：noswap 增益同样全视野平坦（k=5 0.220 vs 0.182 vs frozen 0.023），任务面普遍提升。与官方 DROID 消融（swap 有益）的对照：batch 1024 vs 32、真机 vs LIBERO、动作多模态性差异——**"换动作负例"的收益是设定依赖的**。

**Caveats**：单种子成对比较（seed 42×2）；noswap 训练更快（88.9 vs 134.9min，swap 前向不进图）但步数/曝光相同，不构成混淆；双种子复验可选。

**v2.3 检索能力的归因修订**：分支-anchor 对齐（50% 零视野配对）+ 整批负池 = 增益来源；swap 负例 = 本设定负贡献。主表 cape 行（0.192）仍为预登记主口径，noswap（0.253）以消融臂并列报告。
