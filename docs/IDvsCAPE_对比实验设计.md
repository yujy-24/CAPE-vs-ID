# ID vs CAPE：动作相关表征的两条监督路线——控制变量对比实验设计（预登记）

> 2026-11-24 立项。执行环境：本机 WSL2 + RTX 4070 Laptop（**8GB 显存**）+ lawam 环境（torch 2.6.0+cu124 / transformers 5.2.0 / timm 1.0.22）。
> 代码：`E:\world_model\id_vs_cape\`（WSL 内路径 `/mnt/e/world_model/id_vs_cape`）。过程与结果：`IDvsCAPE_实验记录.md`。

---

## 1. 研究问题与动机

**RQ：同一冻结骨干、同一数据、匹配算力下，"逆向回归目标"（ID，42 号 arXiv:2606.07687）与"正向判别式目标"（CAPE，arXiv:2606.07304）谁更能把动作信息注入表征？各自赢在哪一层？**

空白依据（2026-11-24 已查证，见 WAM驾驶调研 §5 补记）：
- 两篇同期（2606.07304v1=06-05，2606.07687 同周更晚），**双向互不引用**；
- 42 号 Table 6 处决的是**无动作条件**的 temporal contrastive（R² 0.073 塌），不是 CAPE 式 goal-convergent + swapped-action 目标；
- CAPE Table 1 判弱的是 Premier-TACO（固定 horizon、策略向设计），不是 ID；
- **干净正面比较（同 backbone/数据/探针）不存在**——本实验填这个空白。

## 2. 预登记假设

- **H1（复现层）**：ID 与 CAPE 两臂都显著优于 frozen 与 forward 回归两基线（在同一设定里同时复现两篇各自的主张）。
- **H2（分层层，最有意思）**：ID 赢动作可恢复性 R²（其主场），CAPE 赢未来检索 Hit@K（其主场）——即两种目标塑造的几何各有侧重。
- **H3（机制层）**：关键成分挖除后各自退化——CAPE 去掉 swapped-action 负例显著掉点；ID 在动作-帧错位（shuffle 标签）时归零（复现 42 的 action-shuffled 对照）。

**判读规则（预登记）**：
- H1 不成立（某臂不显著优于 forward 回归）⇒ 该目标在此规模/数据上无效，结论仍是可报告的负结果；
- H2 成立 ⇒ "按下游用途选监督路线"的配方边界；H2 不成立（一方全面占优）⇒ 更强的单边结论；
- 探针/检索结论与决策层指标冲突时，以决策层为准（42 自己承认 probe≠policy）。

## 3. 总体设计：四臂

| 臂 | 训练目标 | 可训模块 | 角色 |
|---|---|---|---|
| ① frozen | 不训练 | 无 | 地板 |
| ② fwd | (ctx, a) → 预测 ẑ_T，与冻结 z_T 做余弦对齐 | 同 CAPE 壳 | 稻草人（两篇都打败的重建/回归式正向） |
| ③ id | (z_k, z_{k+1}) → 回归 a_k（L1，42 结论 L1 略优） | IDNet（层数配平） | 逆动作臂 |
| ④ cape | goal-convergent InfoNCE + 三种负例 | 同 ② 壳 | 正向判别臂 |

**共享底座（全部锁死）**：冻结 DINOv3 ViT-B/16（`~/LaWAM/weights/dinov3-vitb16-pretrain-lvd1689m`，CLS 768 + 196 patch tokens @224）；同一 manifest（同一批窗口、同一 shuffle 种子、同一步步数与 batch）；LIBERO 演示数据。

**实现不对称（诚实声明，写进论文局限）**：CAPE 臂上下文用 patch tokens（其原文设计），ID 臂输入用 CLS 池化（42 原文设计）；各臂按其作者的原生形态实现，可训参数量配平到 ±20% 以内并如实报告。

## 4. 变量控制清单

1. 数据 exposure：`manifest.parquet` 一次生成（seed 42），三臂读同一文件、同 shuffle 种子；
2. 算力：同步数（默认 20k）、同 batch（默认 8，8GB 显存约束）、同优化器（AdamW 3e-4, wd 0.05, cosine, warmup 1k）；
3. 参数量：训练前打印各臂可训参数表，超出 ±20% 调整层数配平；
4. 评测：三层指标全部在 val 任务（8 个 held-out 任务，见 §5）上算。

## 5. 数据与窗口构造

- 数据：`~/LaWAM/dataset/libero_merged_no_noops_20hz`（LeRobot v3.0，1693 集 / 273,465 帧 / 40 任务 / 20fps / 动作 7 维）。
- **任务级划分**（防 42 的"小固定子集探针值会骗人"）：40 任务 seed 划 32 train / 8 val；窗口只在本 split 的集内采样。
- 窗口：每样本 = (episode, 终点 T, 两个起点 t1<t2∈[T-H, T-1])，H=10（0.5s）；要求 T≥H 且 T+3<len。
- 缓存内容（一次抽取，三臂共用）：o_{t1}, o_{t2} 的 patch tokens [2,196,768] fp16；[T-H..T]∪[T+1..T+3]∪{rand 帧} 的 CLS [15,768] fp16；窗口动作 [H,7]。
- 规模：train 8000 / val 1000 窗口，缓存约 6GB（E 盘项目目录 cache/）。

## 6. 三层指标与各臂打分规则

| 层 | 指标 | frozen | fwd | cape | id |
|---|---|---|---|---|---|
| 表征 | 动作可恢复性 R²（探针: concat(x_k,x_{k+1})→a_k） | 原始 CLS 探针 | gφ(bridge(·)) 空间探针 | 同左 | **IDNet 自身 held-out R²**（ID 即探针，42 协议） |
| 检索 | Hit@1/5/10（池=真未来+2 时间近邻+6 随机+1 换动作，共 10） | 查询=当前帧 CLS（零动作基线） | 预测 u 与候选态余弦 | 同左 | **ID-一致性**：score=−‖ID(z_{T-1}, z_候选)−a_{T-1}‖₁（修正案 11-24：z_ctx 取 o_{T-1}，使真候选为 1 步转移、分布内；时间近邻为 2~3 步，符合 IDNet 训练协议） |
| 决策 | 离线动作匹配（二期）：给定 (o_t, o_goal) 排序 K 候选动作序列 | – | 预测终点 vs goal 余弦 | 同左 | ID 反演一致性 |

附带分析：错误命中的负例类型分布（错的时候把哪类负例排第一）；诊断脚本复刻 L2 两两 cos 方法（同任务/跨任务分布、同 ctx 不同动作的预测多样性——对照驾驶线 cos 0.9955 的病）。

## 7. 超参与扫描

- InfoNCE τ ∈ {0.05, 0.1, 0.2}；ID form ∈ {L1, L2}；各扫 3 点取 val 最优再进主表（全部报告）。
- 不扫的：lr/steps（三臂同值即公平）；H=10 固定（敏感性留二期 {5,20}）。

## 8. 时间表（第一周）

| 日 | 内容 |
|---|---|
| D1 | 00_check_env 全绿 → build_manifest |
| D2 | build_cache（约 1-2h GPU）→ 抽查可视化对齐（帧↔动作↔任务文案） |
| D3 | ①基线数字：frozen 探针 R² + 零动作 Hit@K；diag_pairwise_cos 原始空间 X 光片 |
| D4-5 | ③id、④cape 训练（各数小时）；②fwd 顺带 |
| D6-7 | 三层指标主表 + H3 消融（cape-noswap、id-shuffle）|

## 9. 风险与坑（直接继承 42 的血泪）

1. **动作-帧对齐**：抽取后必须人工抽查（随机 5 窗口：帧序列动作语义一致，如"夹爪闭合帧对应 gripper 动作分量"）；shuffle 对照（H3）本身也是对齐的自动检验。
2. 视频解码帧数与 `length` 可能差 ±1：钳位并记录差异率。
3. 8GB 显存：backbone 只在 cache 阶段上 GPU（fp16），训练阶段全用缓存特征，显存只剩可训模块（<1GB）。
4. WSL 的 /mnt/e I/O 慢：cache 读写走 batch 大块 npz；如慢，把 cache 挪 ~/ 下（脚本参数支持）。
5. 探针 R² 用闭式 ridge + MLP 双口径（继承双口径文化）。

## 10. 交付物

1. 主表：4 臂 × 3 层指标（+置信区间，val 窗口 bootstrap）；
2. 消融表：H3 两组；
3. 几何分析：两空间 cos 分布/CKA、负例类型混淆矩阵；
4. 结论段落：无论方向如何均可写（见 §2 判读规则）；workshop 论文体量；
5. 意外收获管线：ID-一致性打分器与 CAPE 距离打分器直接移植 A 线（LIBERO best-of-K）。

---

## 11. v2 修正案（2026-11-24 深夜预登记，于任何 v2 数字产生前；动因=首跑 cape 塌缩 + 用户指令"按论文策略来，包括防坍缩"）

> 依据：CAPE 原文 §3.2/§4.1/附录 A·C·D（解密副本 p1-19 全读）与 42 号原文 §3/§4.3/附录 G·M。
> v1 代码/缓存快照：`E:\world_model\id_vs_cape_v1_snapshot\` + `cache\`（v1 结果可复现）。

**v1 偏离审计（塌缩归因）**：

| # | 论文做法 | v1 实现 | 后果 |
|---|---|---|---|
| C1 | 状态编码 = **同一解码器 + 可学习空动作查询**（预测/状态同空间；附录 A "Empty-action state encoding"） | 独立 StateBridge 线性通路 | 负例与预测不同空间，可自由分离 → 塌缩温床 |
| C2 | 批 = **16 样本/组 × 同 episode 组**（硬负例同背景、只差动作轨迹）| 随机跨任务批 bs=8 | 上下文捷径足以区分正负例 → **塌缩直接成因** |
| C3 | 支路 = clip（20-50 步 @0.4s/步）内两个中间上下文，剩余动作长度不同、收敛同 o_{t+H} | t1/t2 均隔 3 帧 @20Hz=0.15s | 支路上下文近同 → 杀动作通路即达标 |
| C4 | τ=0.07 固定不调 | τ=0.1 | 次要 |
| C5 | 检索池 = 同任务 10 集 **~420 候选**（zero-action Hit@1 仅 6%）| 9-10 混合候选（zero-action 51%）| 检索任务不要求动作信息 |
| C6 | lr 1e-4 / wd 1e-4 / batch 2048 / 80 epoch | lr 3e-4 / wd 0.05 / bs 8 | 算力差距，登记不改 |
| C7 | DINOv2 ViT-S/14 (384d) | DINOv3 ViT-B/16 (768d) | **保留**：四臂共享底座是本实验前提，脚注声明 |
| I1 | 探针 = 新 MLP [D→256→128→7]，3k 步 / bs 512 / lr 3e-4 / 3 seeds / **patch-mean 特征** / 全 held-out 任务 | ridge + CLS 对（§9.5 预登记双口径只做了一半）| 口径偏离 |
| I2 | shuffle 对照 = **同任务内置换**动作（保边缘分布）| 批内 roll（跨任务）| 对照不纯 |
| I3 | ID 损失 L2 默认（L1 差 <0.05）| L1 默认 | 微小（sweep 双口径保留）|
| I4 | warmup 2k / eff batch 16 / 2 训练 seeds | warmup 1k / bs 8 / 1 seed | 统计力 |

旁证（42 号附录 M）：冻结探针全任务覆盖下 0.449→**−0.006**——v1 frozen 的 −0.29 属同类现象，绝对值只作地板，主张靠 matched-control Δ。

**v2 协议（改 C1-C5、I1-I2-I4；保留 C7 与任务级划分、matched 步数）**：
1. **窗口/支路**：t2~U[T-12, T-5]，t1~U[t2-12, t2-5]（支路间隔 0.25-0.6s、剩余动作 5-24 步）；双支路预测同一 o_T；缓存 pooled CLS+patch-mean [T-24..T+3]+rand（29 槽）、t1/t2 patch tokens。
2. **空动作状态编码**：解码器加可学习 empty-action query；候选/目标/时间近邻负例全部经 `g(dec(ctx_patches, empty_query))` 编码——与预测同空间；**删除 StateBridge 于 cape 通路**。
3. **同 episode 组批**：每批 G=8 集 × S=4 窗（bs=32）；负例 = 组内他窗 t1'/t2' 编码（同场景异时刻硬负例）+ 组内换动作结果（同集 acts 交换）+ 跨集随机 2 条；τ=0.07、lr 1e-4、wd 1e-4、warmup 2k cosine、20k 步。
4. **检索 v2**：查询 = (ctx_t1, acts_{t1:t2}) 预测 o_{t2}（0.25-0.6s）；真候选 = 查询窗 t2 帧空动作编码；池 = **同任务全部 val 窗的 t1/t2 帧编码（~250/任务）**；zero-action 基线 = CLS 余弦。
5. **探针 v2**：MLP [2D→256→128→7]（patch-mean 帧对拼接）为主口径、ridge 为副口径、3 seeds、逐维+汇总、全 8 val 任务。
6. **id v2**：L2 默认、within-task shuffle 对照、warmup 2k、eff batch 16（梯度累积）、2 seeds。
7. **防塌缩监测**：训练中记录反事实敏感度（真动作 vs 同集他窗动作 vs 随机动作下的 u 两两距离）+ swap-wins；检索报告不含 swap 候选（消除 v1 的恒第一伪影）。

判读规则不变（H1/H2/H3 原文有效）；v1 首跑结果降级为"协议偏离下的预实验"，主表以 v2 为准（两表都报告）。

## 12. v2.1 修正案：目标状态进正对集（2026-11-25，于 v2.1 任何数字之前）

**触发**：v2.0（seed 42）三臂完成后，cape 检索 Hit@1=0.000（≈随机，MRR 0.021≈随机排序期望值 0.022），而反事实监控显示 u 动作敏感（cos_random≈0.02）、u1≈u2=0.86 未塌缩。几何诊断：**cos(u, enc(o_T)) = −0.05（正交）**——预测空间与状态编码空间无对齐，检索随机是其必然结果。

**根因（对照原文）**：CAPE 附录 A 明示空动作编码三用途——candidate-key in retrieval、goal-state in planning、**"zero-horizon training samples where the action sequence length is zero"（进训练）**。即目标状态 o_T 以零动作支路身份参与正对（"汇于同一结果"的最简支路）。v2.0 只用 u1↔u2 作正对、全部状态键放负例侧 ⇒ u 被推离状态流形 ⇒ 检索失效。检索协议本身与附录 C.2 一致（query=动作条件、键=空动作、余弦），无需改。

**修正（仅 cape 臂）**：
1. cache 增存全窗口 o_T patch（+2.7GB；id/fwd/探针不受影响，无需重训）。
2. cape_loss 改多正对 InfoNCE：anchor u1 的正对集 = {u2, enc(o_T)}（logsumexp 合并分子），u2 对称；负例不变（组内他窗状态键 + 组内换动作结果 + 跨批随机状态）。

**如实登记的残留偏离**：论文负例含"目标邻近时刻状态"（o_{T±1..3}，细粒度判别主驱动），本实现以组内他窗状态键近似（同轨迹异时刻），T±1..3 patch 未缓存（+8GB RAM 超 WSL 上限）。此偏离在局限节报告。

**处置**：v2.0 结果如实入表（标注"正对缺目标状态"），v2.1 重训 cape 为检索层主口径；H1/H2/H3 判读规则不变。

## 13. v2.2 修正案：零视野样本为独立分支（2026-11-25，于任何 v2.2 数字之前预登记）

**动因**：v2.1 实测无效（检索 Hit@1 0.008 / MRR 0.0284 仍≈随机；探针 R² −0.335 与 v2.0 分毫不动；goal_cos 全程 −0.12→−0.06 不爬）。回读原文定位为**实现读法错误**而非超参问题：附录 A 的 "zero-horizon training samples where the action sequence length is zero" 是批次中 action 长度为 0 的**独立训练样本**；正文式 (3) 为**成对**形式（每正对独立分子+对称项）。v2.1 把 s_T 塞进 u1 的多正对 logsumexp 分子——u1↔u2 项（cos 0.87 @ τ=0.07 → e^12.4）对 s_T 项（e^−0.6）softmax 权重比≈10⁻⁶，目标正对梯度≈0，数学性死锁。

**修正（仅 cape 臂，cache 与 id/fwd 不动）**：
1. 三分支 {u1, u2, s_T=enc(o_T)} 汇于同一 o_T，逐对入式 (3)：L = ⅓[L(u1,u2) + L(u1,s_T) + L(u2,s_T)]，每对独立分子、对称项，负集同式 (3)。
2. 负键池扩容：组内他窗状态键从 {k1,k2} 扩为 {k1, k2, s_T}（批次训练样本全体互为负例）；换动作结果与跨批随机键不变。残留偏离（T±1..3 邻近负例）不变，仍入局限节。
3. 监控增读数：nce_goal（两目标对损失均值，应显著下降）、goal_cos（应爬升过 0）。

**预登记终止判据**：20k 步后若 goal_cos<0.2 或检索仍随机 ⇒ 判"本数据规模/负池结构下 CAPE 检索优势不可复现"，停止 Cape 实现细节迭代，主线按 42 号 ID 探针对比收口（id/控制组/种子补齐照旧）。

**重训前原文复核（2026-11-25）**：式 (3) 成对+对称逐字确认（全文无多正对 logsumexp，v2.1 共享分子系误读）；附录 A 零视野样本=独立训练样本（空动作查询、同通路）逐字确认；Table 5 无零视野比率、偏移无范围 ⇒ 配对比率为原文未定细节，本实现取确定性三分支（两读法公共核心=独立样本·成对·满梯度，登记为残留歧义）；负集三分结构与 C.1 检索协议逐字吻合；消融表佐证 (u,enc(o_T)) 对齐通道=检索性能来源（纯对齐即达全模型 84~91%）。机制无误，可执行。

## 14. v2.3 修正案：对齐官方开源代码（2026-11-25，于任何 v2.3 数字之前预登记）

**动因**：用户指令"查开源代码，直接套用或参考复刻，尽可能完全复刻论文"。CAPE 官方仓库已放出（`github.com/chen2151643/CAPE`，论文正文无链接、附录仅言 "released upon publication"，arXiv 页与 HF papers 页均无索引）。逐行核对官方 `src/utils/loss.py`、`src/dataset/dataset.py`（`__getitem__` 配对段 384–422 行）、`src/models/cape_model.py`、`train.py`、`conf/`，**官方代码裁决了 §13 登记的全部残留歧义，且与 v2.2 的取法不同**：

1. **零视野配对比率（§13 歧义，官方裁决）**：数据侧随机——每训练样本 = 恰好一对支路汇于同一 anchor；第二支路 **50% 概率取 anchor 自身**（零动作、noop token、同解码通路），否则取异于第一支路的非 anchor 偏移。即 (branch, s_T) 对≈50%、(branch, branch) 对≈50%，**每样本只训一对**（逐 epoch 重采样）。v2.2 的确定性三分支全对（state 对占 2/3、每窗 3 对）废止。
2. **分支长度**：官方 branch 起点 = anchor − i·stride，i∈{1..H}，H=5、stride=4@DROID 15fps（每步 0.267s，剩余动作 1..5 步）。v2.2 为 5..12 帧连续。v2.3 取 stride=4@20fps（每步 0.2s）× i∈{1..5} ⇒ 剩余 4..20 帧连续动作，均在 PAST=24 窗口内 ⇒ **manifest 不变、id/fwd 无需重训**。
3. **负例池**：官方主配置（`conf/experiment/dinov2-S.yaml`，use_group_local_state_nce=false）= **整批 keys 互为负例**（批 1024 ⇒ 每锚 ~1023 负键，含同集组 15 硬负例）+ **全批换动作 query**（derangement，仅 query 支路换动作、作为两个方向的附加负列）。v2.2 为组内 3 键+跨批 2 键（小约两个量级）。v2.3 负池 = 整批 keys + 全批换动作（批 32 ⇒ 每锚 31+31 负键；规模差=8GB 显存约束，如实登记）。
4. **last-token 取位**：官方在 `valid_len−1`（最后**有效**动作位）取终态，padding 位替换为 noop 且被 attention mask；v2.2 取 padding 后序列末位（分布外位置）。v2.3 改为按长度 gather；k=0（零视野）走 encode_state 单空查询通路（与官方零 padding 分支逐位等价：位置 0=noop、因果掩码只见自身、无位置编码混入）。
5. **优化器**：官方 AdamW 1e-4/1e-4、warmup=5% 总步数、cosine→1e-6、无梯度裁剪；v2.3 warmup 2000→1000，梯度裁剪 1.0 保留（本机稳定件）。80 epochs 对齐成立：20k 步 × 32 对 ÷ 7993 窗 ≈ 80 epochs。

**v2.3 实现**（新建 cache_v3，manifest 与 pooled/pmean/act_win/probe 分片内容不变，id/fwd/探针零改动）：
- `build_cache.py`：patch 槽位增存 branch_patch = 帧 [T−20, T−16, T−12, T−8, T−4, T]（6 槽，slot 5=anchor，兼作零视野上下文与检索键上下文）；ctx_patch/goal_patch 原样保留（评测脚本过渡期兼容）。
- `train.py` cape：每窗随机 i∈{1..5} 为 query 支路（首支路必非 anchor，同官方）；key 支路 50% anchor/50% 异支路；损失 = 官方 `compute_contrastive_loss` 单卡全局版逐行移植（单正对 + 对称 + swap 负列）。
- `eval_retrieval.py` v2.3 协议：query k~U{1..5}（seed 1234，四臂同 query 集）；池 = 每窗双键 {分支帧 T−4k, anchor T}（≈250/任务，与 v2 池规模持平）；排除自身分支键、真值 = 自身 anchor 键；frozen/fwd/id 不重训、同协议重评（主表 v2.3 行 = 四臂全新口径，v2.0 表保留并报）。`diag_pairwise_cos.py` B 部分支路同步为 k~U{1..5}、o_T 直接取 branch_patch 锚槽。
- 监控保留 goal_cos/swapwins/反事实敏感度/val InfoNCE（与 v2.0–v2.2 历史可比），增记 state 对占比。

**论文-代码不一致点（如实登记，取论文口径）**：官方 dinov2-S 实验配置 num_visual_layers=0（视觉上下文编码器=Identity，实验名 vislayer0）与论文正文"3 层自注意力"矛盾，且与论文消融"去掉它掉分最多"矛盾；data 配置 droid_12000 vs 论文正文 8000 条。本实现 ctx/dec 层数与 FF 宽度沿用 v2 容量对齐设计（ctx2+dec2@768≈25.2M vs 官方 ViT-S 21.3M；骨干 DINOv3-B vs 官方 DINOv2-S），非机制偏离。

**预登记判据（继承 §13，不重置）**：20k 步后 goal_cos<0.2 或检索仍随机 ⇒ 判"本数据规模/负池结构下 CAPE 检索优势不可复现"，停止 Cape 实现细节迭代，主线按 42 号 ID 探针对比收口。v2.2 代码留档作对照（不再重训）。

**残留偏离清单（v2.3 后局限节报告）**：① 负池 31+31 vs 官方 ~1023+1024（批上限）；② stride 0.2s vs 0.267s（帧率折算取整，保 H=5 且不越 PAST=24）；③ 动作=逐帧 7 维 delta（与 id 臂/探针同语义；官方为 stride 级相对位姿，DROID 有绝对状态可构建，改动作语义会破坏四臂对齐）；④ T±1..3 邻近负例仍以同集他窗键近似（继承 §12）；⑤ 单相机（官方训练随机取 DROID 双外参之一，我们天然单视角等价）。

**§14 补充（2026-11-26，noswap 消融实现口径，于任何 noswap 数字之前登记）**：H3-cape 消融 = `train.py --no-swap`：从损失中移除换动作负例列（等价官方 `use_action_swap=False`），其余全部不变（配对/负池/优化器/20k 步/seed 42）；swapwins 监控保留（no_grad 前向，不进梯度）以便观察"无 swap 监督下分离是否自发涌现"。判读基线：v2.3 主力检索 Hit@1 0.192；noswap 若降幅 ≥5pt 判"swap 负例为检索增益关键成分"（H3-cape 成立），若 <2σ 噪声量级判"无法判定"。
