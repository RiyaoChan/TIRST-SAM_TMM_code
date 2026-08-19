# TIRST-SAM 候选研究 Idea

这些是经 P01–P50 冲突核查后的**待验证假设**，不是论文贡献声明，也不预设结果。排序优先考虑：能否形成清楚的新问题、是否可用现有代码低成本证伪、是否避开“更强 caption / 拟合文本向量”的拥挤叙事。

## 推荐顺序

| 优先级 | Idea | 研究范式 | 首轮筛选成本 | 主要风险 |
|---|---|---|---|---|
| 1 | 可靠性校准的多视图 Prompt 与拒绝机制 | prompt verification / abstention | 中 | gate 可能随 saliency 一起漏小目标 |
| 2 | 文本反事实增量的 Prompt→Mask 行为蒸馏 | functional distillation | 中 | teacher 本身增益不足；已有 prompt-in-loop KD 近邻 |
| 3 | 高分辨率残差语义定位 | dense localization | 中 | 与高分辨率视觉模块增益难分离 |
| 4 | 语义只做候选验证器 | role reassignment | 低 | 文本可能没有独立信息 |
| 5 | 反事实文本不变性与正确文本 margin | robustness learning | 低 | 可能训练成完全忽略文本 |
| 6 | 角色化语义 slot-set 学生 | structured representation | 中 | 与 FIANet/DGSP 近邻 |
| 7 | 自适应 prompt 预算与停止策略 | sequential decision | 中高 | 复杂度大于收益 |
| 8 | 多视图涌现式纯图像 prompt | self-prompt / consistency | 低中 | 与 SAM-SPL/TEP-SAM 邻近 |
| 9 | Tiny-target-aware CLIPSelf | local dense distillation | 中 | CLIP patch teacher看不见极小目标 |
| 10 | 存在性—定位解耦的双头模型 | calibrated detection | 低中 | presence head 可能只是常规分类辅助头 |

## 指南要求的范式覆盖

| 必须覆盖的研究范式 | 对应候选 | 在方案中的角色 |
|---|---|---|
| 更可靠的文本/语义 prompt | Idea 1、5 | 文本作为候选验证与可靠性监督，并显式测试错误文本，而非默认 caption 正确 |
| image-only student / prompt distillation | Idea 2、6 | 蒸馏文本反事实增量或角色化 slot，不把全局 embedding 回归当最终目标 |
| 图像到结构化语义 prompt | Idea 6、10 | 预测 presence、location、uncertainty 等有角色约束的语义变量 |
| 相似图到 point / dense mask | Idea 3、4 | 高分辨率 targetness map 负责空间定位，语义只校准或验证候选 |
| patch-level dense visual-language distillation | Idea 9 | 用 tiny-aware crop 与 hard negative 改造局部蒸馏 |
| 多视图 / 多尺度提示生成 | Idea 1、8 | 跨可逆视图的一致性用于 prompt 可靠性或纯图像 self-prompt |
| prompt 质量控制 / 过滤 / 自适应预算 | Idea 1、7、10 | 允许拒绝、零 prompt 与动态停止，优化 Pd–Fa–成本联合风险 |
| early-fusion / internal semantic conditioning | Idea 3、4 | 语义进入内部 residual/verification 路径，但不直接承担像素坐标生成 |

---

## Idea 1：可靠性校准的多视图 Prompt 与拒绝机制（首选）

### 研究假设

若一个自动 prompt 在尺度、轻微几何增强、局部对比/频域视图下不能稳定指向同一候选，则拒绝或降权该 prompt 比强制送入 SAM 更能降低 Fa，且不会显著损害极小目标 Pd。

### 文献来源

- SPD：saliency prior 校验 noisy prompt，并由上下文形成 consensus set。
- SeViL：文本更适合做 pseudo-label 质量控制，而非直接定位。
- SegEarth-OV3：presence score 过滤场景不存在类别。
- One-shot IRSTS / TEP-SAM：不同视图/时间证据可“涌现”空间提示。

### 与现有工作的实质差异

- 不使用 SPD 的相邻医学切片和外部 noisy point，改为单帧 IR 的多尺度、增强、频率/局部对比视图。
- 不像 SAIST 假定逐图文本正确；输出包含 `accept / abstain`，低置信时允许 empty sparse prompt。
- 不像 SAM-SPL 只生成 self-prompt；额外显式估计 prompt reliability，并以空目标/错误文本监督校准。
- 与当前代码的差异不是新增 CBGA，而是把自动 prompt 从确定张量改成带可信度和拒绝决策的分布。

### 为什么可能适合红外小目标

低信噪比背景会产生孤立高响应，而真实极小目标在合理尺度/轻扰动下应保持位置一致；用 risk–coverage 而非固定 top-k 可直接约束 Pd–Fa 权衡。多视图只改变观测，不要求目标具有可描述纹理。

### 最小实现

1. 从 3–4 个可逆视图得到高分辨率 targetness maps，逆变换到原坐标。
2. 计算 component center 一致性、rank consistency 与 augmentation variance。
3. `ReliabilityHead` 输出 presence、prompt confidence 和每个候选权重。
4. 只对通过阈值的 component 产生 dense prompt/正点；高响应但与其他视图不一致的区域产生负点或被拒绝。
5. 文本 teacher 只参与训练期 candidate verification；纯图像 student 预测相同可靠性。

### 推理依赖

- 训练：mask GT 监督 targetness/presence；teacher text 对候选排序提供软目标；加入 empty-image 或合成 hard-negative crop。
- 推理：仅原图及其确定性视图，不调用 GPT/CLIP，不使用 GT point/box/mask。

不需要文本、外部 VLM、人工 prompt、参考图像或 GT 派生信息；只需原图的确定性尺度/对比/频域视图。训练期可选用离线文本 teacher。

### 可证伪实验

- 在 tiny-area bin 上，prompt component recall 不高于单视图 ASSP，或 Fa 相对升高超过 5%。
- reliability 的 ECE/AUPRC 不优于简单 max-score；abstention-risk 曲线无收益。
- 只在一个数据集/一个 seed 有效。

### 预期主要改善指标

首要是 Fa、False Prompts/MP 和 risk–coverage；约束 tiny-area Component Recall/Pd 不下降。mIoU 只作次级结果。

### 最大风险

真实 1–9 像素目标在增强后本身不稳定，gate 可能把唯一正确 prompt 删除；多视图还会增加推理开销。

### 新颖性风险

中高。除 SPD/SeViL 外，AutoPrompt-SAM3D 已做多层自提示与 confidence filtering，RL-CP 已用 activation stability 纯化正负点。可检验空间收窄到：单帧 IR 的多视图风险校准、tiny-target prompt recall、文本反事实安全性和纯图像部署；必须避免只把“consensus/filter”改名。

### 实施成本

Medium：rule gate 可低成本验证，learned gate 和 student 只在 sanity check 通过后训练。

---

## Idea 2：文本反事实增量的 Prompt→Mask 行为蒸馏（首选）

### 研究假设

纯图像学生只蒸馏正确文本相对 no-text 的**反事实条件增量**，并学习 wrong/shuffle text 应退回 no-text 安全边界；这会比回归 CLIP embedding 或无条件模仿 teacher mask 更好地保留有益 Fa/Pd 改善。

### 文献来源

- SimIR：学生蒸馏多粒度输出与 task queries，而非单一 feature。
- PromptKD：prompt/domain knowledge 可作为蒸馏载体。
- CLIPSelf/AlignEarth：local/region feature 比 global cosine 更适合 dense prediction。
- JinSight：语言可只在训练期塑造视觉表征。
- EdgeSAM：prompt-in-the-loop distillation 已覆盖 prompt encoder/mask decoder 动态。
- SAM-COD 与 MS-SAM-LESS：已分别覆盖 prompt-adaptive KD、dense prompt learner→轻量 mask aggregator。

### 与现有工作的实质差异

- 不像 DGSPNet/JinSight 生成或预训练 image-to-text representation；目标是文本条件前后对 SAM **决策的增量**。
- 不像 AlignEarth 只对 global/CLS/local feature；同时蒸馏 sparse/dense prompt distribution、mask logits、component ranking，以及对 prompt 扰动的响应。
- 不像当前 TASSG 只拟合语义 feature；student 在相同 SAM image embedding 上只复现 `teacher(correct text)-teacher(no text)` 的有益增量。
- 不主张首次 prompt-in-loop / prompt-to-mask 蒸馏；与 EdgeSAM、SAM-COD、MS-SAM-LESS 的差异是文本反事实处理、错误文本安全边界和 IR tiny-component ranking。

### 为什么可能适合红外小目标

文本对当前结果的主要潜在价值是抑制背景/Fa，而极小目标 mask 又高度依赖 prompt 与 decoder 的联合响应。蒸馏正确文本相对无文本的决策增量，比压缩到一个 global embedding 更贴近最终 Pd/Fa。

### 最小实现

- `L_prompt`：teacher/student sparse token 的 set matching + dense prompt focal/KL。
- `L_mask`：temperature KL + Dice，对 teacher 高置信像素加权。
- `L_rank`：候选 component 的 target/background 相对排序。
- `L_delta`：分别给 teacher 正确、shuffle、错误、drop-text 输入，student 学正确文本相对 no-text 的 mask delta，而非绝对 feature。
- `L_jac`（可选）：对小幅 prompt perturbation 的 mask response 做有限差分一致性。

### 推理依赖

- teacher：现有 GPT/CLIP+CBGA/ASSP 冻结；训练数据离线缓存所有 prompt/mask soft targets。
- student：只看 IR image；推理仅保留 student+SAM，无 GPT、CLIP、GT prompt。

teacher 训练/缓存阶段需要离线 GPT/CLIP；student 推理不需要文本、VLM、人工/GT prompt或参考图像，只输入 IR image。

### 可证伪实验

- teacher 相对 no-text baseline 的平均增益小于随机波动，说明无有价值行为可蒸馏。
- behavior student 不优于同参数量的 embedding-regression TASSG，或只复现 teacher 错误。
- 在跨数据集上 Fa/Pd 退化，说明蒸馏过拟合 teacher captions。

### 预期主要改善指标

主要目标是保留 teacher 的 Fa 改善且不损害 Pd；其次是 tiny component ranking/recall 与 mIoU。

### 最大风险

teacher 可能没有超出 no-text baseline 的稳定增量；student 也可能复制 teacher false alarm，或反事实 loss 退化成普通 logit KD。

### 新颖性风险

中高。prompt-in-loop、prompt-adaptive KD 和 dense-prompt→mask aggregator 均已有正式工作；只有“正确文本相对无文本的反事实增量 + 错误文本安全边界 + tiny-target component 排序”仍有待验证空间。

### 实施成本

Medium：复用现有 teacher/student，主要成本是离线缓存 C/N/S/W outputs 和新增对照损失。

---

## Idea 3：高分辨率残差语义定位（首选）

### 研究假设

文本/global semantics 适合抑制背景和判断存在性，但极小目标定位必须来自高分辨率局部残差；将两者以“语义校准局部残差”而非“文本直接产坐标”组合，可同时提升 tiny-bin prompt recall 并控制 Fa。

### 文献来源

- IRSAM：结构保持/边缘与多粒度 decoder。
- SegEarth-OV：SimFeatUp 与 global bias subtraction 恢复局部 dense semantics。
- SOPSeg：1/16 分辨率是小目标 SAM 的直接瓶颈，区域放大与边缘细化有效。
- SAIST：语言贡献主要反映在 Fa，支持语义承担背景抑制角色。

### 与现有工作的实质差异

- 不像 READ/Simple-ViL 直接从低分辨率 VLM similarity/attribution map 取点。
- 不像 DGSP TGSA 用 text dot-product 直接当空间权重；局部 targetness 由 IR shallow residual/frequency contrast 产生，text 只校准通道、阈值和候选类别。
- 不像 SAM-SPL 一般浅层 self-prompt，明确优化“tiny target recall under semantic background suppression”并输出可审核 prompt map。

### 为什么可能适合红外小目标

目标像素占比极低、边界弱且容易在 1/16 tokenization 后消失；浅层局部对比/高频残差保留定位，而全局语义更适合压制云边、建筑灯等杂波。

### 最小实现

1. 从 SAM encoder shallow feature、局部对比和可选高频残差形成 1/4 或 1/8 target evidence。
2. subtract 全局背景原型，得到 local residual map。
3. text/no-text semantic vector 只预测 per-scale gate、presence 和背景原型，不直接预测坐标。
4. 把 residual map 转为 dense prompt；top components 可选地产生正/负点。

### 推理依赖

- teacher 版本可用正确/自动文本；student 版本用图像语义头替代文本。
- 推理主张以 student 为主，完全无 GT point；teacher 文本版本只做上界与分析。

teacher 诊断可使用离线文本；主部署为 image semantic student，不需要外部 VLM、人工提示、参考图像或 GT 派生信息。

### 可证伪实验

- 1/4 residual prompt 对小于 9/16/25 像素三档目标的 center/component recall无显著改善。
- 增益完全可由普通 FPN/上采样复现，则语义残差部分没有必要。
- correct/shuffle text 对结果完全相同但参数更多，则删除文本路径。

### 预期主要改善指标

首要是 `1–9/10–16` 像素桶的 Component Recall@K/Pd；语义校准应降低 Fa，boundary/mIoU 为次级。

### 最大风险

增益可能完全来自普通 FPN/更多分辨率而与文本无关；局部残差也可能把背景纹理放大为 false prompts。

### 新颖性风险

中高。高分辨率小目标模块很多；必须用“语义只校准、不定位”的因果消融，以及 prompt recall/Fa 而非仅 IoU，证明不是普通 FPN。

### 实施成本

Medium：需抽取浅层 feature 与一条轻量 residual path，但不需要新 foundation model。

---

## Idea 4：文本作为候选验证器，而非定位器

### 研究假设

对单帧 IR，图像分支先高召回地产生候选，文本 teacher 只判断每个候选是否符合目标/背景属性，比让全局文本直接生成 prompt 更稳健。

### 文献来源

SeViL 的 TAPF、SAIST 的 Fa 改善、SegEarth-OV3 的 presence、Grounding DINO→SAM 的语义定位分工。

### 与现有工作的实质差异

当前 CBGA/ASSP 在候选产生前就融合文本；本 idea 延后到 component/post-selector，明确分离 recall 与 precision。SAM-SPL 提供 proposal，不提供跨模态 verifier；DGSP 直接调制 feature，不做候选级真假审查。

### 为什么可能适合红外小目标

极小目标外观信息少，先以纯视觉高召回保住 Pd，再让语义判断“该亮点是否像目标”更符合文本的全局优势，也能把复杂背景 false alarms 作为候选级 hard negatives。

### 最小实现

ASSP/高分辨率 head 产生 top-N components；对每个 crop 取视觉 feature，与 GPT structured attributes/CLIP slots 计算兼容度；训练后将 verifier 蒸馏给图像 student。部署无文本、无 GT。

### 推理依赖

最终 student 只需图像，不需 VLM、人工提示、参考图或 GT 派生信息；teacher 训练期需要离线文本。

### 可证伪实验

比较视觉 score、text verifier、shuffle/wrong verifier 的 candidate AUPRC，并测 verifier 前后 Pd–Fa。

### 预期主要改善指标

Fa 与 candidate AUPRC；约束 Pd/Component Recall 不下降。

### 最大风险

GPT 文本来自同图，可能没有超出 candidate crop 的独立信息；teacher verifier 也可能偏好大/高对比目标。

### 新颖性风险

Medium：SeViL TAPF 是直接近邻；关键差异只能是单帧 IR candidate-level 评价、无文本 student 与极小目标 recall 约束。

### 实施成本

Low：可先在现有预测 components 上训练线性/小 MLP probe。

---

## Idea 5：反事实文本不变性与正确文本 Margin

### 研究假设

显式训练 correct / shuffled / wrong / dropped text 的反事实约束，可使模型只在文本与图像一致时使用文本，文本缺失或错误时退化到强 no-text baseline，而非灾难性失效。

### 文献来源

READ 的 false-premise 分析、SEEM 的 prompt modality dropout、SPD 的 noisy prompt robustness；现有 SAIST/DGSP 缺少系统错文本测试。

### 与现有工作的实质差异

不是改 caption encoder，也不是普通 text dropout：要求 `M(no-text)` 是性能下界，wrong/shuffle 不得使 mask 偏离 no-text 超过 margin，而 correct text 只有在 image-text consistency 高时才允许产生正增量。

### 为什么可能适合红外小目标

红外场景中的点状杂波很容易被错误的“有目标/某位置”描述放大；safe fallback 能直接回答审稿人关于无文本和错误文本的部署问题。

### 最小实现

增加 consistency gate；`L_safe = max(0, d(M_wrong,M_no)-m_safe)`，`L_use = max(0, m_gain-d(M_correct,M_gt)+d(M_no,M_gt))`。推理可有文/无文；无文直接走视觉基线。

### 推理依赖

支持可选文本；无文本时不需 VLM/人工/GT prompt。若使用文本，caption 在离线生成；安全主表仍报告纯图像路径。

### 可证伪实验

C/N/S/W 四条件测试 safe degradation、正确文本增益和 gate 开启率；与普通 text dropout/随机模态缺失对照。

### 预期主要改善指标

wrong/shuffle 条件的 Fa、安全退化幅度与 worst-case mIoU；正确文本 Pd 不能下降。

### 最大风险

优化最容易的解是永远忽略文本，correct-text 增益消失。

### 新颖性风险

Medium-Low：false premise/noisy prompt/modal dropout 已有，但系统反事实协议在 IR 中仍可作稳健性贡献；单独作为主方法可能不足。

### 实施成本

Low：主要新增文本配对采样和安全 loss。

---

## Idea 6：角色化语义 Slot-Set 学生

### 研究假设

将 GPT 输出拆成 target-presence、scene/background、appearance/contrast、location-uncertainty 四个可缺失 slots，并让图像学生以 set prediction 拟合其作用，比压成一个 global embedding 更容易诊断和迁移。

### 文献来源

FIANet 的 context/object/position 拆分；DGSP 的 coarse/fine token；JinSight 多任务指令；PixelLM 多 segmentation token。

### 与现有工作的实质差异

FIANet 推理需三角色文本，DGSP 的 fine token 是视觉连续潜变量；本方案最终 image-only，并把 uncertainty/presence 作为显式 slot。与当前单 global/token tensor 的差异是可缺失、可匹配、可独立消融的 set。

### 为什么可能适合红外小目标

低信噪比下“目标是否存在、背景类型、局部对比、位置不确定性”职责不同；将它们分开可避免不可靠位置描述污染存在性/背景抑制。

### 最小实现

GPT 离线输出 schema；CLIP/小文本 encoder 得 teacher slots；Hungarian matching 训练 image slot decoder；presence/background slots调阈值，appearance slot调通道，location slot仅提供弱区域先验。部署只保留 image slot decoder。

### 推理依赖

student 推理纯图像，无 VLM、人工 prompt、参考图或 GT；teacher 训练需要离线 GPT schema。

### 可证伪实验

slot dropout、slot shuffle、单 slot probe、single-vector 等容量对照；测各 slot 对 Pd/Fa 的职责是否可分。

### 预期主要改善指标

presence/background slot 预期改善 Fa；appearance/location slot 只允许以 prompt recall/Pd 证实。

### 最大风险

slots 可能高度冗余，图像 student 仅学习到同一全局向量的重复副本。

### 新颖性风险

High：FIANet、DGSPNet、JinSight、多 token 分割均是直接近邻；只有可缺失职责、uncertainty 和纯图像 set student 可能形成差异。

### 实施成本

Medium：需要重做 schema、slot encoder、matching 和逐 slot 对照。

---

## Idea 7：自适应 Prompt 预算与停止策略

### 研究假设

固定 top-k 点对极小/多目标/空目标都不合适；以 predicted risk 决定 0、1、K 个点以及是否执行第二轮，可在相同平均计算量下降低首轮错误放大。

### 文献来源

Plug-and-Play PPO 的点优化、SAM-SP 的迭代 self-prompt、MedSAM3 的分割—检查—再提示、MaskSAM 的多 prompt 协同。

### 与现有工作的实质差异

不做昂贵 RL 图优化；控制器在现有 ASSP map 上选择预算和 stop，允许空 prompt。文本只训练 quality critic，部署为 image-only critic。

### 为什么可能适合红外小目标

单目标、多个微小目标与空场景需要不同点数；固定 top-k 会在空场景强制 false point，在多目标场景又可能漏检。

### 最小实现

状态包括 component score、mask IoU predictor、跨视图 variance、presence；动作 `{abstain, one-positive, pos+neg, refine-mask}`；训练先用 oracle cost imitation，再用可微 surrogate 微调。

### 推理依赖

纯图像，无文本/VLM、人工 prompt、参考图或 GT；训练期 oracle action 只能由 train GT 构造。

### 可证伪实验

固定 0/1/K 点与 controller 在相同平均 SAM calls 下比较；分析空场景和多目标分桶。

### 预期主要改善指标

Pd–Fa–latency 联合曲线、每图平均点数/SAM calls。

### 最大风险

控制器训练不稳定、复杂度过高，首轮漏检后没有状态可恢复。

### 新颖性风险

Medium：PPO、SAM-SP、MedSAM3 已覆盖点优化/迭代；差异是轻量 risk-budget/stop 与 zero prompt。

### 实施成本

Medium-High：需要多轮推理、成本建模和严格 latency 对照。

---

## Idea 8：多视图涌现式纯图像 Prompt

### 研究假设

同一静态 IR 图像在尺度、局部对比、频域分解前后的稳定差异可模拟 temporal-emerged cue，生成不依赖文本的 task-specific prompt，并作为所有文本方法的强 baseline。

### 文献来源

TEP-SAM 的 temporal emergence、One-shot IRSTS 的相似图、多视图 consistency、SAM-SPL 浅层 self-prompt。

### 与现有工作的实质差异

不声称 language contribution；这是主动构建的无文本基线/可能备选方法。与 SAM-SPL 的差异是 prompt 来源为视图差异与一致性，而非单路浅层 feature。

### 为什么可能适合红外小目标

红外目标常以局部对比/频域残差而非纹理语义出现；增强前后稳定性可在没有文本时提供任务特定定位信号。

### 最小实现

共享 encoder 处理原图、局部对比图和频率残差；跨视图 residual attention 形成 targetness；只保留坐标一致 component。训练/推理均无文本与 GT prompt。

### 推理依赖

仅图像及确定性视图；无 VLM、人工/GT prompt和参考图像。

### 可证伪实验

单路/多路同参数、普通 ensemble、view consistency 三组，比较 prompt recall/Fa/latency。

### 预期主要改善指标

tiny-area Component Recall/Pd；同时约束 False Prompts/MP 与 Fa。

### 最大风险

视图变化可能破坏 1–9 像素目标，且计算开销翻倍。

### 新颖性风险

Medium-High：SAM-SPL、TEP-SAM、RL-CP 都有近邻；更适合作为 Idea 1 的 no-text prompt generator 和强基线。

### 实施成本

Low-Medium：先共享 encoder/离线视图做 probe，通过后才多分支训练。

---

## Idea 9：Tiny-target-aware CLIPSelf

### 研究假设

用 GT-free proposal/crop curriculum 将 crop-level teacher 语义蒸馏到原图高分辨率 region feature，可比 global CLIP feature 更好地区分微小目标与局部杂波。

### 文献来源

CLIPSelf 的 crop teacher→ROI student；AlignEarth local region distill；SegEarth-OV 的 dense upsampling；Curriculum Point Prompting 的难度课程。

### 与现有工作的实质差异

不是逐图文本回归，而是局部视觉教师自蒸馏；tiny-target-aware sampling 对高分候选、hard negatives、不同面积 bins 平衡。下游只将 dense region feature用于 verifier/ASSP。

### 为什么可能适合红外小目标

把微小候选 crop 放大后，CLIP/global teacher 才可能看见其局部模式；hard-negative crop 可针对云边/亮点学习区域语义。

### 最小实现

训练期把候选 crop 放大到 CLIP 可见尺度作为 teacher；student 原图 RoI 对齐 teacher，背景 crop 做 margin；推理只运行 student dense encoder。

### 推理依赖

训练需本地 CLIP teacher，不需自然语言 caption；推理只需 student 图像 encoder，无文本/VLM/GT prompt。

### 可证伪实验

先线性 probe crop teacher 的 target/background AUPRC；再比较 random/grid/tiny-balanced crops 与 global embedding regression。

### 预期主要改善指标

candidate AUPRC、False Prompts/MP 与 Fa；只有 region feature 能命中 tiny components 时才期望 Pd。

### 最大风险

自然图像 CLIP 即使放大也无法区分 IR target 与杂波，teacher 上限过低。

### 新颖性风险

High：CLIPSelf/AlignEarth 已覆盖局部蒸馏；适合作为辅助预训练，不建议主贡献。

### 实施成本

Medium：需 crop pipeline、teacher cache 和 dense RoI alignment。

---

## Idea 10：存在性—定位解耦双头

### 研究假设

将“是否有目标”和“目标在哪里”解耦，可阻止空场景或错误文本强制产生 point，从而在不牺牲定位 head recall 的情况下降低 Fa。

### 文献来源

SegEarth-OV3 presence score、MaskSAM 分类 token、SAIST 背景建模、SeViL 候选置信过滤。

### 与现有工作的实质差异

presence head 只控制 prompt 是否生效，定位来自高分辨率 image evidence；文本 teacher 监督 presence calibration，不直接提供坐标。模型二部署时 image presence head 替代文本。

### 为什么可能适合红外小目标

目标占比极小且空/近空背景中的强杂波会触发强制点；独立存在性风险能在不降低 localization head 分辨率的情况下降低 Fa。

### 最小实现

共享 encoder 后分 `PresenceHead` 与 `LocalizationHead`；最终 prompt 为 `p_presence × targetness`，并以可调风险阈值允许完全 abstain。推理纯图像。

### 推理依赖

student 仅图像；无文本/VLM、人工/GT prompt或参考图。训练可用离线文本 presence soft label。

### 可证伪实验

presence head vs max-targetness threshold；空/hard-negative AUPRC；按阈值画 Pd–Fa 与 risk–coverage。

### 预期主要改善指标

Fa、Presence AUPRC/ECE；约束 Pd 不下降。

### 最大风险

存在性只是 targetness max 的单调变换，独立 head 没有新信息。

### 新颖性风险

Medium-High：SegEarth-OV3/SAM3 与 MaskSAM 已有 presence/class head；更适合并入 Idea 1，而非独立主方法。

### 实施成本

Low-Medium：轻量分类/校准头，但需要可靠空场景与 hard-negative 构造。

## 决策建议

先执行 Idea 1–3 的低成本筛选，不并行堆满所有模块。Idea 4/10 可作为 Idea 1 的简化版本，Idea 8 是必须建立的强无文本 baseline；Idea 5 是所有文本方案都应加入的安全性实验；Idea 6/7/9 只有前三项失败或出现明确证据时再进入完整训练。
