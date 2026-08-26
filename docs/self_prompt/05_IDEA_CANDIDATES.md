# TIRST-SAM Self-Prompt 候选 Idea

这些是**待证伪假设**，不是已经成立的论文贡献。共12个，覆盖A Object-Set、B Foreground–Background、C Response-Conditioned、D Pre-mask/Instance、E Prompt Credit、F Evidential/UQ、G Candidate–Context、H Temporal八类范式。所有方案推理均禁止GT point/box/mask；Idea 12另用多帧协议。

## Idea 1：MicroQuery-SAM——对象集合式微掩码 Self-Prompt（A+D）

### 核心假设
把K个候选从“同一query里的K个正点”改成“K个独立object queries”，并让每个query携带局部micro-mask与`∅`类别，可消除错误候选/多目标互相污染。
### Prompt 表示
`query[B,K,C] + micro_mask[B,K,h,w] + center[B,K,2] + object_logit[B,K,1]`。
### Prompt 生成
沿用现有neck候选坐标，但从early/shallow feature对每个坐标ROIAlign，生成局部mask和query token；不再优化heatmap。
### 一候选/一 query 设计
是；image embedding共享，K个query批量解码出`mask_logits[B,K,H,W]`。
### no-object / background / abstention
Hungarian matching；未匹配query训练为`∅`，推理按objectness×SAM-IoU拒绝。
### reliability 如何进入 decoder 或 mask aggregation
现有view reliability投影为query gate，并用于`objectness × reliability × SAM-IoU`聚合，不只排序。
### 训练 loss
component级Hungarian：object/no-object CE + micro-mask Dice/BCE + final mask Dice/BCE + duplicate penalty + calibration/Brier。
### 推理依赖
单图，K≤8；无文本、GT或外部detector。
### 与当前 PR #3 的实质差异
候选身份一直保留到独立mask和拒绝，不再是`[B,1,K,2]→one mask`。
### 最接近论文
MaskSAM、RSPrompter、Sam2Rad。
### 新颖性冲突
高：object query/no-object已成熟；独立命题必须收窄为“极小连通域的micro-mask state + SAM response校准 + 低K共享解码”，不能宣称首次auto-query SAM。
### 预期改善指标
duplicate prompts/component、False Prompts/MP、Fa、tiny Pd、no-object accuracy。
### 最大风险
EfficientSAM decoder对多个微弱query仍不敏感；K倍mask decoding增加显存。
### 最小可证伪实验
不训练：用现有A3坐标分别做one-query和K独立query；再加oracle center/box/micro-mask上界。
### 通过/停止条件
oracle independent-query相对oracle one-query的tiny Pd或IoU无改善且Fa不降，则停止；predicted K-query至少Fa降10%、Pd下降≤0.5pp才训练set head。
### 实施成本
低到中：先改batch/query shape与聚合；通过后才加micro-mask head。

## Idea 2：TB-Prompt——候选级 Target–Background 成对内生 Prompt（B）

### 核心假设
红外小目标更容易通过“中心相对邻域背景的差异”识别；每个候选同时产生target/background states，并让background在prompt-space非对称抑制，比all-positive points更能降低Fa。
### Prompt 表示
每候选`q+`,`q-`两token，加inner micro-mask与outer-ring background mask；两者保持分离。
### Prompt 生成
中心/inner ring/outer ring/global background从shallow feature池化；可附local contrast/frequency descriptor，但不再找新峰值。
### 一候选/一 query 设计
每候选一组paired query，输出独立mask。
### no-object / background / abstention
background token显式；若`p(background)>p(target)`或pair margin不足则no-object。
### reliability 如何进入 decoder 或 mask aggregation
`q-`经过冻结prompt encoder或专用negative projection产生非对称gate；view reliability只调gate温度，不消失。
### 训练 loss
target/background contrastive margin、object CE、mask loss、wrong/shuffled background反事实一致性、calibration。
### 推理依赖
单图，无文本/GT；背景从当前图像候选外环和全局prototype获得。
### 与当前 PR #3 的实质差异
不是“再加负点”，而是每候选有可区分的target/background条件直至decoder。
### 最接近论文
IP-SAM、EviPrompt、Memory-SAM、SurgicalSAM。
### 新颖性冲突
高：前/背景prompt已被IP-SAM直接覆盖；独立命题必须是candidate-level tiny-target paired state、outer-ring错误压力测试和one-query-per-component。
### 预期改善指标
Fa、False Prompts/MP、candidate AUPRC、background rejection、risk–coverage。
### 最大风险
1–9像素目标会污染outer ring或与背景同质，导致过抑制和漏检。
### 最小可证伪实验
用现有候选，比较target-only、target+outer-ring负点、target+learned background token、shuffled/wrong ring；decoder冻结。
### 通过/停止条件
correct background相对target-only Fa不降≥10%，或shuffled background与correct无差异，则停止paired-prompt主线。
### 实施成本
低到中；最初只需ring采样与token/负点反事实。

## Idea 3：RQ-Adapt——Response-Conditioned Refine–Reject Query Adapter（C+E）

### 核心假设
当前主要瓶颈是prompt–response gap；候选先独立得到低成本mask response，再更新query并决定refine/reject，会比继续优化candidate score有效。
### Prompt 表示
初始candidate query + 首轮mask statistics/low-res response token + refine/reject state。
### Prompt 生成
现有候选只负责初始化；首轮decoder输出SAM-IoU、mask area/compactness、center agreement与feature response，轻量adapter更新query。
### 一候选/一 query 设计
是；每candidate独立首轮/次轮mask。
### no-object / background / abstention
adapter输出`accept/refine/reject`三类；reject即no-object，空图允许全拒绝。
### reliability 如何进入 decoder 或 mask aggregation
view reliability与response quality共同形成更新门；第二轮query和最终mask权重都使用该门。
### 训练 loss
两轮mask loss、action CE（由GT component matching构造）、response consistency、reject calibration、额外forward成本惩罚。
### 推理依赖
单图，最多两轮轻量decoder；无文本/GT。
### 与当前 PR #3 的实质差异
reliability由SAM实际response验证，并可在decoder后拒绝；当前只在decoder前几何筛选一次。
### 最接近论文
SPARK-SAM、AoP-SAM、AlignSAM、PromptPilot、ReSAM、H-SAM。
### 新颖性冲突
高：response adaptation/iterative refinement非常拥挤。最小独立命题是“candidate-level single-step refine/reject under tiny-target Fa risk”，不得宣称首次mask feedback。
### 预期改善指标
prompt-to-mask sensitivity、harmful prompt AUROC、Fa、no-object accuracy、latency-risk curve。
### 最大风险
冻结decoder首轮response本身无辨别力，错误候选与真目标都得到相似mask。
### 最小可证伪实验
缓存每候选独立mask；用mask area/center/SAM-IoU训练或直接评估一个logistic rejector，先不改decoder。
### 通过/停止条件
response features对TP/FP的AUPRC不高于candidate score，或oracle reject也不能降Fa≥10%，停止。
### 实施成本
中；缓存诊断低，双轮推理中等。

## Idea 4：EviSet——证据分布式 Object Query（A+F）

### 核心假设
候选不确定性应区分“支持target”“支持background”“缺乏证据”，而不是单一sigmoid score；显式evidence可改善abstention。
### Prompt 表示
每query输出Dirichlet/Beta evidence `(e_target,e_bg,u)`、micro-mask和latent token。
### Prompt 生成
同一candidate在原图/轻扰动下产生evidence，使用view作为观测而非继续扩增候选。
### 一候选/一 query 设计
是。
### no-object / background / abstention
高background或高uncertainty均拒绝，但原因分开统计。
### reliability 如何进入 decoder 或 mask aggregation
belief缩放query，uncertainty调低mask聚合；训练/测试画risk–coverage。
### 训练 loss
evidential CE、KL regularizer、mask loss、calibration、empty/hard-negative监督。
### 推理依赖
单图，可只用共享feature的轻扰动head。
### 与当前 PR #3 的实质差异
support/dispersion从启发式过滤变为可校准三态分布，并一直参与mask。
### 最接近论文
EviPrompt、UR-SAM、AutoPromptSeg、UncertainSAM。
### 新颖性冲突
中高：UQ本身不新；须证明candidate object-query与mask risk的联合校准。
### 预期改善指标
ECE/Brier、risk–coverage、no-object accuracy、Fa。
### 最大风险
evidential网络可能只把candidate score重新参数化，校准改善但mask不变。
### 最小可证伪实验
对现有cluster离线拟合Beta/evidence calibration，与temperature/isotonic和max-score比较。
### 通过/停止条件
若不优于temperature scaling的Brier/ECE与Fa-risk曲线，停止复杂evidence head。
### 实施成本
低到中。

## Idea 5：CreditDrop——有害 Prompt 边际贡献抑制（E）

### 核心假设
同一query内某些点会降低最终mask质量；由candidate drop的response delta学习credit，可在单次部署中删除有害prompt。
### Prompt 表示
现有point/candidate + scalar/vector credit token。
### Prompt 生成
训练期对K候选做leave-one-out或Shapley近似，缓存每点mask delta；学生只看candidate/image feature预测credit。
### 一候选/一 query 设计
首轮可在当前one-query接口诊断；最终建议每候选query。
### no-object / background / abstention
negative credit候选被拒绝；全负允许empty prompt。
### reliability 如何进入 decoder 或 mask aggregation
credit缩放sparse token或candidate mask，而非只删Top-K。
### 训练 loss
credit ranking/regression、mask loss、calibration、sparsity/预算。
### 推理依赖
单图单次forward；训练期多次decoder生成credit label。
### 与当前 PR #3 的实质差异
直接监督“对mask是否有益”，而非用view一致性做代理。
### 最接近论文
PromptPilot、AlignSAM、PPD、AoP-SAM。
### 新颖性冲突
高：prompt credit已有近邻；差异只能是离线teacher→单步tiny-target defender。
### 预期改善指标
harmful prompt AUROC、False Prompts/MP、Fa、平均prompt数。
### 最大风险
drop delta在多个点相互作用下不稳定，且GT-derived credit可能过拟合。
### 最小可证伪实验
在现有validation对K≤5做exact leave-one-out，检验credit与TP/FP、最终Fa的相关性。
### 通过/停止条件
credit AUPRC不优于rule reliability或删负credit不能降Fa≥10%，停止。
### 实施成本
低诊断、中训练。

## Idea 6：ContextGraph Query——候选—背景关系图 Prompt（G+A）

### 核心假设
孤立候选无法判断重复峰、结构化热边缘和共同背景；候选与background anchors的关系图可减少重复/虚警。
### Prompt 表示
candidate nodes、global/edge/background nodes与关系编码后的object queries。
### Prompt 生成
节点来自现有cluster，不新增峰值；边特征含距离、feature相似、ring contrast和view共现。
### 一候选/一 query 设计
每candidate node对应一个query和mask。
### no-object / background / abstention
background nodes只提供抑制；candidate head有`∅`。
### reliability 如何进入 decoder 或 mask aggregation
relation-updatedquery进入decoder，node reliability参与message gate和mask weight。
### 训练 loss
set mask/object loss、edge consistency、duplicate contrast、calibration。
### 推理依赖
单图；K和background anchors固定小规模。
### 与当前 PR #3 的实质差异
cluster不再只做几何合并，而是保持关系到decoder。
### 最接近论文
GF-SAM、GPRN、GPR、MaskSAM。
### 新颖性冲突
高：graph prompt reasoning已存在；必须证明IRSTD背景node与object-set的必要性，不可只是加GNN。
### 预期改善指标
duplicate prompts/component、Fa、dense edge hard-negative准确率。
### 最大风险
K很小时graph无额外信息，收益来自参数量。
### 最小可证伪实验
固定candidate features，用非学习关系规则/1层attention预测TP，和独立MLP同参数对照。
### 通过/停止条件
relation model不显著优于independent MLP/candidate score，停止。
### 实施成本
中。

## Idea 7：Budgeted Abstention Query——存在性与 Prompt 预算联合决策（F+E）

### 核心假设
每图固定K会在空/简单图制造虚警；先估计presence与最小所需query数，可改善Fa—延迟权衡。
### Prompt 表示
global presence token + per-candidate object token + stop/budget distribution。
### Prompt 生成
global token来自整图background-aware feature，候选沿用现有proposal。
### 一候选/一 query 设计
是，预算`K_i∈{0…Kmax}`。
### no-object / background / abstention
presence低时K=0；每query仍有no-object。
### reliability 如何进入 decoder 或 mask aggregation
budget按calibrated risk选，保留query权重继续进入聚合。
### 训练 loss
presence BCE、set loss、expected cost penalty、Brier/ECE。
### 推理依赖
单图；可减少decoder调用。
### 与当前 PR #3 的实质差异
从固定Top-K变成可为零的结构化预算，而非换阈值。
### 最接近论文
AutoPrompt-SAM3D target-aware filter、AoP-SAM efficiency、MaskSAM no-object。
### 新颖性冲突
中：动态预算常见；需与object-query/no-object联合才有意义。
### 预期改善指标
Fa、no-object accuracy、latency、risk–coverage。
### 最大风险
当前IRSTD split可能几乎没有空目标，presence学习退化。
### 最小可证伪实验
构造严格train-only hard-negative crops/empty images，离线calibrate K=0 gate。
### 通过/停止条件
没有足够empty/hard-negative数据或presence AUROC<0.9且tiny Pd下降>0.5pp，停止。
### 实施成本
低到中。

## Idea 8：Residual Micro-Mask Bank——高分辨率候选形状库（D）

### 核心假设
tiny目标定位损失主要发生在neck下采样；从shallow residual为每候选产生局部mask提示，优于单点且不必重训完整encoder。
### Prompt 表示
`K`个局部`h×w`micro-masks + center + shape/scale token。
### Prompt 生成
ROIAlign early feature与local contrast/frequency residual，预测小局部mask并回贴dense prompt。
### 一候选/一 query 设计
每micro-mask一个query。
### no-object / background / abstention
micro-mask空/低质量→no-object；外环作为background channel。
### reliability 如何进入 decoder 或 mask aggregation
micro-mask quality与view reliability共同缩放dense prompt。
### 训练 loss
local Dice/BCE、boundary/center、object CE、final mask loss。
### 推理依赖
单图，无文本/GT。
### 与当前 PR #3 的实质差异
prompt包含局部shape和背景，不是坐标点。
### 最接近论文
Sam2Rad、UN-SAM、De-LightSAM、Self-Prompt SAM。
### 新颖性冲突
高：pre-mask/dense prompt拥挤；必须与object-set/no-object和IR micro-scale结合。
### 预期改善指标
1–9/10–16像素component recall、boundary IoU、Pd。
### 最大风险
GT局部mask监督让head变成普通小目标分割器，SAM只做后处理。
### 最小可证伪实验
oracle micro-mask vs oracle point/box；若上界无差异，不做learned bank。
### 通过/停止条件
oracle micro-mask在tiny桶不比oracle point提升≥3pp，停止。
### 实施成本
中。

## Idea 9：RepelSet——重复候选对比与实例排斥 Query（A+G）

### 核心假设
同一target周围多个峰应合并/互斥，而不同target应保留；query间repulsion与component matching可减少重复mask且不误删多目标。
### Prompt 表示
K object queries + pairwise relation logits。
### Prompt 生成
现有clusters直接初始化query；不改proposal数量。
### 一候选/一 query 设计
是。
### no-object / background / abstention
重复query可匹配no-object；真实不同component分别匹配。
### reliability 如何进入 decoder 或 mask aggregation
reliability作为matching prior，最终由query objectness与mask overlap共同NMS/aggregation。
### 训练 loss
Hungarian set loss、query contrast/repulsion、duplicate mask penalty。
### 推理依赖
单图。
### 与当前 PR #3 的实质差异
不在坐标cluster阶段硬合并，而由mask/object response判别“重复还是多目标”。
### 最接近论文
MaskSAM、RSPrompter、AoP-SAM。
### 新颖性冲突
高：set prediction和mask NMS成熟；只能作为Idea1消融而非独立主创新。
### 预期改善指标
duplicate prompts/component、multi-target recall、Fa。
### 最大风险
tiny targets彼此邻近时repulsion误伤。
### 最小可证伪实验
用独立SAM masks按overlap合并，比较坐标cluster和mask-cluster。
### 通过/停止条件
mask-based grouping不优于现坐标cluster，降级为Idea1辅助而非主线。
### 实施成本
低。

## Idea 10：Prompt Jacobian Grounding——Mask 对 Query 的敏感性正则（C）

### 核心假设
decoder忽略self-prompt是A3不转化的原因；显式约束正确query的扰动应改变目标mask、错误/background query不应产生大响应，可恢复prompt grounding。
### Prompt 表示
任意candidate query；不新增proposal表示。
### Prompt 生成
沿用Idea1/2 query，训练时做query drop/shuffle/sign-flip小扰动。
### 一候选/一 query 设计
是。
### no-object / background / abstention
no-object query的mask Jacobian应小；target query在对应局部应大。
### reliability 如何进入 decoder 或 mask aggregation
reliability加权sensitivity loss与query gate。
### 训练 loss
mask GT + finite-difference/Jacobian contrast：correct-drop delta、shuffle invariance、background suppression。
### 推理依赖
单图单次；扰动只在训练。
### 与当前 PR #3 的实质差异
直接优化prompt→mask函数，而不是prompt heatmap准确率。
### 最接近论文
SPARK-SAM response adaptation、SAM-REF、perturbed prompt训练。
### 新颖性冲突
中高：鲁棒prompt训练常见；“正确prompt应敏感、错误prompt应不敏感”的双向目标需谨慎定位。
### 预期改善指标
prompt-to-mask sensitivity、shuffled gap、IoU/Fa。
### 最大风险
模型通过扩大所有响应或过拟合GT区域满足loss，造成不稳定。
### 最小可证伪实验
先测现有checkpoint的correct/drop/shuffle mask delta是否与candidate TP相关。
### 通过/停止条件
若oracle correct query都不产生可测delta，先做adapter；若delta与TP无关，停止该正则。
### 实施成本
中。

## Idea 11：Semantic Verifier Distillation——文本仅验证候选（B+F）

### 核心假设
文本不适合生成tiny坐标，但结构化presence/location/contrast角色可以在训练期校准candidate target/background，student部署只看图像。
### Prompt 表示
候选query不变；teacher只输出per-candidate target/background/uncertainty软标签。
### Prompt 生成
图像候选先产生；GPT/CLIP角色token与candidate ROI做匹配，不直接投成全局sparse token。
### 一候选/一 query 设计
依附Idea1/2，每候选独立。
### no-object / background / abstention
student蒸馏teacher正确文本相对zero/shuffle/wrong的**安全增量**；错文本退回no-text。
### reliability 如何进入 decoder 或 mask aggregation
student verifier输出query gate/objectness并进入mask权重。
### 训练 loss
candidate CE/KL、counterfactual margin、mask loss、teacher可信样本mask；不回归单一CLIP global embedding。
### 推理依赖
纯图像student；GPT/CLIP只离线训练。
### 与当前 PR #3 的实质差异
文本不生成点、不直接占SAM token，只监督候选target/background决策。
### 最接近论文
SeViL/JinSight式语言训练视觉部署、IP-SAM、当前反事实蒸馏实验。
### 新颖性冲突
高，且当前已有`C≈S≈W≈O`负证据；只能在teacher候选级增益先成立后继续。
### 预期改善指标
candidate AUPRC、Fa、C>S/W margin、student保留率。
### 最大风险
teacher文本对tiny候选没有独立信息；此前实验已显示文本语义未被模型消费。
### 最小可证伪实验
不训练student：在同一candidate集合上比较C/N/S/W verifier AUPRC与mask delta。
### 通过/停止条件
C不稳定优于N/S/W或teacher Fa改善不超过配对标准误，立即停止。
### 实施成本
低诊断；通过后中等。

## Idea 12：Temporal-Emerged Object Query（H，独立多帧课题）

### 核心假设
多帧IR中单帧不可见target可通过相对global background motion的局部偏差形成object query；query而非热图承担prompt。
### Prompt 表示
temporal candidate queries + frame micro-mask + background-motion state。
### Prompt 生成
短时窗feature做global/local discrepancy，固定K queries读取不同运动残差。
### 一候选/一 query 设计
是。
### no-object / background / abstention
每帧query可no-object；全序列目标可出现/消失。
### reliability 如何进入 decoder 或 mask aggregation
temporal consistency与motion discrepancy门控query/memory更新。
### 训练 loss
frame/set mask、temporal consistency、appearance/disappearance objectness、background motion contrast。
### 推理依赖
真实多帧序列；与单帧IRSTD完全分开。
### 与当前 PR #3 的实质差异
不是同一图像的deterministic transforms，而是真实时间证据和可更新query。
### 最接近论文
TEP-SAM、SAM-DAQ、TSP-SAM。
### 新颖性冲突
高：temporal prompting已有直接先例；需新数据/特定query命题。
### 预期改善指标
hard low-SNR Pd、Fa、query persistence、出现/消失准确率。
### 最大风险
当前项目数据为单帧，无法公平验证；容易变成完全不同论文。
### 最小可证伪实验
只在有真实序列的数据上复现TEP-SAM式query baseline，禁止用flip/scale代替时间。
### 通过/停止条件
没有合规序列split或相对单帧强baseline无hard-subset增益则停止。
### 实施成本
高，暂不进入当前Top-3。

## 加权排名

权重按任务书：跳出heatmap 20%、FP/FN 15%、decoder消费15%、多目标10%、no-object10%、新颖性15%、可行性10%、成本5%。表中单项1–5；总分为加权百分制。

| Rank | Idea | Heatmap | FP/FN | Consume | Multi | NoObj | Novelty | Feasible | Cost | 总分 | 决策 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | MicroQuery-SAM | 5 | 5 | 5 | 5 | 5 | 3 | 4 | 4 | **88** | Top-3；先oracle shape test |
| 2 | TB-Prompt | 5 | 5 | 5 | 4 | 5 | 3 | 4 | 4 | **86** | Top-3；先background反事实 |
| 3 | RQ-Adapt | 5 | 5 | 5 | 4 | 5 | 2 | 4 | 3 | **82** | Top-3；先离线response AUPRC |
| 4 | EviSet | 5 | 5 | 4 | 4 | 5 | 3 | 4 | 4 | 82 | 若Top-3需要校准再并入 |
| 5 | Prompt Jacobian | 5 | 4 | 5 | 4 | 4 | 3 | 3 | 3 | 79 | response adapter辅助 |
| 6 | Budgeted Abstention | 4 | 5 | 4 | 4 | 5 | 3 | 4 | 5 | 79 | 需empty样本 |
| 7 | CreditDrop | 4 | 5 | 5 | 3 | 4 | 2 | 4 | 3 | 75 | 先做exact drop diagnostic |
| 8 | Residual Micro-Mask Bank | 5 | 4 | 5 | 4 | 4 | 2 | 3 | 3 | 74 | 可并入Idea1 |
| 9 | ContextGraph Query | 5 | 4 | 5 | 5 | 5 | 1 | 3 | 3 | 74 | graph新颖性风险高 |
| 10 | RepelSet | 5 | 4 | 4 | 5 | 5 | 1 | 4 | 4 | 74 | 只作Idea1消融 |
| 11 | Semantic Verifier Distillation | 4 | 4 | 4 | 4 | 5 | 2 | 3 | 3 | 68 | 当前文本负证据，条件启动 |
| 12 | Temporal-Emerged Query | 5 | 5 | 5 | 4 | 5 | 1 | 1 | 1 | 68 | 独立多帧项目 |

## 自动淘汰检查

- 12个候选均显式包含background/no-object/uncertainty之一，并设计decoder消费路径。
- Idea 9明确降级为Idea1消融；Idea 12因协议不同不进入当前单帧Top-3；Idea 11因`C≈S≈W≈O`只保留为条件分支。
- 不提出：新显著性算子、新NMS/Top-K、增加视图、support MLP、all-positive one-query、reliability只排序、global text token直投、GT修prompt、detector-box→SAM主创新。
