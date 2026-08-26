# Self-Prompt 候选 Idea 新颖性与冲突审计

本文件判断“已有工作覆盖到哪里”，不声称任何候选已经新颖。S16 IR-SAM2因MDPI来源策略排除，只标记为未纳入，不能据任务书摘要推导碰撞。

## 1. 必比方法的已覆盖机制

| 方法 | 已覆盖的核心命题 | 当前不能再主张 | 尚未直接覆盖的窄问题 |
|---|---|---|---|
| SPARK-SAM | IRSTD自动self-prompt、response knowledge/adaptation、candidate gating、high-res refinement、校准 | 首次发现prompt–response gap；首次自动IR SAM；只加response adapter即新 | 小K component-set matching、显式`∅`、候选独立micro-mask与paired background |
| IP-SAM | 图像生成连续前/背景prompt，经冻结prompt encoder，background非对称抑制 | 首次target/background intrinsic prompt；feature-add就等价 | candidate-level paired state、tiny outer-ring可靠性、多目标独立query |
| SAM-SPL | 浅层`F0/F1/F2`对齐后生成dense prompt `P`，经two-way transformer与SAM2 latent交互；CGKA/MSIC；纯图像IRSTD部署 | 首次纯图像IR self-prompt；首次浅层feature/dense prompt；仅换全局或局部视觉token即新 | candidate identity/no-object、decoder-side候选拒绝、角色/背景反事实 |
| IR-SAM2 | 本轮策略排除 | 不作任何基于其正文的首次/碰撞判断 | 用户若要求，独立附录补审 |
| MaskSAM | per-query mask+box prompts、mask classification、Hungarian matching、no-object、3D auto-prompt | 首次one-query-one-mask或`∅` | IR tiny-component低K共享解码、view reliability与SAM response联合校准 |
| Sam2Rad | image→learned object/class query，联合预测box/mask/latent prompt，原生prompt encoder | 首次image生成latent/box/mask联合prompt | instance set matching、空目标、tiny micro-mask与background pair |
| AoP-SAM | confidence map自动点、coarse-to-fine adaptive sampling、mask-overlap去冗余、效率 | 首次mask feedback筛prompt | candidate-levelcredit/no-object/latent state；IR低K而非AMG |
| RSPrompter | detector/query自动latent prompt、每query独立mask、`C+1`、optional pre-mask | 首次latent query或自动遥感SAM实例分割 | 极小单类对象的简化set head、background pair与风险校准 |
| UN-SAM | coarse mask dense prompt、domain queries、coarse-to-fine prompt-free医学分割 | 首次pre-mask→SAM或abstract query | instance candidate/no-object/空图，IR小目标shape state |
| EviPrompt | reference正负prototype、evidential uncertainty、多视图证据、正负点和mask→box refinement | 首次evidential prompt/正负自动点 | 无reference的candidate evidence、证据直接进入query/mask聚合 |
| PromptPilot | 标注reference+DINOv2迁移初始化，Feature/Physical Agents activate/prune，Manager由SAM DSC与训练期GT-LOO边际贡献优化 | 首次prompt credit、agentic point optimization或GT-LOO harmful-prompt supervision | 无reference、无在线GT、非RL、单次candidate response拒绝是否仍成立 |
| AutoPrompt-SAM3D | SAM2三层feature prompt generator、confidence frame filter、invalid-window filter、序列传播 | 首次可靠性筛选/无目标window | 单帧IR candidate set与decoder response，非序列协议 |
| DVPT | 冻结SAM编码器内的local dense prompt与global group prompt tokens，图像-only部署 | 首次image feature→local/global visual prompt；首次全局learned token | candidate identity、`∅`、字段/角色反事实和tiny-object因果消费 |
| AutoPromptSeg | MC-dropout epistemic/aleatoric uncertainty、低不确定性PSS、3D NMS、每类Top-K自动点 | 首次uncertainty-guided prompt sampling或低UQ Top-K | reliability在decoder/独立mask中的持续状态，而非外部排序 |
| MUP-SAM | 辅助分割→形态学/扩框/NMS boxes→冻结MedSAM→预测fusion；zero-box返回空mask | 首次auxiliary mask/segmentor→box→SAM；把fusion收益归因于prompt | 无重型辅助分割/融合器的轻量candidate state与prompt-only因果收益 |
| S4M | top/bottom/left/right或major/minor四点role embeddings，Canvas token/decoder；人工交互部署 | 首次结构化点角色/type embedding | 新图像自动产生角色可靠的candidate/background states与`∅` |

## 2. 逐 Idea 冲突审计

| Idea | 已有覆盖部分 | 当前新增的最小部分 | 只是换领域？ | 只是换backbone？ | 只是组合模块？ | 可形成论文贡献的最小独立命题 | 风险 |
|---|---|---|---|---|---|---|---|
| 1 MicroQuery-SAM | MaskSAM/RSPrompter已覆盖query set、独立mask、`∅`；Sam2Rad覆盖box+mask+latent | low-K IR component query、shallow micro-mask、view reliability×SAM response校准、共享EfficientSAM解码 | **若只迁移set head则是** | 否 | **当前很像MaskSAM+Sam2Rad** | 在极小连通域中，micro-mask object state+`∅`比point-set显著降低候选污染，且不是增加decoder容量造成 | 中高 |
| 2 TB-Prompt | IP-SAM覆盖前/背景prompt-space gating；EviPrompt/Memory-SAM覆盖正负提示；S4M覆盖点角色embedding | 每candidate inner/outer/background pair + independent mask + role-shuffle/wrong-ring反事实 | **若只把IP-SAM/S4M搬到IR则是** | 否 | IP-SAM+MaskSAM+role token组合风险高 | 局部背景不是普通负点，而是可校准候选条件；其非对称作用须在tiny target上通过shared/specific/shuffled role因果验证 | 高 |
| 3 RQ-Adapt | SPARK-SAM覆盖IR response adaptation；ReSAM/H-SAM覆盖再查询/两阶段；PromptPilot完整覆盖多轮feedback与GT-LOO credit | 无reference、无在线GT、非RL的单次candidate refine/reject与Fa-risk控制 | 部分 | 否 | **很可能是SPARK+PromptPilot的轻量化** | 冻结首轮response是否在单次部署中比candidate/UQ score更能拒绝有害IR candidate | **很高** |
| 4 EviSet | EviPrompt/UR-SAM覆盖evidence/UQ；AutoPromptSeg完整覆盖低不确定性NMS/Top-K；MaskSAM覆盖set | 无reference的per-query三态evidence直接门控独立mask | 若只加evidential loss或UQ排序则是常规迁移 | 否 | EviPrompt+AutoPromptSeg+MaskSAM | decoder内三态risk是否优于AutoPromptSeg式外部UQ ranking及temperature/isotonic | 高 |
| 5 CreditDrop | PromptPilot已直接用GT-LOO marginal contribution；AlignSAM/PPD覆盖credit/删除/RL；AoP-SAM覆盖response筛选 | 只可作exact-drop oracle与无需GT response proxy诊断 | 是，若独立投稿 | 否 | PromptPilot简化/蒸馏 | 不作为独立贡献；仅在单次student跨数据集保留收益时并入RQ-Adapt | **很高** |
| 6 ContextGraph | GPRN/GF-SAM已覆盖graph prompt reasoning；RSPrompter覆盖query set | background anchor nodes + tiny candidate independent masks | **若只加GNN则是** | 否 | GPRN+MaskSAM | IR clutter中的candidate–background关系在同参数独立MLP之外提供可重复增益 | 高 |
| 7 Budgeted Abstention | AutoPrompt-SAM3D有invalid-window filter；MaskSAM有no-object；AoP-SAM做效率 | global presence与query budget联合、K可为0、Fa–latency risk | 部分 | 否 | 常见presence head+set query | 在含真实/合成empty hard negatives的IRSTD中，calibrated K=0/动态K同时改善Fa与延迟而不伤tiny Pd | 中 |
| 8 Residual Micro-Mask Bank | Sam2Rad/UN-SAM/De-LightSAM/Self-Prompt SAM均有pre-mask/dense prompt | candidate-local shallow residual micro-mask而非整图pre-mask | **很容易只是换IR** | 否 | 多篇pre-mask方法组合 | oracle micro-mask相对oracle point的tiny上界及其与object query的交互 | 高 |
| 9 RepelSet | MaskSAM/RSPrompter的Hungarian、AoP-SAM mask去重均覆盖 | IR近邻微目标的mask-level duplicate vs true multi-target判别 | 是，若独立投稿 | 否 | 是 | 只能作为Idea1的一项loss/aggregation消融，不足以独立贡献 | 高 |
| 10 Prompt Jacobian | SPARK-SAM response sensitivity、PP-SAM perturbed prompts、SAM-REF image-prompt synergy | correct-sensitive / wrong-insensitive的双向finite-difference grounding | 部分 | 否 | 高 | prompt内容对mask的局部因果敏感性正则是否解决A3“prompt好而mask不变” | 中高 |
| 11 Semantic Verifier | JinSight/SeViL类训练有语言部署纯图像；当前项目已做反事实文本诊断；IP-SAM有bg gate | 文本只给candidate-level target/bg增量，错文退回no-text | **高度可能** | 否 | language distill+query gate | 只有先证明C>N/S/W的candidate response margin，才可能形成“安全语义验证器”命题 | 高/当前停止 |
| 12 Temporal Query | TEP-SAM直接覆盖temporal-emerged prompt；SAM-DAQ query memory；TSP-SAM motion prompt | 若有新query出现/消失/背景竞争机制 | **当前几乎只是同域近邻** | 否 | 三者组合 | 需新多帧问题/数据或强独立理论；不属于当前单帧论文最小修改 | 极高 |

## 3. Top-3 与逐篇直接碰撞

### Idea 1 MicroQuery-SAM

- **SPARK-SAM**：两者都在IRSTD自动推理；必须用`independent object masks + ∅`区分，且报告SPARK式response adapter控制。
- **IP-SAM**：Idea1若没有background只解决实例隔离；与Idea2组合时必须证明不是“MaskSAM set + IP-SAM gate”的机械拼接。
- **SAM-SPL**：必须与纯视觉shallow prompt同协议；若只加shallow feature，无创新。
- **DVPT**：局部dense prompt和全局group tokens均已覆盖；Idea1若只是ROI特征生成token，没有新颖性。
- **MaskSAM/RSPrompter**：最大碰撞。论文主张只能是针对tiny component的轻量prompt state/证据，不是set prediction本身。
- **Sam2Rad**：box/mask/latent联合状态已有；Idea1需实例Hungarian、no-object与微目标结果。
- **AoP-SAM**：mask去重不是新；Idea1的每候选mask必须服务于set/no-object而非AMG筛选。
- **UN-SAM**：pre-mask已有；区别是local per-candidate而非整图/域query。
- **EviPrompt**：若加uncertainty必须无reference且进入mask聚合。
- **PromptPilot**：若加credit不得宣称首次；可只作离线诊断。
- **MUP-SAM**：auxiliary mask→box→SAM+fusion已覆盖；必须将prompt本身收益与后端分割/fusion容量分离。
- **AutoPrompt-SAM3D**：confidence filtering已有，但序列与单帧协议不同。
- **IR-SAM2**：未纳入，必须在用户补审后再更新最终novelty结论。

### Idea 2 TB-Prompt

- **IP-SAM**是决定性近邻。只做`target feature + background feature + gate`没有新颖性。
- **EviPrompt**已覆盖正负prompt与不确定性；区别必须是无reference、candidate-local pair和原生prompt-space/decoder消费。
- **Memory-SAM**（A33）已覆盖retrieval foreground/background points；不能宣称首次自动正负点。
- **SurgicalSAM**覆盖正/负class embeddings；不能只把prototype改成IR中心/外环。
- **SPARK-SAM**已有可选negative ring；必须证明“paired latent condition”明显不同于加4个负点。
- **S4M**证明角色/type embedding本身会改变结构化点prompt表现；必须增加shared-role、role-specific和role-shuffled控制，不能把角色编码当新贡献。
- **MaskSAM/RSPrompter**提供independent query外壳，不能作为第二项创新重复计数。

### Idea 3 RQ-Adapt

- **SPARK-SAM**已是IR response knowledge/adaptation，最大风险。Idea3必须减少到可独立证伪的`candidate response→refine/reject`，并与SPARK式adapter正面对照。
- **ReSAM/H-SAM**已覆盖refine/requery与two-stage mask；不能只做第二轮decoder。
- **AoP-SAM**已用mask筛prompt；区别是learned candidate state update，而非overlap过滤。
- **PromptPilot**全文显示其Manager同时使用SAM DSC与训练期GT-LOO credit，目标图推理虽无GT但需要一张标注reference并进行多轮博弈；Idea3必须同时满足无reference、无在线GT、非RL、单次/至多选择性二次解码。
- **AutoPromptSeg**给出低不确定性点优于随机/中心/grid的强控制；若response reject不优于UQ/candidate ranking，Idea3停止。
- **AutoPrompt-SAM3D**覆盖错误prompt传播前筛选；区别是单帧per-candidate decoder response，不是frame confidence。

## 4. 必须新增检索的窄题目

在投稿前应再做一次针对以下精确命题的最新检索，而不是泛搜`auto prompt SAM`：

1. `infrared small target object query no-object SAM independent masks`；
2. `candidate-level foreground background prompt space SAM`；
3. `mask response conditioned prompt rejection SAM`；
4. `prompt sensitivity Jacobian segment anything`；
5. `micro-mask prompt tiny object SAM`。

当前结论有效期只到2026-08-26；S01/S02等2026 preprint变化快，正式投稿前必须刷新。

## 5. 结论与门槛

- Top-3都不是低风险“空白”；补充全文后，新颖性风险分别为中高/高/**很高**。尤其RQ-Adapt不能沿PromptPilot的RL/GT-LOO路线展开。
- 最安全的论文命题不是提出一个模块名，而是回答三个可证伪问题：独立query是否消除候选污染；paired background是否真正被decoder消费；response是否能在candidate级预测并拒绝错误prompt。
- 如果这三个问题的最小实验都不成立，应停止Self-Prompt主线，而不是再组合UQ、graph、text或更多views。
