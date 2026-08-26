# S 级 Self-Prompt 论文深读

阅读口径：14 篇基于完整正文，S12/S15 基于摘要+作者代码，S14 基于摘要，S16按来源策略排除。对缺全文条目，未补写不存在的公式、消融或算力。张量 shape 以作者公开代码为准，详见 `03_REFERENCE_CODE_PATH_AUDIT.md`。

## S01 SPARK-SAM

1. **研究问题**：目标覆盖正确的box也不能让冻结SAM2在IRSTD产生正确mask，作者把它定义为 prompt–response gap。
2. **完整架构**：SAM2.1-Tiny + prompt estimator + response guidance/adaptation + local prompt tokens + candidate gating + response/false-alarm calibration + high-resolution prompt refinement。
3. **Prompt 张量/数据流**：图像编码得到多层特征；estimator输出objectness、box size和候选；候选形成点/box/可选背景环，局部图像token与dense residual共同进入decoder。
4. **训练目标**：mask监督、候选/objectness/box、response guidance与quality/calibration目标；训练mask只用于监督和离线response构造。
5. **推理**：单图直接产生候选与joint self-prompt state，不读GT；estimator无候选时的代码有argmax fallback，不能天然保证空输出。
6. **关键消融**：论文分别移除response guidance、prompt refinement与空间干预；matched ablation支持response adaptation是主要增益来源，且简单改坐标/空间提示普遍退化。
7. **失败案例**：暗弱目标、hot clutter、target-like distractor仍有FN/FP。
8. **计算开销**：39.688M总参数；相对SAM2.1-Tiny新增0.726M参数、6.134 GFLOPs、6.80 MiB峰值显存；同一RTX 4090协议均值延迟增加0.051 ms（论文报告）。
9. **与当前项目差异**：PR #3只在decoder前生成点并丢掉可靠性；SPARK-SAM让response knowledge、candidate gate和局部token进入mask生成/校准。
10. **可迁移机制**：先做oracle/frozen prompt sensitivity诊断；candidate gate进入token幅值；response adapter与false-alarm calibration。
11. **不应照搬**：objectness heatmap→Top-K、强制argmax候选、复杂多阶段训练。它们不能解决PR #3“所有候选混入一个query”的核心问题。
12. **复现依赖**：SAM2.1 checkpoints、三套IRSTD数据、作者response cache/训练阶段脚本；代码无根许可证，只可审计/重写。

## S02 IP-SAM

1. **研究问题**：prompt-absent部署时，空prompt无法使SAM2主动区分伪装前景和同纹理背景。
2. **完整架构**：LoRA image encoder + Self-Prompt Generator（SPG）+ 冻结SAM2 prompt encoder + Prompt-Space Gating（PSG）+ task-specific mask decoder。
3. **Prompt 张量/数据流**：SPG产生连续前景/背景mask logits `P+`,`P-`；二者不先二值化，直接经冻结prompt encoder变成dense states；背景state通过sigmoid/complement对前景条件做非对称抑制。
4. **训练目标**：BCE、IoU和L1组合；prompt encoder严格冻结，训练SPG/PSG/task decoder和image LoRA。
5. **推理**：单图、无文本、无人工/GT prompt；21.26M可训练参数。
6. **关键消融**：null prompt baseline→SPG→PSG→lateral/decoder；bypass prompt encoder、feature add或解冻prompt encoder均不如连续logit经过冻结prompt encoder。
7. **失败案例**：极低对比/前背景同质时，`P-`可能包含目标，PSG造成过抑制和欠分割。
8. **计算开销**：supplement报告约29 FPS，冻结prompt encoder占内部CUDA延迟约1.6%；相对baseline有约20.7%端到端延迟增幅。
9. **与当前项目差异**：PR #3无负/背景条件，且reliability只用于选点；IP-SAM把背景变为decoder前的原生prompt-space约束。
10. **可迁移机制**：候选中心/内环/外环构造target-background paired state；冻结prompt encoder通路与feature-add反事实。
11. **不应照搬**：单一全图前景/背景mask无法分隔多个微小目标；直接迁移COD CNN/SPG会成为换领域。
12. **复现依赖**：SAM2、COD数据与task decoder实现；本轮未定位到任务书指定官方代码。

## S03 MaskSAM

1. **研究问题**：把SAM从人工prompt二值分割改造成可自动、多类、集合预测的3D医学分割器。
2. **完整架构**：3D depth adapters + auxiliary prompt generator + 原生prompt encoder + 改造mask decoder/classifier token + Hungarian matching。
3. **Prompt 张量/数据流**：固定`Nq` query产生`[B,Nq,D,H,W]` binary auxiliary masks和boxes；reshape为`B*Nq`，每query的mask/box分别编码；classifier token输出`C+1`类别。
4. **训练目标**：bipartite matching后做class（含no-object）、mask BCE/Dice、box回归与多层辅助loss。
5. **推理**：无人工prompt；每query独立调用decoder，按class/mask得分聚合为语义mask。
6. **关键消融**：论文分解prompt generator、classifier token、3D depth adapters和不同prompt组合；结论是mask+box+classification共同重要。
7. **失败案例**：query不足、相邻器官/小结构的类别混淆和3D显存压力；正文未给IRSTD式空目标分析。
8. **计算开销**：固定query导致图像embedding复制到`B*Nq`，显存/延迟随query线性增加；论文以参数高效adapter缓解但不消除decoder开销。
9. **与当前项目差异**：它显式实现一query一mask与`∅`，而PR #3把K个正点放进同一个query并只出一张mask。
10. **可迁移机制**：小K集合预测、Hungarian component matching、no-object类、独立micro-mask，再做可靠性加权聚合。
11. **不应照搬**：3D adapters、器官多类头和大Nq；IR只需小K binary object/no-object。
12. **复现依赖**：nnUNet、SAM checkpoint、3D数据预处理；公开仓库未声明根许可证。

## S04 Sam2Rad

1. **研究问题**：医学图像域移位下，不依赖人工prompt学习SAM/SAM2可消费的box、mask和latent prompts。
2. **完整架构**：冻结/PEFT image encoder + high-resolution Prompt Predictor Network（PPN）+ 原生prompt encoder + mask decoder。
3. **Prompt 张量/数据流**：learned class queries与多层feature cross-attention；前2 query预测box，interim mask经`_embed_masks`成dense prompt，其余query作为latent sparse prompts。
4. **训练目标**：segmentation与prompt相关目标；训练器可随机混合learned prompt和由GT mask构造的manual box/low-res mask。
5. **推理**：默认只输入图像；可选人工prompt拼接，但不是部署必需。
6. **关键消融**：SAM/SAM2、PPN层级/learned prompts、冻结/PEFT设置与不同模态对比；证据支持多层feature PPN优于单纯人工稀疏prompt。
7. **失败案例**：超声骨界面域差、低质量边界；class-centric query对多实例与空目标没有set-level约束。
8. **计算开销**：高分辨率PPN增加cross-attention；主体可冻结，仅PPN/decoder或adapter训练。
9. **与当前项目差异**：prompt是同一object state的box+mask+latent tokens，不是互不关联的Top-K正点。
10. **可迁移机制**：candidate query联合预测micro-mask、box/center和latent token；同一state进入原生prompt encoder。
11. **不应照搬**：训练时混GT-derived manual prompts会改变部署契约；class query不能直接替代IR instance queries。
12. **复现依赖**：SAM/SAM2/MedSAM权重、PPN配置、医学数据；仓库未声明根许可证。

## S05 AoP-SAM

1. **研究问题**：替代SAM Automatic Mask Generator的密集网格，减少无效点和冗余mask。
2. **完整架构**：轻量Prompt Predictor产生confidence map；粗Adaptive Sampling/Filtering（ASF）选点；SAM mask与score驱动finer ASF去冗余。
3. **Prompt 张量/数据流**：原图与reshape后的SAM embedding拼接进入CNN，sigmoid map采点；点进标准SAM；低分辨率masks用于overlap/PET筛选。
4. **训练目标**：prompt confidence/位置监督；SAM保持冻结。
5. **推理**：先预测候选，再分批SAM，利用predicted-IoU/stability/overlap过滤；可能多次或批量SAM forward。
6. **关键消融**：Prompt Predictor、coarse ASF、finer ASF、prompt密度与不同SAM backbone；主要收益是效率—召回折中。
7. **失败案例**：confidence map漏掉微小/低显著目标时后续无法恢复；多实例mask重叠仍依赖启发式阈值。
8. **计算开销**：比规则grid减少prompt和SAM调用；但仍随候选点数增长。
9. **与当前项目差异**：它让mask response参与prompt筛选；PR #3只在送入SAM之前用view reliability。
10. **可迁移机制**：候选逐个/小批解码，记录每个prompt的marginal mask gain和重复mask惩罚。
11. **不应照搬**：confidence heatmap+采点本身与当前范式相同，不能作为新Idea。
12. **复现依赖**：SAM checkpoint、COCO/LVIS类通用数据与作者predictor权重；Apache-2.0。

## S06 RSPrompter

1. **研究问题**：在遥感实例分割中自动学习SAM prompt embeddings，避免人工点/框。
2. **完整架构**：SAM ViT-H+FPN；anchor版为Mask R-CNN ROI prompt head，query版为Mask2Former query head；二者调用SAM mask decoder。
3. **Prompt 张量/数据流**：query版`[B,Nq,C]→[B*Nq,Np,256]` sparse embeddings；可选`mask_pred_plus`经SAM mask embed形成dense prompt；图像embedding按Nq复制。
4. **训练目标**：anchor assignment/ROI losses或Hungarian set losses；class`C+1`、mask/Dice及auxiliary mask losses。
5. **推理**：anchor版内部RPN/box detector，query版直接set prediction；无人工/GT prompt。
6. **关键消融**：anchor vs query、prompt point数、decoder-plus/pre-mask、不同SAM规模和adapter设置。
7. **失败案例**：小密集遥感对象、query内存和ViT-H计算；latent embedding可被decoder忽略，论文未做shuffled prompt因果测试。
8. **计算开销**：query版显式复制image embedding到`B*Nq`，README也提示显存更高；anchor版依赖检测两阶段。
9. **与当前项目差异**：RSPrompter把每个候选变成独立query/mask并有背景类；当前只有一个query。
10. **可迁移机制**：小K query head、`C+1` objectness、dense micro-mask可选、query-specific SAM-IoU。
11. **不应照搬**：完整Mask2Former/Mask R-CNN或ViT-H；对单类tiny target过重。
12. **复现依赖**：MMDetection/MMEngine、HuggingFace SAM ViT-H、遥感实例数据；Apache-2.0。

## S07 UN-SAM

1. **研究问题**：跨多个细胞核域做统一prompt-free分割，避免逐域人工prompt。
2. **完整架构**：SPGen产生coarse probability map；其作为dense mask prompt；固定domain mask queries经Domain-Query Decoder细化。
3. **Prompt 张量/数据流**：coarse `output_prob[B,1,H,W]`逐图进入PromptEncoder masks；`domain_num+1`个`[256]` mask queries与空sparse embedding拼接。
4. **训练目标**：coarse/fine segmentation和domain-aware目标；代码返回fine masks与coarse map联合监督。
5. **推理**：只输入图像/域设置，不用点框mask。
6. **关键消融**：SPGen、domain queries、query decoder与泛化域组合。
7. **失败案例**：domain标签/预定义query限制开放域；相邻核粘连和小核仍由coarse mask决定。
8. **计算开销**：一个coarse branch+固定少量query，明显轻于实例set decoder。
9. **与当前项目差异**：用dense pre-mask而非Top-K点，但仍非一候选一query/no-object。
10. **可迁移机制**：高分辨率micro-mask作为prompt；coarse→fine深监督。
11. **不应照搬**：domain-specific queries和整图语义mask；不能解决多候选互相污染。
12. **复现依赖**：SAM、多个nuclei数据、域split；MIT。

## S08 De-LightSAM（任务书误称 ESP-MedSAM）

1. **研究问题**：用轻量、模态解耦结构在多医学域做自动分割。
2. **完整架构**：SemiTViT image encoder + patch decoder/pre-mask dense prompter + 7 learned mask queries + query-decoupled mask decoder；另有蒸馏设置。
3. **Prompt 张量/数据流**：`[B,256,64,64]→patch_map[B,1,32,32]`，detach/0.5二值化并上采样后投影为`[B,256,64,64]`，直接加到mask source；7×256 learned queries作为tokens。
4. **训练目标**：fine mask、patch map和knowledge distillation相关目标。
5. **推理**：只输入图像与domain/class序号；原生PromptEncoder仅生成empty embeddings。
6. **关键消融**：self-patch prompt、query-decoupling、轻量encoder与distillation。
7. **失败案例**：二值pre-mask被detach，错误位置不能由最终mask梯度直接纠正；固定query无no-object matching。
8. **计算开销**：轻量encoder；7 queries固定开销，目标是比完整SAM更轻。
9. **与当前项目差异**：dense pre-mask真实进入decoder，但reliability/对象隔离仍不足。
10. **可迁移机制**：micro-mask残差直接进入image source；固定小K query可作最小原型。
11. **不应照搬**：0.5硬阈值+detach和domain id；会放大弱小目标漏检。
12. **复现依赖**：SAM/teacher checkpoints、多医学域数据；Apache-2.0。

## S09 AutoPrompt-SAM3D

1. **研究问题**：SAM2处理完整3D序列时，自动定位首个可靠slice并阻止错误prompt沿序列传播。
2. **完整架构**：sequence partition→Target-aware Filter→三层feature Automatic Prompt Generator→supervised Confidence Frame Filter→SAM2 propagation。
3. **Prompt 张量/数据流**：SAM2多层feature融合成slice prompt/pseudo-mask；frame score选置信slice，再写入SAM2 video state并双向/序列传播。
4. **训练目标**：prompt generator和confidence frame/window filter由tumor mask或pseudo-mask监督。
5. **推理**：完整CT/MRI volume；无需人工prompt，但依赖多slice和多阶段SAM2传播。
6. **关键消融**：三层vs末层feature、confidence filter、target-aware invalid-window filter和full-sequence策略。
7. **失败案例**：首个有效slice评分错误会传播；非连续/极少slice病灶可能被window filter拒绝。
8. **计算开销**：多slice feature提取、筛选和SAM2 memory传播，远高于单图协议。
9. **与当前项目差异**：它显式过滤错误prompt和无目标窗口；当前只按view support排序，且不让mask response复核。
10. **可迁移机制**：presence/window gate、prompt传播前confidence check、错误prompt不进入状态。
11. **不应照搬**：把独立IR图伪装为视频/volume；不能借序列信息与单帧baseline比较。
12. **复现依赖**：SAM2、3D volume预处理、target-aware/filter checkpoints；文章OA但未定位指定代码仓库。

## S10 EviPrompt

1. **研究问题**：在不训练模型的情况下，用少量参考标注和证据不确定性自动生成可靠正负prompt。
2. **完整架构**：reference feature正/负prototype→target多增强feature evidence→Dempster式意见融合→patch-wise正负点→SAM→mask-derived box refinement。
3. **Prompt 张量/数据流**：三视图每像素产生两类evidence，转换belief/uncertainty并融合；各取正/负patch点，拼成坐标/label数组；首轮mask再转box。
4. **训练目标**：training-free，无参数优化。
5. **推理**：需要带mask reference；target图多视图和至少两轮SAM。
6. **关键消融**：evidential融合、augmentation数量、正负点与refinement；论文比较不同shot/器官。
7. **失败案例**：reference域差、prototype污染、极小ROI在SAM低分辨feature上不可分。
8. **计算开销**：多增强feature+FAISS/原型相似+多轮SAM；无需训练但不是低延迟。
9. **与当前项目差异**：它同时建模正/负证据和不确定性；PR #3只生成正点。
10. **可迁移机制**：target/background belief与abstention；用Brier/ECE/risk-coverage评估。
11. **不应照搬**：reference mask与多轮box refinement不满足当前image-only部署。
12. **复现依赖**：SAM、reference masks、FAISS/增强代码；仓库无根许可证。

## S11 AlignSAM

1. **研究问题**：通过RL代理自动把SAM对齐到显著、阴影、模糊、玻璃等开放上下文任务。
2. **完整架构**：VLM产生task attention/初始状态；actor-critic逐步选点；SAM mask/probability更新环境。
3. **Prompt 张量/数据流**：agent从候选动作中选物理点；标准point encoder→SAM decoder；上一轮probability map和VLM attention构成下一状态。
4. **训练目标**：actor PPO-style clipped objective、critic value loss；mask与GT的目标前景得分为reward。
5. **推理**：任务文本/VLM+多轮agent+SAM，无GT。
6. **关键消融**：RL branch、VLM guidance、iteration count和不同开放任务；迭代并非越多越好。
7. **失败案例**：大/多对象场景可能覆盖不完整；初始attention错误导致策略局部最优。
8. **计算开销**：每轮都要SAM推理；agent/VLM额外计算，正文未给可直接迁移到IRSTD的低成本配置。
9. **与当前项目差异**：prompt credit由mask reward直接学习，且反馈闭环；当前reliability在decoder之后消失。
10. **可迁移机制**：低成本离线marginal prompt credit、remove-one候选反事实，不必先上完整RL。
11. **不应照搬**：VLM+PPO+多轮在线优化；成本和GT reward训练都过重。
12. **复现依赖**：SAM、VLM、任务数据、RL环境；论文/补充已读，官方仓库404。

## S12 PromptPilot（待 PDF 补齐）

1. **研究问题**：few-shot条件下联合优化语义一致性、物理覆盖和每个点的边际贡献。
2. **完整架构**：DINOv2特征匹配初始化点图；Feature Agent与Physical Agent提议remove/restore；Manager Agent仲裁；SAM反馈。
3. **Prompt 张量/数据流**：代码state是点图、feature/position/statistics；action编码含操作、正负、坐标和权重；最终是标准正负点集合。
4. **训练目标**：代码包含DQN/manager目标；环境用SAM预测与GT mask计算DSC/global reward及局部marginal contribution。论文精确公式待PDF。
5. **推理**：README称DINOv2和SAM冻结、迭代优化点；是否仍需support mask及停止策略细节待正文。
6. **关键消融**：待PDF，当前不得引用README以外的表格数字。
7. **失败案例**：待PDF；代码层面动作空间随最大节点数增长，且没有显式no-object action。
8. **计算开销**：多agent多step SAM评价，预计明显高于一次decoder；精确值待PDF。
9. **与当前项目差异**：它显式计算点的边际mask贡献并允许删除有害点。
10. **可迁移机制**：离线leave-one-out/候选drop作为credit label；训练轻量gate后单次推理。
11. **不应照搬**：多agent RL整套系统和GT Dice在线环境；不符合当前低成本单图部署。
12. **复现依赖**：DINOv2、SAM、few-shot reference、agent checkpoints；代码无根许可证，论文PDF待用户提供。

## S13 H-SAM

1. **研究问题**：prompt-free医学适配中，单层SAM decoder不能同时利用先验mask和细粒度结构。
2. **完整架构**：image encoder LoRA + empty/default prompt + stage-1 decoder + class-balanced mask-guided self-attention + stage-2 hierarchical decoder + deep supervision。
3. **Prompt 张量/数据流**：无点框；stage-1 probability/mask attention作用于第二阶段transformer和pixel decoder；两阶段mask最后ensemble。
4. **训练目标**：每阶段deep supervision，Dice/CE类mask损失，stage权重随训练变化。
5. **推理**：单图、无人工prompt，两个decoder stage。
6. **关键消融**：hierarchical stages、learnable mask attention、class-balanced attention、decoder layer数和LoRA rank。
7. **失败案例**：stage-1把背景误作器官时stage-2可继续强化错误；supplement展示了该类例子。
8. **计算开销**：双decoder但参数比加深单一SAM decoder更有效；精确表格见论文Table 9/10。
9. **与当前项目差异**：它直接适配mask response而非只改prompt位置，支持Q3优先测decoder sensitivity。
10. **可迁移机制**：stage-1 micro-mask作为stage-2 condition；prompt/query dropout测试response是否依赖condition。
11. **不应照搬**：两套完整医学decoder和整图class-balanced attention；没有candidate/no-object。
12. **复现依赖**：SAM ViT-B、医学预处理、LoRA checkpoints；MIT。

## S14 Semantic AutoSAM（待 PDF）

1. **研究问题**：将SAM的手工二值prompt改为由图像自动生成语义prompt embedding。
2. **完整架构**：摘要可核验的是以轻量cross-attention替换/重载manual prompt encoder，并从image feature预测prompt embeddings。
3. **Prompt 张量/数据流**：具体query数、shape与是否经过原生prompt encoder待PDF/代码。
4. **训练目标**：待PDF；不能从摘要编造。
5. **推理**：摘要明确无需manual prompting。
6. **关键消融**：待PDF。
7. **失败案例**：待PDF；摘要报告其在FLAIR上仍低于MobileSAM+GT box上界。
8. **计算开销**：轻量cross-attention，精确参数/FLOPs待PDF。
9. **与当前项目差异**：latent prompt由图像直接预测，但未证明对象级隔离/拒绝。
10. **可迁移机制**：只作为“image→prompt embedding已有先例”的新颖性边界。
11. **不应照搬**：在证据不完整前不能据此设计shape或宣称复现。
12. **复现依赖**：正式4页PDF、任何supplement/作者代码；请用户补齐。

## S15 SAM-SPL（待 PDF）

1. **研究问题**：在多红外平台上统一保持tiny detail与高层context，纯图像生成self-derived prompts。
2. **完整架构**：SAM2 adaptor consult-guide encoder + shallow self-prompt generator双向交互 + mutual calibration skip decoder。
3. **Prompt 张量/数据流**：代码中浅层dense features与SAM stage features相加进入`pmt_blocks`；输出`sam_backbone_embeds`与三层`dense_embeds`，不是物理点/框。
4. **训练目标**：公开训练以IRSTD mask监督；论文精确loss与权重待PDF。
5. **推理**：`testing.py`只把图像送入model；无GT point/box/mask。
6. **关键消融**：待正式PDF；README结果不能替代论文表格/消融。
7. **失败案例**：待PDF；代码没有显式candidate/no-object，因此多实例归因与空目标是可预见但尚未被论文证实的边界。
8. **计算开销**：Tiny/Small/Large配置存在；精确参数/FLOPs待PDF。
9. **与当前项目差异**：比PR #3更早在encoder多层融合浅层prompt，但没有独立candidate query。
10. **可迁移机制**：作为强纯视觉baseline；浅层feature可供micro-mask query读取。
11. **不应照搬**：仅抽取`pmt_generator`加到现有模型会是模块堆叠，且许可未声明。
12. **复现依赖**：SAM2 checkpoints、四套IRSTD数据；代码可运行边界需实测，正式PDF请用户补齐。

## S16 IR-SAM2（策略排除）

1–12. 来源属于本轮明确排除的MDPI，不读取/转述其正文机制、消融和结果，也不把任务书里的摘要性描述当证据。若用户明确要求独立补充，将在单独附录记录，不影响主排名。

## S17 TEP-SAM

1. **研究问题**：单帧低SNR目标不可见时，用短时序中“逐渐涌现”的目标—背景运动差异产生SAM prompt。
2. **完整架构**：frozen/PEFT SAM encoder + Discrepancy-Enhanced Temporal Encoder（global-local temporal modeling）+ Temporal Prompt Generator（TP-Gen）+ SAM decoder。
3. **Prompt 张量/数据流**：邻帧feature建背景运动均值与局部差异；融合后产生固定数量temporal query tokens（论文最佳为4），与SAM output tokens共同解码。
4. **训练目标**：最终mask损失+temporal auxiliary loss，稳定时序feature学习。
5. **推理**：短时窗图像序列；不读GT prompt。
6. **关键消融**：temporal encoder、TP-Gen、aux loss、global model、query数和window length；过多query会引入背景并提高Fa。
7. **失败案例**：快速场景变化、非相干背景运动、极低SNR与强动态背景；论文明确承认效率仍可提升。
8. **计算开销**：supplement Table 7给参数/GFLOPs/FPS；时序窗口与多帧encoder使成本高于单帧。
9. **与当前项目差异**：query是时序状态而非单帧坐标点；协议不同，不能把flip/scale多视图称为temporal emergence。
10. **可迁移机制**：固定少量query、背景运动/全局背景作为negative context、query数—Fa联合选择。
11. **不应照搬**：多帧数据与运动模块；当前单帧实验必须独立报告。
12. **复现依赖**：M-IRSTD序列、SAM-B/相关checkpoint、时序窗口预处理；本轮未审计代码。

## S18 SAM-DAQ

1. **研究问题**：SAM2原始prompt与memory attention在RGB-D视频显著目标中缺少深度引导和跨帧自适应query。
2. **完整架构**：Parallel Adapter-based Multi-modal Image Encoder + Depth-guided learnable embedding generator + Query Adaptive Module + SAM2 memory/mask decoder + intermediate supervision。
3. **Prompt 张量/数据流**：RGB/depth多层feature生成frame-level与video-level queries，查询在帧间update并作为learnable embeddings与SAM2 memory共同驱动decoder。
4. **训练目标**：最终预测损失`Lpred`+加权intermediate loss`Linter`。
5. **推理**：RGB-D视频流，无GT prompt；依赖depth与memory state。
6. **关键消融**：multi-modal adapter、query generator、frame/video query数、query hidden dimension、update机制与intermediate supervision。
7. **失败案例**：depth噪声、遮挡/背景干扰；过多video queries会引入背景。
8. **计算开销**：冻结SAM2 encoder主体，以adapter/query模块为主；视频memory仍有序列成本。
9. **与当前项目差异**：query是可更新的跨帧状态并被decoder消费；当前候选score只影响排序。
10. **可迁移机制**：候选query在得到首轮micro-mask response后更新；frame query和global background query分工。
11. **不应照搬**：depth/video输入、长期memory和显著对象协议。
12. **复现依赖**：SAM2-L、RGB-D视频数据和depth预处理；本轮未定位任务书指定代码。

## 跨论文可执行结论

- **第一诊断不是继续训prompt head，而是测decoder是否对prompt内容敏感。** SPARK-SAM、IP-SAM、H-SAM都显示response path可能是瓶颈。
- **一候选一query、`∅`和micro-mask可以用小K实现。** MaskSAM/RSPrompter给出完整set版本，Sam2Rad给出box+mask+latent联合状态；TIRST-SAM不必复制重型3D/Mask2Former骨架。
- **背景必须是一等prompt state。** IP-SAM、EviPrompt和Memory-SAM的共同点不是“加负点”，而是让前景与背景条件保持可区分，直到decoder或拒绝器。
- **mask feedback应先做离线credit诊断。** AoP-SAM/AlignSAM/PromptPilot说明有害prompt可以通过mask响应识别；先做candidate-drop/leave-one-out，只有存在稳定信号才训练adapter或policy。
- **停止旧路线**：更换显著性算子、调Top-K/NMS、再加几种deterministic views、把support/dispersion喂MLP、所有正点塞一个query、global文本向量直接投token。
