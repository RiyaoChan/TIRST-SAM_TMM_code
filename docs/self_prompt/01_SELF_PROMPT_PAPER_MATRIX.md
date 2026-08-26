# Self-Prompt 论文机制矩阵

本矩阵优先回答“prompt 到底是什么、如何被 decoder 消费、能否拒绝错误候选”，而不是按论文自称的 `self-prompt` 分类。`Y/N/P/?` 分别表示是、否、部分、无法由现有全文/代码核验。`GT-train` 只描述训练监督，不等于推理泄漏。

## 1. S 级：Prompt 表示与 decoder 接口

| ID | Paper / Domain | Prompt source | Prompt representation | S/D/Q | 坐标 | 一候选一 query | no-object / background | reliability / uncertainty | Prompt encoder | Decoder adaptation | 迭代反馈 |
|---|---|---|---|---|---:|---:|---|---|---|---|---|
| S01 | SPARK-SAM / 单帧 IRSTD | 图像→目标性/框/局部 token；训练期 response guidance | 正点、可选负环、box、local tokens、dense residual | S+D+Q | Y | P：候选可分批解码，但仍从峰值候选起步 | objectness + 可选背景负点；无 DETR 式 no-object | candidate gate、SAM-IoU、校准与 response quality | 原生 SAM2 prompt encoder | response adapter、local token、high-res refinement、校准阶段 | 多阶段离线 response knowledge，正常推理单次主路径 |
| S02 | IP-SAM / 伪装目标 | SPG 从图像特征生成前景/背景连续 mask logits | 正/负 intrinsic dense prompt | D | N | N：语义前/背景状态，不是实例 set | **Y：显式背景 prompt，PSG 非对称抑制** | gate 隐式表达背景风险，无校准概率 | **Y，冻结原生 SAM2 prompt encoder** | task mask decoder + LoRA + PSG | N |
| S03 | MaskSAM / 3D 医学 | 辅助 mask branch + learnable queries | 每 query 的 binary pre-mask + box + classifier token | S+D+Q | box=Y | **Y** | **Y：类别集含 ∅/no-object** | class probability + mask quality；无显式 UQ | **Y：mask/box 进入原生 encoder** | classifier token、3D adapter、query mask decoder | N |
| S04 | Sam2Rad / 医学 | 多层图像特征与 class queries cross-attention | box、mask、latent embeddings | S+D+Q | box=Y | P：每类/每 ROI query，非实例 set matching | SAM2 no-object machinery有限；无显式 background prompt | IoU/质量头；无系统 UQ | **Y：预测 box/mask 进入原生 encoder** | PPN + 可训练 decoder/adapter | N |
| S05 | AoP-SAM / 通用图像 | 图像+SAM embedding→置信图 | 自动正点；SAM mask feedback 筛点 | S | Y | 点批次/AMG，不是对象 query | 过滤后可零 mask；无训练 no-object | 置信图、pred-IoU、stability、mask overlap | Y：标准点 prompt | SAM 冻结；外置 prompt predictor/filter | **Y：粗到细 mask 反馈筛 prompt** |
| S06 | RSPrompter / 遥感实例 | detector anchor ROI 或 Mask2Former queries | learned sparse prompt embeddings；query 版可带 pre-mask dense prompt | S+D+Q | N（anchor proposal 有 box，但 decoder prompt 是 latent） | **Y：query 版每 query 独立 mask** | **Y：Mask2Former `num_classes+1`** | query class score；无 UQ | N：latent 直接作为 sparse embeddings；仅借 no-mask/mask embed | detector/query head + SAM mask decoder | query mask 反哺 transformer attention |
| S07 | UN-SAM / 细胞核 | coarse SPGen + domain mask queries | coarse mask dense prompt + domain queries | D+Q | N | N：domain query，不是实例 query | 有 domain/background分类但非显式 abstention | 无校准 UQ | Y：coarse mask 进入 mask prompt encoder | DQDecoder/domain query | 两阶段 coarse→fine |
| S08 | De-LightSAM / 泛化医学 | patch decoder 从图像特征生 binary pre-mask | dense patch prompt + 7 learned mask queries | D+Q | N | P：固定 mask queries，非实例匹配 | domain/class query；无 no-object | 无显式可靠性 | 仅调用空 prompt；dense pre-mask直接加 feature | query-decoupled decoder | N |
| S09 | AutoPrompt-SAM3D / 3D 医学 | SAM2 三层 feature→prompt，confidence frame filter | frame/slice级 mask/point提示与传播状态 | S+D | P | N：序列 prompt，不是对象 set | target-aware invalid-window filter | **Y：confidence frame与target-aware filtering** | 经 SAM2 prompt/video接口 | 生成器与筛选器，SAM2 sequence propagation | **Y：跨切片传播与筛选** |
| S10 | EviPrompt / 少样本医学 | reference mask 的正/负 feature prototypes；三增强证据融合 | 正点+负点，后续 box refinement | S | Y | N：多点同一 SAM 调用 | **Y：正/负证据** | **Y：evidential belief/uncertainty** | Y：点/box | SAM 冻结 | **Y：首轮 mask→box→再解码** |
| S11 | AlignSAM / 开放上下文 | VLM attention + RL agent | 连续选择/修正点 prompt | S | Y | N：每任务 agent 顺序决策 | 背景动作/奖励，非显式 no-object class | critic/value 与 mask reward | Y：标准点 | SAM 冻结，训练 agent | **Y：SAM mask/probability 作状态反馈** |
| S12 | PromptPilot / few-shot | 标注reference mask经DINOv2双向patch匹配初始化；feature/physical agents+manager | 正/负点集合与 activate/prune 动作 | S | Y | N：点集合 coalition | 正/负点与删除；无显式 no-object | **Y：SAM DSC、逐点LOO边际贡献、EMA、Q-value** | Y：标准点 | SAM/DINO冻结，三个DQN agent学习 | **Y：每步调用SAM并评价动作** |
| S13 | H-SAM / 医学 | 默认空 prompt + 第一阶段预测 | stage-1 pre-mask/attention引导 stage-2 | D+Q | N | N：语义 mask，不是实例 set | class-balanced attention压背景；无 no-object | 无概率校准 | 默认 embeddings；不需要手工 prompt | **Y：两阶段 hierarchical decoder** | **Y：stage-1 mask→stage-2** |
| S14 | Semantic AutoSAM / 医学 | 图像 feature→cross-attention prompt embedding | learned prompt embeddings | Q/feature | N | ? | ? | ? | 摘要称替换手工 prompt encoder | 轻量 cross-attention；细节待全文 | ? |
| S15 | SAM-SPL / 单帧 IRSTD | `F0/F1/F2`按16/8/4倍下采样并拼接→`1×1 Conv` | `P∈R^(H/16×W/16×256)` dense prompt tokens + 1个learned output token | D+Q/feature | N | N：整体 dense segmentation | 无显式 no-object/background prompt | 无显式UQ/候选置信 | 不走物理 point/box prompt encoder | `P/O`与latent `Z`经two-way Transformer双向交互；MSIC skip decoder | N |
| S16 | IR-SAM2 / 单帧 IRSTD | — | — | — | — | — | — | — | — | — | — |
| S17 | TEP-SAM / 多帧 IRSTD | global-local temporal discrepancy→4 temporal query tokens | temporal prompt queries + enhanced feature | Q | N | N：帧/视频 query，不是对象 set | 显式建模背景运动均值但无 no-object class | temporal discrepancy/aux loss | 论文路径向 SAM token stream注入 | temporal prompt generator + adapter | 跨帧联合，不是 iterative click |
| S18 | SAM-DAQ / RGB-D 视频显著目标 | RGB-D encoder→frame/video adaptive queries | frame queries + video queries + memory prompts | Q | N | N：显著对象级查询集合但非实例匹配 | depth抑制背景；无显式 no-object | query quality隐式；无校准 UQ | SAM2 memory/prompt统一路径 | query adapter + intermediate supervision | **Y：视频 memory/update** |

说明：S16 因来源排除，不从 MDPI 页面提取机制；任务书中的机制描述也不当作独立证据。

## 2. S 级：训练、部署、代码与 IRSTD 相关性

| ID | GT use during training | Test-time dependencies | Code/license | IRSTD relevance | 与当前 PR #3 的碰撞 | Main limitation |
|---|---|---|---|---|---|---|
| S01 | mask监督；response guidance、候选/质量监督均由训练 mask 产生 | 单图；无 GT prompt | 有代码；许可未声明 | **直接** | 高：自动 IRSTD+候选可靠性+response adaptation | 仍以目标性峰值/Top-K为入口；hot clutter/暗目标仍失败；preprint |
| S02 | COD mask监督 | 单图；无文本/GT prompt | 未审计官方代码 | 高：前景/背景提示与虚警抑制 | 中高：paired prompt/decoder消费已被覆盖 | 前背景难分时可能把目标当背景而过抑制 |
| S03 | 类别、binary masks、boxes；Hungarian matching | 单 volume；无人工 prompt | 有代码；许可未声明 | 高：object-set/no-object范式 | 高：一候选一 query/no-object不再是空白 | 3D器官尺度与 IR tiny component差异大；固定 query成本 |
| S04 | segmentation mask；训练可混合 GT-derived manual box/mask prompt | 单图；无 GT prompt | 有代码；许可未声明 | 中高：query→box/mask/token | 中高 | class query非instance set；训练手工 prompt augmentation边界需注意 |
| S05 | prompt predictor用标注；SAM不更新 | 单图，多次/批量 SAM AMG | Apache-2.0 | 中：mask feedback/credit | 中：反馈筛选可迁移，但表示仍是点 | dense AMG代价；置信图仍可能漏 tiny target |
| S06 | detection/instance mask GT；Hungarian/ROI assignment | 图像；无人工 prompt，anchor版依赖内部 detector | Apache-2.0 | 高：query set与实例隔离 | 高：one-query-one-mask/no-object已有强先例 | ViT-H/Mask2Former重；遥感实例不等于点目标 |
| S07 | coarse/fine nuclei masks、domain labels | 单图；无 prompt | MIT | 中：coarse pre-mask/domain query | 中 | query按domain而非candidate；不处理空目标/候选拒绝 |
| S08 | segmentation mask、domain/class | 单图；无 prompt | Apache-2.0 | 中：high-res pre-mask + query | 中 | pre-mask阈值detach，可靠性不进入decoder；非实例 |
| S09 | tumor masks/伪mask，frame confidence与window监督 | 完整 volume，多个 slice/SAM2传播 | 正文OA；未审计官方代码 | 中高：极小病灶+错误prompt过滤 | 中：confidence filtering已有先例 | 依赖3D连续性；不适用于独立单帧协议 |
| S10 | 无训练；reference mask构造正负原型 | **需要一张标注 reference**，多次 SAM | 有代码；许可未声明 | 中：不确定性/正负 prompt | 中 | 非image-only；reference域差/原型错误会传递 |
| S11 | task GT mask用于RL reward | 图像+任务文本，迭代多次 SAM | 论文全文；代码404 | 中：prompt credit/feedback | 中高 | 训练/推理成本高；依赖VLM/text；代码不可复核 |
| S12 | **目标图GT mask用于RL训练的DSC与LOO reward**；reference mask用于初始化 | 推理需要1张标注reference、已训练agent和多步SAM；不需要目标图GT/参数更新 | 全文+代码可读；许可未声明 | 中：marginal credit/harmful prompt | **高：SAM response与逐prompt credit已有直接近邻** | 不是无参考单图；200 episodes×100 steps，多次SAM/LOO代价高；无no-object |
| S13 | stage1/stage2 mask deep supervision | 单图；无 prompt | MIT | 中：mask-response feedback | 中高：response adaptation不必来自点 | 无显式候选/no-object；医学语义分割偏大目标 |
| S14 | mask GT（摘要可推断，细节待正文） | 单图；无人工 prompt | 无代码/全文 | 中 | 中 | 仅4页且证据不全；无法核对 shape/loss/失败案例 |
| S15 | BCE式三尺度mask supervision；浅层prompt与CGKA端到端学习；SAM2 Hiera-Tiny部分冻结 | 单图→最终`Y0`阈值0.5；无GT/文本/人工prompt | 全文+代码可读；许可未声明 | **直接** | **极高：浅层纯视觉dense self-prompt直接基线** | 无object query/no-object/候选级信用；300 epochs、双主干，跨域和实时性仍受限 |
| S16 | — | — | 排除 | 直接但未纳入证据 | 不作结论 | 来源策略排除 |
| S17 | multiframe mask + temporal aux loss | 短帧窗；无 GT prompt | 全文；本轮未审计代码 | 直接但协议不同 | 中：temporal query不可冒充单帧创新 | 依赖相干运动；快速场景变化和效率受限 |
| S18 | video saliency masks、intermediate loss | RGB+depth+video memory | 全文；未审计代码 | 邻近：query/memory机制 | 中 | 多模态视频；无instance/no-object，不能直接用于单帧IR |

## 3. A 级机制筛查矩阵

S 级重复项见上表；这里列独立的 A 级工作。`外部依赖`是部署边界，不代表方法不好。

| ID | Paper | Prompt source / representation | S/D/Q | 坐标/每候选 query | background/no-object/UQ | feedback | GT-train / 测试依赖 | 对 IRSTD 的启示 | 与当前仓库碰撞/限制 |
|---|---|---|---|---|---|---|---|---|---|
| A01 | PMG-SAM | 排除 | — | — | — | — | — | 不纳入 | MDPI策略排除 |
| A02 | LDFSAM | 排除 | — | — | — | — | — | 不纳入 | MDPI策略排除 |
| A03 | Self-Prompt SAM | auxiliary multi-scale masks→box+center | S+D | Y/N | N | coarse→prompt | mask / image-only | pre-mask比单点强 | 仍是单语义mask，非object set |
| A04 | AutoPrompt MedSAM | diffusion class prompt→sparse+dense embeddings | S+D | N/N | uncertainty-aware objective | N | class+mask / class label | 结构化sparse+dense | 公开代码未连接 `diffusion_model`，不可直接复现 |
| A05 | MUP-SAM | MSVM-UNet mask→腐蚀/膨胀、box expansion、NMS→class boxes；另有mask融合 | S | Y/P：多box分别解码 | box分数；零box输出零mask，无learned no-object | **aux mask与MedSAM mask经learned pixel fusion** | 两阶段mask监督 / image-only | box prompt错误可由response fusion补救 | 本质是强aux segmentor→box→冻结MedSAM；不是轻量self-prompt，且融合贡献很大 |
| A07 | PA-SAM | prompt adapter加强point/box与image细节 | S+D | Y/N | N | N | prompt GT / prompt required | 验证decoder sensitivity | 不是自动prompt源 |
| A08 | DVPT | CNN stem生成local dense prompt；SAM四层global-attention features提取GGP | D+Q | N/N | 无background/no-object/UQ | N | mask / image-only；500 epochs | 局部dense+全局learned token双prompt | **强碰撞image→global prompt feature**；绕开native prompt encoder、自定义CNN decoder，非candidate-level |
| A09 | HSP-SAM | hierarchical abstract prompts | Q | N/N | P | hierarchical | mask / image-only | 摆脱物理坐标 | 与UN-SAM/H-SAM近，需object/no-object差异 |
| A10 | SurgicalSAM | class prototypes→dense+sparse class tokens | S+D+Q | N/N | 正/负class embedding，无no-object | N | masks/prototypes / image+learned prototypes | paired prototype prompt | 医疗类别固定；不是实例候选 |
| A17 | UR-SAM | augmented prompts + uncertainty rectification | S | Y/N | **UQ/rectification** | correction | mask+perturbed prompt / prompt | 错prompt鲁棒 | 需要初始prompt；不是image-only生成 |
| A18 | FNPC-SAM | uncertainty定位false negative/positive→纠正点 | S | Y/N | **aleatoric UQ+正负纠错** | iterative | mask / initial prompt | harmful prompt correction | 医学交互式，非自动候选set |
| A19 | AutoPromptSeg | V-Net MC dropout→epistemic/aleatoric uncertainty；`PSS×class probability`→3D NMS Top-K | S | 3D坐标/N | **低不确定度可靠点；每类K=20** | V-Net/SAM-Med3D双分支一致性 | labeled+unlabeled / image-only；1200 epochs、多次MC | UQ必须进入选点且低风险点优于高风险点 | **可靠性采点已拥挤**；医学3D/半监督、MC开销大，摘要与Table 1有数字不一致 |
| A20 | UncertainSAM | post-hoc aleatoric/epistemic/task UQ | — | — | **UQ** | N | 无需重训/已有SAM输出 | 可作可靠性评估器 | 不生成prompt，不解决候选污染 |
| A23 | PPD | attack/defense agents增删正负点 | S | Y/N | harmful prompt score | **Y** | 代码训练环境用GT Dice / 多步SAM | 负点与prompt credit | GT reward/多步成本；论文入口待核验 |
| A24 | PP-SAM | perturbed boxes/points训练 | S | Y/N | robustness | N | GT prompt扰动 / prompt | decoder对误差鲁棒 | 不解决无prompt部署 |
| A25 | Self-Prompting LVM | support mask→pixel classifier→points | S | Y/N | foreground/background classifier | P | reference GT / reference required | 正负像素分类 | 不是无参考部署；近EviPrompt |
| A26 | PerSAM | one-shot target similarity→points+mask refinement | S+D | Y/N | positive/negative points | cascade | reference mask / reference required | target-guided attention | 需要标注参考；相似图→点已覆盖 |
| A27 | Med-PerSAM | reference warping prompt tuning | S/D | Y/N | P | iterative | reference mask / reference required | 几何对齐 | 医学先验强；不适合单帧无参考 |
| A28/A38 | OP-SAM | one-shot correlation prior→Euclidean prompt evolution | S+D | Y/N | P | **iterative** | one reference / reference required | prompt evolution | 与PerSAM/AlignSAM邻近，外部reference |
| A29 | Segment Any Tissue | dual-space cyclic prompt engineering | S | Y/N | 正负点/自动选择 | **cyclic** | one reference mask / reference required | feature+physical双空间 | 无参考IR不可直接部署 |
| A30 | GBMSeg | feature/physical-space prompt optimization | S | Y/N | 正负点 | iterative | reference mask / reference required | 点集联合优化 | 仍依赖few-shot reference |
| A31 | PGP-SAM | support prototype→prompt | S/Q | P/N | foreground prototype | P | support mask / support required | prototype grounding | 近SurgicalSAM/PerSAM |
| A32 | GF-SAM | graph-based point selection/propagation | S+graph | Y/P | relational background | graph propagation | support mask / support required | candidate关系建模 | reference依赖；图关系不是自动新颖性 |
| A33 | Memory-SAM | DINOv3+FAISS retrieval→前/背景点 | S | Y/N | **foreground/background** | retrieval | memory masks / memory required | 显式正负点 | 外部标注memory；不适合作为image-only主线 |
| A34 | ViRefSAM | visual contextual prompt + target alignment adapter | Q/D | N/N | P | reference-target interaction | few-shot refs / refs required | contextual query | 参考图依赖；遥感目标尺度仍大 |
| A39 | ReSAM | prediction→refine/requery/reinforce box+embedding | S+Q | Y/P | quality/reject P | **Y** | mask / image-only iterative | response-conditioned requery | 与Top3-C近；需避免同构 |
| A40 | GPRN | SAM masks→visual prompts→graph reasoning→adaptive points | Q+S | Y/P | graph suppress/consistency | **Y** | support/query masks / few-shot | candidate-context graph | 已覆盖graph+feedback；非单图无监督 |
| A42 | SAM-REF | image-prompt synergy refiner | S/D | 与输入prompt相同 | N | **interaction refinement** | mask+prompt / prompt required | decoder响应适配 | 不是prompt generator |
| A43 | SAM-RSIS | detector boxes→SAM | S | Y/Y（box实例） | detector score/no explicit no-object in SAM | N | box/mask / detector | 强自动实例基线 | 外部detector→SAM，任务书明确不能当主创新 |
| A44 | GeoSAM | task model points + LLM text | S+text | Y/N | P | N | task labels / text+task model | 多源prompt | 外部模型与文本依赖重 |
| A45 | TSP-SAM | motion self-prompt→mask+box/point | S+D | Y/N | temporal consistency | temporal | video masks / video | box比点更稳、运动涌现 | 多帧协议，不能与单帧混评 |
| A47 | SPT | coarse anomaly draft→relation refinement | D/Q | N/N | background relation | two-stage | mask / image-only | draft+关系 | 非SAM核心prompt，近pre-mask范式 |
| A48 | SACM | prompt-free dual-level adapters + dual-stage masks | D/Q | N/N | curvilinear background | two-stage | mask / image-only | prompt-free响应适配 | 结构域完全不同；不是object set |
| A49 | OFL-SAM2 | online few-shot target feature learner→prompt-free SAM2 | Q/D | N/N | target/background P | online memory | support/online target / few-shot | 在线target state | 外部在线样本，部署协议不同 |
| A50 | S4M | GT/人工 extreme 或 major/minor 四点 | S+relation | Y/N：单实例4点 | role-specific type embeddings；无no-object/UQ | 训练可加2轮error-region refinement；Canvas为训练only | GT mask模拟点+convex-hull Canvas / 测试仍需人工4点 | prompt应保留角色与几何关系 | **不是self-prompt部署**；但证明把所有点当同类型会产生表示冲突，限制TB/多点创新表述 |

## 4. 机制结论

1. **“图像生成 prompt”不是创新空白。** 单图 image-only 路线已覆盖置信图点（AoP-SAM）、pre-mask（UN-SAM/De-LightSAM/H-SAM）、learned latent prompt（Semantic AutoSAM）、object query（MaskSAM/RSPrompter/Sam2Rad）和直接 dense self-prompt（SAM-SPL）。
2. **真正与 PR #3 不同的最小单位是 candidate state，而不是更好的 heatmap。** 最有价值的共同结构是：每候选独立 query/mask、显式 background/no-object、candidate score在 decoder 内仍存在，以及用 mask response 反向检验 prompt。
3. **原生 prompt encoder 是否被使用必须单独报告。** IP-SAM 的消融表明“连续前/背景状态经过冻结 prompt encoder”与简单 feature addition 不等价；RSPrompter/De-LightSAM又证明可以绕开坐标 prompt。实验必须直接测 prompt-to-mask sensitivity，不能只看 prompt recall。
4. **单帧 IRSTD 的空白不是 TTA consistency。** SPARK-SAM 已把自动IR提示与response adaptation结合；SAM-SPL已覆盖浅层纯图像self-prompt；IP-SAM已覆盖前景/背景非对称提示。可发表空间要同时满足对象级隔离、拒绝/背景、decoder消费和tiny-target证据。
5. **“可靠性选点”与“全局图像prompt”也不是空白。** AutoPromptSeg已系统比较低/高不确定度及random/center/grid点；DVPT已从多层全局图像特征生成learned prompt tokens。新方法必须证明可靠性/语义状态被decoder消费并产生可归因的mask变化。
6. **结构化点的角色不能在decoder前丢失。** S4M的消融表明共享positive embedding会与极值点角色冲突；当前把全部候选塞进`[B,1,K,2]`不仅混对象，也混角色。即使不采用交互式4点，其role-aware结论仍支持candidate identity与target/background type分离。
