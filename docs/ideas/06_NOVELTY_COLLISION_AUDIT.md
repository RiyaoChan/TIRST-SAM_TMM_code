# 新颖性冲突审计

审计对象是 `05_TIRST_SAM_IDEA_CANDIDATES.md`。结论仅基于本轮 P01–P50 和可核验公开代码，不等于最终全领域 novelty search；论文立项前仍需检索 2026-08-13 之后工作与目标投稿期最新文献。

## 1. 原始两个想法的判定

| 原始想法 | 最直接先行工作 | 冲突等级 | 可保留角色 |
|---|---|---|---|
| GPT 替代 Qwen 生成更准确描述，再把文本 feature 送入 SAM | SAIST（自动 caption+CLIP+SAM）、DGSPNet（固定+逐图 I2T semantics）、EVF-SAM/RS2-SAM（图文 prompt） | 高 | teacher/data quality ablation，不作为主创新 |
| 训练 CLIP/图像编码器拟合 GPT 文本 feature，推理无文本 | DGSPNet、JinSight、SeViL、MoPKL、AlignEarth、CLIPSelf、PromptKD | 高 | image-only student baseline；若升级为 downstream behavior distillation 才有研究空间 |

## 2. 候选 Idea 冲突矩阵

等级含义：`高`=问题+接口+训练目标已有近邻，换模块名不足；`中`=组合或任务化差异存在，但必须由专门实验支持；`低`=本轮范围内无直接同构机制，仍需扩展检索。

| Idea | SAIST | SAM-SPL / IRSAM | DGSP / JinSight / SeViL | READ / Simple-ViL / RS2 | SPD / prompt works | Distillation works | 综合 | 需要守住的实质差异 |
|---|---|---|---|---|---|---|---|---|
| 1 多视图可靠性+拒绝 | 中：错文未测 | 中：均无显式 abstain | 中：SeViL 过滤伪标签 | 中：均会强制产生 prompt | 高：SPD consensus；AutoPrompt-SAM3D/RL-CP | 中 | **中高** | 单帧多视图、tiny prompt recall、risk calibration、文本反事实与无外部 noisy prompt |
| 2 prompt→mask 行为蒸馏 | 中 | 中 | 高：语言训/图像测 | 中 | 高：SAM-COD | 高：SimIR/PromptKD/EdgeSAM/MS-SAM-LESS | **中高** | 仅蒸馏“正确文本相对 no-text”的反事实函数增量、错误文本安全边界和 tiny component ranking |
| 3 高分辨率残差语义定位 | 中：CG-SAM | 高：IRSAM/SAM-SPL | 中：DGSP TGSA | 高：dense map 已有 | 低 | 中：local distill | **中高** | 文本只校准不定位；tiny-area prompt recall；普通 FPN 等容量控制 |
| 4 文本候选验证器 | 中：更像 Fa 抑制 | 中 | 高：SeViL TAPF | 低 | 中 | 中 | **中** | 单帧 IR component-level verifier + image-only distill，而非 pseudo-label 过滤改名 |
| 5 反事实文本安全性 | 低 | 低 | 中 | 中：READ false premise | 中：noisy prompt | 低 | **中低** | correct/shuffle/wrong/drop 全协议和 safe fallback；不能只有 text dropout |
| 6 角色化 slot-set student | 中 | 低 | 高：DGSP/JinSight | 中：multi-token | 低 | 高 | **高** | uncertainty/presence slot、可缺失 set matching 与 role-specific downstream action |
| 7 自适应 prompt 预算/停止 | 低 | 中 | 低 | 中 | 高：PPO/SAM-SP | 低 | **中** | 无 RL 重模块、允许 zero prompt、以 Pd–Fa–算力联合风险作动作目标 |
| 8 多视图纯图像 prompt | 低 | 高：self-prompt | 中：纯图像部署 | 低 | 中 | 低 | **中高** | 视图差异涌现、坐标一致性；更适合作为强 baseline |
| 9 tiny-aware CLIPSelf | 中 | 低 | 中 | 中 | 低 | 高：CLIPSelf/AlignEarth | **高** | tiny candidate curriculum/hard negatives；只适合辅助预训练 |
| 10 presence-location 解耦 | 中 | 中 | 高：SeViL | 中 | 高：SegEarth-OV3/MaskSAM | 低 | **中高** | risk-calibrated abstention 与极小目标 Pd–Fa，而非普通分类辅助头 |

## 2.1 十个机制维度 × 十三篇直接近邻

下表不是用“像/不像”的单一分数替代阅读，而是把指南指定的十三篇工作作为列逐项核对。`—` 表示论文没有该机制；`未核`表示当前可得材料不足，不能作肯定判断。这里“同问题设定”指其原始任务是否直接覆盖“红外小目标、自动语义/提示、测试无 GT prompt”三项，而不是仅看是否使用 SAM。

| 碰撞维度 | SAIST | SAM-SPL | DGSPNet | JinSight | SeViL | IRSAM | SimIR | EVF-SAM | READ | Simple-ViL | RS2-SAM 2 | SegEarth-R2 | AlignEarth |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 同问题设定 | IR 小目标+自动文本+SAM | IR 小目标+自提示 SAM | IR 小目标+语言引导 | IR 小目标+语言训/图像测 | 运动 IR 小目标+语言 teacher | IR 小目标+prompt-free SAM | IR 小目标+轻量 student | 通用 referring segmentation | reasoning segmentation | 医学类别文本分割 | 遥感 referring segmentation | 遥感 referring segmentation | 遥感图文表征/分割 |
| 文本来源 | 自动 caption→模板 | — | 固定模板+I2T token | 结构化多任务指令 | GPT-4o 固定格式 | — | — | 人工 referring expression | 问题/表达 | 类别名 | referring expression | referring expression | 遥感图文对/类别语义 |
| 融合位置 | SR-CLIP+SAM decoder | encoder/skip 自提示调制 | TGCA/TGSA | 语言预训练视觉编码器 | teacher TATE/TAPF/ACM | encoder+decoder | teacher→student 表征 | BEIT-3 early fusion→SAM | LMM token 层→SAM | attribution+双向交互 decoder | BEIT-3+BHFM | `[SEG]` token attention | 全局/局部视觉—语言对齐 |
| SAM prompt 形式 | 文本/视觉 prompt token | 浅层 dense self-prompt | 非 SAM | 部署非 SAM 文本 prompt | student 部署非文本 prompt | 无外部 point/box；mask/edge token | 学习 sparse/dense query | 多模态 `[CLS]` sparse token | similarity 正负点+语义 token | attribution dense map | pseudo-mask dense+多模态 sparse | `[SEG]` attention/语义 token | 非直接 SAM prompt |
| 空间化方式 | CG-SAM/视觉特征承接语义 | 浅层高分辨率 prompt | channel/spatial cost map | 视觉检测/分割头 | 运动候选+文本筛选 | WPMD/PMD+GAD | 多尺度 query→mask | early-fused global token 由 SAM 解码 | patch similarity→DtoC points | attribution→dense prompt | pseudo mask+双向层融合 | token-to-pixel attention supervision | local feature/pixel alignment |
| 蒸馏对象 | 文本/视觉 prompt 分布 MMD | — | 对比+重建预训练，非 teacher KD | 语言语义进入视觉表征 | language teacher→纯图像 student | — | logits、通道、query/表征 | — | — | — | — | attention/segmentation supervision，非 teacher KD | global/local dense feature distillation |
| 训练/推理依赖 | 训/测均用逐图文本；caption 可离线 | 训/测仅图像 | 训/测图像+固定语义/I2T | 训练语言；推理图像 | 训练语言 teacher；推理图像 student | 训/测图像 | 训练 teacher；推理图像 student | 训/测需文本 | 训/测需文本/LMM | 训/测需类别文本/map | 训/测需表达 | 训/测需表达 | 训练图文对齐；下游依任务 |
| 小目标/细节机制 | CG-SAM 背景抑制 | 浅层高分辨率+skip calibration | 多层细粒度 I2T/TGSA | LSI 全局—局部交换 | 运动特征+候选筛伪 | WPMD 保细节+多粒度解码 | 多尺度 query+轻量 backbone | 双分辨率 VLM/SAM | patch similarity 与点采样 | SAM affinity refinement | 多层双向融合+边界建模 | pixel-level attention | local dense distill/尺度对齐 |
| 主要 loss/监督 | 分割损失+prompt 分布 MMD | 代码有任务 loss；正文细项未核 | 分割+图文对比/重建 | 多任务语言预训+下游任务监督 | teacher 语言/候选监督+student 任务监督 | mask/edge Dice+BCE | 分割+logit/通道/query KD | mask BCE/Dice 类监督 | 语言任务+mask supervision | 分割监督 | mask+边界/多层监督 | segmentation+attention supervision | global/local distillation objectives |
| 论文主要实验 claim | 文本缩小 IR 语义鸿沟 | 浅层 self-prompt 改善细节 | 逐图语义提升 IR 检测 | 语言预训可图像-only 部署 | 语言 teacher 改善纯视觉运动 IR student | 无外部 prompt 的 IR SAM | 通用 teacher 压缩为高效 IR student | early fusion 优于 late fusion prompt | LMM 相似度可转为空间点 | 简单文本+稠密 attribution 有效 | dense+sparse 图文 prompt 协同 | attention supervision 改善 grounding | global/local 蒸馏改善 dense 遥感表征 |

直接结论：原始“GPT caption 更准”与 SAIST 重叠；原始“图像拟合文本 feature”与 DGSPNet、JinSight、SeViL、SimIR、AlignEarth 形成机制簇；“相似图转点/稠密提示”已有 READ、Simple-ViL、RS2-SAM 2；“prompt-free/自提示”已有 SAM-SPL、IRSAM。因而剩余可检验空间主要在**错误语义下的风险校准、允许拒绝的自动 prompt，以及文本相对 no-text 的反事实行为蒸馏**。

## 3. 禁止作为“本文首次”的 claim

- 首次把 CLIP/文本引入红外小目标 SAM。
- 首次自动生成红外图像描述并用于分割。
- 首次在推理阶段不使用人工点/GT 点。
- 首次让图像分支产生或拟合文本特征。
- 首次训练时用语言、推理时只输入图像。
- 首次把文本 token 投影成 SAM sparse prompt。
- 首次把图文相似图转换为 points/dense mask prompt。
- 首次用高分辨率/浅层特征生成 SAM self-prompt。
- 首次通过 prompt distillation 或 local feature distillation 去除文本依赖。
- 首次过滤/拒绝自动 SAM prompt（AutoPrompt-SAM3D、SAM-COD、SPD 已有直接机制）。
- 首次进行 prompt-in-the-loop 或 prompt→mask 决策蒸馏（EdgeSAM、SAM-COD、MS-SAM-LESS 已有近邻）。

## 3.1 指南外的高冲突补充文献

| 工作 | 直接冲突 | 对方案的约束 |
|---|---|---|
| IFP, CVPR 2026 Findings | text + DINOv3 focus→similarity→迭代 structural points | Idea 3 不得把“结构化文本点”本身作为创新 |
| AutoPrompt-SAM3D, BMC Bioinformatics 2026 | 多层 SAM2 自提示 + confidence filtering + non-salient pre-screen | Idea 1 必须用单帧 IR 多视图校准和 tiny Pd–Fa 指标区分 |
| SAM-COD, ECCV 2024 | response filter + semantic matcher + prompt-adaptive KD | Idea 1/2 必须超出常规 filter/KD |
| MS-SAM-LESS, Pattern Recognition Letters 2026 | dense prompt learner + frozen SAM training +轻量 mask aggregator 部署 | Idea 2 不能泛称“蒸馏 prompt→mask”；需限定文本反事实增量 |
| RL-CP, BSPC 2026 | activation stability + RL/contrastive purification→正负点 | Idea 1/8 的稳定性与正负点生成已有近邻 |
| EdgeSAM | prompt encoder/mask decoder 在环蒸馏 | “完整决策函数蒸馏”表述必须收窄到 text-conditioned delta |

## 4. 可以尝试形成的保守 claim（需实验后才能使用）

1. **可靠性问题定义**：红外极小目标自动 prompt 中，定位错误和“目标不存在”需要显式 abstention，而不是强制 top-k。
2. **行为蒸馏对象**：蒸馏文本条件对 SAM prompt/mask decision 的增量行为，比回归 global/token embedding 更有效。
3. **语义与定位职责解耦**：全局语言先验用于 presence/background calibration，高分辨率局部残差负责坐标与轮廓。

上述每条必须同时满足：无 GT-derived inference prompt；与同容量 self-prompt baseline 比较；correct/no/shuffle/wrong/oracle text 控制；至少两个 IR 数据集和三个 seeds；报告 IoU/nIoU/Pd/Fa 及 prompt-level recall/calibration。

## 5. 审稿人视角的高风险问题

| 问题 | 必须准备的证据 |
|---|---|
| GPT 描述来自同一图像，究竟增加什么信息？ | correct/shuffle/wrong/no/oracle，候选验证 AUPRC，以及 teacher 相对 no-text 的 mask delta |
| 为什么不直接用 SAM-SPL/IRSAM？ | 同输入、同分辨率、无 GT prompt 的强纯图像基线；参数/FLOPs/延迟公平 |
| student 只是在拟合 teacher feature 吗？ | embedding regression vs prompt distill vs behavior distill 的逐级消融 |
| 高分辨率模块本身就有效，文本是否多余？ | ordinary FPN/upsampler、no-text residual、fixed text、correct text 的正交消融 |
| 错文本会不会把背景亮点变成目标？ | wrong/shuffle stress test、空场景、risk-coverage/Pd–Fa 曲线、可视化拒绝案例 |
| 自动 prompt 是否暗中来自测试 GT？ | 数据流表、启动命令、checkpoint metadata、代码断言；把 loss sampling 与 decoder prompt 明确分开 |

## 6. 当前推荐

- 主线优先：Idea 1（可靠性/拒绝）或 Idea 2（行为蒸馏）。两者能直接回答审稿人的“无文本怎么办”和现有 GT prompt 质疑，又不把 GPT 本身当贡献。
- Idea 3 只有在 prompt-level tiny-bin 诊断确认空间分辨率是主要瓶颈时进入主线。
- Idea 5 应成为统一实验协议；Idea 8 应成为所有文本方案必须击败的强 no-text baseline。
- Idea 6/9 当前碰撞过高，不建议优先投入完整训练。
