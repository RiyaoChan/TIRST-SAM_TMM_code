# 文本—图像—SAM 相关研究阅读与 Idea 挖掘任务书

> **面向项目**：`RiyaoChan/TIRST-SAM_TMM_code`  
> **当前相关分支**：`codex/update-cbga-experiment-record`  
> **文献检索截止日期**：2026-08-13  
> **任务性质**：文献精读、代码追踪、创新边界核查与候选 Idea 整理；**本阶段不修改模型代码，不直接生成最终方法**。

---

## 0. 研究背景与硬约束

当前项目希望解决以下问题：

1. **模型一：有文本教师的自动提示模型**  
   输入红外图像及由图像获得的高质量文本语义，在推理阶段不使用 GT 点、GT 框或 GT mask，为 SAM 提供有效提示并完成红外小目标分割。

2. **模型二：无文本部署模型**  
   在训练阶段利用模型一的跨模态信息，通过蒸馏将文本提示能力迁移到纯图像模型；部署时只输入红外图像。

3. **当前困难**  
   模型一虽然已经避免了部分人工文本依赖，但在“有文本、无 GT 点”的设定下尚未达到期望性能。现阶段不应继续局限于“把单个文本 token 投影成 SAM sparse prompt”这一条路线，而应系统考察：
   - 文本如何转化为**空间定位信息**；
   - 文本如何进入 SAM 的**图像编码器、prompt encoder 或 mask decoder**；
   - 如何自动生成 point、box、mask、dense map、query 等不同 prompt；
   - 如何处理**文本不准确、文本缺失、目标极小、背景高杂波**；
   - 如何把跨模态教师能力蒸馏给纯图像学生。

### 0.1 必须遵守的实验边界

后续筛选出的任何方法均需满足或明确标注以下条件：

- 推理阶段不得从 GT mask 采点、生成 box、构造 coarse mask 或得到任何 GT 派生提示；
- 必须区分“训练时使用 GT 监督”和“推理时使用 GT prompt”，二者不能混淆；
- 必须区分文本来源：人工输入、固定类别模板、图像自动生成文本、可学习软文本、图像到文本映射、外部报告或描述；
- 必须明确模型二部署时是否真正不再调用 CLIP 文本编码器、LLM/VLM 或离线 caption 文件；
- 红外小目标的关键评价不只看 mIoU，还要同时关注 `Pd`、`Fa`、目标级召回、极小目标召回和边界质量；
- 不得仅凭论文摘要判断可迁移性，必须阅读正文、补充材料并追踪公开代码中的真实推理路径。

---

## 1. 本轮阅读不预设最终 Idea

本任务的目标不是把若干论文模块机械拼接，而是建立一个**机制空间**：

| 机制维度 | 需要回答的问题 |
|---|---|
| 文本来源 | 文本是人工、自动 caption、类别名、属性集合、软提示，还是图像内部生成的伪文本？ |
| 文本粒度 | 全局场景、目标类别、目标属性、空间关系、候选级描述、像素级语义分别如何表示？ |
| 融合位置 | 文本在图像编码前、编码过程中、prompt encoder、mask decoder，还是结果选择阶段发挥作用？ |
| 空间化方式 | 文本如何变为 similarity map、CAM、attention map、point、box、mask、query 或 dense prompt？ |
| 自动提示 | 提示由图像分支、VLM、检测器、显著性图、上一轮 mask、参考图像还是强化学习产生？ |
| 小目标保护 | 如何避免 1/16 特征分辨率、全局语义偏置和多次下采样抹除极小目标？ |
| 提示可靠性 | 文本错误、弱提示、噪声点、目标不存在时如何校验、拒绝或降权？ |
| 蒸馏对象 | 蒸馏 global embedding、dense map、region feature、prompt token、候选集合、mask logits 还是完整决策函数？ |
| 部署依赖 | 测试时是否需要文本、外部 VLM、参考图像、人工点击或额外模型？ |
| 创新冲突 | 与 SAIST、SAM-SPL、DGSPNet、JinSight、SeViL 等直接红外工作是否实质重合？ |

---

## 2. 优先级总览

### S 级：必须先精读，直接决定创新边界

1. SAIST（CVPR 2025）
2. A Unified SAM-Guided Self-Prompt Learning Framework / SAM-SPL（TGRS 2025）
3. DGSPNet（2025 预印本）
4. JinSight / Understand Before Detect（2026-08 最新预印本）
5. SeViL（AAAI 2026）
6. IRSAM（ECCV 2024）
7. SimIR（通用分割模型蒸馏到红外小目标）
8. EVF-SAM（ECCV 2024）
9. READ（CVPR 2025）
10. Simple-ViLMedSAM（CVPR 2026）
11. RS2-SAM 2（AAAI 2026 / arXiv）
12. SegEarth-R2（CVPR 2026）
13. SegEarth-OV / AlignEarth（CVPR 2025 及后续版本）
14. SPD：Saliency-Guided Prompt Distillation（2026 预印本）

### A 级：用于扩展可行机制

- Curriculum Point Prompting
- MedCLIP-SAM / MedCLIP-SAMv2
- RSRefSeg
- RSPrompter
- MaskSAM
- SAM-SP
- Plug-and-Play PPO
- SOPSeg
- CLIP-Guided SAM
- F-LMM
- CLIPSelf
- PromptKD
- KnowSAM
- Grounding DINO-US-SAM
- TextSAM-EUS

### B 级：作为通用接口、数据和推理范式补充

- LISA
- GLaMM
- PixelLM
- Emerging Pixel Grounding without Grounding Supervision
- MedSAM3
- One Shot is Enough for Sequential IRSTS
- TEP-SAM
- MoPKL
- Training-Free Text-Based Remote Sensing Segmentation
- SegEarth-OV3

---

# 第一部分：与当前课题创新边界最接近的红外研究

## P01. SAIST: Segment Any Infrared Small Target Model Guided by Contrastive Language-Image Pretraining

- **领域/出处**：红外小目标检测，CVPR 2025
- **论文**：https://ieeexplore.ieee.org/document/11092723
- **核心机制**：由 Scene Recognition CLIP 与 CLIP-guided SAM 两部分构成，目标是让语言—图像预训练知识参与红外小目标的场景理解和分割。
- **为什么必须优先阅读**：这是“CLIP + SAM + 红外小目标”最直接的先行工作。当前项目若仍以文本特征调制 SAM 为主，必须逐层核查与 SAIST 的实质差异，而不能只在模块名称上区分。
- **Codex 必查**：
  1. SR-CLIP 的训练数据、文本模板与正负样本如何构造；
  2. CLIP 信息进入 SAM 的具体层级、张量形状和融合算子；
  3. 训练与测试是否都需要文本；
  4. 其增益主要来自降低 Fa、提高 Pd，还是改善 mask 边界；
  5. 与当前 CBGA、ASSP、自动文本教师的重叠程度。
- **可借鉴但不可直接复刻的机制**：场景级语义用于抑制复杂背景；红外域专用语言—图像对齐；语言语义与 SAM 的联合适配。

## P02. A Unified SAM-Guided Self-Prompt Learning Framework for Infrared Small Target Detection（SAM-SPL）

- **领域/出处**：红外小目标检测，IEEE TGRS 2025
- **论文**：https://ieeexplore.ieee.org/document/11172325
- **代码**：https://github.com/fuyimin96/SAM-SPL
- **核心机制**：在编码阶段以 consult–guide 方式引入 SAM；利用浅层特征产生 self-derived prompts，并与编码后的潜在表示双向交互；通过 skip connection 中的 mutual calibration 缓解分辨率恢复过程中的语义不一致。
- **为什么重要**：它并不依赖外部文本，却直接解决“无人工提示推理”的问题。当前项目必须证明文本带来的价值超出强 self-prompt SAM 基线。
- **Codex 必查**：
  1. self-derived prompt 是 sparse token、dense map 还是多尺度特征；
  2. prompt 是否由浅层高分辨率特征产生，如何避免目标被深层语义抹除；
  3. SAM 是教师、冻结先验还是端到端组成部分；
  4. 公开代码的真实模块与论文描述是否完全一致；
  5. 与当前“图像分支自己预测 prompt”的任何候选方案是否冲突。
- **直接启发点**：红外小目标 prompt 未必应由全局文本产生，浅层细节本身可能是更可靠的空间提示源；文本可以转为校验或调制，而不是承担全部定位职责。

## P03. IRSAM: Advancing Segment Anything Model for Infrared Small Target Detection

- **领域/出处**：红外小目标检测，ECCV 2024
- **论文**：https://arxiv.org/abs/2407.07520
- **代码**：https://github.com/IPIC-Lab/IRSAM
- **核心机制**：在 SAM 编码器多个层级加入 Perona–Malik diffusion 相关模块，以保留结构并抑制噪声；使用 Granularity-Aware Decoder 融合多粒度特征。
- **为什么重要**：文本是否有效，首先取决于 SAM 的红外视觉表征是否保留了极小目标。IRSAM 是当前项目必须采用的视觉侧强基线和结构参考。
- **Codex 必查**：
  1. 哪些 encoder stage 被改动；
  2. 是否仍依赖外部点/框；
  3. 多粒度特征如何进入 decoder；
  4. 公开代码是否完整实现论文中的 WPMD/PMD；
  5. 文本模块若接入 IRSAM，最小侵入位置在哪里。

## P04. Unleashing the Power of Generic Segmentation Models: A Simple Baseline for Infrared Small Target Detection（SimIR）

- **领域/出处**：红外小目标检测，2024 预印本
- **论文**：https://arxiv.org/abs/2409.04714
- **代码**：https://github.com/O937-blip/SimIR
- **核心机制**：研究通用分割模型向红外小目标任务的适配与蒸馏；设计 dense/sparse queries 编码多尺度信息，并让较小学生模型通过蒸馏超过教师。
- **为什么重要**：与“模型一跨模态教师 → 模型二纯图像学生”的总体设想高度相关。它提示学生不必只拟合文本 embedding，也可以蒸馏通用分割教师的多尺度查询和输出函数。
- **Codex 必查**：
  1. 教师与学生分别是什么模型；
  2. 蒸馏发生在 logits、features、queries 还是多个层级；
  3. dense/sparse query 的生成、监督及推理依赖；
  4. 学生超过教师的条件；
  5. 与当前模型二方案是否存在直接重叠。

## P05. Dual-Granularity Semantic Prompting for Language Guidance Infrared Small Target Detection（DGSPNet）

- **领域/出处**：红外小目标检测，arXiv 2025
- **论文**：https://arxiv.org/abs/2511.19306
- **核心机制**：同时使用粗粒度固定文本先验（如 infrared image、small target）与由视觉到文本映射获得的细粒度个性化语义；通过 text-guided channel attention 与 spatial attention 增强不同层级的目标响应；推理不依赖人工标注。
- **为什么必须优先阅读**：它已经覆盖“固定通用文本 + 图像内部生成个性化文本 + 通道/空间引导”的核心思路。任何 soft prompt、图像拟合文本特征或双粒度文本方案都需要与其严格区分。
- **Codex 必查**：
  1. fine-grained personalized semantics 是否是真实自然语言、连续 embedding 还是 visual-to-text latent；
  2. 文本监督从哪里来；
  3. TGCA/TGSA 的具体作用位置；
  4. 推理阶段是否仍调用文本编码器；
  5. 是否可将其视为当前模型二的近邻工作。

## P06. Understand Before Detect: Vision--Language Learning for Omni-Domain Infrared Small Target Detection（JinSight）

- **领域/出处**：全域红外小目标检测，arXiv 2026-08-07
- **论文**：https://arxiv.org/abs/2608.07015
- **核心机制**：提出“先理解、后检测”的范式，通过语言监督学习跨红外域保持稳定的全局语义，再将其迁移到精确小目标检测；Latent Semantic Interaction 在低秩空间交换语言对齐的全局语义与细粒度空间特征；同时构建 OmniIRST-VL 数据集，包含场景理解和目标推理等多种指令任务。
- **为什么极其重要**：这是当前检索中最新、与“语言帮助红外小目标泛化”最直接的工作。它可能改变项目叙事：文本价值不一定在推理时替代点，而可能在预训练阶段形成跨域、可迁移的视觉表征。
- **Codex 必查**：
  1. 六类指令任务及标注生成流程；
  2. 语言监督是否只用于预训练，部署是否纯图像；
  3. LSI 的低秩交互公式与插入位置；
  4. 跨域评测协议；
  5. 是否为模型二提供更自然的“训练有语言、测试无语言”范式。

## P07. SeViL: Semi-supervised Vision-Language Learning with Text Prompt Guiding for Moving Infrared Small Target Detection

- **领域/出处**：运动红外小目标，AAAI 2026
- **论文**：https://ojs.aaai.org/index.php/AAAI/article/view/37372
- **核心机制**：自适应文本 prompt 用于增强目标区域，并筛除无标注数据中的低质量伪标签。
- **为什么重要**：文本不一定要直接作为 SAM 的分割 prompt，也可以作为**训练期质量控制器、伪标签过滤器或置信度校准器**。这为模型二提供了另一条路径。
- **Codex 必查**：
  1. 文本模板是否固定；
  2. 文本如何判断伪标签质量；
  3. 测试阶段是否需要文本；
  4. 文本增益主要来自哪类样本；
  5. 是否存在可迁移到单帧分割的伪标签或教师置信度机制。

## P08. Motion Prior Knowledge Learning with Homogeneous Language Descriptions for Moving Infrared Small Target Detection（MoPKL）

- **领域/出处**：运动红外小目标，AAAI 2025
- **论文**：https://ojs.aaai.org/index.php/AAAI/article/view/32217
- **代码**：https://github.com/UESTC-nnLab/MoPKL
- **核心机制**：使用同质化语言描述表达位置、方向、速度和运动关系等先验，增强运动小目标表征。
- **阅读价值**：重点不是直接迁移运动模块，而是学习它如何将难以直接监督的运动先验转为结构化语言，并与视觉特征对齐。
- **Codex 必查**：语言字段、模板生成、语言编码器、视觉—语言对齐损失和测试依赖。

## P09. One Shot is Enough for Sequential Infrared Small Target Segmentation

- **领域/出处**：序列红外小目标，ICASSP 2025 / arXiv 2024
- **论文**：https://arxiv.org/abs/2408.04823
- **代码**：https://github.com/D-IceIce/one-shot-IRSTS
- **核心机制**：由局部特征匹配产生置信图，以最高响应点自动替代人工点提示；再通过 point-prompt-centric focusing 与多级集成降低过分割、漏检和虚警。
- **阅读价值**：这是“连续空间图 → 自动点 → SAM”的直接范例。虽然需要一个参考帧，但其 prompt 自动化和 prompt 后校正过程值得拆解。

## P10. Temporal-Emerged Prompting for Segment Anything in Multiframe Infrared Small Target Detection（TEP-SAM）

- **领域/出处**：多帧红外小目标，arXiv 2026
- **论文**：https://arxiv.org/abs/2606.27655
- **核心机制**：联合建模全局运动与局部运动偏差，形成 temporal-emerged cues 调制并提示 SAM，实现非交互分割。
- **阅读价值**：如果项目未来扩展视频，此工作说明 prompt 可以来自“目标相对背景逐渐显现”的任务属性，而不必来自文本。对单帧任务也可启发基于增强前后、尺度前后或频域前后的“差异涌现提示”。

---
# 第二部分：文本如何真正获得空间定位能力

## P11. EVF-SAM: Early Vision-Language Fusion for Text-Prompted Segment Anything Model

- **领域/出处**：通用 referring segmentation，ECCV 2024
- **论文**：https://arxiv.org/abs/2406.20076
- **代码**：https://github.com/hustvl/EVF-SAM
- **核心机制**：不是只用独立文本编码器产生一个 prompt，而是用具备早期视觉—语言融合能力的预训练 VLM 同时编码图像与文本，再将融合表示用于提示 SAM。
- **关键价值**：论文系统比较了纯文本编码器、LLM 与早期融合 VLM，说明“prompt 是否已经看到图像”可能比单纯扩大文本模型更重要。
- **Codex 必查**：
  1. 图像是否同时送入 VLM 和 SAM；
  2. multimodal class token 如何投影到 SAM；
  3. 该 token 是否包含可解释空间信息；
  4. SAM image encoder 是否冻结；
  5. 早期融合相较 late fusion 的增益来自哪里。

## P12. Reasoning to Attend: Try to Understand How `<SEG>` Token Works（READ）

- **领域/出处**：通用推理分割，CVPR 2025
- **论文**：https://arxiv.org/abs/2412.17741
- **官方页面**：https://openaccess.thecvf.com/content/CVPR2025/html/Qian_Reasoning_to_Attend_Try_to_Understand_How_SEG_Token_Works_CVPR_2025_paper.html
- **代码**：https://github.com/rui-qian/READ
- **核心机制**：分析 `<SEG>` token 与图像 token 的相似度图，发现该 token 的主要作用是查询与文本语义一致的图像 patch；Similarity-as-Points 模块从高响应区域得到点提示，并用于增强下游分割。
- **为什么高度相关**：它直接回答“一个全局语义 token 为什么可能不能替代空间点”，并给出 token → similarity map → point prompt 的可微转换方式。
- **Codex 必查**：
  1. similarity map 在 LMM 端和 SAM 端分别如何计算；
  2. 正点数量、阈值、top-k 与负点策略；
  3. point 选择是否可微；
  4. 对多目标和小目标是否稳定；
  5. 能否把当前 ASSP/CBGA 中的文本 token 先转为多尺度相似图，再决定是否进入 SAM。

## P13. Curriculum Point Prompting for Weakly-Supervised Referring Image Segmentation

- **领域/出处**：弱监督 referring segmentation，CVPR 2024
- **论文**：https://arxiv.org/abs/2404.11998
- **核心机制**：结合 CLIP 的图文对齐与 SAM 的 mask 生成能力，训练 point generator；不仅生成正点，也显式生成负点以减轻噪声和只关注目标局部的问题，并采用由简单目标中心图像到复杂场景的 curriculum learning。
- **关键价值**：说明文本产生的空间提示不能只追求最高响应正点，负点和训练课程同样重要。
- **Codex 必查**：point generator 的监督、正负点标签来源、噪声处理、curriculum 构造，以及是否需要 mask GT。

## P14. MedCLIP-SAM: Bridging Text and Image Towards Universal Medical Image Segmentation

- **领域/出处**：医学图像，MICCAI 2024
- **论文**：https://arxiv.org/abs/2403.20253
- **代码**：https://github.com/HealthX-Lab/MedCLIP-SAM
- **核心机制**：通过 DHN-NCE 微调 BiomedCLIP，再使用 gScoreCAM 将文本—图像对齐转为可供 SAM 使用的视觉提示；支持 zero-shot 与弱监督分割。
- **关键价值**：这是“领域 CLIP → 可解释激活图 → SAM prompt”的典型路径，可重点观察热图分辨率、阈值和噪声如何影响小病灶。

## P15. MedCLIP-SAMv2: Towards Universal Text-Driven Medical Image Segmentation

- **领域/出处**：医学图像，Medical Image Analysis 2025 / arXiv
- **论文**：https://arxiv.org/abs/2409.19483
- **代码**：https://github.com/HealthX-Lab/MedCLIP-SAMv2
- **核心机制**：在 MedCLIP-SAM 基础上使用 Multi-modal Information Bottleneck（M2IB）从图文信息中生成视觉 prompt，覆盖多种成像模态与 zero/few/weak supervision。
- **为什么重要**：与简单 CAM 相比，信息瓶颈可主动保留与文本相关、又具有空间判别性的视觉信息，可能比把文本 embedding 直接投影为 token 更适合极小目标。
- **Codex 必查**：M2IB 的输入输出、互信息目标、prompt 类型、空间分辨率、是否需要类别文本以及对小病灶的表现。

## P16. Simple-ViLMedSAM: Simple Text Prompts Meet Vision-Language Models for Medical Image Segmentation

- **领域/出处**：医学图像，CVPR 2026
- **论文**：https://openaccess.thecvf.com/content/CVPR2026/html/Qian_Simple-ViLMedSAM_Simple_Text_Prompts_Meet_Vision-Language_Models_for_Medical_Image_CVPR_2026_paper.html
- **代码**：https://github.com/qcc001/Simple-ViLMedSAM
- **核心机制**：只使用简单类别文本；Implicit Pos-Prompter 利用多模态信息瓶颈与 affinity refinement 产生带隐式位置的 attribution map；Bidirectional Interaction Decoder 对齐位置图与 SAM 像素特征。
- **为什么高度相关**：它不是要求文本本身给出坐标，而是让文本和图像共同形成**隐式位置图**，再与 SAM 像素特征双向交互。此机制与当前“文本足够精确就能提供 prompt”的原始假设有本质区别。
- **Codex 必查**：
  1. attribution map 的分辨率和监督；
  2. affinity refinement 如何传播区域而不扩散到背景；
  3. bidirectional decoder 的具体交互方向；
  4. zero-shot 与 few-shot 配置差异；
  5. 对小病灶、低对比病灶的错误案例。

## P17. RSRefSeg: Referring Remote Sensing Image Segmentation with Foundation Models

- **领域/出处**：遥感 referring segmentation，arXiv 2025
- **论文**：https://arxiv.org/abs/2501.06809
- **代码**：https://github.com/KyanChen/RSRefSeg
- **核心机制**：利用 CLIP 编码图像和文本，将全局文本语义与局部文本语义作为过滤器，在潜空间生成与指代表达相关的视觉激活特征，再以这些激活特征提示 SAM。
- **关键价值**：需要重点理解“全局文本”和“局部文本”是否承担不同角色，以及其过滤机制能否避免全局语义淹没小目标。

## P18. Customized SAM 2 for Referring Remote Sensing Image Segmentation（RS2-SAM 2）

- **领域/出处**：遥感 referring segmentation，AAAI 2026 / arXiv 2025
- **论文**：https://arxiv.org/abs/2503.07266
- **核心机制**：使用 union encoder 联合编码视觉与文本；通过双向层级融合对齐遥感视觉特征和视觉增强文本；mask prompt generator 根据视觉 embedding 与 multimodal class token 生成 pseudo-mask dense prompt；另以 text-guided boundary loss 约束边界。
- **为什么高度相关**：它将文本作用拆成了三部分：联合理解、dense mask prompt、边界监督，而不是只产生 sparse token。
- **Codex 必查**：
  1. pseudo-mask 是否在推理时完全自动生成；
  2. dense prompt 的监督和尺寸；
  3. class token 是否保留多目标信息；
  4. boundary loss 如何由文本加权；
  5. 对极小遥感目标是否有独立结果。

## P19. Grounding DINO-US-SAM: Text-Prompted Multi-Organ Segmentation in Ultrasound

- **领域/出处**：超声医学，IEEE TUFFC 2025
- **论文**：https://arxiv.org/abs/2506.23903
- **核心机制**：用 LoRA 适配到超声域的 Grounding DINO 根据文本产生目标定位，再由 SAM2 完成分割。
- **关键价值**：代表“文本语义检测器负责定位、SAM 负责轮廓”的两阶段范式。应与端到端文本 token 注入方案比较性能、复杂度和误差传播。
- **Codex 必查**：检测器输出是 box 还是点；域适配数据量；未见器官泛化；检测漏检对最终 mask 的影响；是否适合红外极小目标。

## P20. CLIP-Guided SAM: Parameter-Efficient Semantic Conditioning for Promptable Segmentation

- **领域/出处**：通用与专业域分割，arXiv 2026
- **论文**：https://arxiv.org/abs/2605.24807
- **核心机制**：不只在外部生成空间 prompt，而是通过轻量多模态语义 adapter 将 CLIP 文本、视觉及相似度特征直接注入 SAM image encoder；同时保留 SAM 原有 prompt 接口，并强调训练—测试 prompt 类型一致性。
- **关键价值**：代表“内部语义条件化 + 外部空间提示互补”的路线，可检验当前 CBGA 是否应该从独立跨注意力改为更轻量、层级选择性的 semantic adapter。
- **Codex 必查**：adapter 插入层、三类 CLIP 特征的分工、manual 与 text-only 模式、prompt consistency 实验，以及错误文本下的鲁棒性。

## P21. F-LMM: Grounding Frozen Large Multimodal Models

- **领域/出处**：通用像素级视觉 grounding，CVPR 2025
- **论文**：https://arxiv.org/abs/2406.05821
- **核心机制**：冻结现成 LMM，不训练特殊 segmentation token；直接利用 LMM 中已有的 word–pixel attention，通过少量 CNN 转成 mask logits，再由 SAM-based refiner 优化。
- **关键价值**：文本空间信息可能已经存在于 attention 中，无需把最终文本 embedding 重新投影成 prompt。对当前项目，应重点考察 Qwen/CLIP 中间层 attention 或 image-token similarity 是否比最终全局文本向量更有空间价值。

## P22. Emerging Pixel Grounding in Large Multimodal Models Without Grounding Supervision

- **领域/出处**：通用视觉 grounding，arXiv 2024
- **论文**：https://arxiv.org/abs/2410.08209
- **项目页**：https://groundLMM.github.io
- **核心机制**：从没有显式 grounding 监督的标准 LMM 中提取 attention map，再执行 attend-and-segment；同时探索扩散视觉编码器以改善像素 grounding。
- **阅读价值**：用于判断当前 GPT/Qwen 教师是否可以直接输出或暴露空间注意，而不是只保留 caption 文本。

---

# 第三部分：自动提示、自提示与提示优化

## P23. RSPrompter: Learning to Prompt for Remote Sensing Instance Segmentation

- **领域/出处**：遥感实例分割，TGRS 2024 / arXiv 2023
- **论文**：https://arxiv.org/abs/2306.16269
- **代码**：https://github.com/KyanChen/RSPrompter
- **核心机制**：在遥感域中学习为 SAM 自动生成带语义类别信息的 prompt，使 SAM 不再依赖人工 point/box/mask。
- **关键价值**：它是自动 prompt generator 的早期强基线。Codex 必须追踪其 prompt 头到底预测什么，以及推理时是否真正无人工提示。

## P24. MaskSAM: Towards Auto-prompt SAM with Mask Classification for Medical Image Segmentation

- **领域/出处**：三维医学分割，ICCV 2025 / arXiv 2024
- **论文**：https://arxiv.org/abs/2403.14103
- **核心机制**：prompt generator 同时生成辅助分类 token、二值 mask 和 box；每个辅助 mask/box 与类别预测关联，实现 prompt-free 语义分割；另以 3D adapter 适配医学体数据。
- **关键价值**：prompt 不必是单一形式。对红外小目标可重点考察“类别/存在性 token + coarse mask + box/point”联合提示是否优于单一 sparse token。

## P25. AutoProSAM: Automated Prompting SAM for 3D Multi-Organ Segmentation

- **领域/出处**：医学图像，WACV 2025 / arXiv 2023
- **论文**：https://arxiv.org/abs/2308.14936
- **代码**：https://github.com/ChengyinLee/AutoProSAM_2024
- **核心机制**：通过参数高效适配和自动 prompt learning，使 SAM 在不依赖医生手工提示的情况下完成 3D 多器官分割。
- **阅读重点**：自动 prompt 是显式几何提示还是 latent prompt；训练监督如何定义；测试时依赖哪些外部信息。

## P26. SAM-SP: Self-Prompting Makes SAM Great Again

- **领域/出处**：医学与领域适配，arXiv 2024
- **论文**：https://arxiv.org/abs/2408.12364
- **核心机制**：将模型上一轮输出作为下一轮 prompt，迭代自提示；同时使用 self-distillation 强化自提示过程，减少测试阶段专家 prompt 依赖。
- **关键价值**：提示可以是动态迭代状态，不必一次生成。对小目标应重点评估首轮漏检是否会被不断放大，以及如何设计“出现新目标”的恢复机制。

## P27. Plug-and-Play PPO: An Adaptive Point Prompt Optimizer Making SAM Greater

- **领域/出处**：通用 SAM 提示优化，CVPR 2025
- **官方页面**：https://openaccess.thecvf.com/content/CVPR2025/html/Liu_Plug-and-Play_PPO_An_Adaptive_Point_Prompt_Optimizer_Making_SAM_Greater_CVPR_2025_paper.html
- **核心机制**：在特征空间与物理坐标空间构建异质图，通过强化学习迭代优化初始点提示分布，并可作为即插即用模块而不重训任务模型。
- **关键价值**：若初始自动点不够准，可以研究“prompt optimizer”而非不断增强文本编码器。需特别考察其计算开销、奖励定义和对小目标的点数需求。

## P28. Learning from Noisy Prompts: Saliency-Guided Prompt Distillation for Robust Segmentation with SAM（SPD）

- **领域/出处**：医学图像，arXiv 2026
- **论文**：https://arxiv.org/abs/2604.23314
- **核心机制**：轻量 saliency head 学习定位先验；Contextual Prompt Distillation 验证并丰富粗糙、偏移或有噪声的 prompt，形成 consensus prompt set；另以相邻切片一致性提升稳定性。
- **为什么高度相关**：当前自动文本或自动点不可能始终准确。SPD 提供的是“提示校验与蒸馏”机制，而不是假设输入 prompt 完美。
- **Codex 必查**：
  1. saliency map 与 prompt 的相互关系；
  2. consensus prompt set 如何生成；
  3. 错误点如何被拒绝；
  4. 是否可将相邻切片上下文替换为多尺度、增强视图或多模型一致性；
  5. 蒸馏目标是否可以用于模型二。

## P29. TextSAM-EUS: Text Prompt Learning for SAM to Segment Pancreatic Tumor in EUS

- **领域/出处**：低对比超声，ICCV Workshop 2025 / arXiv
- **论文**：https://arxiv.org/abs/2507.18082
- **核心机制**：通过 BiomedCLIP context optimization 学习文本 prompt，并用 LoRA 适配 SAM；推理时不需要人工几何提示。
- **关键价值**：与当前项目的“有文本、无点”设定相近，但其文本可能是学习到的领域软提示而非每张图的自动 caption。需要比较哪一种文本更稳定。

## P30. Enabling Training-Free Text-Based Remote Sensing Segmentation

- **领域/出处**：遥感，CVPR Workshop 2026 / arXiv
- **论文**：https://arxiv.org/abs/2602.17799
- **代码**：https://github.com/josesosajs/trainfree-rs-segmentation
- **核心机制**：一条路线用 CLIP 从 SAM 网格 mask proposals 中选择目标 mask；另一条路线用 GPT-5 或 LoRA Qwen-VL 产生 SAM click prompts，支持 reasoning/referring segmentation。
- **为什么重要**：这是“VLM 直接生成点击”和“CLIP 后验选择 SAM 候选 mask”两种截然不同接口的同框比较。Codex 应重点分析哪种路线在小目标上失效，以及失败来自 localization、proposal recall 还是语义选择。

## P31. MedSAM3: Delving into Segment Anything with Medical Concepts

- **领域/出处**：医学图像/视频，arXiv 2025
- **论文**：https://arxiv.org/abs/2511.19046
- **核心机制**：以开放词汇医学概念驱动可提示分割，并引入 MLLM agent 进行复杂推理与迭代修正。
- **阅读价值**：不建议直接迁移整套大模型，但可研究“存在性判断—分割—质量检查—再次提示”的 agent-in-the-loop 结构，尤其适合处理文本错误和 false alarm。

---
# 第四部分：遥感小目标、细粒度对齐和高分辨率语义

## P32. Exploring Fine-Grained Image-Text Alignment for Referring Remote Sensing Image Segmentation（FIANet）

- **领域/出处**：遥感 referring segmentation，2024
- **论文**：https://arxiv.org/abs/2409.13637
- **代码**：https://github.com/Shaosifan/FIANet
- **核心机制**：将原始表达拆分为 context、ground object 和 spatial position 三类文本；通过 Fine-grained Image-text Alignment 同时对齐不同语义角色，并用 Text-aware Multi-scale Enhancement 处理遥感目标尺度变化。
- **为什么重要**：文本分解不应只是更多句子，而应形成不同职责的语义变量。当前自动 caption 可被审查是否把场景、目标外观、位置和置信度混在一个 global embedding 中。

## P33. SegEarth-R2: Towards Comprehensive Language-guided Segmentation for Remote Sensing Images

- **领域/出处**：遥感语言引导分割，CVPR 2026
- **论文**：https://arxiv.org/abs/2512.20013
- **代码/数据**：https://github.com/earth-insights/SegEarth-R2
- **核心机制**：构建覆盖层级粒度、多目标、推理需求与语言变化的 LaSeRS 数据；模型使用 spatial attention supervision 专门处理小目标及其部件，并用灵活 segmentation queries 同时支持单目标和多目标。
- **为什么高度相关**：对红外小目标而言，普通图文对齐的主要短板正是 attention 不够精细。该文把“小目标定位”直接作为空间注意监督问题，而不是依赖文本自然产生精确坐标。
- **Codex 必查**：
  1. spatial attention supervision 的标签与 loss；
  2. segmentation query 数量和多目标匹配；
  3. 小目标及部件的单独评测；
  4. 是否可在不引入完整 MLLM 的情况下抽取该机制；
  5. 与 PixelLM/LISA 的 query 接口差异。

## P34. SegEarth-OV: Towards Training-Free Open-Vocabulary Segmentation for Remote Sensing Images

- **领域/出处**：遥感开放词汇分割，CVPR 2025 Oral
- **论文**：https://arxiv.org/abs/2410.01768
- **项目页**：https://earth-insights.github.io/SegEarth-OV
- **核心机制**：SimFeatUp 在无需任务训练的情况下从粗糙深层特征恢复高分辨率空间信息；通过从局部 patch token 中减去全局 `[CLS]` 偏置，增强局部语义忠实度。
- **为什么重要**：红外小目标极易被 CLIP/SAM 的低分辨率和全局语义偏置淹没。即使文本正确，若 dense visual-language feature 不保留局部响应，文本仍无法形成有效 prompt。
- **Codex 必查**：SimFeatUp 的结构、是否需要高分辨率教师、global bias subtraction 的数学形式、对小目标类别的实际增益。

## P35. Annotation-Free Open-Vocabulary Segmentation for Remote-Sensing Images（含 AlignEarth）

- **领域/出处**：遥感光学与 SAR，arXiv 2025
- **论文**：https://arxiv.org/abs/2508.18067
- **核心机制**：在 SegEarth-OV 基础上提出 AlignEarth，以蒸馏方式把光学 VLM 编码器的语义知识迁移给 SAR 编码器，无需从头构建 SAR 视觉语言大模型。
- **为什么高度相关**：这是“有强语义模态教师 → 缺少对应 VLM 的专业传感模态学生”的直接类比。红外模型二可以研究迁移的对象究竟应是全局语义、局部 dense feature、cost map 还是多尺度空间响应。
- **Codex 必查**：教师/学生输入是否成对；蒸馏层级；对齐损失；是否保留文本编码器；推理是否纯 SAR；跨模态注册误差如何处理。

## P36. SOPSeg: Prompt-based Small Object Instance Segmentation in Remote Sensing Imagery

- **领域/出处**：遥感小目标实例分割，CVPR Findings 2026 / arXiv 2025
- **论文**：https://arxiv.org/abs/2509.03002
- **核心机制**：针对 SAM 1/16 分辨率造成的小目标细节丢失，引入区域自适应放大；定制 decoder 联合边缘预测和渐进细化；设计适配旋转框的 prompt 机制。
- **为什么高度相关**：它直接说明小目标 SAM 的瓶颈可能主要在视觉分辨率与边界恢复，而非文本质量。任何文本模块都应与 region magnification 或细粒度 decoder 分开消融。

## P37. SegEarth-OV3: Exploring SAM 3 for Open-Vocabulary Semantic Segmentation in Remote Sensing Images

- **领域/出处**：遥感开放词汇，arXiv 2025，2026 修订
- **论文**：https://arxiv.org/abs/2512.08730
- **核心机制**：融合 SAM 3 的语义头和实例头；使用 presence score 过滤场景中不存在的类别，从而降低大词表与 patch 预测造成的 false positives。
- **阅读价值**：红外小目标方法通常同时面对 Pd 与 Fa。presence head 提示：应将“场景是否存在目标”与“目标在哪里”分开建模，尤其要考察文本是否更适合 presence calibration 而非像素定位。

## P38. Exploring Efficient Open-Vocabulary Segmentation in Remote Sensing（RSKT-Seg）

- **领域/出处**：遥感开放词汇，arXiv 2025
- **论文**：https://arxiv.org/abs/2509.12040
- **核心机制**：从多方向计算 vision-language cost maps，联合建模空间与语义依赖，并通过知识迁移和增强上采样适配遥感域。
- **阅读价值**：多方向 cost map 对旋转和极小遥感目标有效的原因，可能为红外小目标的多尺度/多方向文本相似图提供参考。

## P39. ConInfer: Context-Aware Inference for Training-Free Open-Vocabulary Remote Sensing Segmentation

- **领域/出处**：遥感开放词汇，arXiv 2026
- **论文**：https://arxiv.org/abs/2603.29271
- **核心机制**：不再对 patch 独立预测，而是在多个空间单元之间进行联合推断，显式建模其空间和语义依赖。
- **阅读价值**：极小目标的单 patch 响应容易不稳定；联合上下文推断可用于判断孤立高响应是目标还是杂波。

---

# 第五部分：跨模态蒸馏、文本缺失与纯图像部署

## P40. PromptKD: Unsupervised Prompt Distillation for Vision-Language Models

- **领域/出处**：视觉语言模型蒸馏，CVPR 2024
- **官方页面**：https://openaccess.thecvf.com/content/CVPR2024/html/Li_PromptKD_Unsupervised_Prompt_Distillation_for_Vision-Language_Models_CVPR_2024_paper.html
- **核心机制**：利用较大 CLIP 教师的领域 prompt，把知识蒸馏给轻量学生；可利用无标注领域图像进行 prompt/domain knowledge transfer。
- **阅读价值**：模型二不应只做普通 cosine regression。需研究 prompt 本身如何作为领域知识载体，以及教师、学生的文本/视觉分支分别保留什么。

## P41. CLIPSelf: Vision Transformer Distills Itself for Open-Vocabulary Dense Prediction

- **领域/出处**：开放词汇密集预测，ICLR 2024 Spotlight
- **论文**：https://arxiv.org/abs/2310.01403
- **代码**：https://github.com/wusize/CLIPSelf
- **核心机制**：将图像 crop 的全局 CLIP 表示作为教师，与原图 dense feature map 中对应区域的表示对齐，使原本擅长图像级识别的 CLIP 获得局部 region-language alignment，且不需要 region-text pairs。
- **为什么高度相关**：当前模型二若只拟合全局文本特征，仍可能缺乏像素定位。CLIPSelf 提供“全局语义 → 局部密集语义”的自蒸馏范式。
- **Codex 必查**：crop 对应关系、区域采样、teacher/student 是否同一 ViT、局部对齐 loss、对小物体区域的采样策略。

## P42. Learnable Prompting SAM-induced Knowledge Distillation for Semi-supervised Medical Image Segmentation（KnowSAM）

- **领域/出处**：医学半监督分割，TMI 2025 / arXiv 2024
- **论文**：https://arxiv.org/abs/2412.13742
- **代码**：https://github.com/taozh2017/KnowSAM
- **核心机制**：两个子网络进行 multi-view co-training；learnable prompt strategy 动态产生 dense prompt 并适配 SAM；SAM 再将知识蒸馏回子网络。子网络预测还会作为 SAM 的 mask prompt，形成双向信息交换。
- **关键价值**：教师与学生不必是单向固定关系，可以通过学生生成 prompt、SAM 修正、再蒸馏回学生的闭环学习。

## P43. Learning from Noisy Prompts: Saliency-Guided Prompt Distillation（SPD）

- **论文**：https://arxiv.org/abs/2604.23314
- **在蒸馏部分的阅读重点**：不要只看其医学上下文，应提取“噪声 prompt → saliency prior → consensus prompt → robust mask”的蒸馏对象和拒绝机制，评估能否用于自动文本不可靠时的教师净化。

## P44. SimIR 的教师—学生蒸馏

- **论文**：https://arxiv.org/abs/2409.04714
- **在蒸馏部分的阅读重点**：重点追踪学生为何能够超过教师；如果原因来自任务特定结构、query 设计或教师软标签，而非单纯 feature imitation，则模型二应蒸馏“任务决策”而不是“文本向量”。

## P45. JinSight 的训练有语言、部署检测范式

- **论文**：https://arxiv.org/abs/2608.07015
- **在蒸馏部分的阅读重点**：确认其检测阶段是否仍需要语言输入。如果语言只用于表征预训练而部署为视觉检测器，它可能比当前“先训练文本教师，再单独训练学生”的两模型结构更简洁。

---

# 第六部分：通用像素级语言模型与 SAM 接口

这些工作不一定适合直接部署到红外小目标，但用于理解“语言 token 怎样与像素建立对应关系”。

## P46. LISA: Reasoning Segmentation via Large Language Model

- **领域/出处**：推理分割，CVPR 2024
- **官方页面**：https://openaccess.thecvf.com/content/CVPR2024/html/Lai_LISA_Reasoning_Segmentation_via_Large_Language_Model_CVPR_2024_paper.html
- **核心机制**：在多模态 LLM 中引入 segmentation token，并将其 hidden state 连接到 SAM 类分割器。
- **阅读重点**：`<SEG>` token 的监督、是否包含空间信息、为何 READ 认为还需 similarity-as-points。

## P47. GLaMM: Pixel Grounding Large Multimodal Model

- **领域/出处**：像素 grounding 与 grounded conversation，CVPR 2024
- **论文**：https://arxiv.org/abs/2311.03356
- **核心机制**：语言回答与对应目标 mask 交织输出，同时接受文本及可选视觉区域 prompt；以大规模 grounded 数据训练像素级语言 grounding。
- **阅读重点**：大规模区域—文本数据如何构造；其像素 grounding 是否主要依赖数据规模；对当前小规模红外数据是否可蒸馏而非全量训练。

## P48. PixelLM: Pixel Reasoning with Large Multimodal Model

- **领域/出处**：像素级多目标推理，CVPR 2024
- **官方页面**：https://openaccess.thecvf.com/content/CVPR2024/html/Ren_PixelLM_Pixel_Reasoning_with_Large_Multimodal_Model_CVPR_2024_paper.html
- **核心机制**：使用 segmentation codebook / 多个分割 token 表达多目标，并通过轻量像素 decoder 输出 mask。
- **阅读重点**：单个 `<SEG>` token 与多 token/codebook 的差别；多目标匹配；目标级 refinement loss。

## P49. F-LMM: Grounding Frozen Large Multimodal Models

- **论文**：https://arxiv.org/abs/2406.05821
- **在通用接口部分的阅读重点**：对比“训练专用 `<SEG>` token”与“读取冻结 LMM 的 word–pixel attention”两种接口，判断当前项目是否不应继续把最终文本 embedding 当作唯一知识源。

## P50. SEEM: Segment Everything Everywhere All at Once

- **领域/出处**：通用多提示分割，NeurIPS 2023
- **项目/论文**：https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once
- **核心机制**：将文本、点、框、涂鸦、mask 等统一到共享视觉—语义 prompt 空间，并允许不同提示组合。
- **阅读重点**：不同 prompt 是否可互相补充或替代；如何在训练中随机缺失某一模态，使模型在无文本或无点时仍可工作。

---

# 第七部分：建议重点抽取的候选机制，而非直接形成最终方案

Codex 在完成精读前不得将以下内容直接写成“本文方法”。它们只是需要被验证的机制方向：

1. **文本—图像早期联合编码**  
   参考 EVF-SAM、RS2-SAM 2。核查早期融合是否比单独 CLIP 文本编码更能形成空间 prompt。

2. **文本 token 的相似图空间化**  
   参考 READ、F-LMM、RSRefSeg。核查 token→pixel similarity、attention map 或 attribution map 是否能替代无坐标 sparse token。

3. **隐式位置图与 dense mask prompt**  
   参考 Simple-ViLMedSAM、MedCLIP-SAMv2、RS2-SAM 2。核查文本是否应生成 dense prompt，而非直接模拟人工点。

4. **多角色文本或结构化语义**  
   参考 FIANet、DGSPNet、JinSight。分离场景、目标外观、空间关系、存在性和置信度，避免全部压缩进一个 global embedding。

5. **自动 prompt 的多类型联合表达**  
   参考 MaskSAM、RSPrompter。研究 query、coarse mask、box/point、存在性 token 是否应协同，而不是只使用一种提示。

6. **提示可靠性与拒绝机制**  
   参考 SPD、SegEarth-OV3、SAM-SP。自动文本或自动点错误时，模型是否有 presence gate、consensus、uncertainty 或迭代修正。

7. **高分辨率密集语义恢复**  
   参考 SegEarth-OV、SOPSeg、IRSAM、SegEarth-R2。先解决 tiny target 在 CLIP/SAM 中的空间信息丢失，再评估文本价值。

8. **训练期语言监督、测试期纯图像**  
   参考 JinSight、AlignEarth、PromptKD、CLIPSelf、SimIR。比较预训练迁移、feature distillation、prompt distillation 与 mask-function distillation。

9. **文本作为伪标签质量控制而非直接定位器**  
   参考 SeViL。核查文本是否更适合判断“是否像目标/伪标签是否可信”，而不是预测精确位置。

10. **上下文或多视图产生任务特定 prompt**  
    参考 TEP-SAM、One Shot IRSTS、ConInfer。即使当前为单帧，也可研究增强视图、尺度视图、频率视图或模型视图之间的一致性提示。

---
# 第八部分：Codex 的具体执行要求

## 8.1 工作分支与范围

建议新建独立文献分支：

```bash
git checkout -b codex/text-image-sam-literature-review
```

本阶段只允许新增或修改：

```text
docs/literature/
docs/ideas/
docs/experiments/
references/
```

**禁止在本阶段直接修改训练代码、模型结构、配置文件或现有实验结果。**  
若发现当前分支实现与实验记录不一致，只在审计文档中记录文件路径、代码行和问题，不立即修复。

## 8.2 检索与核验原则

1. 优先使用论文官方页面、arXiv、CVF Open Access、AAAI、IEEE 和作者官方代码；
2. 每篇论文至少阅读：摘要、Introduction、Method、Experiments、Ablation、Limitations、Supplementary；
3. 有代码时必须追踪真实推理入口，不能只根据论文流程图总结；
4. 若论文使用 SAM，必须明确：
   - SAM / MobileSAM / SAM2 / SAM3 的具体版本；
   - image encoder、prompt encoder、mask decoder 哪些冻结、微调或替换；
   - prompt 在训练与推理阶段分别来自哪里；
   - 是否存在隐藏的 GT 派生 point、box 或 coarse mask；
5. 若论文使用文本，必须明确：
   - 文本是 free-form、类别名、模板、属性、软 prompt 还是自动 caption；
   - 文本是否逐图变化；
   - 推理是否调用文本编码器或外部 VLM；
   - 文本如何获得空间信息；
6. 若论文宣称 prompt-free、automatic 或 text-only，必须用代码路径验证，不接受标题层面的判断；
7. 所有“可迁移 idea”必须注明其原始论文来源和潜在创新冲突，禁止把已有模块重新命名后视为新方法。

## 8.3 每篇论文统一记录模板

为每篇论文建立以下条目：

```markdown
### [Paper ID] Title

- Citation:
- Venue / year:
- Paper URL:
- Code URL:
- Code license:
- Domain / task:
- Base SAM version:
- Other foundation models:
- Input modalities:
- Text source:
- Training supervision:
- Train-time prompt source:
- Test-time prompt source:
- Requires GT-derived prompt at inference: Yes / No / Unclear
- Requires text at inference: Yes / No / Optional / Unclear
- Requires external VLM at inference: Yes / No / Offline only / Unclear
- Trainable and frozen modules:
- Text–image fusion location:
- Text spatialization mechanism:
- SAM prompt type: point / box / sparse token / mask / dense map / query / none
- Main losses:
- Small-object or fine-detail mechanism:
- False-alarm suppression mechanism:
- Handling of wrong/missing text or noisy prompt:
- Distillation mechanism:
- Main datasets and metrics:
- Most relevant ablations:
- Failure cases / limitations:
- Exact code path of the key mechanism:
- Difference from current TIRST-SAM implementation:
- Transferable mechanism:
- Novelty collision risk: High / Medium / Low
- Evidence supporting the above judgment:
```

### 强制说明

不得使用以下模糊句式作为主要总结：

- “该方法融合了文本和图像”；
- “该方法利用 CLIP 提升 SAM”；
- “该方法生成自动 prompt”；
- “该方法具有较好效果”。

必须写清楚**哪个张量，在什么位置，通过什么算子，变成了什么 prompt，推理时依赖什么输入**。

## 8.4 需要生成的文件

### 文件 1：文献矩阵

路径：

```text
docs/literature/01_TEXT_IMAGE_SAM_PAPER_MATRIX.md
```

至少包含以下列：

| 字段 | 内容 |
|---|---|
| Paper ID | P01–P50 |
| Domain | IR / remote sensing / medical / general |
| Text source | manual / fixed / learned / auto-caption / image-to-text / none |
| Fusion stage | pre-encoder / encoder / prompt encoder / decoder / post-selector |
| Spatialization | CAM / similarity / attention / point / box / mask / query / dense feature |
| Test-time GT prompt | yes / no / unclear |
| Test-time text | yes / no / optional |
| External VLM inference | yes / no / offline |
| Small-object mechanism | 具体机制 |
| Distillation | 具体蒸馏对象 |
| Code available | yes / no / partial |
| Relevance | S / A / B |
| Collision risk | high / medium / low |

### 文件 2：S 级论文精读笔记

路径：

```text
docs/literature/02_S_TIER_DEEP_READING_NOTES.md
```

必须至少精读以下 14 篇：

```text
P01 SAIST
P02 SAM-SPL
P03 IRSAM
P04 SimIR
P05 DGSPNet
P06 JinSight
P07 SeViL
P11 EVF-SAM
P12 READ
P16 Simple-ViLMedSAM
P18 RS2-SAM 2
P33 SegEarth-R2
P35 AlignEarth
P28 SPD
```

每篇必须包含：方法流程、关键公式、网络插入位置、训练/推理差异、核心消融、失败案例和与当前项目的冲突分析。

### 文件 3：公开代码路径审计

路径：

```text
docs/literature/03_REFERENCE_CODE_PATH_AUDIT.md
```

至少审计有公开代码的以下项目：

```text
SAM-SPL
IRSAM
SimIR
MoPKL
EVF-SAM
READ
MedCLIP-SAM
MedCLIP-SAMv2
Simple-ViLMedSAM
RSRefSeg
RSPrompter
FIANet
SegEarth-R2
SegEarth-OV
CLIPSelf
KnowSAM
One Shot IRSTS
Training-Free RS Segmentation
```

每个项目记录：

```text
repository
commit hash inspected
license
inference entry
model construction entry
prompt generation file/function
text encoder file/function
fusion file/function
loss file/function
checkpoint dependency
whether code matches paper
reusable component boundaries
```

### 文件 4：当前项目机制对照图

路径：

```text
docs/ideas/04_CURRENT_METHOD_VS_LITERATURE_MECHANISM_MAP.md
```

将当前分支的实际实现拆成：

```text
image encoder
text generation / loading
text encoder
CBGA
ASSP / prompt projection
SAM prompt encoder
SAM mask decoder
losses
train-time prompt source
test-time prompt source
```

然后与 S 级论文逐项对齐。必须指出：

- 已有研究已经覆盖的机制；
- 当前实现只是工程差异、尚不足以构成创新的部分；
- 尚未被当前实现利用的文献机制；
- 文本在当前系统中到底提供了新信息，还是只重编码了图像信息。

### 文件 5：候选 Idea 空间，不少于 8 个方向

路径：

```text
docs/ideas/05_TIRST_SAM_IDEA_CANDIDATES.md
```

每个候选方向必须使用以下模板：

```markdown
## Idea X: 暂定名称

### 研究假设
一句可证伪的假设，不写宣传性语言。

### 文献来源
列出直接启发论文及具体机制，不得只列标题。

### 与现有工作的实质差异
对比 SAIST、SAM-SPL、DGSPNet、JinSight、SeViL 和当前代码。

### 为什么可能适合红外小目标
从低信噪比、目标像素占比、背景杂波、尺度和边界角度解释。

### 最小实现
只列最少必要模块、输入输出张量和插入位置。

### 推理依赖
明确是否需要文本、外部 VLM、人工提示、参考图像或 GT 派生信息。

### 可证伪实验
用 1–3 个最小实验判断该方向是否值得继续。

### 预期主要改善指标
Pd / Fa / mIoU / boundary / cross-domain 中只能选择有依据的指标。

### 最大风险
包括文本不可靠、空间分辨率不足、教师上限、训练不稳定、创新重叠等。

### 新颖性风险
High / Medium / Low，并给出证据。

### 实施成本
Low / Medium / High。
```

候选 Idea 必须覆盖不同范式，不能全部是 cross-attention 的变体。至少包括：

1. 一种 text-to-dense-localization 范式；
2. 一种 early-fusion 或 internal semantic conditioning 范式；
3. 一种 prompt validation / rejection 范式；
4. 一种高分辨率 dense vision-language feature 范式；
5. 一种训练有语言、测试纯图像的蒸馏范式；
6. 一种自提示或迭代提示范式；
7. 一种多目标 query / role-token 范式；
8. 一种文本不直接定位、只做存在性或伪标签质量控制的范式。

### 文件 6：创新冲突矩阵

路径：

```text
docs/ideas/06_NOVELTY_COLLISION_AUDIT.md
```

至少以以下工作为列进行冲突核查：

```text
SAIST
SAM-SPL
DGSPNet
JinSight
SeViL
IRSAM
SimIR
EVF-SAM
READ
Simple-ViLMedSAM
RS2-SAM 2
SegEarth-R2
AlignEarth
```

冲突维度包括：

```text
same problem statement
same text source
same fusion location
same prompt form
same spatialization
same distillation target
same training/inference dependency
same small-target enhancement
same loss
same experimental claim
```

### 文件 7：最小验证实验排序

路径：

```text
docs/experiments/07_IDEA_SCREENING_EXPERIMENTS.md
```

只为排名前 3 的候选 Idea 设计实验。每个 Idea 先给出：

- 一个低成本 sanity check；
- 一个能判断文本是否真正被使用的 control；
- 一个能判断空间提示质量的 oracle/control；
- 一个 no-text / shuffled-text / wrong-text 对照；
- 一个不使用 GT prompt 的严格推理脚本；
- 明确的停止条件。

本文件完成前，不开始大规模多数据集训练。

### 文件 8：BibTeX

路径：

```text
references/text_image_sam_related.bib
```

只使用论文官方 BibTeX 或出版社 BibTeX。预印本与正式发表版本不得重复保留；若尚无正式版本，明确标注 `@article` 或 `@misc` 的 arXiv 信息。

---

# 第九部分：推荐阅读顺序

## Phase R1：直接创新冲突核查

按顺序阅读：

```text
SAIST → SAM-SPL → DGSPNet → JinSight → SeViL → IRSAM → SimIR
```

完成后先回答：

1. 当前项目“自动文本 + SAM”是否已经被 SAIST 或 DGSPNet 覆盖；
2. “无人工提示”是否已被 SAM-SPL 更简洁地解决；
3. “训练有语言、测试无语言”是否已被 JinSight 或 SimIR 覆盖；
4. 当前文章还可以在哪个明确问题上形成新的、可证伪的研究命题。

未回答这四点前，不进入最终 Idea 设计。

## Phase R2：文本空间化

```text
EVF-SAM → READ → Curriculum Point Prompting → MedCLIP-SAMv2
→ Simple-ViLMedSAM → RSRefSeg → RS2-SAM 2 → F-LMM
```

重点比较：

- global token；
- similarity / attention map；
- CAM / attribution map；
- point/negative point；
- pseudo-mask dense prompt；
- multimodal class query。

## Phase R3：自动提示与小目标保持

```text
RSPrompter → MaskSAM → SAM-SP → PPO → IRSAM
→ SegEarth-R2 → SegEarth-OV → SOPSeg → SPD
```

重点回答：

- prompt generator 的召回上限；
- 高分辨率细节是否先于文本成为瓶颈；
- 错误 prompt 如何校验；
- 多轮 prompt 是否会放大漏检。

## Phase R4：纯图像部署与蒸馏

```text
AlignEarth → PromptKD → CLIPSelf → KnowSAM → SimIR → JinSight
```

重点比较蒸馏对象：

```text
global text embedding
region-language feature
dense cost map
prompt embedding
query set
mask logits
teacher uncertainty
complete prompt-to-mask behavior
```

## Phase R5：Idea 归纳

只有在完成 R1–R4 后，才生成候选 Idea。每个候选必须能用一句可证伪假设表述，例如：

```text
在无 GT prompt 推理条件下，机制 A 是否能显著提高 Top-k 目标级 prompt recall，
并在不降低 Pd 的前提下降低 Fa？
```

不要使用以下不可证伪表述：

```text
充分利用文本语义增强红外小目标特征；
有效融合多模态信息；
提升模型对复杂背景的感知能力。
```

---

# 第十部分：必须进行的真实性与公平性审计

## 10.1 Prompt 来源审计

对每篇方法和当前项目分别填写：

| 阶段 | point | box | mask | text | reference image | external VLM |
|---|---|---|---|---|---|---|
| train | 来源 | 来源 | 来源 | 来源 | 来源 | 来源 |
| val | 来源 | 来源 | 来源 | 来源 | 来源 | 来源 |
| test | 来源 | 来源 | 来源 | 来源 | 来源 | 来源 |

任何由 GT mask 中心、连通域、外接框、距离变换或随机前景点产生的提示，都必须标为 **GT-derived prompt**。

## 10.2 文本真实性审计

必须区分：

- GT mask/标签直接生成的 oracle 文本；
- 利用整张测试图像自动生成的 caption；
- 利用目标 crop 生成的文本；
- 人工描述；
- 固定模板；
- 训练得到的 soft prompt；
- 图像分支预测的连续“文本空间”向量。

后两者不能直接称为“自然语言理解”，除非确实经过语言解码或有文本级监督。

## 10.3 公平评测审计

- val 仅用于选 epoch、threshold 和超参数；
- test 不用于训练中选 best checkpoint；
- 同一数据划分、输入尺寸和后处理；
- no-text、shuffled-text、wrong-text、fixed-text、oracle-text 必须使用完全相同的视觉模型容量；
- 统计文本对 `Pd` 与 `Fa` 的独立作用；
- 对目标像素面积分桶，至少报告极小、小、中等目标；
- 对背景复杂度或 SCR 分桶；
- 最终关键结果至少 3 个随机种子。

---

# 第十一部分：本轮文献工作的验收标准

本任务只有同时满足以下条件才算完成：

- [ ] P01–P50 均进入文献矩阵，无法访问者说明原因；
- [ ] 14 篇 S 级论文完成全文精读；
- [ ] 至少 18 个公开仓库完成真实代码路径追踪；
- [ ] 当前项目的 train/test prompt 来源被代码级确认；
- [ ] 至少 8 个候选 Idea，且来自不同机制范式；
- [ ] 每个 Idea 都有 novelty collision audit；
- [ ] 排名前 3 的 Idea 各有低成本可证伪实验；
- [ ] 不把已有模块的改名、串联或简单 cross-attention 视为新颖贡献；
- [ ] 不在文献阶段擅自改动模型代码或启动大规模训练；
- [ ] 所有结论均附论文页码、公式、表格或代码路径作为证据。

---

# 第十二部分：给 Codex 的直接执行指令

可将以下内容作为执行摘要：

```text
请以本文件为唯一主任务文档，对文本—图像—SAM、自动提示、遥感小目标、医学文本分割、
像素级视觉语言 grounding 和跨模态蒸馏研究开展系统精读。

当前阶段只做文献、补充材料和公开代码核验，不修改 TIRST-SAM 模型代码，不启动训练，
不提前替用户决定最终方法。

首要目标不是罗列论文，而是回答：
1. 文本到底通过什么机制获得像素空间定位能力；
2. 哪些方法在推理中真正不使用 GT 派生提示；
3. 极小目标情况下，瓶颈是文本语义、视觉分辨率、prompt recall 还是 mask decoder；
4. 如何把训练期跨模态能力迁移到测试期纯图像模型；
5. 哪些候选方向与 SAIST、SAM-SPL、DGSPNet、JinSight、SeViL 已经重合。

严格生成任务书中规定的 8 个文件。所有结论必须有论文页码/公式/表格或代码文件/函数证据。
对于有公开代码的论文，必须核验真实 inference path，不能只根据摘要和网络图总结。
完成文献与创新冲突审计后，再提出不少于 8 个相互有实质差异的候选 Idea，
并仅为排名前 3 的方向设计最小可证伪实验。
```

---

## 参考说明

本文档中的论文链接优先采用 arXiv、CVF Open Access、AAAI、IEEE 或作者官方仓库。部分 2025–2026 工作仍为预印本，Codex 在生成 BibTeX 和引用状态时需再次核实是否已有正式发表版本。
