# 文本—图像—SAM 相关论文矩阵

更新日期：2026-08-13。矩阵按 `manuscript_guide/TIRST_SAM_TEXT_IMAGE_LITERATURE_READING_FOR_CODEX.md` 的 P01–P50 编号整理。判断优先采用论文正文、补充材料与作者仓库；“无 GT”仅表示推理 prompt 不由测试 GT mask/box/point 派生，不代表训练无需标注。

缩写：`I2T`=图像生成连续文本潜变量，`VLM`=视觉语言大模型，`T`=文本，`V`=图像，`离线`=部署前已缓存。碰撞风险指与当前 TIRST-SAM“文本条件 SAM + 图像学生/自动提示”的实质重叠，而非论文质量。

字段映射说明：为控制表宽，“论文（领域）”同时给出论文名与 Domain；“推理文本 / 外部 VLM”严格按“测试时是否需要文本 / 测试时是否需要运行外部 VLM”的顺序编码，因此它对应指南中的两个独立审计字段，而不是一个混合判断。

| ID | 论文（领域） | 文本来源 | 融合位置 | 空间化 / SAM prompt | 推理 GT prompt | 推理文本 / 外部 VLM | 小目标或细节机制 | 蒸馏对象 | 代码 | 相关性 | 碰撞 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P01 | SAIST（IR） | 自动 caption→模板描述 | SR-CLIP + SAM decoder | 文本/视觉 prompt token | 否 | 是 / 否（文本离线） | CG-SAM 背景建模 | MMD 对齐文本/视觉 prompt 分布 | 无 | S | 高 |
| P02 | SAM-SPL（IR） | 无 | encoder consult-guide、skip calibration | 浅层特征生成 self-derived dense prompts | 否 | 否 / 否 | 浅层高分辨率提示、互校准 | 无 | 是；正文受限 | S | 高 |
| P03 | IRSAM（IR） | 无 | SAM encoder + decoder | 无外部 prompt；mask/edge token | 否 | 否 / 否 | WPMD/PMD 保结构，GAD 多粒度 | 无 | 是 | S | 中 |
| P04 | SimIR（IR） | 无 | 通用分割 teacher→轻量 student | 学习的 sparse/dense query | 否 | 否 / 否 | 多尺度 query、RepViT | logits、通道 KL、query/表征 | 部分；未放蒸馏脚本 | S | 高 |
| P05 | DGSPNet（IR） | 固定模板 + I2T 连续 token | encoder TGCA、decoder TGSA | channel/spatial cost map，不接 SAM | 否 | 固定模板 / 否 | 多层细粒度 I2T token | 视觉—文本对比和重建预训练 | 无 | S | 高 |
| P06 | JinSight（IR） | 多任务结构化指令 | InternVL 预训练、LSI 检测器 | 语言只作训练表征监督 | 否 | 否 / 否 | LSI 低秩交换全局/局部语义 | 语言监督进入视觉编码器 | 未公开 | S | 高 |
| P07 | SeViL（moving IR） | GPT-4o 生成、固定格式描述 | teacher 的 TATE/TAPF/ACM | text-motion attention、候选过滤 | 否 | 否 / 否 | 运动特征 + 文本筛伪标签 | teacher→纯图像 student | 未确认公开完整度 | S | 高 |
| P08 | MoPKL（moving IR） | 同质结构化运动描述 | 训练期运动潜变量 | 无 SAM；语言约束运动先验 | 否 | 否 / 否 | 方向/速度/关系运动知识 | 语言先验→视觉潜变量 | 是 | B | 中 |
| P09 | One-shot IRSTS（IR video） | 无；一帧带 mask 参考 | 特征匹配→SAM | 相似图 top/bottom points | 否（但需参考帧 GT） | 否 / 否 | patch 匹配、多尺度投票 | 无 | 是 | B | 中 |
| P10 | TEP-SAM（IR video） | 无 | temporal cue→SAM | 时间涌现 dense/sparse prompt | 否 | 否 / 否 | 全局运动+局部偏差 | 无 | 无 | B | 中 |
| P11 | EVF-SAM（general） | 人工 referring expression | BEIT-3 early fusion→SAM | multimodal `[CLS]` sparse token | 否 | 是 / 否 | 双分辨率 VLM/SAM 特征 | 无 | 是 | S | 高 |
| P12 | READ（general） | 推理问题/表达 | LMM image-text token 层→SAM | `<SEG>` similarity→正负点 + token | 否 | 是 / 否 | patch similarity、DtoC 点选择 | 无 | 是 | S | 高 |
| P13 | Curriculum Point Prompting（general） | referring expression | CLIP→point generator→SAM | 课程式正/负点 | 否 | 是 / 否 | 从简单中心到复杂场景 | 弱监督 prompt 学习 | 无 | A | 高 |
| P14 | MedCLIP-SAM（medical） | 类别/器官名 | BiomedCLIP 后处理 | gScoreCAM→点/框→SAM | 否 | 是 / 否 | 域 CLIP、显著图阈值 | 无 | 是；多脚本流水线 | A | 中 |
| P15 | MedCLIP-SAMv2（medical） | 类别/器官名 | CLIP M2IB→SAM | 信息瓶颈 attribution map→点/框 | 否 | 是 / 否 | M2IB 保留判别区域 | 无 | 是；流水线分离 | A | 中 |
| P16 | Simple-ViLMedSAM（medical） | 简单类别名 | IPP + BID | attribution dense map | 否 | 是 / 否 | SAM affinity refinement | 无 | 部分；map 需预计算 | S | 高 |
| P17 | RSRefSeg（RS） | referring expression | SigLIP global/local filter→SAM | 激活 dense feature 作 sparse token 集 | 否 | 是 / 否 | 全局/局部文本分工 | 无 | 是 | A | 高 |
| P18 | RS2-SAM 2（RS） | referring expression | BEIT-3 union encoder、BHFM | pseudo-mask dense + multimodal sparse | 否 | 是 / 否 | 多层双向融合、边界损失 | 无 | 无 | S | 高 |
| P19 | Grounding DINO-US-SAM（medical） | 器官文本 | 检测器→SAM2 | Grounding DINO box | 否 | 是 / 否 | 超声域 LoRA | 无 | 无 | A | 中 |
| P20 | CLIP-Guided SAM（general） | 类别/表达 | semantic adapter 注入 encoder | CLIP T/V/similarity；兼容几何 prompt | 视模式 | 是 / 否 | 分层轻量 adapter | 无 | 无 | A | 高 |
| P21 | F-LMM（general） | 问题/短语 | frozen LMM attention→refiner | word–pixel attention map | 否 | 是 / 是（本地 LMM） | 多层 attention + SAM refinement | 无 | 无 | A | 中 |
| P22 | Emerging Pixel Grounding（general） | 问题/短语 | LMM/扩散 encoder attention | attend-and-segment map | 否 | 是 / 是（本地 LMM） | 扩散特征恢复局部性 | 无 | 项目页 | B | 中 |
| P23 | RSPrompter（RS） | 无 | SAM encoder→自动 prompt head | query/anchor 特征→latent point embeddings | 否 | 否 / 否 | FPN、多尺度 RoI/query | 无 | 是 | A | 高 |
| P24 | MaskSAM（medical） | 类别 token（学习） | auto-prompt generator→SAM | 类别 token + coarse mask + box | 否 | 否 / 否 | 3D adapter、多 prompt 协同 | 无 | 无 | A | 中 |
| P25 | AutoProSAM（medical） | 无/学习 prompt | 3D adapter + SAM | latent automatic prompt | 否 | 否 / 否 | 3D 多器官适配 | 无 | 是 | A | 中 |
| P26 | SAM-SP（medical） | 无 | 上轮 mask→下轮 SAM | 迭代 mask self-prompt | 否 | 否 / 否 | 自提示迭代 | self-distillation | 无 | A | 中 |
| P27 | Plug-and-Play PPO（general） | 无 | SAM 外部点优化器 | 异质图 + RL 更新 point | 初始点依来源 | 否 / 否 | 特征/坐标联合优化 | 策略学习 | 无 | A | 中 |
| P28 | SPD（medical） | 无；输入 noisy prompt | saliency prior→prompt filter→SAM | consensus point set | 否（仍需外部 noisy prompt） | 否 / 否 | 相邻切片一致性 | saliency/consensus prompt | 无 | S | 中 |
| P29 | TextSAM-EUS（medical） | CoOp 学习软文本 | BiomedCLIP + SAM LoRA | text-conditioned latent prompt | 否 | 软 prompt / 否 | 低对比超声域适配 | 无 | 无 | A | 高 |
| P30 | Training-Free RS Segmentation（RS） | 类别/推理文本或 VLM 自动 click | 后选择或 VLM→SAM | CLIP 选 mask；GPT/Qwen 产 click | 否 | 可选 / 是 | SAM 网格 proposals | 无 | 仅 README（无代码） | B | 中 |
| P31 | MedSAM3（medical） | 医学概念/agent 指令 | MLLM agent→SAM3 | 存在判断、分割、再提示 | 否 | 是 / 是 | 多轮质量修正 | 无 | 无 | B | 中 |
| P32 | FIANet（RS） | context/object/position 三角色文本 | 每层 PWAM/OPAB + TMEM | token-pixel attention dense feature | 否 | 是 / 否 | text-aware multiscale enhancement | 无 | 是 | A | 高 |
| P33 | SegEarth-R2（RS） | 自由语言指令 | MLLM `[SEG]`→Mask2Former | 多 `[SEG]` query + spatial attention | 否 | 是 / 是（本地 MLLM） | attention supervision、小目标 query | 无 | 是 | S | 中 |
| P34 | SegEarth-OV（RS） | 类名模板 | CLIP dense feature→classifier | patch-text similarity dense logits | 否 | 是 / 否 | SimFeatUp + global bias subtraction | 无 | 是 | S | 中 |
| P35 | AlignEarth（RS/SAR） | 类名模板（测试） | optical teacher→SAR encoder | dense patch-text cost map | 否 | 是 / 否 | local region distill 抗错位 | global/CLS/local region feature | 无 | S | 高 |
| P36 | SOPSeg（RS） | 几何 prompt | region zoom + custom decoder | oriented box/region prompt | 取决于框来源 | 否 / 否 | 区域放大、边缘渐进细化 | 无 | 无 | A | 低 |
| P37 | SegEarth-OV3（RS） | 类名模板 | SAM3 semantic+instance heads | presence score + mask | 否 | 是 / 否 | presence gate 降 false positives | 无 | 无 | B | 中 |
| P38 | RSKT-Seg（RS） | 类名模板 | VL cost maps + decoder | 多方向 dense cost map | 否 | 是 / 否 | 方向建模、增强上采样 | VLM→高分辨率 student | 无 | A | 中 |
| P39 | ConInfer（RS） | 类名模板 | 训练免费后推理 | patch 间上下文联合 logits | 否 | 是 / 否 | 空间单元联合推断 | 无 | 无 | B | 低 |
| P40 | PromptKD（general） | teacher domain prompts | CLIP teacher→student prompt | global VLM logits/prompt | 否 | 类名 / 否 | 域无标注迁移 | teacher logits、prompt/domain knowledge | 无 | A | 中 |
| P41 | CLIPSelf（general） | 无新增文本对 | crop teacher→dense student | region feature，不接 SAM | 否 | 类名用于下游 / 否 | crop 全局语义→局部区域 | 同一 ViT crop/ROI cosine | 是 | A | 高 |
| P42 | KnowSAM（medical） | 无 | 双子网↔SAM | 学习 dense/box embedding、mask prompt | 否 | 否 / 否 | multi-view co-training | SAM 修正结果回蒸馏子网 | 是；部分路径不一致 | A | 中 |
| P43 | SPD（P28 重复） | 同 P28 | 同 P28 | 同 P28 | 同 P28 | 同 P28 | 同 P28 | 同 P28 | 同 P28 | S | 中 |
| P44 | SimIR 蒸馏（P04 重复） | 无 | 同 P04 | 同 P04 | 否 | 否 / 否 | 同 P04 | 同 P04 | 部分 | S | 高 |
| P45 | JinSight 部署（P06 重复） | 训练期指令 | 同 P06 | 纯图像检测 | 否 | 否 / 否 | 同 P06 | 同 P06 | 未公开 | S | 高 |
| P46 | LISA（general） | reasoning instruction | LMM `<SEG>`→SAM | `<SEG>` hidden state sparse token | 否 | 是 / 是（本地 LMM） | SAM 像素 decoder | 无 | 是 | B | 中 |
| P47 | GLaMM（general） | grounded conversation | LMM region token→mask decoder | `[SEG]`/region token + 可选 region prompt | 否 | 是 / 是（本地 LMM） | 大规模 region-text 数据 | 无 | 是 | B | 中 |
| P48 | PixelLM（general） | reasoning instruction | LMM token codebook→decoder | 多 segmentation token/query | 否 | 是 / 是（本地 LMM） | 多目标 token、像素 refinement | 无 | 是 | B | 中 |
| P49 | F-LMM（P21 重复） | 同 P21 | 同 P21 | word-pixel attention | 否 | 是 / 是 | 同 P21 | 无 | 无 | A | 中 |
| P50 | SEEM（general） | 文本/点/框/涂鸦/mask | unified prompt space | shared query/prompt embedding | 取决于使用模态 | 可选 / 否 | 多模态随机缺失、组合 prompt | 无 | 是 | B | 中 |

## 指南外补充发现（不改 P01–P50 编号）

检索 P01–P50 时发现以下高冲突正式工作，虽不在原指南清单中，但不能在新颖性判断中忽略：

| 工作 | 正式来源 | 与候选方向的关系 |
|---|---|---|
| Instruction-Focus-Prompt (IFP) | [CVPR 2026 Findings](https://openaccess.thecvf.com/content/CVPR2026F/html/Xia_Instruction-Focus-PromptSemantics-Driven_Structural_Prompts_for_Universal_SAM_Segmentation_CVPRF_2026_paper.html) | DINOv3 semantic focus + text instruction→dense similarity→迭代 structural points；进一步压缩“文本空间化为点”的创新空间 |
| AutoPrompt-SAM3D | [BMC Bioinformatics 2026](https://link.springer.com/article/10.1186/s12859-026-06390-7) | SAM2 多层 feature 生成 prompt，并以 confidence frame filter/target-aware pre-screen 拒绝错误；Idea 1 必须强调单帧 IR、多视图 reliability 与 risk calibration |
| SAM-COD | [ECCV 2024](https://eccv.ecva.net/virtual/2024/poster/1896) | response filter、semantic matcher、prompt-adaptive KD；Idea 1/2 不能泛称首次 prompt 过滤或 prompt-aware distillation |
| SAM-guided prompt learning for MS lesions | [Pattern Recognition Letters 2026](https://www.sciencedirect.com/science/article/pii/S0167865525003708) | 图像生成 dense prompt，冻结 SAM 训练，部署以轻量 aggregator 从 prompt 到 mask；进一步说明自动 prompt 与 mask-decision 蒸馏并非空白 |
| RL-CP | [Biomedical Signal Processing and Control 2026](https://www.sciencedirect.com/science/article/abs/pii/S174680942601493X) | 跨样本 activation stability + RL/contrastive purification 生成正负点；多视图稳定性 prompt 需与其区分 |
| EdgeSAM | [arXiv 2312.06660](https://arxiv.org/abs/2312.06660) | prompt-in-the-loop distillation 已蒸馏 prompt encoder/mask decoder 动态；Idea 2 只能聚焦**文本反事实增量行为**，不能泛称首次完整 prompt-to-mask 蒸馏 |

这些补充发现使 Idea 1/2 的综合碰撞风险上调，但并不直接覆盖“红外极小目标 + 无文本部署 + correct/no/shuffle/wrong 反事实语义增量 + prompt-level Pd/Fa 风险校准”这一完整问题组合。

## R1：直接创新冲突核查

1. **“自动文本 + SAM”不是空白。** SAIST 已在红外小目标中使用自动生成并人工核查的描述；DGSPNet 更直接地用固定粗粒度文本加逐图 I2T 连续 token，并在推理时由图像内部产生个性语义。因此，仅把 Qwen 换成 GPT 并提高 caption 准确率属于教师/数据工程改进，不能单独构成方法创新。
2. **“无人工/无 GT prompt”已有更简单路线。** SAM-SPL、IRSAM、RSPrompter 和 SimIR 均可不依赖测试 GT 几何提示。当前方法必须与强纯图像 self-prompt 基线同协议比较，而不是只与 GT point 版本比较。
3. **“训练有语言、推理无语言”已有直接先例。** JinSight、SeViL、MoPKL 与 AlignEarth 分别覆盖语言表征监督、文本筛伪标签、语言运动先验和跨传感器局部蒸馏。单纯 cosine 回归 CLIP/GPT 文本 embedding 的碰撞风险为高。
4. **尚可验证的缺口。** 文献尚未同时解决：红外极小目标的高分辨率 prompt recall；错误/缺失文本与错误自动点的显式拒绝；将文本教师的**prompt→mask 行为**而非单一向量蒸馏给纯图像学生。后续 idea 围绕这三个可证伪问题展开。

## 全文与重复项状态

- 已获得并阅读正文：除 P02 外的全部唯一论文条目；P43=P28、P44=P04、P45=P06、P49=P21，不重复建立“新工作”结论。
- **需要用户提供：P02 SAM-SPL 正式论文 PDF**，DOI `10.1109/TGRS.2025.3610919`。IEEE 页面可核验题录和摘要，但当前环境无法下载正文。P02 的公式、表格、完整消融和论文—代码一致性均标为待补，不用摘要替代全文证据。
- P30 的论文正文可读，但作者仓库在审计时只有 README 与 teaser，代码仍标记为不可审计，而不是“论文不可读”。

## 主要来源入口

- [SAIST（CVPR Open Access）](https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_SAIST_Segment_Any_Infrared_Small_Target_Model_Guided_by_Contrastive_CVPR_2025_paper.html)
- [READ（CVPR Open Access）](https://openaccess.thecvf.com/content/CVPR2025/html/Qian_Reasoning_to_Attend_Try_to_Understand_How_SEG_Token_Works_CVPR_2025_paper.html)
- [PromptKD（CVPR Open Access）](https://openaccess.thecvf.com/content/CVPR2024/html/Li_PromptKD_Unsupervised_Prompt_Distillation_for_Vision-Language_Models_CVPR_2024_paper.html)
- [SAM-SPL 题录（DBLP）](https://dblp.org/rec/journals/tgrs/FuLMLN25)
- [SeViL（AAAI）](https://ojs.aaai.org/index.php/AAAI/article/view/37372)
- 其他 arXiv / 代码入口沿用阅读指南中的官方链接；引用键见 `references/text_image_sam_related.bib`。
