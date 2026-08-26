# 用户补充 PDF 的扩展全文阅读

回填日期：2026-08-26。本文记录本次新增的四篇 A 级全文；S12 PromptPilot 与 S15 SAM-SPL 已直接回填到 `02_S_TIER_DEEP_READING.md`。所有数值均来自用户提供的正式出版 PDF，不使用 README 数字替代论文证据。本地 PDF 不进入 Git。

## 1. A05 MUP-SAM

正式题名：*MUP-SAM: Multi-scale Vision Mamba UNet Prompt Generation for SAM in Multi-organ Medical Image Segmentation*，*Neural Networks* 204 (2026) 109106，DOI `10.1016/j.neunet.2026.109106`。

1. **研究问题**：把依赖人工box的MedSAM改为全自动多器官分割，并处理网络自动生成box时的错框、漏框和碎片。
2. **完整架构**：MSVM-UNet先产生多类`y_prompt`；形态学、box expansion和NMS把连通域变成带类别的box；冻结MedSAM逐box产生`y_sam`；另一个U形prediction fusion network根据`[image,y_prompt,y_sam]`学习逐像素融合权重。
3. **Prompt表示**：标准box坐标，经MedSAM原生prompt encoder；每个连通域/box独立解码后按类别写回logits。无box时直接输出全零mask，不是learned no-object。
4. **训练和GT边界**：先以mask GT训练MSVM-UNet，再冻结它和MedSAM，以mask GT训练fusion network。推理只需要图像，但其“self-prompt”本质是一个完整监督分割器产生box。
5. **关键消融**：仅prompt generator时Synapse/ACDC Dice为80.85/56.40；加入prompt post-processing后为83.02/56.47；再加prediction fusion后为86.43/93.19。尤其ACDC的大幅提升来自输出融合，而不是box prompt改进。
6. **后处理结论**：box expansion、形态学和NMS单独均有小幅收益，三者联合达到Synapse 86.43、ACDC 93.17。论文指出扩张5–15像素有效，过大则引入背景。
7. **结果边界**：Synapse 86.43 Dice/13.31 mm HD95，ACDC 93.17 Dice；但Synapse仍低于其表中的nnU-Net 88.10 Dice。MedSAM在ACDC的自动box结果很差，最终融合主要由auxiliary branch兜底。
8. **计算开销**：MSVM-UNet约55.82M参数、约120G FLOPs；两阶段各最多300 epochs；冻结MedSAM并不意味着整个系统轻量。
9. **对当前项目的启示**：`aux segmentor→box→SAM`和`aux mask+SAM mask融合`都已有正式先例。若当前方法依赖一个强IRSTD网络先分割再产生box，贡献很容易退化为外部检测器/分割器包装。
10. **可迁移但不宜主张创新的部分**：零候选直接输出空mask、错prompt后response fusion、box扩张的容错实验。

质量评分：洞察4/5，完整性4/5，数值证据4/5。主要风险是fusion network承担了大部分最终性能，prompt机制的独立贡献有限。

## 2. A08 Dual Visual Prompt Tuning（DVPT）

正式题名：*Taming Large Vision Model for Medical Image Segmentation via Dual Visual Prompt Tuning*，*Computerized Medical Imaging and Graphics* 124 (2025) 102608，DOI `10.1016/j.compmedimag.2025.102608`。

1. **研究问题**：在不人工点击的情况下，同时补偿SAM对医学局部纹理的不敏感和全局边界/噪声建模不足。
2. **完整架构**：冻结SAM ViT-B image encoder；LFPT由六层卷积stem产生局部dense feature prompt，并在每个ViT layer后通过cross-attention调制encoder feature；GGP从SAM四个global-attention layer的feature中抽取全局prompt tokens；自定义CNN mask decoder消费两类prompt。
3. **Prompt表示**：local prompt为`H/64×W/64×D_lp` dense feature，`D_lp=32`；GGP为四组`C_g×D_g` learned continuous tokens，`D_g=256`，多类任务按每类32 tokens配置。它不输出文本，也不依赖CLIP。
4. **Decoder接口**：最终local prompt与SAM image feature相加作为dense condition；GGP经self-attention后作为key/value，与feature做cross-attention。方法绕开SAM原生point/box prompt encoder，并使用自定义CNN decoder。
5. **训练和GT边界**：主模型以mask GT端到端训练LFPT、GGP和decoder，SAM image encoder冻结；500 epochs，单RTX 3090。正常推理只需要图像。
6. **关键消融**：ISIC2017 baseline为66.25 IoU，LFPT-only 72.21，GGP-only 72.68，二者联合75.24；说明global learned tokens本身已是强分支，并且local/global互补。
7. **GT prompt对照的正确解释**：论文在额外消融中用GT mask centroid/GT box与GGP比较；这仅是人工prompt上界对照，不是DVPT主推理路径。GGP优于该实验中的point/box，但不能外推到红外tiny目标。
8. **结果与限制**：ISIC2017为75.24 IoU/85.87 Dice；作者承认HD边界误差仍不理想、模型需要modality-specific pretraining，SAM encoder也不利于实时部署。
9. **对当前项目的启示**：`图像feature→全局learned prompt tokens`与`浅层local dense prompt`均已有正式方法。用CLIP image embedding或GPT/text embedding变成单一全局token，不能作为主要新颖性；必须有字段/候选级因果敏感性与tiny-target专属证据。
10. **可迁移机制**：把全局状态作为key/value而非直接feature addition；correct/zero/shuffled global-token反事实；local/global branch独立消融。

质量评分：洞察4/5，完整性5/5，数值证据4/5。主要碰撞对象是任何“image encoder自动生成global prompt feature”的方案。

## 3. A19 AutoPromptSeg

正式题名：*AutoPromptSeg: Automated Decoupling of Uncertainty Prompts with SAM for Semi-supervised Medical Image Segmentation*，*Computerized Medical Imaging and Graphics* 128 (2026) 102708，DOI `10.1016/j.compmedimag.2026.102708`。

1. **研究问题**：半监督3D医学数据没有人工prompt，如何由模型自身的不确定性自动选可靠点，并让V-Net与SAM-Med3D互补。
2. **完整架构**：Model A为V-Net coarse branch；Model B为SAM-Med3D 3D encoder/prompt encoder/mask decoder；DUPG从V-Net不确定性产生point prompts；CAFA对齐V-Net与3D Transformer feature；两个分支做supervised与consistency learning。
3. **不确定性分解**：MC dropout产生多次概率和log-variance；类别概率的方差之和表示epistemic uncertainty，预测方差的期望表示aleatoric uncertainty。
4. **Prompt Suitability Score**：`PSS=w_e(1-H_epistemic)+w_a(1-H_aleatoric)`；再与每类mean probability逐元素相乘，使用3D max-pooling实现NMS，最后每类选择Top-K三维坐标，默认`K=20`。
5. **训练和GT边界**：先只用标注数据预训练V-Net，再用标注/未标注数据训练双分支；GT不直接产生推理点。训练总计1200 epochs，4×RTX 3090；推理仍需MC dropout多次forward。
6. **关键反事实**：LA 10%标注时，低不确定度点为90.02 Dice/6.59 HD95，高不确定度点只有75.39/23.18；20%时为91.66/4.97 vs 86.05/10.88。低不确定度点明显优于“去困难区域纠错”的直觉。
7. **采点baseline**：10%标注时random/center/grid/DUPG Dice为85.24/83.50/84.22/90.02；20%时为87.14/85.98/86.19/91.66。`K=20`优于10/15/25，证明Top-K仍需任务级校准。
8. **模块消融**：去DUPG或CAFA均下降；10%标注时full/without DUPG/without CAFA Dice为90.02/88.13/87.78。可靠选点有效，但不是唯一增益来源。
9. **数字一致性风险**：摘要宣称Amos 2022为68.78%/71.28% Dice，正文Table 1对应AutoPromptSeg行是68.05%/70.68%。在作者勘误前，不引用这组数值做强对比。
10. **对当前项目的启示**：epistemic/aleatoric分解、低不确定度选点、NMS与Top-K都已有完整先例。当前PR #3如果只是把多视图support解释为uncertainty并据此筛点，新颖性不足；必须把可靠性作为query状态带入decoder，并证明置零/打乱会改变mask。

质量评分：洞察4/5，完整性4/5，数值证据4/5；因摘要与正文表格不一致降置信一级。

## 4. A50 S4M

正式题名：*S4M: 4-points to segment anything*，*International Journal of Computer Assisted Radiology and Surgery* 21 (2026) 1311–1319，DOI `10.1007/s11548-026-03689-x`。

1. **研究问题**：普通positive points没有角色和几何关系，医学交互分割常陷入逐点修错；如何用固定4点一次描述对象范围、方向和粗形状。
2. **Prompt表示**：extreme points分别使用top/bottom/left/right type embeddings；major/minor axis endpoints使用major/minor embeddings，并与位置编码相加。它扩展的是SAM prompt encoder的type vocabulary。
3. **四点生成**：训练/模拟测试从GT mask边界产生。major/minor使用PCA主轴和正交轴、投影排序、ROI膨胀后随机采样；extreme使用图像水平/垂直轴。真实用户实验则由三名临床人员人工点击。
4. **Canvas任务**：训练only的独立decoder只接收prompt embedding和位置编码、不接图像，预测GT mask的convex hull；独立Canvas token避免辅助任务污染正常mask token。推理时整个Canvas分支丢弃。
5. **GT和部署边界**：它不是自动self-prompt。训练prompt由GT mask模拟；测试需要人工4点或GT模拟点。其目标是提高标注/交互效率，不回答“新图像没有prompt怎么办”。
6. **关键结果**：八个超声/内镜数据集平均相对强SAM baseline提升3.42 mIoU；Endoscapes上S4M extreme/major-minor为77.2/77.3 mIoU，普通SAM+直接接这类未见prompt只有32.7/32.5。
7. **关键消融**：仅把4点加入训练仍有prompt-type interference；分离embedding、进一步role-aware encoding和带独立token的Canvas逐步改善。共享positive embedding会把“对象内部点”和“边界角色点”混为一类。
8. **人工有效性**：三名临床人员的manual extreme/major-minor结果为76.58/77.19，与模拟点77.24/77.32接近；但样本量只有三名标注者，结论应保持谨慎。
9. **失败边界**：多叶、碎片或V形对象的轴定义可能歧义；极值点重合时人工认知负担上升；4点不普适。
10. **对当前项目的启示**：S4M不能作为image-only baseline，但它提供一个强表示证据：candidate的角色不能在进入decoder前被压成同一种positive point。TB-Prompt若采用target/background或inner/outer角色，必须显式区分type embedding并做role shuffle反事实。

质量评分：洞察4/5，完整性4/5，数值证据4/5。其部署协议与当前IRSTD不同，只能作为prompt representation先例。

## 5. 合并后对研究路线的修订

| 原拟议命题 | 新全文带来的碰撞 | 修订后的最低可发表边界 |
|---|---|---|
| 浅层高分辨率feature生成self-prompt | SAM-SPL已在单帧IRSTD完整覆盖 | 必须增加candidate identity、独立mask、`∅`或response reject；不再把dense prompt本身当贡献 |
| 图像生成全局语义/text-like token | DVPT已覆盖multi-level image feature→global learned tokens | 必须证明字段/候选级correct-vs-shuffled敏感性，且优于同参数视觉tokenbaseline |
| 可靠性或uncertainty选点 | AutoPromptSeg已覆盖两类UQ、低不确定度、NMS和Top-K | reliability必须进入decoder/query并可通过mask反事实验证，不只用于keep/rank |
| auxiliary网络自动box→SAM | MUP-SAM已覆盖后处理box和输出融合 | 只能作为baseline；若主性能来自aux mask/fusion，不能宣称SAM self-prompt创新 |
| 结构化多点 | S4M已覆盖role-aware type与shape Canvas | 当前价值是candidate/role identity，不是“四点”本身；image-only仍要有自动来源 |
| SAM response/逐prompt credit | PromptPilot已覆盖DSC+LOO+agent动作 | RQ-Adapt仅保留单步、无在线GT的response proxy/refine-reject，不做完整RL/LOO主创新 |

## 6. 仍缺全文/代码

1. S14 Semantic AutoSAM正式4页PDF；当前只能用摘要说明`image→prompt embedding`，不能核验shape、loss和消融。
2. A43 SAM-RSIS正式PDF或supplement；当前只有正式题名、DOI和摘要级机制。
3. AlignSAM官方源码或作者重新发布链接；指定仓库仍返回404。
