# S 级论文精读笔记

本文件记录 14 个直接决定创新边界的条目。公式均为结构化转述，便于定位而非替代原文；页码采用论文印刷页或 PDF 页中更稳定者。`证据状态` 分为正文、正文+代码、摘要+代码（仅 P02）。

## P01 SAIST

- **证据状态**：正文、补充材料；未发现作者公开代码。
- **流程与插入点**：Fig. 2（CVPR 论文 p.9551）显示两条支路：SR-CLIP 由 CLIP 文本编码器和冻结的 CLIP 图像编码器产生文字/视觉 prompt；SAM image encoder 以 LoRA 适配，prompt 送入 mask decoder。CG-SAM 在 SAM 路径中建模红外背景。
- **关键公式**：Sec. 3.1、Eq. (1)–(6) 构造描述及文本/视觉 prompt；Eq. (7)–(8) 用 MMD 对齐两类 prompt 分布；Sec. 3.2、Eq. (9)–(14) 是 CG-SAM 的背景条件增强。
- **文本来源与部署**：MIRSTD 描述由 VLM 生成，作者说明把“移除 target mask 的图像”以 base64 输入并由 ChatGPT/人工检查，再固定为约 25 词模板。原文措辞不足以确认是仅移除可视化 mask 还是对目标像素 inpaint，不能扩写成“严格无 GT 文本生成”。推理需要对应描述/缓存特征，但不使用 GT 点框。
- **核心消融**：Table 2：完整模型 IoU/nIoU/Fa 为 80.82/99.56/0.87；去 SR-CLIP 为 80.02/98.77/6.61；去 CG-SAM 为 78.87/97.42/8.53；两者都去除为 76.82/96.50/12.37。语义模块主要显著降低 Fa，而非证明其能独立精确定位。
- **失败/缺口**：未报告无文本、错文本、shuffle 文本、空场景和 caption 质量分层；逐图文本的真实获取成本未纳入部署比较。
- **与当前项目冲突**：当前 GPT 结构化 caption→CLIP feature→CBGA/ASSP 与其“自动描述 + CLIP + SAM”高度重叠。可区分点不能只是换更强 GPT 或改投影层，必须落在可靠性、高分辨率定位或纯图像行为蒸馏。

## P02 SAM-SPL

- **证据状态**：**摘要+作者代码；正式 PDF 未能下载**。下面仅陈述可由摘要和 commit `1bde7b5` 验证的内容，公式、表格和论文消融待用户提供 PDF 后补齐。
- **可核验流程**：摘要称 SAM image encoder 以 consult–guide 方式参与编码；浅层特征生成 self-derived prompts，并与潜表征双向交互；skip connections 使用 mutual calibration。代码 `sam_spl/image_encoder.py:68-89` 把浅层 dense feature 与 SAM 特征送入 `pmt_blocks`，`sam_spl/hieradet.py:270-272` 接回主干，prompt 生成器在 `sam_spl/pmt_generator.py`。
- **部署路径**：`testing.py:9-15` 仅把图像送入 predictor；`sam_spl/base_model.py:435` 从图像端到端 forward，没有测试 GT point/box/mask 输入。SAM adaptor 构造位于 `sam_spl/base_model.py:133`。
- **风险**：在没有正文的情况下，不能确认 dense/sparse prompt 的论文定义、双向交互公式、各 loss 权重、使用的具体 SAM2 变体及表格数字。代码根目录没有发现许可证文件。
- **与当前项目冲突**：它已经提供“纯图像、无人工 prompt、浅层高分辨率 self-prompt”的直接红外基线。若当前模型二只是图像→prompt 回归，创新冲突高；必须证明语言教师带来的决策知识超出 self-prompt。

## P03 IRSAM

- **证据状态**：正文+代码（commit `ec65744`）。
- **流程与插入点**：WPMD 在 SAM encoder 多层进行扩散式细节/噪声调节；GAD 用浅层和深层特征、mask token 与 edge token 产生动态卷积核。论文 Method Eq. (1)–(4) 为 WPMD，Eq. (5)–(9) 为 GAD，Eq. (10)–(12) 为 mask/edge Dice+BCE 目标。
- **训练/推理**：代码 `demo.py:133` 直接调用 `net(batched_input)`；`segment_anything_training/modeling/IRSAM_edge.py:112-115` 明确将 `points=None`。因此公开推理路径不是 GT-point SAM。
- **核心消融**：Table 3 分解 WPMD/GAD，Table 4 比较插入 block 数，Table 5 比较浅/深特征融合。说明视觉细节和 decoder 粒度本身是强贡献，文本实验必须在其上或与其公平比较。
- **代码疑点**：论文称 WPMD，但仓库关键文件名为 `PMD.py`，实现主要是梯度/扩散算子；需按公式复核而不能只按类名判断“完整一致”。根目录未发现许可证。
- **与当前项目冲突**：文本并非小目标 SAM 的必要条件。IRSAM 是隔离“视觉适配收益”和“文本收益”的强无文本参照。

## P04 SimIR

- **证据状态**：正文+推理代码（commit `47172b1`）；蒸馏训练脚本未公开。
- **流程与公式**：Semantic-SAM teacher 在约 1% SA-1B 上提供多粒度输出；RepViT-M1.1 student 学习 sparse/dense queries。LDIS 的 Eq. (1) 是 BCE 加权 Dice、vanilla KL 和 channel-wise KL；早期 mask prediction 作为 dense SAM prompt，学习 query 贯穿 encoder/FPN/decoder。
- **训练/推理差异**：teacher 仅训练期存在；最终 student 纯图像推理。`train_net.py:304-357` 调 `ARCH_SIRST.evaluate_sirst`，`semantic_sam/architectures/arch_sirst.py:352` 进入 SIRST 路径；`repvit11_query_deform.py:522-523` 返回 mask、early mask 与 sparse query。
- **消融**：Table 2 是从 generic model 到 SIRST student 的 design journey，并分析 query/蒸馏组合。正文核心证据是“任务结构化 query + 多级决策蒸馏”可让小模型超过通用 teacher，而非简单 feature cosine imitation。
- **可复现缺口**：README `95-97` 只发布 distilled backbone checkpoint，未发现生成 teacher targets/LDIS 的完整脚本，因此论文的关键蒸馏环节只能记为部分匹配。
- **与当前项目冲突**：模型二若仅回归 CLIP 文本向量，比 SimIR 的任务决策蒸馏更弱且不新。可行差异是蒸馏“文本改变 prompt 与 mask 的行为”。

## P05 DGSPNet

- **证据状态**：正文；未发现公开代码。
- **流程与插入点**：粗粒度文本为固定场景模板（如 sky/ground/ocean 中的 infrared target）；细粒度 token 不是自然语言，而是由前三个视觉 stage 经 visual-to-text inversion 得到的逐图连续潜变量。Eq. (2)–(3) 为 TGCA，Eq. (4)–(5) 生成 fine tokens，Eq. (7)–(9) 用 token-feature 点积形成 TGSA 空间调制。
- **训练/推理**：预训练 Eq. (10) 结合图文对比与重建 MSE；检测训练 Eq. (11) 用 BCE+SoftIoU。推理无需人工 caption/GT prompt，但仍使用固定模板的 text encoder 和图像生成 token。
- **消融**：Table 2 拆粗/细 prompt、TGCA/TGSA；Table 3 比较 token 数量。它已覆盖“固定通用文本 + 图像个性文本潜变量 + 通道/空间引导”。
- **失败/缺口**：视觉生成的 fine token 很可能重编码图像而非引入外部新知识；固定场景模板在未知场景的来源未完全交代；没有与 caption 错误/缺失的压力测试。
- **与当前项目冲突**：图像编码器拟合/生成 text-space feature 属于其近邻。必须把目标从“复现文本 embedding”升级为带空间、可靠性或行为约束的对象。

## P06 JinSight

- **证据状态**：正文；作者写明代码/数据将在发表后公开，当前没有可审计仓库。
- **流程与公式**：Stage I 对 InternVL2.5-1B 做六类 infrared instruction tuning；Stage II 移除 language model/projector，只保留视觉表征进入检测。Eq. (1) 为 instruction loss；Eq. (2)–(3) 的 LSI 在低秩空间双向更新全局视觉语义 `V` 与细粒度空间特征 `S`。
- **数据/监督**：OmniIRST-VL 含 39,701 条标注，覆盖详细 caption、物理 VQA、全局/区域 grounding/counting；构造过程使用 mask 与元数据，不能称为无标注 caption。
- **部署**：正文明确语言是训练期 representation supervision，dense prediction 时只输入 infrared image。这是“训练有语言、部署无语言”的直接红外先例。
- **消融**：Table 2 跨域结果；Table 4 IVIT/LSI；Table 5 低秩 rank；Table 6 指令组合。主要证据是多任务语义预训练和局部交互，而不是单一 GPT 描述准确率。
- **与当前项目冲突**：单独训练 image→CLIP text embedding 的叙事高度碰撞。差异必须落到 SAM prompt/function distillation、可靠性或可量化的高分辨率空间保持。

## P07 SeViL

- **证据状态**：AAAI 2026 正文；未完成公开代码可复现确认。
- **流程与公式**：GPT-4o 生成 13 种固定格式描述，CLIP 编码后由 TATE（Eq. (1)–(3)）形成 pixelwise text-motion attention。TAPF（Eq. (4)–(8)）对 pseudo box crop 取 CLIP image feature，与文本余弦相似，结合 harmonic confidence 和 GMM 阈值筛选伪标签；ACM（Eq. (9)–(11)）做跨模态 masking，总损失见 Eq. (12)/(13)。
- **训练/推理**：完整视觉语言 teacher 只用于半监督训练；部署只保留 Student，纯图像输入。
- **消融**：Table 3 分模块，Table 4 比较文本施加位置，Fig. 7 用 prompt Shapley 分析不同描述贡献。文本最清晰的作用是伪标签质量控制和难样本选择。
- **失败/限制**：moving IR 与单帧 SIRST 存在域差；teacher 参数量大；固定描述未做系统错文本/空文本测试。
- **与当前项目冲突**：它支持“文本做 verifier 而非坐标生成器”，也已覆盖语言训练、纯图像部署。新工作需对 prompt 可靠性做更直接、可测的建模。

## P11 EVF-SAM

- **证据状态**：正文+代码（commit `9935dc5`）。
- **流程与插入点**：BEIT-3 同时接收 224×224 图像与文本，early fusion 后的 multimodal `[CLS]` 经 projector 变为 `B×1×D` sparse prompt；原图另以 1024 分辨率进入冻结的 SAM image encoder。SAM prompt encoder 不接 points/boxes。
- **代码证据**：`inference.py:176-185` 传图像和表达；`model/evf_sam.py:170-197` 生成融合 embedding，`220-246` 计算 BCE/Dice。SAM mask decoder 接 sparse prompt 的路径可由同文件复核。
- **消融**：Table 4 比 early/late fusion，Table 5 比不同 foundation models，Table 6 比训练/冻结组件。证据支持“prompt 是否看过图像”比单纯扩大文本编码器重要。
- **部署/限制**：推理必须有 referring expression；全局 `[CLS]` 的空间定位依赖 early fusion，没有错误/缺失文本兜底。双编码图像增加计算。
- **与当前项目冲突**：CBGA 若只做 token-level late fusion，在机制上弱于其 early fusion；直接替换成联合 VLM 仍只是移植。可迁移的是“图像条件 prompt”，需结合 tiny-target 空间/可靠性新问题。

## P12 READ

- **证据状态**：正文+代码（commit `2549240`）。
- **流程与公式**：Eq. (5) 计算 `<SEG>` 与 image tokens 的相似度；从 top/bottom 响应提取正负点，Eq. (7)–(10) 用 differentiable top-k-to-coordinate（DtoC）保持可训练；Eq. (6) 将 `<SEG>` sparse token 与 points 一起送入 SAM。
- **代码证据**：`test_read.py:60` 是推理入口；`model/READ.py:212` 后构建语义 token，约 `298-314` 将点和 `<SEG>` embedding 送入 prompt encoder/decoder。
- **消融**：Table 6 中，仅 `<SEG>` 为 57.6 cIoU，加入 points 为 64.6，加入 DtoC 为 67.6；Table 5 分析 false-premise queries。说明 global semantic token 不等价于可靠空间点。
- **部署/限制**：推理需文本问题；对小于一个 LMM patch 的红外目标，top-k similarity 的 recall 未得到证明；多目标/空目标可能产生强制点。
- **与当前项目冲突**：把 token 转 similarity map/point 已有直接实现。新意应是 tiny-target 高分辨率恢复和“低置信不出点”的拒绝机制，而不是重复 top-k。

## P16 Simple-ViLMedSAM

- **证据状态**：CVPR 2026 正文+部分代码（commit `76d95d6`）。
- **流程与公式**：论文 p.30044–30049：Eq. (1)–(2) 得 CLIP image/text 表示，Eq. (3) 为 SAM encoder LoRA；M2IB Eq. (4)–(5) 生成 attribution map；SAM feature affinity Eq. (6)–(7) 传播/阈值细化；BID Eq. (8)–(11) 双向交互位置图和 SAM pixel feature；Eq. (12) 为训练损失。
- **训练/推理**：已知类别的简单文本在推理时仍必需。公开 `model/model.py:75-111` 的 BID 接收 `sam_image_feats` 与 `attribution_map`；`train.py:197-206`、`test.py:199-208` 从数据集中读取 attribution map，并未包含 M2IB 在线生成脚本。README 要求每个 split 预放 `attribution_map/`。
- **消融**：Table 3 比简单/复杂文本，Table 4 拆 IPP/affinity/BID。其价值是 image-text 共同形成 dense implicit position，而不是文本向量直接替代点。
- **限制**：论文承认 2D、单目标、已知标签限制；医学对象尺度和纹理与红外点目标不同；公开代码不能端到端复现 IPP。
- **与当前项目冲突**：dense text-conditioned prompt 与 ASSP/CBGA 高重叠。若迁移，必须证明针对极小目标的分辨率和拒绝机制，而不能仅复刻 attribution map。

## P18 RS2-SAM 2

- **证据状态**：正文；未发现官方公开代码。
- **流程与公式**：BEIT-3 union encoder 联合图文；BHFM 在 SAM2 encoder 层间做双向 cross-attention（Eq. (1)–(2)）；Mask Prompt Generator 根据 visual embedding 与 multimodal class token 生成 pseudo-mask dense prompt，同时 class token 作 sparse prompt。分割目标为 Eq. (3)，text-guided boundary loss 为 Eq. (4)。
- **训练/推理**：推理需要 referring text，但 pseudo mask 完全由当前图文输入生成，不依赖 GT point/box/mask。
- **消融**：Table 3 分 MPG/BHFM，Table 4 比融合位置/设置。核心证据是 dense+sparse prompt 协同以及文本参与边界约束。
- **限制**：遥感对象通常比红外点目标更有纹理；class token 对空目标/多同类目标的可靠性不明；无代码无法验证具体 checkpoint 与数据流。
- **与当前项目冲突**：token-level CBGA + global ASSP 与“层级融合 + dense/sparse prompt”机制碰撞高。候选方案需从可靠性或完整行为蒸馏重新定义问题。

## P33 SegEarth-R2

- **证据状态**：CVPR 2026 正文+代码（commit `72e1d18`）。
- **流程与插入点**：MLLM 接收图像与指令，多个 `[SEG]` hidden states 充当 segmentation queries；spatial attention supervision 直接约束 `[SEG]` 到 image patches 的 attention。最终 mask backend 使用 Swin-B + Mask2Former，而非必须使用 SAM。
- **公式/代码**：Method 的 spatial attention loss `L_S` 让目标区域 attention 与背景拉开；多 `[SEG]` 对应多目标 query。代码 `segearth_r2/datasets/dataset.py:275-311` 构建 `[SEG]` indices，`model/language_model/llava_phi.py:43-68` 实现 AttentionLoss，`eval/eval.py:226-250` 将指令与 `[SEG]` 送入模型。
- **消融**：Table 5 比 attention loss 权重；Table 6 显示 Swin+Mask2Former 后端优于 SAM/SAM2+ViT-H；减少冗余 queries 反而改善定位。
- **限制**：训练/部署均需大型 MLLM 与文本；attention 监督使用 mask GT，不能称为弱空间监督；系统远重于 TIRST-SAM。
- **与当前项目冲突**：它表明“文本 token 自然含空间”不可靠，需要显式 spatial supervision。可迁移的是 prompt attention 监督，但直接加 mask-supervised attention 可能只属常规辅助 loss。

## P35 AlignEarth

- **证据状态**：正文；未发现独立训练仓库。
- **流程与公式**：成对 optical–SAR 图像中，冻结 optical CLIP teacher，训练 SAR encoder student。Eq. (10) 为 global contrastive，Eq. (11) 为 CLS cosine distillation，Eq. (12) 对区域池化的 local feature 做蒸馏以缓解跨模态配准误差。
- **训练/推理**：训练需要成对跨传感器图像；测试仍使用 class-name templates 与文本编码器计算 open-vocabulary logits，但不需要逐图 caption 或外部 VLM，不能称为“完全无文本分类器”。
- **结果证据**：Table 9 报告 AlignEarth 平均 mIoU 30.2，对应 SkyCLIP 15.4；应理解为跨传感器 dense alignment 的价值，不等价于红外小目标性能。
- **相关前作机制**：SegEarth-OV 的 Eq. (9) 做 global bias subtraction（局部 patch 减 CLS 偏置），SimFeatUp 恢复高分辨率 dense semantics。
- **限制**：需要光学/SAR配对；SAR 结构与 IR 不同；没有 prompt→SAM 行为。将其类比到 GPT/CLIP teacher 时，只有 global+local feature regression 仍高度碰撞。
- **与当前项目冲突**：模型二“图像拟合文本特征”已被其多层蒸馏范式覆盖。潜在差异是 distill 高分辨率 prompt distribution 与 downstream mask response。

## P28 SPD

- **证据状态**：正文；未定位到官方公开代码。
- **流程与公式**：Stage I 用 Dice+Focal（Eq. (1)）训练 saliency prior；Stage II 依据 saliency 阈值过滤本切片/相邻切片 noisy prompts（Eq. (6) 附近），形成 consensus prompts，再由 prompt saliency consistency 约束 SAM。
- **训练/推理**：训练使用 GT mask 学 saliency；推理虽然不使用 GT-derived point，但仍假设有外部 noisy point/prompt，并利用医学相邻切片。它不是无提示模型。
- **消融**：Table 2 拆 saliency、contextual prompt distillation/consistency；另有 prompt 来源/噪声设置比较。贡献在“先校验再使用提示”。
- **限制**：相邻切片不适用于单帧 IR；若 saliency head 漏掉极小目标，过滤会删除唯一正确点；缺少空目标 abstention 的系统分析。
- **与当前项目冲突**：直接照搬 saliency filter 不新。可转换为多尺度/增强/频域多视图共识，并把“拒绝提示”作为首要输出，需用 prompt recall/Fa 验证。

## 跨论文结论

| 问题 | 证据结论 | 对 TIRST-SAM 的约束 |
|---|---|---|
| 更准确 caption 是否足够？ | SAIST/DGSP 已说明语义可降 Fa，但 READ/SegEarth-R2 说明全局 token 不自动等于空间定位 | GPT 只能作为离线 teacher/实验变量，不能是主创新 |
| 无文本如何部署？ | SAM-SPL/IRSAM 是直接 self-prompt；JinSight/SeViL 是训练有语言、部署纯图像 | 必须有强纯图像基线和 teacher→student 对照 |
| 蒸馏什么？ | SimIR 蒸馏任务决策，CLIPSelf/AlignEarth 蒸馏局部特征，PromptKD 蒸馏 prompt/domain knowledge | 仅 global embedding cosine 不足；优先 prompt distribution、mask logits 与扰动响应 |
| 小目标关键瓶颈？ | IRSAM、SegEarth-OV、SOPSeg 指向高分辨率视觉证据；SAIST 更像背景/Fa 改善 | 先测 prompt recall 与 tiny-area bin，再讨论最终 IoU |
| 错 prompt 怎么办？ | SPD/SeViL/SegEarth-OV3 提供过滤、质量控制、presence gate | 自动提示必须允许“拒绝/不出点”，并测试错文、shuffle、空场景 |
