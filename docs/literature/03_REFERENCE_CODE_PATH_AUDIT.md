# 参考代码路径审计

审计日期：2026-08-13。所有仓库均以浅克隆的固定 commit 只读检查。`未发现 LICENSE` 表示该 commit 根目录没有许可证文件，不等同于作品无版权。行号会随上游更新变化，复现时优先使用 commit+路径+函数名。

统一判读规则：每个项目均核对构建/依赖、推理入口、prompt 构造、文本编码器、融合位置、loss、checkpoint、论文—代码匹配度与可复用边界；若正文、训练脚本或实现未公开，则明确写为“不可核验/未公开”，不以论文叙述补写代码事实。无文本项目的文本编码器和文本融合记为“无”。

## 总表

| 项目 | 仓库 / commit | 许可证 | 推理入口 | 论文匹配度 | 可复用边界 |
|---|---|---|---|---|---|
| SAM-SPL | [fuyimin96/SAM-SPL](https://github.com/fuyimin96/SAM-SPL) `1bde7b5` | 未发现 LICENSE | `testing.py` | 部分；缺正文核对 | 浅层 self-prompt blocks |
| IRSAM | [IPIC-Lab/IRSAM](https://github.com/IPIC-Lab/IRSAM) `ec65744` | 未发现 LICENSE | `demo.py` | 大体匹配；WPMD 命名/实现待复核 | PMD、GAD decoder |
| SimIR | [O937-blip/SimIR](https://github.com/O937-blip/SimIR) `47172b1` | 未发现 LICENSE | `train_net.py` eval | 部分；缺蒸馏训练 | student query/inference |
| MoPKL | [UESTC-nnLab/MoPKL](https://github.com/UESTC-nnLab/MoPKL) `dfcfebf` | 未发现 LICENSE | `predict.py`/model inference | 基本匹配 | 训练期语言→纯视觉部署结构 |
| EVF-SAM | [hustvl/EVF-SAM](https://github.com/hustvl/EVF-SAM) `9935dc5` | Apache-2.0 | `inference.py` | 匹配 | BEIT-3 early-fusion sparse prompt |
| READ | [rui-qian/READ](https://github.com/rui-qian/READ) `2549240` | Apache-2.0 | `test_read.py` | 匹配 | similarity-as-points / DtoC |
| MedCLIP-SAM | [HealthX-Lab/MedCLIP-SAM](https://github.com/HealthX-Lab/MedCLIP-SAM) `7c51122` | MIT | 多脚本 | 部分；非一键流水线 | scoremap→points/box |
| MedCLIP-SAMv2 | [HealthX-Lab/MedCLIP-SAMv2](https://github.com/HealthX-Lab/MedCLIP-SAMv2) `07047e3` | MIT | 多脚本 | 部分；阶段断开 | M2IB/scoremap prompt |
| Simple-ViLMedSAM | [qcc001/Simple-ViLMedSAM](https://github.com/qcc001/Simple-ViLMedSAM) `76d95d6` | 未发现 LICENSE | `test.py` | 部分；IPP map 需预计算 | BID decoder |
| RSRefSeg | [KyanChen/RSRefSeg](https://github.com/KyanChen/RSRefSeg) `7ce3e7a` | Apache-2.0 | `tools/test.py`/demo | 匹配 | SigLIP global/local dense activation |
| RSPrompter | [KyanChen/RSPrompter](https://github.com/KyanChen/RSPrompter) `7c676fe` | Apache-2.0 | `tools/test.py` | 匹配 | query/anchor latent prompt heads |
| FIANet | [Shaosifan/FIANet](https://github.com/Shaosifan/FIANet) `beaf1c9` | 未发现 LICENSE | `test.py` | 基本匹配 | role-aware text alignment/TMEM |
| SegEarth-R2 | [earth-insights/SegEarth-R2](https://github.com/earth-insights/SegEarth-R2) `72e1d18` | 未发现 LICENSE | `segearth_r2/eval/eval.py` | 匹配 | `[SEG]` attention supervision |
| SegEarth-OV | [likyoo/SegEarth-OV](https://github.com/likyoo/SegEarth-OV) `3e22a96` | 根目录未发现；vendored BLIP 有 LICENSE | `eval.py`/`demo.py` | 匹配 | GBA、SimFeatUp、dense CLIP logits |
| CLIPSelf | [wusize/CLIPSelf](https://github.com/wusize/CLIPSelf) `1c7fe9c` | Apache-2.0 | downstream configs/eval | 匹配 | crop teacher→ROI dense distill |
| KnowSAM | [taozh2017/KnowSAM](https://github.com/taozh2017/KnowSAM) `b5de98d` | MIT | `prediction.py` | 部分；SAM path 与发布推理松耦合 | image-learned prompt embedding |
| One Shot IRSTS | [D-IceIce/one-shot-IRSTS](https://github.com/D-IceIce/one-shot-IRSTS) `7e2c1c9` | 未发现 LICENSE | `main.py` | 匹配 | reference similarity→point→SAM |
| Training-Free RS | [josesosajs/trainfree-rs-segmentation](https://github.com/josesosajs/trainfree-rs-segmentation) `2a1d1e9` | 未发现 LICENSE | 无 | 不可审计；只有 README/teaser | 无可直接复用代码 |

## 1. SAM-SPL

- **构建/依赖**：`sam_spl/base_model.py:133` 构建 `SamAdaptor`，同文件约 `306` 处理 checkpoint；依赖 SAM2 权重下载。
- **prompt 路径**：`sam_spl/image_encoder.py:68-89` 把浅层 dense feature 与 SAM feature 送入 `pmt_blocks`；`sam_spl/hieradet.py:270-272` 将 prompt 调制接回主干；生成器类在 `sam_spl/pmt_generator.py`。
- **文本/融合/loss**：无文本编码器；skip/decoder 交互在 `sam_spl/UpBlock_layer.py`。loss 与数据流程可运行，但因 P02 正文不可读，暂不声明与所有论文公式完全一致。
- **推理真实性**：`testing.py:9-15` 仅输入 image；`sam_spl/base_model.py:435` 为端到端图像 forward，未见 GT point/box/mask prompt。

## 2. IRSAM

- **构建/推理**：`demo.py:133` 调 `net(batched_input)`；`segment_anything_training/modeling/IRSAM_edge.py:112-115` 显式 `points=None`。
- **关键机制**：`segment_anything_training/modeling/PMD.py` 为 encoder 细节/扩散模块；`IRSAM_decoder.py` 实现多粒度 decoder 和 edge/mask token；loss 随 decoder 输出计算 Dice/BCE。
- **checkpoint**：需要原始 SAM 权重和作者训练 checkpoint。
- **边界**：可单独借鉴 PMD/GAD，但必须重新实现/授权确认；论文称 WPMD 而仓库主类为 PMD，需进一步公式单测。

## 3. SimIR

- **推理**：`train_net.py:304-357` 进入 `ARCH_SIRST.evaluate_sirst`；路由在 `semantic_sam/architectures/arch_sirst.py:352`。
- **模型/提示**：student backbone `semantic_sam/backbone/repvit11_query_deform.py:522-523` 返回 `outputs_mask`、早期 mask 与 `sparse_query`；FPN 见 `transformer_encoder_fpn_test.py`，decoder 见 `istdecoder.py`。
- **文本**：无。**蒸馏缺口**：README `95-97` 发布的是 distilled backbone checkpoint；未找到 Semantic-SAM teacher targets 和 LDIS 的完整训练入口，不能直接复现论文最关键的 teacher→student 过程。
- **依赖**：Semantic-SAM teacher、作者 student checkpoint、RepViT 权重。

## 4. MoPKL

- **模型构建/融合**：`nets/MoPKL.py:12` 的 `MotionModel`；`train_forward:64-91` 把文本描述和 motion prior 融合，并计算 KL、重建和 latent consistency；`inference_forward:93-101` 只接视觉 feature。
- **训练/推理切换**：同文件主 `forward:244`，训练路径约 `255-272`，推理路径约 `274`。这由代码确认“训练有语言、部署纯图像”。
- **文本**：数据加载器从外部 pickle 读取预计算 descriptions，路径存在硬编码，未提供生成器/文本 encoder 的完整可移植封装。
- **依赖**：运动 backbone checkpoint、描述 pickle。可复用边界是双路径接口而非硬编码数据层。

## 5. EVF-SAM

- **入口/构建**：`inference.py:176-185` 同时传 image 与 text；`model/evf_sam.py` 构建 BEIT-3、projector 与 SAM。
- **prompt/fusion**：`model/evf_sam.py:170-197` 做 early vision-language fusion，multimodal `[CLS]` 投影为一个 sparse prompt；不采点/框。图像另走 SAM encoder。
- **loss**：`model/evf_sam.py:220-246` 为 BCE+Dice。
- **依赖/边界**：BEIT-3、SAM/efficient-SAM 权重；early fusion+projector 可以清晰拆分，但模型计算较重。

## 6. READ

- **入口/构建**：`test_read.py:60`；模型在 `model/READ.py:212` 后构建 LMM、vision tower、SAM。
- **prompt**：`model/READ.py:298-314` 附近计算 `<SEG>`–image similarity，生成 points，并把 points + semantic token 送入 SAM prompt path。DtoC/选择逻辑在同文件对应函数。
- **文本/融合/loss**：输入文本必需；融合发生在 LMM token 层；训练包含 language modeling 与 mask BCE/Dice 目标。
- **依赖**：LLaVA/LISA 风格 LMM、SAM 权重、作者 checkpoint。可复用边界是 similarity→正负点模块，不能复制整套 LMM。

## 7. MedCLIP-SAM

- **入口**：不是单一 `infer.py`。CLIP/GradCAM 位于 `saliency_maps/predict.py`、`pytorch_grad_cam/gscore_cam_test.py`，模型加载在 `model_loader/clip_loader.py`。
- **prompt**：`segment-anything/prompt_sam.py:153` `scoremap2bbox`；约 `181` `get_prompts`；约 `199` 从正显著 mask 取 points；约 `221` 调 SAM 得 final mask；main 约 `291`。
- **loss/依赖**：DHN-NCE 的 CLIP 适配和 SAM 分割分属不同阶段；依赖 BiomedCLIP/SAM 权重及中间 scoremap 文件。
- **匹配度**：算法组件存在，但从文本到最终 mask 的数据交换依赖落盘和手工阶段，适合机制参照，不宜作为即插即用库。

## 8. MedCLIP-SAMv2

- **M2IB**：`saliency_maps/scripts/eval.py:60-64` 在单张图/文本条件下优化 bottleneck；随后生成 attribution/score map。
- **prompt/SAM**：沿用 `segment-anything/prompt_sam.py` 的 scoremap→point/box→SAM 路径。
- **loss/依赖**：信息瓶颈目标在 saliency 阶段，SAM 阶段与其解耦；依赖 CLIP、SAM 和作者 checkpoint/中间文件。
- **匹配度**：主要机制代码存在，但不是端到端共同训练；复用需要重构 I/O，并先确认 MIT 许可证覆盖范围。

## 9. Simple-ViLMedSAM

- **入口**：`test.py:199-208` 读取 image 与 `attribution_map`，提取 SAM feature，再交给 BID；训练同样见 `train.py:197-206`。
- **模型/融合**：`model/model.py:75-111` 对 SAM pixel affinity 细化 map，并做两次双向 cross-attention。
- **prompt/text**：公开仓库没有从 simple text 在线生成 M2IB attribution map 的完整路径；README 明确数据目录需要预先准备 `attribution_map/`。因此 text encoder 不是发布推理的一体化组成。
- **依赖**：SAM、CLIP、LoRA/作者 checkpoint和预计算 maps。BID 可分离，但根目录无许可证，不能直接拷贝实现。

## 10. RSRefSeg

- **入口/模型**：`tools/test.py` 或 demo；核心 `rsris/models/models.py:24` `RefSegEncoderDecoder`。
- **文本与融合**：`models.py:148-189` 同时跑 SigLIP vision/text encoder，约 `170-186` 计算局部 token/全局 pooled text 与视觉 patch 的相似激活并拼接。
- **prompt**：`models.py:198-218` 获得 SAM dense embedding，并把激活 feature 展平为 sparse token 集送入 mask decoder。`predict:241-265` 明确测试 data sample 仍读取 text。
- **loss/依赖**：MMSeg decode loss；依赖 SigLIP、SAM checkpoint。global/local activation 可独立研究，但高分辨率 IR 需要重新设计。

## 11. RSPrompter

- **入口/构建**：`tools/test.py`；配置 `configs/rsprompter/_base_/rsprompter_anchor.py` 构建 SAM ViT-H、FPN、RPN/RoI 或 query 头。
- **prompt**：`mmdet/rsprompter/models.py:275-373` 的 query 版本将 decoder outputs 投影为多点 latent embeddings并输入 SAM mask decoder；anchor 版本见 `1597-1689`。这些不是物理坐标点，而是学习的 SAM sparse prompt embeddings。
- **文本**：无。loss 在同文件 mask/query heads，训练用检测/实例 GT，测试无需人工 prompt。
- **依赖/边界**：HuggingFace SAM ViT-H 与作者 checkpoint；prompt head 结构可参照，但不是红外点目标专用。

## 12. FIANet

- **入口**：`test.py:52-77` 读取 expression/token masks 并调用模型；测试仍需文本。
- **构建/融合**：`lib/_utils.py:45-69` 分别编码原表达、ground object 和 spatial position；`lib/backbone.py:648-693` 的 PWAM/OPAB 在层内对齐，`lib/text_aware_multiscale_enhancement.py:173-240` 实现 TMEM。
- **prompt**：不是 SAM，文本形成多尺度 dense feature；`lib/mask_predictor.py` 输出 mask。
- **loss/依赖**：`loss/loss.py` 和 `train.py:174-184`；依赖 BERT/Swin checkpoint。角色拆分思想可复用，整网不可当成 SAM prompt 模块。

## 13. SegEarth-R2

- **入口/构建**：`segearth_r2/eval/eval.py:172-250`；`utils/builder.py` 从 checkpoint 构建 `SegEarthR2`。
- **文本/query**：`datasets/dataset.py:275-311` 标记多个 `[SEG]` token；`model/language_model/llava_phi.py:121` 定义主模型，mask backend 为简化 Mask2Former。
- **attention loss**：`llava_phi.py:43-68` 的 `AttentionLoss` 用 GT mask 区域与背景 attention 构造监督。
- **依赖/边界**：MLLM、CLIP vision tower、Swin/Mask2Former checkpoint；attention supervision 函数边界清楚，但不能脱离其 token/patch 定义直接搬用。

## 14. SegEarth-OV

- **入口/构建**：`eval.py`、`demo.py`；`segearth_segmentor.py:23` `SegEarthSegmentation`。
- **文本/融合**：`segearth_segmentor.py:124-141` 从 class-name templates 生成 text features；`162-204` 提取 dense image features、可选 CLS 并计算 patch-text logits；`279-311` 输出分割。
- **高分辨率**：`simfeatup_dev/upsamplers.py`；模型参数 `cls_token_lambda` 控制 global bias，相关逻辑约 `172-192`。
- **依赖/边界**：OpenCLIP/BLIP、SimFeatUp 权重；GBA/upsampler 适合作为无文本定位前的视觉诊断，但仓库根目录未发现总许可证。

## 15. CLIPSelf

- **训练入口/蒸馏**：`src/training/clipself.py:6-49`。teacher 对真实 crop 调 `dist_model.encode_image`，student 对原图 RoI 调 `encode_pseudo_boxes`，归一化后以 `1-cosine` 对齐。
- **数据**：`src/training/data.py:30` ProposalDistillDataset，约 `135` GridDistillDataset。无需 region-text pair。
- **推理**：论文模型作为 dense CLIP backbone 接下游 open-vocabulary detector/segmentor，不是 SAM prompt 生成器。
- **依赖/边界**：OpenCLIP teacher/student checkpoint；训练 method 类边界清晰，可改为 tiny-target-aware crops，但直接做局部 feature cosine 已属已有范式。

## 16. KnowSAM

- **入口**：发布的 `prediction.py:98-110` 实际评估双子网融合 `SGDL_model(test_image)`；没有在该路径逐样本调用 SAM prompt loop。
- **prompt 模块**：`Model/prompt.py:183-212` 由 feature 生成 box embeddings，`215-239` `Super_Prompt` 汇总每类 embeddings；`Model/sam/build_sam.py:72-108` 把它构建进 SAM。
- **不一致点**：`Model/sam/modeling/sam.py:61-123` 的标准 forward 仍优先读取输入 records 中 points/boxes/masks，公开推理主要走融合学生网络。论文的闭环训练逻辑散落在训练脚本，部署路径不是一个干净的“prompt generator→SAM”入口。
- **依赖/边界**：SAM、UNet/VNet/作者 checkpoints；可参考可学习 prompt embedding，但复用前必须重构并写集成测试。

## 17. One Shot IRSTS

- **入口/依赖**：`main.py:34`；加载 MobileSAM/SAM checkpoint（约 `47-52`）和一个带 GT mask 的 reference frame（`43-64`）。
- **prompt**：reference target features 与测试 patch 相似度产生候选；`main.py:195-213` 取 top/bottom point；`168-172` 将正点送入 SAM，最后多尺度投票 `182-189`。
- **真实性**：测试帧本身无 GT prompt，但任务依赖同序列一个标注参考帧，不能作为单帧无标注部署的同等基线。
- **边界**：相似图→point 和多尺度共识可参照；reference-GT 依赖必须在实验表中单列。

## 18. Training-Free RS Segmentation

- **仓库状态**：固定 commit 只有 `README.md` 与 `teaser.png`，无推理、模型、prompt、text encoder、loss 或 checkpoint 下载脚本。
- **论文与仓库关系**：论文描述 CLIP 选择 SAM proposals，以及 GPT-5/Qwen-VL 生成 click 的两条路线，但目前不能从代码核验 prompt 坐标格式、外部 API 调用、SAM 版本和错误处理。
- **结论**：论文正文可作为机制/实验对照；代码不可复用、不可称“公开实现已验证”。

## 当前项目 prompt 路径审计（对照）

- 官方实验记录和 shell 脚本采用 `assp_only`：`EXPERIMENT_RECORD.md:13`、`scripts/tmm_eval_tassg_student.sh:22-24`。该模式在 `train_sirst_hq_ubuntu.py:1420-1436` / validate `2274-2277` 构造 empty point prompt，再加入 ASSP/TASSG/text prompts；不是用 GT 点提示 SAM。
- 但公共 CLI 默认仍为 `gt_points`：`train_sirst_hq_ubuntu.py:2824`、`scripts/eval_accuracy_metrics.py:39-43`。评测中 `assp_only` 在约 `325` 走 empty prompt，其他分支约 `334-341` 从 GT mask 采点。`scripts/infer_hq_sirst_test_vis.py:553-581` 也保留同类 legacy 分支。这是**可复现性风险**，本阶段按指南只记录、不改代码。
- `train_sirst_hq_ubuntu.py:1920-1929` 的 point sampling 用于 point loss/监督诊断，不是 SAM inference prompt，不能与 GT point prompt 混为一谈。
- 文本生成链：`scripts/generate_gpt_structured_prompts.py:292` 把 GPT 属性渲染成确定性 caption，约 `482` 后生成 CLIP global/token features；dataset 在 `sirst_dataset.py:372-494` 加载缓存。
- 当前文本模块：`efficient_sam/text_conditioner.py:33` global conditioner，`:83` sparse projector，`:189/:247` dense generators，`:417` BiFusion，`:660/:843` CBGA 变体，`:1029` TASSG student，`:1156` 后为蒸馏 losses。研究脚本中 TASSG 的部署契约在 `train_sirst_hq_ubuntu.py:3972-3978`，checkpoint 保存约 `4306-4309`。
