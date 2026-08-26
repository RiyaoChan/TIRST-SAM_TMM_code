# 参考 Self-Prompt 代码路径审计

审计日期：2026-08-26。15 个指定仓库中，14 个完成浅克隆固定快照的只读审计；AlignSAM官方链接404。以下是代码事实，不用README补全缺失实现。shape中的`B`为batch，`K/Nq`为候选/query数，`C=256`通常是SAM prompt维度。

## 总览

| # | Repo / snapshot | Prompt实际类型 | 原生prompt encoder | 坐标 | 每候选query | bg/no-object | response feedback | 推理外部依赖 | License |
|---:|---|---|---|---:|---:|---|---|---|---|
| 1 | SPARK-SAM `adceab6` | point+box+local token+dense residual | Y | Y | P | objectness/可选负环 | response cache/calibration | 单图 | 未声明 |
| 2 | MaskSAM `c6860c5` | per-query mask+box+classifier query | Y | box | Y | `C+1` no-object | query mask→prompt | 单volume | 未声明 |
| 3 | SamRadiology `ccf881d` | class query→box+mask+latent token | Y | box | P | P | interim mask | 单图 | 未声明 |
| 4 | AoP-SAM `de503b2` | confidence map→points | Y | Y | N | filter only | Y，mask overlap/IoU | 多候选SAM | Apache-2.0 |
| 5 | RSPrompter `7c676fe` | learned sparse tokens，optional dense pre-mask | N/P | N | Y(query版) | `C+1` | query mask作attention | detector/query head | Apache-2.0 |
| 6 | UN-SAM `1c16218` | coarse mask+dense prompt+domain queries | Y(mask) | N | N | domain/background P | coarse→fine | 单图/domain id | MIT |
| 7 | EviPrompt `b8a25a5` | evidential正负点→box | Y | Y | N | Y | Y，两轮SAM | 标注reference | 未声明 |
| 8 | AlignSAM | 不可访问 | — | — | — | — | 论文为RL迭代 | VLM/text | 不可核验 |
| 9 | PromptPilot `c99739e` | 正负点图+remove/restore actions | Y | Y | N | 正负点 | Y，SAM DSC/marginal | reference/多步SAM | 未声明 |
| 10 | H-SAM `5bdf491` | empty prompt+stage1 mask state | 默认empty | N | N | attention抑制 P | Y，stage1→stage2 | 单图 | MIT |
| 11 | SAM-SPL `1bde7b5` | shallow dense feature prompts | N | N | N | N | 双向feature交互 | 单图 | 未声明 |
| 12 | AutoPromptMedSAM `ece3ce2` | diffusion sparse+dense class prompt | 绕过point/box | N | N | N | N | class label；实现不完整 | 未声明 |
| 13 | De-LightSAM `d2260d7` | binary patch pre-mask+7 queries | empty only | N | P | class/domain query | N | domain/class id | Apache-2.0 |
| 14 | SurgicalSAM `4b4c655` | prototype dense prompt+正负class tokens | N | N | 按class，不按candidate | 正/负class | N | learned prototypes | MIT |
| 15 | PPD `f040aab` | agent增删正负点 | Y | Y | N | 正负/harmful | Y，多步SAM | DINOv3初始点/agent | 未声明 |

## 1. SPARK-SAM

- **构建/入口**：`sparksam/models/prompt_estimator.py` 的prompt estimator；`sparksam/models/sam2_adapter.py`包装SAM2；`sparksam/models/response_guidance.py`生成response状态；训练/校准/高分辨率refinement集中在`scripts/train_sparksam.py`。
- **forward与shape**：estimator输出objectness logits、box size、point candidates；`LearnedAutoPrompt`保存`point [B,1,2]`、`box [B,4]`、`points [B,K,2]`、`point_labels [B,K]`、`objectness [B,K]`。候选局部feature投成`[B,K,C]` tokens；dense refinement与image embedding同空间。
- **Prompt encoder/decoder**：point/box经过SAM2 native prompt path；local tokens/candidate gates与dense residual进入adapter/decoder。candidate gate不只排序，还缩放局部token。
- **loss/GT boundary**：mask监督生成response guidance、candidate/quality标签；正常inference只读图像。代码 `_topk_candidates` 在无候选时回退argmax，空目标不是真正abstain。
- **checkpoint**：SAM2.1-Large用于离线response guidance，部署以SAM2.1-Tiny等配置；分阶段cache/checkpoint必须匹配。
- **敏感性证据**：论文有coordinate/local-token replacement和matched response ablations，但未见完整shuffled query/no-object set反事实。
- **复用结论**：架构思想可参考；无LICENSE，不能复制。尤其不能把它误写成摆脱heatmap：高分辨率refinement仍对objectness map做`topk`。

## 2. MaskSAM

- **构建/入口**：nnUNet路径中的`nnUNet/nnunetv2/sam/sam_model_2024_*_tqreshape.py`；`nnUNet/nnunetv2/sam/modified_mask_decoder.py`（或同名classifier decoder）；`matcher.py`与`criterion.py`。
- **forward与shape**：aux branch输出`[B,Nq,T,H,W]` masks；按query求bbox并与直接回归bbox融合。reshape为`B*Nq`后，binary mask与box进入prompt encoder。classifier decoder为每query输出`C+1`，binary mask保持独立。
- **Prompt encoder/decoder**：box→sparse，mask→dense，均走native SAM prompt encoder；image embedding按query复制，decoder一query一mask。
- **loss/GT boundary**：Hungarian matcher对class/mask/Dice/box cost做匹配；criterion把`num_classes`当background/no-object并用`eos_coef`；训练mask/类别/box来自GT，inference无GT prompt。
- **checkpoint**：依赖SAM/nnUNet预训练权重；3D adapters和decoder checkpoint必须同配置。
- **敏感性证据**：论文有prompt/classifier/adapter消融；未见shuffle query或oracle spatial prompt专门实验。
- **复用结论**：`one-query-one-mask + ∅ + Hungarian`是最强可迁移结构；无根许可证，只能独立实现。

## 3. Sam2Rad / SamRadiology

- **构建/入口**：`sam2rad/models/sam2rad/model.py`；`sam2rad/models/prompt_predictor/highres_prompt_predictor.py`、cross-attention/PPN文件；训练prompt采样器在trainer/data路径。
- **forward与shape**：class/learned prompts为`[B,num_prompts,256]`（实现也支持`B*num_classes`展平）。`TwoWayCrossAttention/HighResPPN`用multi-level image feature作K/V、learned prompts作Q；前2 tokens预测box，其余是latent tokens；interim binary mask送`_embed_masks`。
- **Prompt encoder/decoder**：predicted box经`_embed_boxes`，interim mask经`_embed_masks`，均调用冻结native prompt encoder内部接口；latent tokens拼到sparse prompt。
- **loss/GT boundary**：训练`PromptSampler`可随机混入由GT mask得到的manual box/low-res mask；部署PPN不需要GT。必须在复现表单独标注这类train-time teacher prompt。
- **checkpoint**：SAM/SAM2/MedSAM及PPN checkpoints；部分实验冻结native encoder/prompt encoder，只训PPN/decoder或PEFT。
- **敏感性证据**：有manual+learned prompt组合/冻结策略消融；无no-object、shuffle prompt因果测试。
- **复用结论**：联合box+micro-mask+latent query很适合最小prototype；仓库无许可证。

## 4. AoP-SAM

- **构建/入口**：`segment_anything/modeling/PromptFilter.py`的Prompt Predictor；`segment_anything/automatic_mask_generator.py`与ASF逻辑。
- **forward与shape**：原图`[B,3,H,W]`与flatten/重排后的SAM embedding形成CNN输入，输出`[B,1,H,W]`confidence map；采点成`[B,K,2]`及全正labels。
- **Prompt encoder/decoder**：标准physical points进入SAM；SAM冻结。
- **loss/GT boundary**：predictor由目标区域监督；推理无GT。`ASF_finer`读取low-res masks、pred-IoU、stability/overlap和Prompt Evaluation Threshold去掉冗余prompt。
- **checkpoint**：SAM checkpoint+Prompt Predictor checkpoint。
- **敏感性证据**：有采样密度/ASF消融；不是candidate drop/shuffle因果实验。
- **复用结论**：Apache-2.0允许合规复用；但confidence-map→point不能当新贡献，真正可借的是response-based duplicate filtering。

## 5. RSPrompter

- **构建/入口**：`mmdet/rsprompter/models.py`。query版`RSMask2FormerHead`约275行，anchor版`RSPrompterAnchorRoIPromptHead`约1367行、`RSPrompterAnchorMaskHead`约1597行。
- **forward与shape**：query decoder state`[B,Nq,C]`经`point_emb`变`[B,Nq,Np,256]`，重排成`[B*Nq,Np,256]` sparse embeddings。`decoder_plus`时`mask_pred_plus [B,Nq,H,W]`经SAM mask embed为dense prompt；否则用`no_mask_embed [B*Nq,256,h,w]`。
- **Prompt encoder/decoder**：learned tokens直接作为`sparse_prompt_embeddings`，没有physical point encoding；dense pre-mask使用从native prompt encoder取出的`mask_embed`。image embedding按Nq复制。
- **loss/GT boundary**：query版继承Mask2Former Hungarian/class/mask/Dice，class head输出`num_classes+1`；anchor版用RPN/ROI assignment和mask loss。测试无GT prompt，但anchor版依赖内部detector proposals。
- **checkpoint**：HuggingFace SAM ViT-H、MMDetection configs与作者checkpoint；冻结/适配策略由config控制。
- **敏感性证据**：prompt point数/anchor-query/decoder-plus等消融；没有shuffled latent prompt。
- **复用结论**：Apache-2.0。可以参考query接口，但不要移入整套Mask2Former；小K二类head足够。

## 6. UN-SAM

- **构建/入口**：`model.py`中的`SPGen`、`UNSAM`；`SAM/modeling/mask_decoder.py`中的`DQDecoder`。
- **forward与shape**：SPGen输出coarse map；逐样本以`masks=output_prob[idx].unsqueeze(0)`进入PromptEncoder，points/boxes均None。`domain_num+1`个`mask_query [*,256]`与empty sparse embeddings拼成tokens。
- **Prompt encoder/decoder**：coarse mask走native mask prompt；domain queries进入自定义DQDecoder。
- **loss/GT boundary**：训练fine/coarse mask loss；inference只输入图像和domain setting，无GT prompt。
- **checkpoint**：SAM backbone与作者UN-SAM checkpoint。
- **敏感性证据**：SPGen/domain query消融；无candidate-level counterfactual。
- **复用结论**：MIT；dense pre-mask路径可借，但domain query和整图语义mask不解决多实例。

## 7. EviPrompt

- **构建/入口**：`main2d.py`、`main3d.py`；`evidence_tools.py`；`cluster.py`；point选择在`vis_tools.py`。
- **forward与shape**：reference GT mask把SAM feature分成positive/negative anchors；target多增强得到每像素`evidence[...,2]`，`evidence2opinion`得到belief和uncertainty；选正/负坐标后拼`points_tgt[K,2]`,`point_lbl[K]`。
- **Prompt encoder/decoder**：standard point prompts；首轮mask非空时取bbox再第二次SAM。
- **loss/GT boundary**：training-free；但部署需要一张标注reference mask，不是image-only。target GT只用于evaluation。
- **checkpoint**：SAM checkpoint、reference image/mask、FAISS环境。
- **敏感性证据**：evidence fusion/points/refinement消融；没有无reference部署。
- **复用结论**：无根许可证；可重写evidential target/background state，不可复制源码。

## 8. AlignSAM

- 指定仓库`https://github.com/Duojun-Huang/AlignSAM-CVPR2024`在审计日返回404，无法核验forward、shape、loss、checkpoint和license。
- 论文与supplement确认的是VLM-guided actor-critic、物理point prompt、SAM mask/probability反馈和GT-derived RL reward；这些属于论文证据而非代码证据。
- 用户若提供源码压缩包，应优先检查：候选动作空间、point labels、每轮是否重算image embedding、reward是否直接看GT、测试iteration/stop条件和权重许可。

## 9. PromptPilot

- **构建/入口**：`feature_matching/generate_points.py`生成DINOv2初始点；`agents/node_env.py`为点图环境；`agents/node_agent.py`、`manager_agent.py`、`scheduler.py`；`train_game.py`与`arg_game_multi.py`。
- **forward与shape**：NodeAgent的动作空间`max_nodes*4`，操作为remove/restore positive/negative；候选action encoding包含`[is_restore,is_neg,x,y,weight]`。状态是图/统计flatten后的向量，不是SAM latent query。
- **Prompt encoder/decoder**：环境把当前正负点交给标准SAM；DINOv2/SAM按README冻结。
- **loss/GT boundary**：DQN/manager学习Q-value；环境用GT mask计算SAM DSC/global reward与局部边际贡献。论文精确reward/训练协议待PDF。
- **checkpoint**：DINOv2、SAM、各agent checkpoints；多agent权重必须成套。
- **敏感性证据**：代码天然做remove/restore与marginal credit；是否有正式oracle/shuffle表待PDF。
- **复用结论**：无许可证。适合把leave-one-out credit离线化，不适合直接复用RL系统。

## 10. H-SAM

- **构建/入口**：`sam_lora_image_encoder_mask_decoder.py`的`LoRA_Sam.forward`；`segment_anything/modeling/mask_decoder_224.py`及stage-2/hierarchical decoder相关文件；训练`trainer.py`，测试`test.py`。
- **forward与shape**：默认empty sparse/dense prompt；stage-1 mask decoder输出`[B,M,256,256]`、attention/mask feature/upscaled embedding；stage-1概率/attention进入stage-2 class-balanced mask-guided attention。
- **Prompt encoder/decoder**：不需要physical points/boxes；prompt encoder用default embeddings。image encoder LoRA，mask decoder/attention部分可训练。
- **loss/GT boundary**：stage1/stage2 deep supervision；训练GT mask，推理无GT prompt。
- **checkpoint**：SAM ViT-B+LoRA/hierarchical decoder checkpoint。
- **敏感性证据**：stage/attention/LoRA消融；无candidate query drop。
- **复用结论**：MIT；可借stage1 response condition思想，但不应复制重型医学decoder。

## 11. SAM-SPL

- **构建/入口**：`sam_spl/base_model.py`的模型forward；`sam_spl/image_encoder.py`；`sam_spl/pmt_generator.py`；`sam_spl/mask_decoder.py`；`testing.py`。
- **forward与shape**：`image_encoder.forward`先得到主干浅层`out_feats`和SAM stages；stage boundary处`pmt_blocks[i](sam_out + shallow_out)`生成dense prompt并反加到后续SAM block。返回`sam_backbone_embeds`和前三层`dense_embeds`。
- **Prompt encoder/decoder**：不是point/box/mask prompt，不调用native prompt encoder；prompt是feature-space状态，decoder用skip mutual calibration。
- **loss/GT boundary**：训练脚本用mask GT；`testing.py`只传图像，无GT点框mask。
- **checkpoint**：SAM2 Hiera checkpoints与SPL-T/S/L config/checkpoint。
- **敏感性证据**：正式论文PDF缺失，不能核验prompt block消融与论文表；代码无counterfactual/shuffle测试。
- **复用结论**：无根许可证；只能作为强baseline/重写，不可拷贝。

## 12. AutoPromptMedSAM

- **构建/入口**：`AutoMedSAM/AutoMedSAM.py`、`AutoMedSAM/class_prompt.py`。
- **forward与shape**：设计上image encoder输出`[B,256,64,64]`；DiffusionModel应输出`sparse [B,2,256]`与`dense [B,256,64,64]`，直接交给SAM mask decoder。
- **真实实现断点**：`AutoMedSAM.__init__`接收`class_prmpt_encoder`却没有赋给`self.diffusion_model`，`forward`却调用`self.diffusion_model(...)`；返回顺序在`class_prompt.py`与主类注释也不一致。当前快照不能按README直接运行。
- **Prompt encoder/decoder**：只用`prompt_encoder.get_dense_pe()`；sparse/dense embeddings绕过物理prompt编码。
- **loss/GT boundary**：公开文件没有完整训练入口/loss/checkpoint恢复链；输入`label`是测试依赖，不能称完全class-agnostic image-only。
- **敏感性/许可**：无反事实代码，未声明许可证。
- **复用结论**：只把论文当机制参考；不能从当前代码复制或声称复现成功。

## 13. De-LightSAM

- **构建/入口**：`model.py`中的`ESPMedSAM.forward`；`SAM/modeling/student_encoder.py`、`mask_decoder.py`；`train.py`,`eval.py`。
- **forward与shape**：image embedding`[B,256,64,64]`；`patch_decoder→[B,1,32,32]`，detach后二值化、×2上采样、`dense_prompter→[B,256,64,64]`；`mask_query[7,256]`与empty sparse prompt拼接。
- **Prompt encoder/decoder**：points/boxes/masks均None；patch dense embedding直接加到`mask_src`，7 learned queries进自定义decoder。
- **loss/GT boundary**：mask/patch/KD训练监督；推理图像+`domain_seq`，无GT prompt。
- **checkpoint**：teacher/student encoder与作者checkpoint；domain config需一致。
- **敏感性证据**：pre-mask/query/KD消融；无shuffled query/no-object。
- **复用结论**：Apache-2.0；可复用合规，但硬二值+detach对tiny target风险高。

## 14. SurgicalSAM

- **构建/入口**：`surgicalSAM/model.py`的`Prototype_Prompt_Encoder`、`Learnable_Prototypes`；`model_forward.py`；训练`train.py`，推理`inference.py`。
- **forward与shape**：SAM feature展平为`[B,4096,256]`；7 class prototypes做相似；当前class feature变`dense [B,256,64,64]`；所有class生成`sparse [B,7*num_tokens,256]`，默认`num_tokens=8`，再叠加positive/negative class embeddings。
- **Prompt encoder/decoder**：learned prompt embeddings直接送SAM mask decoder，仅使用native prompt encoder的dense PE；无physical coordinate。
- **loss/GT boundary**：训练prototype/segmentation/contrastive相关目标；数据预处理用GT masks提取class embeddings，但inference使用learned prototypes与class ids，不读GT prompt。
- **checkpoint**：SAM ViT-H、prototype/decoder checkpoint、固定EndoVis尺寸约定。
- **敏感性证据**：prototype/token/contrastive消融；无no-object/candidate shuffle。
- **复用结论**：MIT；target/background成对token可借，但固定7类与GT prototype预处理不适合IRSTD。

## 15. PPD

- **构建/入口**：`feature_matching/gen_points.py`用DINOv3生成初始正负点；`prompt_env.py`维护prompt图；`attack_agent.py`,`defense_agent.py`；`seg/segment.py`调用SAM；`run_evaluation.py`。
- **forward与shape**：points是坐标list及正负label；graph node保存coord/feature/point_type。attacker可激活/误标候选，defender删除/恢复点；每step重新调用SAM。
- **Prompt encoder/decoder**：标准点prompt；不改SAM decoder。
- **loss/GT boundary**：`PromptEnv`构造时直接接收`gt_mask`，每step用其算Dice/IoU与reward；这可作为训练/研究环境，不能把同一reward路径当部署算法。
- **checkpoint**：DINOv3、SAM和attack/defense agents；外部模型和多step开销高。
- **敏感性证据**：显式attack/defense是有害prompt反事实；但no-object、单图无GT停止策略与正式论文协议待核验。
- **复用结论**：无根许可证；可重写离线harmful-prompt diagnostic，不可复制。

## 当前仓库应采用的最小接口契约

```text
CandidateState:
  query          [B,K,C]
  micro_mask     [B,K,h,w]
  center_or_box  [B,K,2|4]        # optional
  object_logit   [B,K,1]          # includes no-object/abstain
  background     [B,K,C|h,w]      # distinguishable until decoder
  reliability    [B,K,1]          # consumed by token/mask aggregation

Decoder output:
  mask_logits    [B,K,H,W]
  sam_iou        [B,K,1]
  response_delta [B,K,*]          # optional diagnostic
```

这一契约来自MaskSAM/RSPrompter的set结构、Sam2Rad的联合prompt state、IP-SAM的background condition、SPARK-SAM的response adaptation；它刻意不包含“单热图再调Top-K”的旧路径。
