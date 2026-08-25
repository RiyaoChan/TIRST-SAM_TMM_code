# 当前 TIRST-SAM 与文献机制对照

## 1. 当前分支真实数据流

```text
IR image ── EfficientSAM image encoder ── image embedding ───────────────┐
    │                                                                   │
    ├─(离线) GPT structured attributes → deterministic caption         │
    │                         → CLIP global/token features               │
    │                                     │                             │
    │                                     ├─ TextConditioner / CBGA ────┤
    │                                     └─ sparse/dense projection ───┤
    │                                                                   ▼
    └─ TASSG image student（可选，部署替代缓存文本） ── semantic prompts → SAM mask decoder

point prompt：正式实验 `assp_only` 为 empty；legacy CLI 默认仍可从 GT mask 采点。
```

## 2. 组件级对照

| 当前组件 | 当前实现与代码证据 | 最接近已有机制 | 判断 |
|---|---|---|---|
| image encoder | EfficientSAM；主训练入口 `train_sirst_hq_ubuntu.py` | IRSAM 的 WPMD/GAD、SAM-SPL 浅层 self-prompt、SimIR RepViT query、SegEarth-OV 高分辨率 dense feature | 目前没有显式证明 encoder 在 tiny-area bin 保留了目标；这是文本前的首要瓶颈 |
| text generation/loading | `scripts/generate_gpt_structured_prompts.py`：GPT 属性→固定 caption→CLIP；`sirst_dataset.py:372-494` 读缓存 | SAIST 自动描述；DGSP 粗/细 prompt；JinSight 多任务指令；SeViL GPT-4o 固定描述 | 更强 GPT 只提高教师标签质量，属于数据/teacher 选择，不是独立方法创新 |
| text encoder | 缓存 CLIP global/token features | SAIST CLIP；DGSP text encoder；EVF-SAM/RS2-SAM 早期联合 encoder；FIANet 三角色 BERT | CLIP global 主要做图文全局对齐；token feature 仍未自动获得红外像素定位能力 |
| CBGA | `efficient_sam/text_conditioner.py:660/:843`，text token 与 backbone feature 融合 | DGSP TGCA/TGSA、FIANet PWAM/TMEM、RS2-SAM BHFM、EVF-SAM early fusion | 仅算子/插入位置差异通常是工程差异；若无新问题与专门证据，难构成强创新 |
| ASSP / prompt projection | sparse projector `:83`，dense generator `:189/:247`，BiFusion `:417` | SAIST prompt、EVF-SAM sparse `[CLS]`、READ token+points、Simple-ViL/RS2-SAM dense map | global text→sparse token 已拥挤；dense prompt 必须说明空间证据来自哪里、错时如何拒绝 |
| SAM prompt encoder | `assp_only` 时先构造 empty point prompt，再叠加 learned semantic prompt | RSPrompter latent sparse embeddings、MaskSAM 多 prompt、SAM-SPL self-derived prompt | latent prompt 本身不是新接口；关键是生成与验证机制 |
| SAM mask decoder | EfficientSAM decoder 接 image embedding + semantic prompt | SAIST/EVF/READ/RS2-SAM；SimIR/SegEarth-R2 甚至显示非 SAM decoder 可更强 | 需隔离“更强 decoder”与“文本”的收益，避免把普通适配算作语义贡献 |
| losses | mask loss、point-related loss、TASSG distill losses `text_conditioner.py:1156+` | SimIR LDIS、AlignEarth global/local、CLIPSelf crop-region、PromptKD prompt、SeViL pseudo-label filter | 当前若只做 feature/embedding imitation，碰撞高；可升级到 prompt distribution 与 mask-function behavior |
| train-time prompt | 支持 `gt_points` 和 `assp_only`；正式记录用后者 | 许多 SAM 论文训练用 GT 派生 prompt；SAM-SPL/IRSAM/RSPrompter 自动化 | 必须在所有主表明示 prompt mode，并禁止不同模式混表 |
| test-time prompt | 正式脚本 `assp_only`，但公共 CLI 默认 `gt_points` | SAM-SPL/IRSAM 无外部 prompt；SAIST/EVF text prompt；READ text→point | 科学设定已可无 GT，但默认值是复现风险；待文献阶段结束后单独修复 |
| TASSG student | `text_conditioner.py:1029`，图像产生 semantic features，部署替代缓存 Qwen/CLIP | JinSight/SeViL/MoPKL 语言训练→纯图像部署；DGSP I2T；AlignEarth/CLIPSelf distill | “图像拟合文本空间”不是新颖点；需重新定义蒸馏对象与可证伪目标 |

## 3. 与 14 个 S 级工作的逐项对齐

| 工作 | 已覆盖当前哪一部分 | 文献仍提供、当前未利用的机制 | 冲突后允许保留的研究问题 |
|---|---|---|---|
| SAIST | 自动描述、CLIP、SAM、背景抑制 | prompt 分布 MMD、明确的 Fa 贡献分析 | 错/缺文本时的可靠性；无文本 student 的行为保持 |
| SAM-SPL | 纯图像 self-prompt | 浅层高分辨率 prompt、skip mutual calibration | 语言 teacher 是否能超过同容量 self-prompt，而非只超过空 prompt |
| IRSAM | 无文本 SAM 视觉适配 | diffusion 细节保持、edge/mask multi-granularity | 文本加入前是否修复 tiny-target visual bottleneck |
| SimIR | teacher→纯图像 student | query/多层 mask decision distillation | 蒸馏文本对 prompt→mask 函数的增量行为 |
| DGSPNet | 图像生成 text-space token、channel/spatial guidance | 粗/细语义职责分离、I2T 反演 | 不再主张“图像生成文本特征”本身新颖 |
| JinSight | 训练有语言、测试图像 | 多任务指令预训练、LSI 全局/局部交换 | 更轻量 SAM-specific behavior distill，而非完整 VLM 预训练 |
| SeViL | GPT 文本 teacher、纯图像 student | 文本筛伪标签/置信控制 | 将语言定位职责改成 candidate verifier，并显式 abstain |
| EVF-SAM | 图文共同产生 SAM prompt | early fusion `[CLS]` | 仅作为 teacher，上层研究高分辨率/无文本部署 |
| READ | token→相似图→points | 正负点和可微坐标 | tiny-target 多尺度 similarity + reject，而非普通 top-k |
| Simple-ViLMedSAM | dense text-conditioned prompt | affinity refinement、双向 decoder | 低分辨率 attribution 对 tiny targets 的修复与可靠性 |
| RS2-SAM 2 | token fusion + dense/sparse prompt | pseudo-mask generator、边界 loss | prompt source 的校准和纯图像蒸馏 |
| SegEarth-R2 | semantic token 与空间监督 | `[SEG]` spatial attention supervision、多 query | 轻量化 tiny-target attention supervision，但不能只新增普通辅助 loss |
| AlignEarth | teacher→专业传感器 student | global/CLS/local region distill、抗配准误差 | 从 feature distill 升级到 downstream prompt/mask behavior |
| SPD | noisy prompt 的过滤/共识 | saliency validation、contextual consensus | 用单帧多视图替代相邻切片，并让模型可以拒绝 prompt |

## 4. 文本到底提供了新信息吗？

当前离线 GPT 输入本身仍是同一张红外图像，所以至少有两种不同含义，实验中必须分开：

1. **重编码信息**：caption 中的 scene/position/brightness 都能由图像直接估计。若 TASSG 从图像恢复这些特征且不低于文本 teacher，文本更像训练正则或昂贵中间表示，而不是部署新模态。
2. **外部先验**：GPT/CLIP 可能通过预训练知识提供“何种亮点更像目标、何种背景易误报”的类别/场景先验。这一价值更适合做 presence/候选校验，而不是声称文本产生精确坐标。

判别实验：固定图像分别输入正确文本、shuffle 文本、语义相反文本、无文本和 oracle 属性。若正确文本只改善 Fa 而不改善 prompt center/component recall，则它是 verifier；若 shuffle 与正确相近，则模块主要是额外参数/正则；若 oracle 仍不能提升定位，则瓶颈在视觉分辨率而非 caption 质量。

## 5. 当前最需要避免的表述

- 不再把“GPT-5.x 比 Qwen caption 更准确”写作算法贡献；它是 teacher/data ablation。
- 不再把“CLIP image encoder 学会生成文本 feature”写作首次纯图像部署；DGSPNet/JinSight/SeViL/AlignEarth 已覆盖邻近范式。
- 不把 `assp_only` 的 empty point 与“完全无 prompt”混称：系统仍有 learned sparse/dense semantic prompts。
- 不把用于 loss 的 GT point sampling 误写成 inference prompt；同时也不能忽略 public CLI 默认 `gt_points` 的复现风险。
- 不把未发布的 P02 正文机制写成已完全核验结论。
