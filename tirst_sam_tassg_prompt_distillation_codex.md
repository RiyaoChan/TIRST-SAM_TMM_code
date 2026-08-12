# Codex 修改任务：为 TIRST-SAM 加入 Targetness-aware Semantic Slot Generator + Qwen-CLIP Prompt Distillation

## 0. 背景与目标

当前 TIRST-SAM 的核心问题是：原始流程依赖 `Qwen3-VL -> CLIP text encoder -> token/global text features -> CBGA/ASSP -> SAM mask decoder`。这会导致审稿人质疑：**新图像推理时是否仍然需要 Qwen3-VL 离线生成文本？**

本次修改目标是将 Qwen3-VL 从“推理阶段组件”改成“训练阶段离线语义教师”。最终模型在推理阶段只输入红外图像，不再需要 Qwen3-VL、不再需要 CLIP text encoder、不再需要 `mllm_features_path`。

最终推理流程应为：

```text
Infrared image
  -> image encoder
  -> Targetness-aware Semantic Slot Generator, TASSG
  -> predicted semantic tokens / predicted global semantic embedding
  -> CBGA and/or ASSP text sparse prompt projector
  -> SAM mask decoder
  -> IR small-target mask
```

训练阶段仍然可以读取已有的缓存文件：

```text
Qwen3-VL-8B-Instruct_mllm_clip_token_features.pt
```

但它只作为 teacher，用于蒸馏 TASSG 预测的 semantic tokens 和 global semantic embedding。

---

## 1. 必须保持的原则

1. **不要破坏现有 Qwen-CLIP teacher 流程。**
   - 原有 `--mllm_features_path`、ASSP、CBGA、text-sensitivity 相关实验需要继续能跑。
   - 新功能通过显式参数开启，例如 `--use_tassg`、`--semantic_source student`。

2. **推理阶段 student-only 必须不依赖 Qwen 特征文件。**
   - `scripts/eval_accuracy_metrics.py` 和 `scripts/infer_hq_sirst_test_vis.py` 在 `--use_tassg --semantic_source student` 时，不应强制要求 `--mllm_features_path`。

3. **TASSG 不生成自然语言。**
   - 它从图像特征直接预测 latent semantic tokens 和 latent global semantic feature。
   - 它的输出替代原来的 `clip_text_token_feat` 和 `clip_text_feat`。

4. **第一版优先跑通，不追求一次性最优。**
   - 先完成 single-pass student sparse prompt。
   - 再支持 two-pass CBGA。

---

## 2. 需要修改的主要文件

请先检查实际代码中的函数签名，再按相同风格集成。

建议修改：

```text
efficient_sam/text_conditioner.py
train_sirst_hq_ubuntu.py
scripts/eval_accuracy_metrics.py
scripts/infer_hq_sirst_test_vis.py
scripts/tmm_required_ablations.sh 或新增独立 launcher
```

通常不需要改：

```text
sirst_dataset.py
```

原因是 dataset 已经能读取 cached Qwen-CLIP features，并返回类似字段：

```python
clip_text_feat
clip_text_token_feat
clip_text_attn_mask
clip_text_token_ids
```

这些字段可以直接作为 teacher supervision。

---

## 3. 新增模块：TargetnessAwareSemanticSlotGenerator

### 3.1 放置位置

在：

```text
efficient_sam/text_conditioner.py
```

新增类：

```python
class TargetnessAwareSemanticSlotGenerator(nn.Module):
    ...
```

以及 factory：

```python
def build_targetness_aware_semantic_slot_generator(...):
    ...
```

### 3.2 输入输出

输入：

```python
img_emb: Tensor[B, C, H, W]
```

输出 dict：

```python
{
    "global": pred_global,          # Tensor[B, text_dim]
    "tokens": pred_tokens,          # Tensor[B, K, text_dim]
    "attn_mask": pred_attn_mask,    # Tensor[B, K], 1 means valid
    "targetness_logits": logits,     # Tensor[B, 1, H, W]
    "targetness_prob": prob,         # Tensor[B, 1, H, W]
}
```

其中：

```text
K = tassg_num_slots，建议默认 8
text_dim = CLIP text feature dim，通常 512；不要写死，优先从 args 或 teacher feature 推断
C = SAM image embedding channel，通常 256；不要写死，提供 args
```

### 3.3 推荐实现

```python
class TargetnessAwareSemanticSlotGenerator(nn.Module):
    def __init__(
        self,
        img_dim: int = 256,
        text_dim: int = 512,
        num_slots: int = 8,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_slots = int(num_slots)

        self.targetness_head = nn.Sequential(
            nn.Conv2d(img_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )

        self.vis_proj = nn.Linear(img_dim, hidden_dim)

        self.ctx_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.slot_queries = nn.Parameter(torch.randn(num_slots, hidden_dim) * 0.02)

        self.slot_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.slot_norm = nn.LayerNorm(hidden_dim)

        self.token_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, text_dim),
            nn.LayerNorm(text_dim),
        )

        self.global_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, text_dim),
            nn.LayerNorm(text_dim),
        )

    def forward(self, img_emb: torch.Tensor) -> dict:
        if img_emb.dim() != 4:
            raise ValueError(f"img_emb must be [B,C,H,W], got {tuple(img_emb.shape)}")

        b, c, h, w = img_emb.shape

        targetness_logits = self.targetness_head(img_emb)
        targetness_prob = torch.sigmoid(targetness_logits)

        # Visual tokens: [B, HW, C] -> [B, HW, hidden]
        v = img_emb.flatten(2).transpose(1, 2)
        v = self.vis_proj(v)

        # Targetness-aware pooling
        a = targetness_prob.flatten(2).transpose(1, 2)  # [B, HW, 1]
        z_target = (v * a).sum(dim=1) / a.sum(dim=1).clamp_min(1e-6)
        z_global = v.mean(dim=1)
        z_context = torch.cat([z_target, z_global, z_target - z_global], dim=-1)
        z_context = self.ctx_proj(z_context)

        # Semantic slots read from image tokens
        q = self.slot_queries.unsqueeze(0).expand(b, -1, -1)
        q = q + z_context.unsqueeze(1)
        slots, _ = self.slot_attn(q, v, v, need_weights=False)
        slots = self.slot_norm(slots + q)

        pred_tokens = self.token_proj(slots)
        pred_global = self.global_proj(slots.mean(dim=1))

        pred_attn_mask = torch.ones(
            b,
            self.num_slots,
            dtype=torch.long,
            device=img_emb.device,
        )

        return {
            "global": pred_global,
            "tokens": pred_tokens,
            "attn_mask": pred_attn_mask,
            "targetness_logits": targetness_logits,
            "targetness_prob": targetness_prob,
        }
```

Factory：

```python
def build_targetness_aware_semantic_slot_generator(
    img_dim: int = 256,
    text_dim: int = 512,
    num_slots: int = 8,
    hidden_dim: int = 256,
    num_heads: int = 4,
    dropout: float = 0.0,
):
    return TargetnessAwareSemanticSlotGenerator(
        img_dim=img_dim,
        text_dim=text_dim,
        num_slots=num_slots,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dropout=dropout,
    )
```

---

## 4. 新增蒸馏损失函数

建议在 `efficient_sam/text_conditioner.py` 或新文件 `efficient_sam/tassg_losses.py` 中实现。为了减少文件数量，第一版可直接放在 `text_conditioner.py`。

### 4.1 Global feature cosine distillation

```python
def cosine_distill_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    student = F.normalize(student.float(), dim=-1)
    teacher = F.normalize(teacher.float(), dim=-1)
    return 1.0 - F.cosine_similarity(student, teacher, dim=-1).mean()
```

### 4.2 Token-level masked set cosine loss

不要强行逐 token 对齐，因为 teacher token sequence 长度和 student semantic slots 数量不同。使用 masked Chamfer-style cosine set loss：

```python
def masked_token_set_cosine_loss(
    student_tokens: torch.Tensor,      # [B, K, D]
    teacher_tokens: torch.Tensor,      # [B, L, D]
    teacher_mask: torch.Tensor = None, # [B, L], 1 valid, 0 pad
    bidirectional: bool = True,
) -> torch.Tensor:
    s = F.normalize(student_tokens.float(), dim=-1)
    t = F.normalize(teacher_tokens.float(), dim=-1)
    sim = torch.matmul(s, t.transpose(1, 2))  # [B, K, L]

    if teacher_mask is not None:
        valid = teacher_mask.to(device=sim.device).bool().unsqueeze(1)  # [B,1,L]
        sim = sim.masked_fill(~valid, -1e4)

    # each student slot should match at least one teacher token
    s2t = 1.0 - sim.max(dim=2).values.mean()

    if not bidirectional:
        return s2t

    # optional coverage: each valid teacher token should be covered by one student slot
    t2s_sim = sim.transpose(1, 2)  # [B,L,K]
    t2s = 1.0 - t2s_sim.max(dim=2).values

    if teacher_mask is not None:
        mask = teacher_mask.to(device=t2s.device).float()
        t2s = (t2s * mask).sum() / mask.sum().clamp_min(1.0)
    else:
        t2s = t2s.mean()

    return 0.5 * (s2t + t2s)
```

### 4.3 Targetness auxiliary loss

目标很小，单纯 BCE 容易被背景压倒。建议使用 BCEWithLogits + Dice：

```python
def targetness_aux_loss(
    targetness_logits: torch.Tensor, # [B,1,h,w]
    gt_mask: torch.Tensor,           # [B,H,W] or [B,1,H,W]
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
) -> torch.Tensor:
    if gt_mask.dim() == 3:
        gt = gt_mask.unsqueeze(1)
    elif gt_mask.dim() == 4:
        gt = gt_mask
    else:
        raise ValueError(f"gt_mask must be [B,H,W] or [B,1,H,W], got {tuple(gt_mask.shape)}")

    gt = gt.float().to(device=targetness_logits.device)
    gt_small = F.interpolate(gt, size=targetness_logits.shape[-2:], mode="nearest")

    bce = F.binary_cross_entropy_with_logits(targetness_logits.float(), gt_small.float())

    prob = torch.sigmoid(targetness_logits.float())
    inter = (prob * gt_small).sum(dim=(1, 2, 3))
    union = prob.sum(dim=(1, 2, 3)) + gt_small.sum(dim=(1, 2, 3))
    dice = 1.0 - ((2.0 * inter + 1.0) / (union + 1.0)).mean()

    return bce_weight * bce + dice_weight * dice
```

### 4.4 Sparse prompt distillation

直接复用现有 `TextSparsePromptProjector`。训练时：

```python
with torch.no_grad():
    teacher_sparse = text_sparse_prompt(
        teacher_tokens_or_global,
        attention_mask=teacher_mask,
        use_global_prompt_enhance=...,  # 按原代码参数传
    )

student_sparse = text_sparse_prompt(
    student_tokens,
    attention_mask=student_mask,
    use_global_prompt_enhance=...,      # 按原代码参数传
)

loss_prompt = F.mse_loss(student_sparse.float(), teacher_sparse.float())
```

注意：如果 `text_sparse_prompt` 的 forward 参数名与上面不同，以实际代码为准。

---

## 5. 训练脚本修改：train_sirst_hq_ubuntu.py

### 5.1 新增 imports

从 `efficient_sam.text_conditioner` 导入：

```python
from efficient_sam.text_conditioner import (
    build_targetness_aware_semantic_slot_generator,
    cosine_distill_loss,
    masked_token_set_cosine_loss,
    targetness_aux_loss,
)
```

如果避免循环或文件过长，也可以放到 `efficient_sam/tassg_losses.py` 再导入。

### 5.2 新增 argparse 参数

```python
p.add_argument("--use_tassg", action="store_true", help="Enable Targetness-aware Semantic Slot Generator.")
p.add_argument("--semantic_source", type=str, default="teacher", choices=["teacher", "student", "none"],
               help="Semantic source for CBGA/ASSP. teacher=old Qwen-CLIP features; student=TASSG output; none=no semantic prompt.")
p.add_argument("--tassg_num_slots", type=int, default=8)
p.add_argument("--tassg_hidden_dim", type=int, default=256)
p.add_argument("--tassg_num_heads", type=int, default=4)
p.add_argument("--tassg_dropout", type=float, default=0.0)
p.add_argument("--tassg_img_dim", type=int, default=256)
p.add_argument("--tassg_text_dim", type=int, default=None,
               help="Text feature dimension. If None, use --mllm_text_dim or infer from teacher feature.")
p.add_argument("--tassg_two_pass_backbone", action="store_true",
               help="Run image encoder once to generate TASSG tokens, then run text-conditioned image encoder again with student tokens.")
p.add_argument("--student_only_start_epoch", type=int, default=-1,
               help="If >=0, from this epoch onward use student semantics for mask forward even if teacher features exist.")

p.add_argument("--lambda_tassg_global", type=float, default=0.1)
p.add_argument("--lambda_tassg_token", type=float, default=0.1)
p.add_argument("--lambda_tassg_prompt", type=float, default=0.5)
p.add_argument("--lambda_tassg_targetness", type=float, default=0.2)
```

Default `semantic_source="teacher"` keeps old behavior. New experiments should use:

```bash
--use_tassg --semantic_source student
```

### 5.3 构建 TASSG

在 model、ASSP/text prompt、CBGA adapter 初始化之后或附近：

```python
tassg = None
if args.use_tassg:
    text_dim = args.tassg_text_dim
    if text_dim is None:
        text_dim = getattr(args, "mllm_text_dim", 512)

    tassg = build_targetness_aware_semantic_slot_generator(
        img_dim=args.tassg_img_dim,
        text_dim=text_dim,
        num_slots=args.tassg_num_slots,
        hidden_dim=args.tassg_hidden_dim,
        num_heads=args.tassg_num_heads,
        dropout=args.tassg_dropout,
    ).to(device)
```

### 5.4 加入 optimizer

把 `tassg.parameters()` 加入 head/new module 参数组。不要把它放进 frozen SAM backbone 参数组。

伪代码：

```python
if tassg is not None:
    head_params += list(p for p in tassg.parameters() if p.requires_grad)
```

实际变量名按原代码修改。

### 5.5 从 batch 中读取 teacher features

训练 loop 中：

```python
teacher_global = batch.get("clip_text_feat", None)
teacher_tokens = batch.get("clip_text_token_feat", None)
teacher_mask = batch.get("clip_text_attn_mask", None)

if teacher_global is not None:
    teacher_global = teacher_global.to(device, non_blocking=True).float()
if teacher_tokens is not None:
    teacher_tokens = teacher_tokens.to(device, non_blocking=True).float()
if teacher_mask is not None:
    teacher_mask = teacher_mask.to(device, non_blocking=True)
```

字段名如果不同，按 dataset 实际返回名适配。

### 5.6 训练 forward：推荐逻辑

需要支持三种语义来源：

```text
teacher: old Qwen-CLIP teacher path
student: TASSG student path
none: no semantic prompt / no text fusion
```

#### 5.6.1 student path, single-pass MVP

第一版先跑通：

```python
# 1. get image embedding without text
img_emb0, interms0, detail_ms0 = get_image_embedding_by_existing_code(...)

# 2. generate student semantics
student_out = tassg(img_emb0)
student_global = student_out["global"]
student_tokens = student_out["tokens"]
student_mask = student_out["attn_mask"]

# 3. do not run CBGA in MVP; use original image embedding
img_emb = img_emb0
fused_global = student_global
fused_tokens = student_tokens
fused_mask = student_mask

# 4. build text sparse prompt from student tokens/global using existing helper
text_sparse_embeddings = text_sparse_prompt(
    fused_tokens,
    attention_mask=fused_mask,
    ...
)

# 5. call model.predict_masks or existing decode path
```

This is the minimum version required to prove: inference no longer depends on Qwen.

#### 5.6.2 student path, two-pass CBGA

After MVP works, support:

```python
if args.tassg_two_pass_backbone:
    # Pass 1: no-text image embedding -> TASSG
    img_emb0, interms0, detail_ms0 = get_image_embedding_by_existing_code(...)
    student_out = tassg(img_emb0)

    # Pass 2: re-run text-conditioned image encoder / CBGA with student tokens
    img_emb, interms, detail_ms = get_image_embedding_with_text_by_existing_code(
        images,
        text_tokens=student_out["tokens"],
        text_attention_mask=student_out["attn_mask"],
        text_global=student_out["global"],
    )

    fused_global = student_out["global"]
    fused_tokens = student_out["tokens"]
    fused_mask = student_out["attn_mask"]
```

Do not invent a new CBGA interface if the code already has helpers such as `_apply_backbone_bifusion_adapter(...)`. Reuse existing helper and pass `student_global/student_tokens/student_mask` where teacher CLIP features were previously passed.

### 5.7 Distillation losses

Add these losses only when `args.use_tassg` and teacher features exist.

```python
loss_tassg_global = torch.tensor(0.0, device=device)
loss_tassg_token = torch.tensor(0.0, device=device)
loss_tassg_prompt = torch.tensor(0.0, device=device)
loss_tassg_targetness = torch.tensor(0.0, device=device)

if args.use_tassg:
    if teacher_global is not None:
        loss_tassg_global = cosine_distill_loss(student_global, teacher_global)

    if teacher_tokens is not None:
        loss_tassg_token = masked_token_set_cosine_loss(
            student_tokens,
            teacher_tokens,
            teacher_mask,
            bidirectional=True,
        )

    if text_sparse_prompt is not None and teacher_tokens is not None:
        with torch.no_grad():
            teacher_sparse = text_sparse_prompt(
                teacher_tokens,
                attention_mask=teacher_mask,
                ...
            )
        student_sparse = text_sparse_prompt(
            student_tokens,
            attention_mask=student_mask,
            ...
        )
        loss_tassg_prompt = F.mse_loss(student_sparse.float(), teacher_sparse.float())

    loss_tassg_targetness = targetness_aux_loss(
        student_out["targetness_logits"],
        masks,
    )
```

Total loss：

```python
loss = loss_main

if args.use_tassg:
    loss = loss \
        + args.lambda_tassg_global * loss_tassg_global \
        + args.lambda_tassg_token * loss_tassg_token \
        + args.lambda_tassg_prompt * loss_tassg_prompt \
        + args.lambda_tassg_targetness * loss_tassg_targetness
```

### 5.8 日志

训练日志中加入：

```text
loss_tassg_global
loss_tassg_token
loss_tassg_prompt
loss_tassg_targetness
semantic_source
student_only_start_epoch
tassg_two_pass_backbone
```

### 5.9 checkpoint 保存与加载

保存：

```python
if tassg is not None:
    ckpt["tassg"] = tassg.state_dict()
```

加载：

```python
if args.use_tassg and "tassg" in ckpt:
    tassg.load_state_dict(ckpt["tassg"], strict=True)
```

如果现有 checkpoint 保存方式是 flat state dict，也可以把 key prefix 写成 `tassg.`，但训练和评估必须一致。

---

## 6. 评估脚本修改：scripts/eval_accuracy_metrics.py

### 6.1 新增参数

```python
p.add_argument("--use_tassg", action="store_true")
p.add_argument("--semantic_source", type=str, default="teacher", choices=["teacher", "student", "none"])
p.add_argument("--tassg_num_slots", type=int, default=8)
p.add_argument("--tassg_hidden_dim", type=int, default=256)
p.add_argument("--tassg_num_heads", type=int, default=4)
p.add_argument("--tassg_img_dim", type=int, default=256)
p.add_argument("--tassg_text_dim", type=int, default=None)
p.add_argument("--tassg_two_pass_backbone", action="store_true")
```

### 6.2 mllm_features_path 逻辑

当前 teacher path 需要 `--mllm_features_path`。新逻辑：

```python
if args.semantic_source == "teacher":
    assert args.mllm_features_path is not None, "teacher semantic_source requires --mllm_features_path"

if args.semantic_source == "student":
    # must not require mllm_features_path
    pass
```

### 6.3 student-only eval forward

复用训练中的 student forward helper，避免复制大量代码。建议抽出一个函数到训练脚本或 shared utils，例如：

```python
def build_semantic_inputs_for_forward(...):
    return img_emb, text_sparse_embeddings, text_dense_embeddings, debug_dict
```

如果不想大改，先在 eval 脚本内实现同样逻辑。

必须满足：

```bash
python scripts/eval_accuracy_metrics.py \
  --ckpt outputs/.../best.pt \
  --data_root /path/to/dataset/IRSTD-1k \
  --split 50_50/test.txt \
  --use_tassg \
  --semantic_source student \
  --prompt_mode assp_only
```

不传 `--mllm_features_path` 也可以跑。

---

## 7. 可视化推理脚本修改：scripts/infer_hq_sirst_test_vis.py

同 eval 脚本。

必须支持：

```bash
python scripts/infer_hq_sirst_test_vis.py \
  --ckpt outputs/.../best.pt \
  --data_root /path/to/dataset/IRSTD-1k \
  --split 50_50/test.txt \
  --use_tassg \
  --semantic_source student \
  --prompt_mode assp_only \
  --save_dir outputs_vis/tassg_student
```

不传 `--mllm_features_path`。

建议额外保存 TASSG targetness map：

```text
save_targetness=True 或 --save_tassg_targetness
```

保存为灰度 PNG，便于验证 TASSG 是否关注小目标区域。

---

## 8. 新增训练与评估 launcher

新增文件：

```text
scripts/tmm_train_tassg_student.sh
scripts/tmm_eval_tassg_student.sh
```

### 8.1 训练脚本示例

```bash
#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-$(pwd)}
DATA_BASE=${DATA_BASE:-${PROJECT_DIR}/dataset}
DATASET=${DATASET:-IRSTD-1k}
GPU=${GPU:-0}

export CUDA_VISIBLE_DEVICES=${GPU}

python train_sirst_hq_ubuntu.py \
  --data_root "${DATA_BASE}/${DATASET}" \
  --split "50_50/train.txt" \
  --val_split "50_50/test.txt" \
  --mllm_features_path "${DATA_BASE}/${DATASET}/Qwen3-VL-8B-Instruct_mllm_clip_token_features.pt" \
  --use_tassg \
  --semantic_source student \
  --tassg_num_slots 8 \
  --tassg_hidden_dim 256 \
  --tassg_num_heads 4 \
  --lambda_tassg_global 0.1 \
  --lambda_tassg_token 0.1 \
  --lambda_tassg_prompt 0.5 \
  --lambda_tassg_targetness 0.2 \
  --prompt_mode assp_only \
  --exp_name "tassg_student_${DATASET}"
```

具体参数名按现有脚本实际参数修正。

### 8.2 评估脚本示例

```bash
#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-$(pwd)}
DATA_BASE=${DATA_BASE:-${PROJECT_DIR}/dataset}
DATASET=${DATASET:-IRSTD-1k}
CKPT=${CKPT:?Please set CKPT=/path/to/best.pt}
GPU=${GPU:-0}

export CUDA_VISIBLE_DEVICES=${GPU}

python scripts/eval_accuracy_metrics.py \
  --ckpt "${CKPT}" \
  --data_root "${DATA_BASE}/${DATASET}" \
  --split "50_50/test.txt" \
  --use_tassg \
  --semantic_source student \
  --tassg_num_slots 8 \
  --prompt_mode assp_only
```

---

## 9. 推荐实现顺序

### Step 1：MVP student sparse prompt

实现：

```text
image embedding -> TASSG -> student semantic tokens/global -> existing TextSparsePromptProjector -> mask decoder
```

暂时不启用 two-pass CBGA。

验收：

```text
训练能跑
eval 不传 mllm_features_path 能跑
checkpoint 能保存/加载 tassg
```

### Step 2：加入蒸馏 loss

实现：

```text
global cosine loss
token set cosine loss
sparse prompt MSE loss
targetness BCE+Dice loss
```

验收：日志能看到这些 loss，loss 不为 NaN。

### Step 3：two-pass CBGA

实现：

```text
Pass 1: image -> image embedding -> TASSG
Pass 2: image + TASSG tokens -> text-conditioned image encoder / CBGA
```

验收：`--tassg_two_pass_backbone` 能跑，但默认可关闭。

### Step 4：推理脚本 student-only

实现：

```text
eval_accuracy_metrics.py: --use_tassg --semantic_source student without --mllm_features_path
infer_hq_sirst_test_vis.py: same
```

### Step 5：补 ablation launcher

实现以下实验配置。

---

## 10. 需要跑的消融实验

| Setting | 推理用 Qwen | 推理用 cached text features | 说明 |
|---|---:|---:|---|
| baseline no semantic | No | No | 纯视觉 / no text prompt |
| teacher Qwen-CLIP upper bound | Yes/offline cached | Yes | 原始方法上限，不作为最终部署主结果 |
| TASSG student-only | No | No | 主方法 |
| TASSG w/o targetness loss | No | No | `lambda_tassg_targetness=0` |
| TASSG w/o token distill | No | No | `lambda_tassg_token=0` |
| TASSG w/o prompt distill | No | No | `lambda_tassg_prompt=0` |
| TASSG single-pass | No | No | no CBGA two-pass |
| TASSG two-pass CBGA | No | No | with `--tassg_two_pass_backbone` |

主文结果应报告：

```text
TIRST-SAM + TASSG student-only
Inference input: image only
No Qwen3-VL during inference
No CLIP text encoder during inference
No external text description during inference
```

---

## 11. 常见坑与处理

### 11.1 text_dim 不一致

如果 teacher global 是 `[B, 512]`，TASSG 也必须输出 `[B, 512]`。

如果实际 CLIP 是 768 或 1024，不要写死 512。优先：

```text
args.tassg_text_dim
args.mllm_text_dim
teacher_global.shape[-1]
```

### 11.2 image embedding channel 不一致

如果 SAM image embedding 不是 256，用 `--tassg_img_dim` 或从 `img_emb.shape[1]` 动态初始化。但动态初始化会影响 checkpoint，推荐先显式传参。

### 11.3 targetness map 全黑或全白

如果 targetness loss 不稳定：

```text
lambda_tassg_targetness 从 0.2 降到 0.05
BCE + Dice 改为 only Dice
targetness_head 最后一层 bias 初始化为负数，例如 -4.0
```

建议初始化：

```python
nn.init.constant_(self.targetness_head[-1].bias, -4.0)
```

这样初始目标响应较低，适合小目标稀疏场景。

### 11.4 prompt distillation 不稳定

先只用 teacher tokens 做 prompt distill，不要混用 teacher global：

```python
teacher_sparse = text_sparse_prompt(teacher_tokens, attention_mask=teacher_mask)
student_sparse = text_sparse_prompt(student_tokens, attention_mask=student_mask)
```

如果 MSE 太大，改成 SmoothL1：

```python
F.smooth_l1_loss(student_sparse.float(), teacher_sparse.float(), beta=0.1)
```

### 11.5 two-pass 太慢

two-pass 是为了最大程度复用 CBGA，不是最终必须。论文中可以把 two-pass 作为性能上限，把 single-pass student 作为效率版本。

---

## 12. 代码验收命令

### 12.1 语法检查

```bash
python -m py_compile efficient_sam/text_conditioner.py
python -m py_compile train_sirst_hq_ubuntu.py
python -m py_compile scripts/eval_accuracy_metrics.py
python -m py_compile scripts/infer_hq_sirst_test_vis.py
```

### 12.2 单 batch smoke test

用一个小 batch 跑 1-2 iteration：

```bash
python train_sirst_hq_ubuntu.py \
  --data_root /path/to/dataset/IRSTD-1k \
  --split 50_50/train.txt \
  --val_split 50_50/test.txt \
  --mllm_features_path /path/to/dataset/IRSTD-1k/Qwen3-VL-8B-Instruct_mllm_clip_token_features.pt \
  --use_tassg \
  --semantic_source student \
  --tassg_num_slots 8 \
  --lambda_tassg_global 0.1 \
  --lambda_tassg_token 0.1 \
  --lambda_tassg_prompt 0.5 \
  --lambda_tassg_targetness 0.2 \
  --max_epochs 1 \
  --batch_size 2 \
  --exp_name debug_tassg_student
```

参数名按原训练脚本实际名字修正。

### 12.3 student-only eval smoke test

```bash
python scripts/eval_accuracy_metrics.py \
  --ckpt outputs_sam_sirst_hq/debug_tassg_student/best.pt \
  --data_root /path/to/dataset/IRSTD-1k \
  --split 50_50/test.txt \
  --use_tassg \
  --semantic_source student \
  --prompt_mode assp_only
```

该命令不得要求 `--mllm_features_path`。

---

## 13. 论文方法表述对应修改

实现后，方法部分可以改成：

```text
During training, Qwen3-VL is used as a frozen offline semantic teacher to generate image-specific descriptions for infrared small-target scenes. The descriptions are encoded by the CLIP text encoder to obtain token-level and global semantic features, which provide distillation targets for the proposed Targetness-aware Semantic Slot Generator. During inference, Qwen3-VL and the CLIP text encoder are removed. Given a new infrared image, TASSG directly predicts image-conditioned semantic slots and a global semantic embedding from visual features. These predicted semantic representations are then injected into CBGA and transformed by ASSP into sparse prompt embeddings for the SAM mask decoder.
```

审稿回复核心句：

```text
In the revised framework, Qwen3-VL is not used during inference. It is used only offline during training as a semantic teacher. For a new image, the trained TASSG predicts semantic prompt tokens directly from the image feature, so the deployed model requires only the infrared image as input.
```

---

## 14. 最终交付要求

请 Codex 完成后给出：

```text
1. 修改文件列表
2. 新增参数列表
3. 训练命令
4. student-only 评估命令
5. 是否不传 mllm_features_path 也能评估
6. 一次 debug 训练日志中的 loss_main、loss_tassg_global、loss_tassg_token、loss_tassg_prompt、loss_tassg_targetness
7. 如果 two-pass CBGA 未完成，请明确说明，并保证 single-pass TASSG 已可运行
```
