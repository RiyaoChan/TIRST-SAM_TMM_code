# TIRST-SAM TMM Code

This is a cleaned code package extracted from the mixed EfficientSAM working
repository for the TIRST-SAM/TMM experiments.

## What Is Included

- `train_sirst_hq_ubuntu.py`: main training entry for the paper experiments.
- `sirst_dataset.py`: IRSTD dataset loader and preprocessing.
- `efficient_sam/`: EfficientSAM-HQ backbone, decoder, prompt encoder, CBGA,
  ASSP/text-prompt, and auxiliary modules required by the training script.
- `scripts/tmm_required_ablations.sh`: required TMM ablation launcher.
- `scripts/tmm_train_assp_no_gt_points.sh`: ASSP-only training without valid
  GT point prompts.
- `scripts/tmm_train_tassg_student.sh`: single-pass TASSG student-only training.
- `scripts/tmm_train_tassg_student_slots.sh`: TASSG student-only training with
  8 semantic slots injected as sparse prompts.
- `scripts/tmm_train_tassg_twopass_cbga.sh`: two-pass TASSG+CBGA training from
  the latest single-pass checkpoint.
- `scripts/tmm_train_tassg_twopass_cbga_slots.sh`: two-pass TASSG+CBGA with
  `fused_tokens` and 8 sparse prompt tokens.
- `scripts/eval_accuracy_metrics.py`: IoU/nIoU/F1/Pd/Fa evaluation.
- `scripts/infer_hq_sirst_test_vis.py`: checkpoint inference and visualization.
- `EXPERIMENT_RECORD.md`: verified server results, run status, failures, and
  protocol-matched ablation comparisons.
- `scripts/tmm_make_text_feature_variants.py` and
  `scripts/tmm_eval_text_variants.sh`: text-sensitivity evaluation helpers.

## What Is Not Included

The following are intentionally not copied:

- datasets;
- model weights and checkpoints;
- `outputs_*`, logs, visualizations, and analysis folders;
- `.git` history and caches;
- server credentials.
- historical command notebooks and server-status notes.

Put the EfficientSAM baseline weight at:

```text
weights/efficient_sam_vitt.pt
```

Put datasets under either `${PROJECT_DIR}/dataset` or pass `DATA_BASE` to the
launcher scripts. Expected dataset names are:

```text
dataset/IRSTD-1k
dataset/NUAA-SIRST
dataset/NUDT-SIRST
```

Each dataset should contain the split files used by the paper, for example:

```text
50_50/train.txt
50_50/test.txt
Qwen3-VL-8B-Instruct_mllm_clip_token_features.pt
```

## Environment

Install PyTorch for your CUDA version first, then install the remaining packages:

```bash
pip install -r requirements.txt
```

The experiments were run with 256x256 inputs and an RTX 3090-class GPU.

## GPT-5.6 Structured Semantic Teachers

`scripts/generate_gpt_structured_prompts.py` replaces unconstrained Qwen
descriptions with a reproducible three-stage offline pipeline:

```text
infrared image -> GPT-5.6 Sol structured attributes
               -> deterministic fixed-template caption
               -> CLIP token/global text features
```

GPT is used only to predict a closed set of visual attributes (presence, count,
position, size, shape, background, contrast, and confidence). The caption is
rendered locally with fixed vocabulary and field order, so wording changes do
not become an uncontrolled experimental variable. CLIP features are still
generated because the current TIRST-SAM code consumes text-encoder embeddings,
not raw captions.

Install the dependencies, then validate a dataset and output plan without an
API key or any file writes:

```bash
python scripts/generate_gpt_structured_prompts.py \
  --data_root /path/to/dataset/IRSTD-1k \
  --split_txt 50_50/train.txt 50_50/test.txt \
  --dry_run
```

After configuring `OPENAI_API_KEY` in the local environment, first run a small
GPT-only smoke test:

```bash
python scripts/generate_gpt_structured_prompts.py \
  --data_root /path/to/dataset/IRSTD-1k \
  --prefix gpt5p6_smoke \
  --max_images 5 \
  --skip_clip
```

Inspect the JSON outputs, then generate CLIP features from the cached structured
attributes without making another API request:

```bash
python scripts/generate_gpt_structured_prompts.py \
  --data_root /path/to/dataset/IRSTD-1k \
  --prefix gpt5p6_smoke \
  --max_images 5 \
  --skip_gpt \
  --clip_device cuda
```

For a full resumable run, remove `--max_images` and use a production prefix such
as `gpt5p6`. Add `--all_images` to cover every supported file in the dataset's
`images` directory even when a split file is incomplete. For a compatible gateway,
`--workers 4` can issue a small number of concurrent requests; the default remains
one request at a time. The generated artifacts are:

- `<prefix>_structured_attributes.json`: auditable GPT output;
- `<prefix>_fixed_descriptions.json`: deterministic captions;
- `<prefix>_generation_log.jsonl`: per-image response ID, usage, and errors;
- `<prefix>_generation_manifest.json`: schema, prompts, and run configuration;
- `<prefix>_clip_token_features.pt`: token features plus `global_feat`, directly
  usable as the model's `--mllm_features_path`;
- optional `<prefix>_clip_features.pt`, enabled by `--save_global_clip`.

The script reads the key only from an environment variable. Do not put keys in
commands, configuration files, logs, or Git. Images are sent to the OpenAI API
during the GPT stage; `--skip_gpt` is entirely local.

For an explicitly trusted OpenAI-compatible gateway, pass its root with
`--base_url` or set `OPENAI_BASE_URL`. A bare host is normalized to `/v1`:

```bash
python scripts/generate_gpt_structured_prompts.py \
  --data_root /path/to/dataset/IRSTD-1k \
  --base_url https://gateway.example.com \
  --prefix gpt5p6_smoke \
  --max_images 5 \
  --skip_clip
```

The API key and input images are sent to that gateway, so only use a provider
whose data handling and billing terms you accept.

## Main Training Examples

Run the required ablation launcher:

```bash
cd /path/to/TIRST-SAM_TMM_code
DATA_BASE=/path/to/SIRST-5K-main/dataset \
GPU_LIST=auto \
RUN_GROUPS=all \
bash scripts/tmm_required_ablations.sh
```

Gated backbone CBGA runs use the stable delta-only path by default in this
launcher. The gate is applied to the cross-modal attention delta before the
output projection, so a closed gate is an identity mapping instead of a learned
unimodal residual. Set `CBGA_DELTA_ONLY=0` only to reproduce legacy checkpoints.
The launcher also accepts `MLLM_FEATURE_FILE`, `GRAD_ACCUM_STEPS`, and
`GRAD_CLIP_NORM`. For a 24 GB GPU, `BS=2 GRAD_ACCUM_STEPS=2
GRAD_CLIP_NORM=1.0` preserves an effective batch size of four while reducing
CBGA activation memory.

Run the no-GT-point ASSP-only ablation:

```bash
cd /path/to/TIRST-SAM_TMM_code
DATA_BASE=/path/to/SIRST-5K-main/dataset \
GPU_LIST=auto \
MAX_PARALLEL=3 \
bash scripts/tmm_train_assp_no_gt_points.sh
```

## TASSG Student-Only Training

```bash
cd /path/to/TIRST-SAM_TMM_code
DATA_BASE=/path/to/SIRST-5K-main/dataset \
DATASET=IRSTD-1k \
GPU=0 \
bash scripts/tmm_train_tassg_student.sh
```

The command above uses cached Qwen/CLIP features only as training-time teacher
supervision. Deployed student inference does not require Qwen, CLIP, captions,
or `mllm_features_path`.

To inject semantic slots directly as sparse prompts, use:

```bash
DATA_BASE=/path/to/SIRST-5K-main/dataset \
DATASET=IRSTD-1k \
GPU=0 \
bash scripts/tmm_train_tassg_student_slots.sh
```

To start two-pass CBGA from a single-pass checkpoint:

```bash
DATA_BASE=/path/to/SIRST-5K-main/dataset \
DATASET=IRSTD-1k \
SINGLE_PASS_CKPT=/path/to/single-pass/best.pt \
GPU=0 \
bash scripts/tmm_train_tassg_twopass_cbga.sh
```

## Evaluation Examples

Student-only deployment evaluation, with no cached Qwen/CLIP feature file:

```bash
python scripts/eval_accuracy_metrics.py \
  --ckpt outputs_tassg_student/EXP_NAME/RUN_ID/best.pt \
  --data_root /path/to/SIRST-5K-main/dataset/IRSTD-1k \
  --split 50_50/test.txt \
  --prompt_mode assp_only \
  --use_tassg \
  --semantic_source student
```

Teacher/cached-feature evaluation is only for the old Qwen-CLIP upper-bound or
text-sensitivity experiments:

```bash
python scripts/eval_accuracy_metrics.py \
  --ckpt outputs_sam_sirst_hq/EXP_NAME/best.pt \
  --data_root /path/to/SIRST-5K-main/dataset/IRSTD-1k \
  --split 50_50/test.txt \
  --mllm_features_path /path/to/SIRST-5K-main/dataset/IRSTD-1k/Qwen3-VL-8B-Instruct_mllm_clip_token_features.pt \
  --prompt_mode assp_only \
  --semantic_source teacher
```

Use `--prompt_mode gt_points` only for legacy GT-pointed comparisons. For
automatic IRSTD results, use `--prompt_mode assp_only` or checkpoints trained
without GT point prompts.
