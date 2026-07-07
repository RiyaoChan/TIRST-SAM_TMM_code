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
- `scripts/eval_accuracy_metrics.py`: IoU/nIoU/F1/Pd/Fa evaluation.
- `scripts/infer_hq_sirst_test_vis.py`: checkpoint inference and visualization.
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

## Main Training Examples

Run the required ablation launcher:

```bash
cd /path/to/TIRST-SAM_TMM_code
DATA_BASE=/path/to/SIRST-5K-main/dataset \
GPU_LIST=auto \
RUN_GROUPS=all \
bash scripts/tmm_required_ablations.sh
```

Run the no-GT-point ASSP-only ablation:

```bash
cd /path/to/TIRST-SAM_TMM_code
DATA_BASE=/path/to/SIRST-5K-main/dataset \
GPU_LIST=auto \
MAX_PARALLEL=3 \
bash scripts/tmm_train_assp_no_gt_points.sh
```

## Evaluation Example

```bash
python scripts/eval_accuracy_metrics.py \
  --ckpt outputs_sam_sirst_hq/EXP_NAME/best.pt \
  --data_root /path/to/SIRST-5K-main/dataset/IRSTD-1k \
  --split 50_50/test.txt \
  --mllm_features_path /path/to/SIRST-5K-main/dataset/IRSTD-1k/Qwen3-VL-8B-Instruct_mllm_clip_token_features.pt \
  --prompt_mode assp_only
```

Use `--prompt_mode gt_points` only for legacy GT-pointed comparisons. For
automatic IRSTD results, use `--prompt_mode assp_only` or checkpoints trained
without GT point prompts.
