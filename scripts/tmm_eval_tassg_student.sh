#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
DATA_BASE=${DATA_BASE:-/DATA20T/bip/cry/code/SIRST-5K-main/dataset}
DATASET=${DATASET:-IRSTD-1k}
DATA_ROOT=${DATA_ROOT:-${DATA_BASE}/${DATASET}}
VAL_TXT=${VAL_TXT:-50_50/test.txt}
CKPT=${CKPT:?Please set CKPT=/path/to/best.pt}
GPU=${GPU:-0}
MASK_SUFFIX=${MASK_SUFFIX:-}
PYTHON=${PYTHON:-/home/bip/cry/anaconda3/bin/python}

export CUDA_VISIBLE_DEVICES=${GPU}
cd "${PROJECT_DIR}"

"${PYTHON}" scripts/eval_accuracy_metrics.py \
  --ckpt "${CKPT}" \
  --data_root "${DATA_ROOT}" \
  --split "${VAL_TXT}" \
  --mask_suffix "${MASK_SUFFIX}" \
  --prompt_mode assp_only \
  --use_tassg \
  --semantic_source student
