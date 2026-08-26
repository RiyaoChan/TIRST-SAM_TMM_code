#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 DATA_ROOT PROBE_CHECKPOINT points|dense|dense_points OUTPUT_DIR [PYTHON]" >&2
  exit 2
fi

data_root=$1
probe_checkpoint=$2
prompt_input=$3
output_dir=$4
python_exe=${5:-python}

"${python_exe}" scripts/train_experiment1_single_view.py \
  --data_root "${data_root}" \
  --generator probe \
  --probe_checkpoint "${probe_checkpoint}" \
  --prompt_input "${prompt_input}" \
  --prompt_budget 5 \
  --output_dir "${output_dir}" \
  --epochs 100 \
  --batch_size 4 \
  --workers 4 \
  --seed 20260825 \
  --amp_dtype bfloat16
