#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 8 ]]; then
  echo "usage: $0 VAL_FEATURES VAL_TARGETS OBJECTNESS_CKPT CANDIDATE_CACHE VAL_SPLIT A1_CKPT PROBE_CKPT OUTPUT_DIR [DATA_ROOT]" >&2
  exit 2
fi

python scripts/eval_microquery_component_safe_cache.py \
  --features "$1" --targets "$2" --objectness_checkpoint "$3" \
  --candidate_cache "$4" --val_split "$5" --a1_checkpoint "$6" \
  --probe_checkpoint "$7" --output_dir "$8" --data_root "${9:-}"
