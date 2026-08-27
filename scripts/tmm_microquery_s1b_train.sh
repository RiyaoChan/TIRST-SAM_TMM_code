#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 7 ]]; then
  echo "usage: $0 TRAIN_FEATURES TRAIN_TARGETS VAL_FEATURES VAL_TARGETS OLD_OBJECTNESS_CKPT ONE_QUERY_SUMMARY OUTPUT_DIR [STAGE]" >&2
  exit 2
fi

python scripts/train_microquery_component_safe.py \
  --train_features "$1" --train_targets "$2" --val_features "$3" \
  --val_targets "$4" --old_objectness_checkpoint "$5" \
  --one_query_summary "$6" --output_dir "$7" --stage "${8:-b1_b2}"
