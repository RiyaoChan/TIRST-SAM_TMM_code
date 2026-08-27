#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/path/to/IRSTD-1k}"
EPOCHS="${EPOCHS:-100}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${REPO_ROOT}/outputs/microquery/end2end_full/IRSTD-1k"
variants=(c0_one_query c1_independent_aux f1_soft_gate f2_gate_token)
directories=(C0_one_query C1_independent_aux F1_soft_gate F2_gate_token)

cd "${REPO_ROOT}"
"${PYTHON_BIN}" scripts/train_microquery_end2end.py --make_shared_init
for index in 0 1 2 3; do
  "${PYTHON_BIN}" scripts/train_microquery_end2end.py \
    --variant "${variants[$index]}" --epochs "${EPOCHS}" \
    --batch_size 4 --gradient_accumulation 1 --data_root "${DATA_ROOT}"
done
for index in 0 1 2 3; do
  run="${OUTPUT_ROOT}/${directories[$index]}"
  "${PYTHON_BIN}" scripts/eval_microquery_end2end.py \
    --checkpoint "${run}/best_fixed05_global_iou.pt" --data_root "${DATA_ROOT}"
done
for index in 2 3; do
  run="${OUTPUT_ROOT}/${directories[$index]}"
  "${PYTHON_BIN}" scripts/eval_microquery_counterfactuals.py \
    --checkpoint "${run}/best_fixed05_global_iou.pt" \
    --main_evaluation_summary "${run}/evaluation_summary.json" \
    --output_dir "${OUTPUT_ROOT}/counterfactuals" --data_root "${DATA_ROOT}"
done
"${PYTHON_BIN}" scripts/compare_microquery_end2end.py \
  --root "${OUTPUT_ROOT}" --bootstrap_samples 2000
