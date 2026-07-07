#!/usr/bin/env bash
set -euo pipefail

# Evaluate full TIRST-SAM checkpoints with controlled cached-text variants.
# Run after M4 checkpoints exist:
#   cd /path/to/TIRST-SAM_TMM_code
#   GPU_LIST=auto bash scripts/tmm_eval_text_variants.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_DIR="${PROJECT_DIR:-${DEFAULT_PROJECT_DIR}}"
DATA_BASE="${DATA_BASE:-${PROJECT_DIR}/dataset}"
OUT_ROOT="${OUT_ROOT:-./outputs_sam_sirst_hq}"
EVAL_ROOT="${EVAL_ROOT:-./outputs_tmm_required/text_sensitivity}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU_LIST="${GPU_LIST:-auto}"
IDLE_MEM_MB="${IDLE_MEM_MB:-1024}"
IDLE_UTIL_PCT="${IDLE_UTIL_PCT:-10}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-60}"
EXCLUDE_GPUS="${EXCLUDE_GPUS:-}"
if [[ "$GPU_LIST" != "auto" ]]; then
  IFS=',' read -r -a GPUS <<< "$GPU_LIST"
else
  GPUS=()
fi

cd "$PROJECT_DIR"
mkdir -p "$EVAL_ROOT"

find_latest_m4() {
  local dataset="$1"
  find "$OUT_ROOT" -path "*${dataset}_TMM_M4_CBGA_ASSP_noTDP_split50_50*/best.pt" \
    | sort | tail -n 1
}

is_gpu_excluded() {
  local gpu="$1"
  local item
  IFS=',' read -r -a excluded <<< "$EXCLUDE_GPUS"
  for item in "${excluded[@]}"; do
    [[ -n "$item" && "$item" == "$gpu" ]] && return 0
  done
  return 1
}

pick_gpu() {
  if [[ "$GPU_LIST" != "auto" ]]; then
    echo "${GPUS[0]}"
    return 0
  fi
  while true; do
    while IFS=',' read -r idx mem util; do
      idx="$(echo "$idx" | xargs)"
      mem="$(echo "$mem" | xargs)"
      util="$(echo "$util" | xargs)"
      [[ -z "$idx" ]] && continue
      is_gpu_excluded "$idx" && continue
      if (( mem <= IDLE_MEM_MB && util <= IDLE_UTIL_PCT )); then
        echo "$idx"
        return 0
      fi
    done < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)
    echo "[$(date '+%F %T')] no idle GPU found for text evaluation; waiting ${GPU_POLL_SECONDS}s..." >&2
    sleep "$GPU_POLL_SECONDS"
  done
}

eval_dataset() {
  local dataset="$1"
  local root="${DATA_BASE}/${dataset}"
  local ckpt_var
  local ckpt
  local mask_suffix=""
  local gpu

  ckpt_var="CKPT_${dataset//-/_}"
  ckpt="${!ckpt_var:-}"
  if [[ -z "$ckpt" ]]; then
    ckpt="$(find_latest_m4 "$dataset")"
  fi
  if [[ -z "$ckpt" || ! -f "$ckpt" ]]; then
    echo "[warn] Missing M4 checkpoint for ${dataset}; set ${ckpt_var}=<best.pt> or run M4 first." >&2
    return 0
  fi
  if [[ "$dataset" == "NUAA-SIRST" ]]; then
    mask_suffix="_pixels0"
  fi

  local original="${root}/Qwen3-VL-8B-Instruct_mllm_clip_token_features.pt"
  local variant_dir="${EVAL_ROOT}/${dataset}_features"
  local log_dir="${EVAL_ROOT}/${dataset}_logs"
  mkdir -p "$variant_dir" "$log_dir"
  gpu="$(pick_gpu)"

  echo "[info] generating text variants for ${dataset}"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" scripts/tmm_make_text_feature_variants.py \
    --input "$original" \
    --out_dir "$variant_dir" \
    --device "cuda"

  declare -A feature_paths
  feature_paths[correct_qwen]="$original"
  feature_paths[no_text]="${variant_dir}/Qwen3-VL-8B-Instruct_mllm_clip_token_features__no_text.pt"
  feature_paths[generic_text]="${variant_dir}/Qwen3-VL-8B-Instruct_mllm_clip_token_features__generic_text.pt"
  feature_paths[mismatched_caption]="${variant_dir}/Qwen3-VL-8B-Instruct_mllm_clip_token_features__mismatched_caption.pt"
  feature_paths[random_caption]="${variant_dir}/Qwen3-VL-8B-Instruct_mllm_clip_token_features__random_caption.pt"
  feature_paths[contradictory_caption]="${variant_dir}/Qwen3-VL-8B-Instruct_mllm_clip_token_features__contradictory_caption.pt"
  feature_paths[blank_caption]="${variant_dir}/Qwen3-VL-8B-Instruct_mllm_clip_token_features__blank_caption.pt"

  for variant in correct_qwen no_text generic_text mismatched_caption random_caption contradictory_caption blank_caption; do
    echo "[info] evaluating ${dataset} / ${variant}"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" scripts/eval_accuracy_metrics.py \
      --ckpt "$ckpt" \
      --data_root "$root" \
      --split "50_50/test.txt" \
      --mask_suffix "$mask_suffix" \
      --mllm_features_path "${feature_paths[$variant]}" \
      --pd_fa_dist 3 \
      | tee "${log_dir}/${variant}.log"
  done
}

eval_dataset "IRSTD-1k"
eval_dataset "NUDT-SIRST"
eval_dataset "NUAA-SIRST"

echo "[info] text-sensitivity evaluation finished."
