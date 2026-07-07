#!/usr/bin/env bash
set -euo pipefail

# Train ASSP-only models without valid GT point prompts.
# This is a from-baseline training protocol for automatic-prompt ablation:
# image + cached text features + ASSP sparse prompt, no GT-derived point prompt.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_DIR="${PROJECT_DIR:-${DEFAULT_PROJECT_DIR}}"
DATA_BASE="${DATA_BASE:-${PROJECT_DIR}/dataset}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TRAIN_ENTRY="${TRAIN_ENTRY:-train_sirst_hq_ubuntu.py}"
OUT_ROOT="${OUT_ROOT:-./outputs_sam_sirst_hq}"
LOG_ROOT="${LOG_ROOT:-./outputs_tmm_required/logs}"

GPU_LIST="${GPU_LIST:-auto}"
IDLE_MEM_MB="${IDLE_MEM_MB:-1024}"
IDLE_UTIL_PCT="${IDLE_UTIL_PCT:-10}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-60}"
EXCLUDE_GPUS="${EXCLUDE_GPUS:-}"
MAX_PARALLEL="${MAX_PARALLEL:-3}"
DRY_RUN="${DRY_RUN:-0}"

SIZE="${SIZE:-256}"
BS="${BS:-4}"
EPOCHS="${EPOCHS:-1000}"
MODEL="${MODEL:-vitt}"
SPLIT_DIR="${SPLIT_DIR:-50_50}"
BASELINE_WEIGHTS="${BASELINE_WEIGHTS:-weights/efficient_sam_vitt.pt}"

cd "$PROJECT_DIR"
mkdir -p "$LOG_ROOT"

RUNNING_PIDS=()
RUNNING_GPUS=()

common_args() {
  local root="$1"
  shift
  echo \
    --data_root "$root" \
    --train_txt "${SPLIT_DIR}/train.txt" \
    --val_txt "${SPLIT_DIR}/test.txt" \
    --size "$SIZE" --keep_ratio_pad \
    --batch_size "$BS" --epochs "$EPOCHS" --model "$MODEL" \
    --hq_warmup_epochs 30 --freeze_encoder_epochs 60 \
    --sctransnet_preproc --sc_use_gamma --sc_pos_prob 0.5 \
    --val_thr_search --pd_fa_dist 3 \
    --init_from_baseline "$BASELINE_WEIGHTS" \
    --out_dir "$OUT_ROOT" \
    --prompt_mode assp_only \
    "$@"
}

dataset_extra_args() {
  local dataset="$1"
  if [[ "$dataset" == "NUAA-SIRST" ]]; then
    echo --n_pos 12 --n_neg 24 --mask_suffix "_pixels0" \
      --val_thr_min 0.40 --val_thr_max 0.55 --val_thr_step 0.05
  fi
}

assp_no_gt_args() {
  local root="$1"
  echo --use_point_loss --point_loss_points 4096 --point_loss_weight 0.3 \
    --use_mllm_prompt --disable_text_conditioner \
    --mllm_features_path "${root}/Qwen3-VL-8B-Instruct_mllm_clip_token_features.pt" \
    --use_text_sparse_prompt \
    --text_sparse_num_tokens 1 \
    --text_sparse_prompt_source raw_global \
    --text_sparse_raw_global_gate \
    --text_sparse_raw_global_gate_init_bias -2.0
}

reap_finished_jobs() {
  local new_pids=()
  local new_gpus=()
  local i
  for i in "${!RUNNING_PIDS[@]}"; do
    if kill -0 "${RUNNING_PIDS[$i]}" 2>/dev/null; then
      new_pids+=("${RUNNING_PIDS[$i]}")
      new_gpus+=("${RUNNING_GPUS[$i]}")
    else
      echo "[$(date '+%F %T')] finished PID=${RUNNING_PIDS[$i]} GPU=${RUNNING_GPUS[$i]}" >&2
    fi
  done
  RUNNING_PIDS=("${new_pids[@]}")
  RUNNING_GPUS=("${new_gpus[@]}")
}

is_gpu_excluded_or_running() {
  local gpu="$1"
  local item
  IFS=',' read -r -a excluded <<< "$EXCLUDE_GPUS"
  for item in "${excluded[@]}"; do
    [[ -n "$item" && "$item" == "$gpu" ]] && return 0
  done
  for item in "${RUNNING_GPUS[@]}"; do
    [[ "$item" == "$gpu" ]] && return 0
  done
  return 1
}

pick_gpu() {
  if [[ "$GPU_LIST" != "auto" ]]; then
    IFS=',' read -r -a fixed_gpus <<< "$GPU_LIST"
    echo "${fixed_gpus[${#RUNNING_PIDS[@]} % ${#fixed_gpus[@]}]}"
    return 0
  fi

  while true; do
    reap_finished_jobs
    if (( ${#RUNNING_PIDS[@]} >= MAX_PARALLEL )); then
      echo "[$(date '+%F %T')] MAX_PARALLEL=${MAX_PARALLEL} reached; waiting ${GPU_POLL_SECONDS}s..." >&2
      sleep "$GPU_POLL_SECONDS"
      continue
    fi
    while IFS=',' read -r idx mem util; do
      idx="$(echo "$idx" | xargs)"
      mem="$(echo "$mem" | xargs)"
      util="$(echo "$util" | xargs)"
      [[ -z "$idx" ]] && continue
      is_gpu_excluded_or_running "$idx" && continue
      if (( mem <= IDLE_MEM_MB && util <= IDLE_UTIL_PCT )); then
        echo "$idx"
        return 0
      fi
    done < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)
    echo "[$(date '+%F %T')] no idle GPU found (mem<=${IDLE_MEM_MB}MB, util<=${IDLE_UTIL_PCT}%). waiting ${GPU_POLL_SECONDS}s..." >&2
    sleep "$GPU_POLL_SECONDS"
  done
}

launch_dataset() {
  local dataset="$1"
  local root="${DATA_BASE}/${dataset}"
  local exp="${dataset}_TMM_M3_ASSP_noGTpts_fromBaseline_noTDP_split${SPLIT_DIR}"
  local log_file="${LOG_ROOT}/${exp}.log"
  local gpu
  local cmd

  cmd="${PYTHON_BIN} -u ${TRAIN_ENTRY} $(common_args "$root" $(dataset_extra_args "$dataset")) $(assp_no_gt_args "$root") --exp_name ${exp}"
  gpu="$(pick_gpu)"

  echo "[$(date '+%F %T')] GPU=${gpu} EXP=${exp}"
  echo "$cmd" > "${log_file}.cmd"

  if [[ "$DRY_RUN" == "1" ]]; then
    cat "${log_file}.cmd"
    return
  fi

  nohup env \
    PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    bash -lc "$cmd" > "$log_file" 2>&1 &
  RUNNING_PIDS+=("$!")
  RUNNING_GPUS+=("$gpu")
}

for dataset in IRSTD-1k NUAA-SIRST NUDT-SIRST; do
  launch_dataset "$dataset"
done

if [[ "$DRY_RUN" != "1" ]]; then
  wait
fi

echo "[$(date '+%F %T')] submitted/completed ASSP no-GT-point jobs."
