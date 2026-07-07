#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
DATA_BASE=${DATA_BASE:-/DATA20T/bip/cry/code/SIRST-5K-main/dataset}
SINGLE_PASS_OUT_DIR=${SINGLE_PASS_OUT_DIR:-${PROJECT_DIR}/outputs_tassg_student}
TWO_PASS_OUT_DIR=${TWO_PASS_OUT_DIR:-${PROJECT_DIR}/outputs_tassg_twopass_cbga}
LOG_DIR=${LOG_DIR:-${PROJECT_DIR}/logs}
POLL_SECONDS=${POLL_SECONDS:-300}
SINGLE_PASS_EPOCHS=${SINGLE_PASS_EPOCHS:-1000}
TWO_PASS_EXTRA_EPOCHS=${TWO_PASS_EXTRA_EPOCHS:-300}
PYTHON=${PYTHON:-/home/bip/cry/anaconda3/bin/python}

# Free-GPU policy. Override these on launch if needed:
#   GPU_POOL=0,1,2,3,4,5,6 GPU_MAX_USED_MB=1200 GPU_MAX_UTIL=15 ./scripts/...
GPU_POOL=${GPU_POOL:-0,1,2,3,4,5,6}
GPU_MAX_USED_MB=${GPU_MAX_USED_MB:-1000}
GPU_MAX_UTIL=${GPU_MAX_UTIL:-10}

mkdir -p "${LOG_DIR}" "${TWO_PASS_OUT_DIR}"
cd "${PROJECT_DIR}"

DATASETS=("IRSTD-1k" "NUAA-SIRST" "NUDT-SIRST")
MASK_SUFFIXES=("" "_pixels0" "")
SINGLE_PASS_EXPS=(
  "IRSTD1k_TASSG_student_ASSPonly_noGTpoints_split50_50"
  "NUAA_TASSG_student_ASSPonly_noGTpoints_split50_50"
  "NUDT_TASSG_student_ASSPonly_noGTpoints_split50_50"
)
TWO_PASS_EXPS=(
  "IRSTD1k_TASSG_twopassCBGA_ASSPonly_noGTpoints_split50_50"
  "NUAA_TASSG_twopassCBGA_ASSPonly_noGTpoints_split50_50"
  "NUDT_TASSG_twopassCBGA_ASSPonly_noGTpoints_split50_50"
)
SINGLE_PASS_LOGS=(
  "${LOG_DIR}/IRSTD1k_TASSG_student_ASSPonly_noGTpoints_split50_50.log"
  "${LOG_DIR}/NUAA_TASSG_student_ASSPonly_noGTpoints_split50_50.log"
  "${LOG_DIR}/NUDT_TASSG_student_ASSPonly_noGTpoints_split50_50.log"
)

IFS=',' read -r -a GPU_CANDIDATES <<< "${GPU_POOL}"

latest_epoch_from_log() {
  local log_file=$1
  if [[ ! -f "${log_file}" ]]; then
    echo 0
    return
  fi
  local line epoch
  line=$(grep -F "[Epoch" "${log_file}" | tail -n 1 || true)
  if [[ -z "${line}" ]]; then
    echo 0
    return
  fi
  epoch=$(printf "%s\n" "${line}" | sed -E 's/.*\[Epoch[[:space:]]+0*([0-9]+)\].*/\1/')
  if [[ "${epoch}" =~ ^[0-9]+$ ]]; then
    echo $((10#${epoch}))
  else
    echo 0
  fi
}

has_log_error() {
  local log_file=$1
  [[ -f "${log_file}" ]] && grep -E "Traceback|RuntimeError|CUDA out of memory|nan|NaN" "${log_file}" >/dev/null 2>&1
}

train_process_running_for_exp() {
  local exp_name=$1
  pgrep -af "train_sirst_hq_ubuntu.py" | grep -F -- "--exp_name ${exp_name}" >/dev/null 2>&1
}

latest_best_ckpt() {
  local exp_name=$1
  shopt -s nullglob
  local ckpts=("${SINGLE_PASS_OUT_DIR}/${exp_name}"/*/best.pt)
  shopt -u nullglob
  if (( ${#ckpts[@]} == 0 )); then
    return 1
  fi
  ls -1t "${ckpts[@]}" | head -n 1
}

two_pass_already_started() {
  local exp_name=$1
  local log_file="${LOG_DIR}/${exp_name}.log"
  if train_process_running_for_exp "${exp_name}"; then
    return 0
  fi
  if [[ -f "${log_file}" ]] && grep -F "Run directory:" "${log_file}" >/dev/null 2>&1; then
    return 0
  fi
  shopt -s nullglob
  local ckpts=("${TWO_PASS_OUT_DIR}/${exp_name}"/*/best.pt)
  shopt -u nullglob
  (( ${#ckpts[@]} > 0 ))
}

gpu_is_free() {
  local gpu=$1
  local stats mem util
  stats=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
    awk -F',' -v target="${gpu}" '
      {
        gsub(/ /, "", $1); gsub(/ /, "", $2); gsub(/ /, "", $3);
        if ($1 == target) { print $2 " " $3; exit }
      }')
  if [[ -z "${stats}" ]]; then
    return 1
  fi
  read -r mem util <<< "${stats}"
  [[ "${mem}" =~ ^[0-9]+$ ]] || return 1
  [[ "${util}" =~ ^[0-9]+$ ]] || return 1
  (( mem <= GPU_MAX_USED_MB && util <= GPU_MAX_UTIL ))
}

find_free_gpu() {
  local gpu
  for gpu in "${GPU_CANDIDATES[@]}"; do
    gpu="${gpu//[[:space:]]/}"
    [[ -z "${gpu}" ]] && continue
    if gpu_is_free "${gpu}"; then
      echo "${gpu}"
      return 0
    fi
  done
  return 1
}

launch_two_pass() {
  local idx=$1
  local gpu=$2
  local ckpt=$3
  local dataset=${DATASETS[$idx]}
  local mask_suffix=${MASK_SUFFIXES[$idx]}
  local single_exp=${SINGLE_PASS_EXPS[$idx]}
  local two_exp=${TWO_PASS_EXPS[$idx]}
  local log_file="${LOG_DIR}/${two_exp}.log"

  if two_pass_already_started "${two_exp}"; then
    echo "$(date '+%F %T') ${two_exp} already started; skip."
    return
  fi

  echo "$(date '+%F %T') launching ${two_exp} on free GPU ${gpu} from ${ckpt}"
  nohup bash -lc "
    cd '${PROJECT_DIR}' &&
    GPU='${gpu}' DATASET='${dataset}' DATA_BASE='${DATA_BASE}' \
    SINGLE_PASS_EXP='${single_exp}' SINGLE_PASS_CKPT='${ckpt}' \
    SINGLE_PASS_EPOCHS='${SINGLE_PASS_EPOCHS}' TWO_PASS_EXTRA_EPOCHS='${TWO_PASS_EXTRA_EPOCHS}' \
    EXP_NAME='${two_exp}' OUT_DIR='${TWO_PASS_OUT_DIR}' MASK_SUFFIX='${mask_suffix}' \
    PYTHON='${PYTHON}' ./scripts/tmm_train_tassg_twopass_cbga.sh
  " > "${log_file}" 2>&1 &
  echo $! > "${LOG_DIR}/${two_exp}.pid"
}

pending_count() {
  local count=0 idx
  for idx in "${!DATASETS[@]}"; do
    if ! two_pass_already_started "${TWO_PASS_EXPS[$idx]}"; then
      count=$((count + 1))
    fi
  done
  echo "${count}"
}

while true; do
  progress=0

  for idx in "${!DATASETS[@]}"; do
    single_exp=${SINGLE_PASS_EXPS[$idx]}
    two_exp=${TWO_PASS_EXPS[$idx]}
    single_log=${SINGLE_PASS_LOGS[$idx]}

    if two_pass_already_started "${two_exp}"; then
      continue
    fi
    if has_log_error "${single_log}"; then
      echo "$(date '+%F %T') detected error in ${single_log}; skip ${two_exp}." >&2
      continue
    fi

    latest_epoch=$(latest_epoch_from_log "${single_log}")
    if (( latest_epoch < SINGLE_PASS_EPOCHS )); then
      if ! train_process_running_for_exp "${single_exp}" && (( latest_epoch > 0 )); then
        echo "$(date '+%F %T') ${single_exp} stopped at epoch ${latest_epoch}/${SINGLE_PASS_EPOCHS}; skip ${two_exp}." >&2
      else
        echo "$(date '+%F %T') waiting for ${single_exp}: epoch ${latest_epoch}/${SINGLE_PASS_EPOCHS}"
      fi
      continue
    fi

    ckpt=$(latest_best_ckpt "${single_exp}" || true)
    if [[ -z "${ckpt}" ]]; then
      echo "$(date '+%F %T') ${single_exp} reached epoch ${latest_epoch}; waiting for best.pt..."
      continue
    fi

    free_gpu=$(find_free_gpu || true)
    if [[ -z "${free_gpu}" ]]; then
      echo "$(date '+%F %T') ${two_exp} is ready, but no free GPU in pool ${GPU_POOL} (mem<=${GPU_MAX_USED_MB}MB, util<=${GPU_MAX_UTIL}%)."
      continue
    fi

    launch_two_pass "${idx}" "${free_gpu}" "${ckpt}"
    progress=1
    sleep 10
  done

  if (( $(pending_count) == 0 )); then
    echo "$(date '+%F %T') all two-pass jobs have been started."
    exit 0
  fi

  if (( progress == 0 )); then
    sleep "${POLL_SECONDS}"
  fi
done
