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

mkdir -p "${LOG_DIR}" "${TWO_PASS_OUT_DIR}"
cd "${PROJECT_DIR}"

DATASETS=("IRSTD-1k" "NUAA-SIRST" "NUDT-SIRST")
GPU_IDS=("2" "3" "4")
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

single_pass_running() {
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

two_pass_already_running() {
  local exp_name=$1
  pgrep -af "train_sirst_hq_ubuntu.py" | grep -F -- "--exp_name ${exp_name}" >/dev/null 2>&1
}

launch_two_pass() {
  local idx=$1
  local dataset=${DATASETS[$idx]}
  local gpu=${GPU_IDS[$idx]}
  local mask_suffix=${MASK_SUFFIXES[$idx]}
  local single_exp=${SINGLE_PASS_EXPS[$idx]}
  local two_exp=${TWO_PASS_EXPS[$idx]}
  local ckpt=$2
  local log_file="${LOG_DIR}/${two_exp}.log"

  if two_pass_already_running "${two_exp}"; then
    echo "$(date '+%F %T') ${two_exp} is already running; skip launch."
    return
  fi
  if [[ -f "${log_file}" ]] && grep -F "Run directory:" "${log_file}" >/dev/null 2>&1; then
    echo "$(date '+%F %T') ${two_exp} appears to have been launched before; skip launch."
    return
  fi

  echo "$(date '+%F %T') launching ${two_exp} on GPU ${gpu} from ${ckpt}"
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

wait_and_launch_one() {
  local idx=$1
  local single_exp=${SINGLE_PASS_EXPS[$idx]}
  local log_file=${SINGLE_PASS_LOGS[$idx]}
  local latest_epoch ckpt

  while true; do
    if has_log_error "${log_file}"; then
      echo "$(date '+%F %T') detected error in ${log_file}; not launching ${single_exp} two-pass." >&2
      return 1
    fi

    latest_epoch=$(latest_epoch_from_log "${log_file}")
    if (( latest_epoch >= SINGLE_PASS_EPOCHS )); then
      ckpt=$(latest_best_ckpt "${single_exp}") || {
        echo "$(date '+%F %T') ${single_exp} reached epoch ${latest_epoch}, waiting for best.pt..."
        sleep "${POLL_SECONDS}"
        continue
      }
      launch_two_pass "${idx}" "${ckpt}"
      return 0
    fi

    if ! single_pass_running "${single_exp}" && (( latest_epoch > 0 )); then
      echo "$(date '+%F %T') ${single_exp} is not running and only reached epoch ${latest_epoch}/${SINGLE_PASS_EPOCHS}; not launching." >&2
      return 1
    fi

    echo "$(date '+%F %T') waiting for ${single_exp}: epoch ${latest_epoch}/${SINGLE_PASS_EPOCHS}"
    sleep "${POLL_SECONDS}"
  done
}

for idx in "${!DATASETS[@]}"; do
  wait_and_launch_one "${idx}" &
done

wait
echo "$(date '+%F %T') watcher finished."
