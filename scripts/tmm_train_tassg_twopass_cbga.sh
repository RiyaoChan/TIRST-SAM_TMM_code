#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
DATA_BASE=${DATA_BASE:-/DATA20T/bip/cry/code/SIRST-5K-main/dataset}
DATASET=${DATASET:-IRSTD-1k}
DATA_ROOT=${DATA_ROOT:-${DATA_BASE}/${DATASET}}
TRAIN_TXT=${TRAIN_TXT:-50_50/train.txt}
VAL_TXT=${VAL_TXT:-50_50/test.txt}
FEATURE_PATH=${FEATURE_PATH:-${DATA_ROOT}/Qwen3-VL-8B-Instruct_mllm_clip_token_features.pt}
GPU=${GPU:-0}
TWO_PASS_EXTRA_EPOCHS=${TWO_PASS_EXTRA_EPOCHS:-300}
OUT_DIR=${OUT_DIR:-${PROJECT_DIR}/outputs_tassg_twopass_cbga}
INIT_CKPT=${INIT_CKPT:-${PROJECT_DIR}/weights/efficient_sam_vitt.pt}
MASK_SUFFIX=${MASK_SUFFIX:-}
PYTHON=${PYTHON:-/home/bip/cry/anaconda3/bin/python}
TEXT_SPARSE_NUM_TOKENS=${TEXT_SPARSE_NUM_TOKENS:-1}
TEXT_SPARSE_PROMPT_SOURCE=${TEXT_SPARSE_PROMPT_SOURCE:-raw_global}
TEXT_SPARSE_RAW_GLOBAL_GATE=${TEXT_SPARSE_RAW_GLOBAL_GATE:-1}
BATCH_SIZE=${BATCH_SIZE:-2}

case "${DATASET}" in
  IRSTD-1k) DATASET_SLUG=${DATASET_SLUG:-IRSTD1k} ;;
  NUAA-SIRST) DATASET_SLUG=${DATASET_SLUG:-NUAA} ;;
  NUDT-SIRST) DATASET_SLUG=${DATASET_SLUG:-NUDT} ;;
  *) DATASET_SLUG=${DATASET_SLUG:-${DATASET//[^A-Za-z0-9]/}} ;;
esac

if [[ "${TEXT_SPARSE_PROMPT_SOURCE}" == "fused_tokens" ]]; then
  PROMPT_TAG=${PROMPT_TAG:-slots${TEXT_SPARSE_NUM_TOKENS}}
else
  PROMPT_TAG=${PROMPT_TAG:-global${TEXT_SPARSE_NUM_TOKENS}}
fi
EXP_NAME=${EXP_NAME:-${DATASET_SLUG}_TASSG_twopassCBGA_${PROMPT_TAG}_ASSPonly_noGTpoints_split50_50}

TEXT_SPARSE_GATE_ARGS=()
if [[ "${TEXT_SPARSE_RAW_GLOBAL_GATE}" == "1" || "${TEXT_SPARSE_RAW_GLOBAL_GATE}" == "true" ]]; then
  TEXT_SPARSE_GATE_ARGS+=(--text_sparse_raw_global_gate)
fi

SINGLE_PASS_OUT_DIR=${SINGLE_PASS_OUT_DIR:-${PROJECT_DIR}/outputs_tassg_student}
SINGLE_PASS_EXP=${SINGLE_PASS_EXP:-${DATASET_SLUG}_TASSG_student_${PROMPT_TAG}_ASSPonly_noGTpoints_split50_50}
SINGLE_PASS_CKPT=${SINGLE_PASS_CKPT:-}

if [[ -z "${SINGLE_PASS_CKPT}" ]]; then
  shopt -s nullglob
  ckpts=("${SINGLE_PASS_OUT_DIR}/${SINGLE_PASS_EXP}"/*/best.pt)
  if (( ${#ckpts[@]} == 0 )); then
    legacy_exp="${DATASET_SLUG}_TASSG_student_ASSPonly_noGTpoints_split50_50"
    ckpts=("${SINGLE_PASS_OUT_DIR}/${legacy_exp}"/*/best.pt)
  fi
  if (( ${#ckpts[@]} == 0 )); then
    legacy_exp="${DATASET}_TASSG_student_ASSPonly_noGTpoints_split50_50"
    ckpts=("${SINGLE_PASS_OUT_DIR}/${legacy_exp}"/*/best.pt)
  fi
  shopt -u nullglob
  if (( ${#ckpts[@]} == 0 )); then
    echo "Could not find single-pass best.pt under ${SINGLE_PASS_OUT_DIR}/${SINGLE_PASS_EXP} or legacy names." >&2
    exit 1
  fi
  SINGLE_PASS_CKPT=$(ls -1t "${ckpts[@]}" | head -n 1)
fi

if [[ ! -f "${SINGLE_PASS_CKPT}" ]]; then
  echo "SINGLE_PASS_CKPT does not exist: ${SINGLE_PASS_CKPT}" >&2
  exit 1
fi

CKPT_EPOCH=$("${PYTHON}" -c "import torch; ckpt=torch.load('${SINGLE_PASS_CKPT}', map_location='cpu'); print(int(ckpt.get('epoch', 0)))")
EPOCHS=${EPOCHS:-$((CKPT_EPOCH + TWO_PASS_EXTRA_EPOCHS))}
if (( EPOCHS <= CKPT_EPOCH )); then
  echo "EPOCHS must be greater than checkpoint epoch. CKPT_EPOCH=${CKPT_EPOCH}, EPOCHS=${EPOCHS}" >&2
  exit 1
fi
echo "Two-pass resume plan: single_pass_ckpt=${SINGLE_PASS_CKPT}"
echo "Two-pass resume plan: ckpt_epoch=${CKPT_EPOCH}, target_epochs=${EPOCHS}, actual_extra_epochs=$((EPOCHS - CKPT_EPOCH))"
echo "Two-pass prompt config: source=${TEXT_SPARSE_PROMPT_SOURCE}, tokens=${TEXT_SPARSE_NUM_TOKENS}, raw_global_gate=${TEXT_SPARSE_RAW_GLOBAL_GATE}"
echo "Two-pass runtime config: batch_size=${BATCH_SIZE}, cuda_visible_devices=${GPU}"

export CUDA_VISIBLE_DEVICES=${GPU}
cd "${PROJECT_DIR}"

"${PYTHON}" train_sirst_hq_ubuntu.py \
  --data_root "${DATA_ROOT}" \
  --train_txt "${TRAIN_TXT}" \
  --val_txt "${VAL_TXT}" \
  --mllm_features_path "${FEATURE_PATH}" \
  --use_mllm_prompt \
  --disable_text_conditioner \
  --use_text_sparse_prompt \
  --text_sparse_num_tokens "${TEXT_SPARSE_NUM_TOKENS}" \
  --text_sparse_prompt_source "${TEXT_SPARSE_PROMPT_SOURCE}" \
  "${TEXT_SPARSE_GATE_ARGS[@]}" \
  --use_tassg \
  --semantic_source student \
  --tassg_two_pass_backbone \
  --tassg_num_slots 8 \
  --tassg_hidden_dim 256 \
  --tassg_num_heads 4 \
  --lambda_tassg_global 0.1 \
  --lambda_tassg_token 0.1 \
  --lambda_tassg_prompt 0.5 \
  --lambda_tassg_targetness 0.2 \
  --use_gated_bifusion_backbone_blocks \
  --bifusion_gate_init_bias -2.0 \
  --prompt_mode assp_only \
  --size 256 \
  --keep_ratio_pad \
  --batch_size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --model vitt \
  --hq_warmup_epochs 30 \
  --freeze_encoder_epochs 60 \
  --sctransnet_preproc \
  --sc_use_gamma \
  --sc_pos_prob 0.5 \
  --val_thr_search \
  --init_from_baseline "${INIT_CKPT}" \
  --resume_ckpt "${SINGLE_PASS_CKPT}" \
  --resume_reset_optimizer \
  --resume_reset_best \
  --mask_suffix "${MASK_SUFFIX}" \
  --out_dir "${OUT_DIR}/${EXP_NAME}" \
  --exp_name "${EXP_NAME}"
