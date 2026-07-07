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
SINGLE_PASS_EPOCHS=${SINGLE_PASS_EPOCHS:-1000}
TWO_PASS_EXTRA_EPOCHS=${TWO_PASS_EXTRA_EPOCHS:-300}
EPOCHS=${EPOCHS:-$((SINGLE_PASS_EPOCHS + TWO_PASS_EXTRA_EPOCHS))}
EXP_NAME=${EXP_NAME:-${DATASET}_TASSG_twopassCBGA_ASSPonly_noGTpoints_split50_50}
OUT_DIR=${OUT_DIR:-${PROJECT_DIR}/outputs_tassg_twopass_cbga}
INIT_CKPT=${INIT_CKPT:-${PROJECT_DIR}/weights/efficient_sam_vitt.pt}
MASK_SUFFIX=${MASK_SUFFIX:-}
PYTHON=${PYTHON:-/home/bip/cry/anaconda3/bin/python}

SINGLE_PASS_OUT_DIR=${SINGLE_PASS_OUT_DIR:-${PROJECT_DIR}/outputs_tassg_student}
SINGLE_PASS_EXP=${SINGLE_PASS_EXP:-${DATASET}_TASSG_student_ASSPonly_noGTpoints_split50_50}
SINGLE_PASS_CKPT=${SINGLE_PASS_CKPT:-}

if [[ -z "${SINGLE_PASS_CKPT}" ]]; then
  shopt -s nullglob
  ckpts=("${SINGLE_PASS_OUT_DIR}/${SINGLE_PASS_EXP}"/*/best.pt)
  shopt -u nullglob
  if (( ${#ckpts[@]} == 0 )); then
    echo "Could not find single-pass best.pt under ${SINGLE_PASS_OUT_DIR}/${SINGLE_PASS_EXP}" >&2
    exit 1
  fi
  SINGLE_PASS_CKPT=$(ls -1t "${ckpts[@]}" | head -n 1)
fi

if [[ ! -f "${SINGLE_PASS_CKPT}" ]]; then
  echo "SINGLE_PASS_CKPT does not exist: ${SINGLE_PASS_CKPT}" >&2
  exit 1
fi

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
  --text_sparse_num_tokens 1 \
  --text_sparse_prompt_source raw_global \
  --text_sparse_raw_global_gate \
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
  --batch_size "${BATCH_SIZE:-4}" \
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
