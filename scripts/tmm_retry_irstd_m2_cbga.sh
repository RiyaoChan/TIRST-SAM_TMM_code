#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_DIR="${PROJECT_DIR:-${DEFAULT_PROJECT_DIR}}"
DATA_BASE="${DATA_BASE:-${PROJECT_DIR}/dataset}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU_ID="${GPU_ID:-0}"
EXP="${EXP:-IRSTD-1k_TMM_M2_CBGA_only_retry1_noTDP_split50_50}"
LOG_ROOT="${LOG_ROOT:-outputs_tmm_required/logs}"

cd "$PROJECT_DIR"
mkdir -p "$LOG_ROOT"

LOG_FILE="${LOG_ROOT}/${EXP}.log"
CMD_FILE="${LOG_FILE}.cmd"

cmd=(
  "$PYTHON_BIN" -u train_sirst_hq_ubuntu.py
  --data_root "${DATA_BASE}/IRSTD-1k"
  --train_txt 50_50/train.txt
  --val_txt 50_50/test.txt
  --size 256
  --keep_ratio_pad
  --batch_size 4
  --epochs 1000
  --model vitt
  --hq_warmup_epochs 30
  --freeze_encoder_epochs 60
  --sctransnet_preproc
  --sc_use_gamma
  --sc_pos_prob 0.5
  --val_thr_search
  --pd_fa_dist 3
  --init_from_baseline weights/efficient_sam_vitt.pt
  --out_dir ./outputs_sam_sirst_hq
  --boundary_prior_sampling
  --boundary_ratio 0.5
  --use_point_loss
  --point_loss_points 4096
  --point_loss_weight 0.3
  --use_mllm_prompt
  --disable_text_conditioner
  --mllm_features_path "${DATA_BASE}/IRSTD-1k/Qwen3-VL-8B-Instruct_mllm_clip_token_features.pt"
  --use_gated_bifusion_backbone_blocks
  --bifusion_hidden_dim 128
  --bifusion_num_heads 4
  --bifusion_block_apply_every 1
  --bifusion_block_vision_res_scale 1.0
  --bifusion_block_text_res_scale 1.0
  --bifusion_gate_hidden_dim 0
  --bifusion_gate_init_bias -2.0
  --exp_name "$EXP"
)

printf "%q " "${cmd[@]}" > "$CMD_FILE"
printf "\n" >> "$CMD_FILE"

if pgrep -af "train_sirst_hq_ubuntu.py.*${EXP}" >/dev/null; then
  echo "Retry job already running for ${EXP}."
  pgrep -af "train_sirst_hq_ubuntu.py.*${EXP}" || true
  exit 0
fi

nohup env \
  PYTHONUNBUFFERED=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CUDA_VISIBLE_DEVICES="$GPU_ID" \
  "${cmd[@]}" > "$LOG_FILE" 2>&1 &

echo "RETRY_PID:$!"
echo "GPU_ID:${GPU_ID}"
echo "EXP:${EXP}"
echo "LOG:${PROJECT_DIR}/${LOG_FILE}"
