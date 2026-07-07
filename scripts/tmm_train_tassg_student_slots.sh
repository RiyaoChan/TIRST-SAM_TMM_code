#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}

export TEXT_SPARSE_NUM_TOKENS=${TEXT_SPARSE_NUM_TOKENS:-8}
export TEXT_SPARSE_PROMPT_SOURCE=${TEXT_SPARSE_PROMPT_SOURCE:-fused_tokens}
export TEXT_SPARSE_RAW_GLOBAL_GATE=${TEXT_SPARSE_RAW_GLOBAL_GATE:-0}
export PROMPT_TAG=${PROMPT_TAG:-slots${TEXT_SPARSE_NUM_TOKENS}}

exec "${PROJECT_DIR}/scripts/tmm_train_tassg_student.sh"
