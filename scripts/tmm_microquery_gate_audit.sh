#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

device="${1:-cuda}"
python scripts/audit_microquery_gate_deployment.py --device "${device}" --bootstrap_repeats 2000
python scripts/eval_microquery_online_probe.py --device "${device}"
