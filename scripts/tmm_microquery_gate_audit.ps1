param(
    [string]$Device = "cuda",
    [int]$BootstrapRepeats = 2000
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

python scripts/audit_microquery_gate_deployment.py `
    --device $Device `
    --bootstrap_repeats $BootstrapRepeats

python scripts/eval_microquery_online_probe.py `
    --device $Device
