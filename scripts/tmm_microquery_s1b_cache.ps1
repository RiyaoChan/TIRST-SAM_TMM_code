param(
    [Parameter(Mandatory = $true)][string]$ValFeatures,
    [Parameter(Mandatory = $true)][string]$ValTargets,
    [Parameter(Mandatory = $true)][string]$ObjectnessCheckpoint,
    [Parameter(Mandatory = $true)][string]$CandidateCache,
    [Parameter(Mandatory = $true)][string]$ValSplit,
    [Parameter(Mandatory = $true)][string]$A1Checkpoint,
    [Parameter(Mandatory = $true)][string]$ProbeCheckpoint,
    [Parameter(Mandatory = $true)][string]$OutputDir,
    [string]$DataRoot = '',
    [string]$PythonExe = 'python'
)

$ErrorActionPreference = 'Stop'
& $PythonExe scripts/eval_microquery_component_safe_cache.py `
    --features $ValFeatures --targets $ValTargets `
    --objectness_checkpoint $ObjectnessCheckpoint `
    --candidate_cache $CandidateCache --val_split $ValSplit `
    --a1_checkpoint $A1Checkpoint --probe_checkpoint $ProbeCheckpoint `
    --output_dir $OutputDir --data_root $DataRoot
if ($LASTEXITCODE -ne 0) { throw "S1b cache audit failed: $LASTEXITCODE" }
