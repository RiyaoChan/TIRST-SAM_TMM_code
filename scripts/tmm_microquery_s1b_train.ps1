param(
    [Parameter(Mandatory = $true)][string]$TrainFeatures,
    [Parameter(Mandatory = $true)][string]$TrainTargets,
    [Parameter(Mandatory = $true)][string]$ValFeatures,
    [Parameter(Mandatory = $true)][string]$ValTargets,
    [Parameter(Mandatory = $true)][string]$OldObjectnessCheckpoint,
    [Parameter(Mandatory = $true)][string]$OneQuerySummary,
    [Parameter(Mandatory = $true)][string]$OutputDir,
    [ValidateSet('b1_b2', 'b3', 'b4')][string]$Stage = 'b1_b2',
    [string]$PythonExe = 'python'
)

$ErrorActionPreference = 'Stop'
& $PythonExe scripts/train_microquery_component_safe.py `
    --train_features $TrainFeatures --train_targets $TrainTargets `
    --val_features $ValFeatures --val_targets $ValTargets `
    --old_objectness_checkpoint $OldObjectnessCheckpoint `
    --one_query_summary $OneQuerySummary --output_dir $OutputDir --stage $Stage
if ($LASTEXITCODE -ne 0) { throw "S1b $Stage training failed: $LASTEXITCODE" }
