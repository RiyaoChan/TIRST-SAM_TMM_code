param(
    [Parameter(Mandatory = $true)][string]$DataRoot,
    [Parameter(Mandatory = $true)][string]$ProbeCheckpoint,
    [Parameter(Mandatory = $true)][ValidateSet('points', 'dense', 'dense_points')][string]$PromptInput,
    [Parameter(Mandatory = $true)][string]$OutputDir,
    [string]$PythonExe = 'python',
    [int]$Epochs = 100,
    [int]$Seed = 20260825
)

$ErrorActionPreference = 'Stop'
$arguments = @(
    'scripts/train_experiment1_single_view.py',
    '--data_root', $DataRoot,
    '--generator', 'probe',
    '--probe_checkpoint', $ProbeCheckpoint,
    '--prompt_input', $PromptInput,
    '--prompt_budget', '5',
    '--output_dir', $OutputDir,
    '--epochs', $Epochs,
    '--batch_size', '4',
    '--workers', '4',
    '--seed', $Seed,
    '--amp_dtype', 'bfloat16'
)

& $PythonExe @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Experiment 1 single-view training failed with exit code $LASTEXITCODE"
}

