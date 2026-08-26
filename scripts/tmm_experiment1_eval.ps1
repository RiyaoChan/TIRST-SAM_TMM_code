param(
    [Parameter(Mandatory = $true)][string]$DataRoot,
    [Parameter(Mandatory = $true)][string]$ProbeCheckpoint,
    [Parameter(Mandatory = $true)][string]$OutputRoot,
    [string]$Split = 'splits/experiment1_seed20260825/val.txt',
    [string]$A1Checkpoint = '',
    [string]$PythonExe = 'python'
)

$ErrorActionPreference = 'Stop'

function Invoke-CheckedPython {
    param([string[]]$Arguments)
    & $PythonExe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE`: $($Arguments -join ' ')"
    }
}

Invoke-CheckedPython @(
    'scripts/eval_prompt_quality.py',
    '--data_root', $DataRoot,
    '--split', $Split,
    '--generator', 'probe',
    '--probe_checkpoint', $ProbeCheckpoint,
    '--output_dir', (Join-Path $OutputRoot 'A1_prompt')
)

Invoke-CheckedPython @(
    'scripts/eval_multiview_prompt_quality.py',
    '--data_root', $DataRoot,
    '--split', $Split,
    '--generator', 'probe',
    '--probe_checkpoint', $ProbeCheckpoint,
    '--gate', 'none',
    '--score_mode', 'mean_max',
    '--output_dir', (Join-Path $OutputRoot 'A2_prompt')
)

Invoke-CheckedPython @(
    'scripts/eval_multiview_prompt_quality.py',
    '--data_root', $DataRoot,
    '--split', $Split,
    '--generator', 'probe',
    '--probe_checkpoint', $ProbeCheckpoint,
    '--gate', 'rule',
    '--min_support', '2',
    '--max_dispersion', '2.0',
    '--alpha', '0.5',
    '--beta', '0',
    '--gamma', '0',
    '--output_dir', (Join-Path $OutputRoot 'A3_prompt')
)

if ($A1Checkpoint) {
    foreach ($mode in @('A1', 'A2', 'A3')) {
        $maskArguments = @(
            'scripts/eval_experiment1_masks.py',
            '--data_root', $DataRoot,
            '--split', $Split,
            '--checkpoint', $A1Checkpoint,
            '--probe_checkpoint', $ProbeCheckpoint,
            '--mode', $mode,
            '--output_dir', (Join-Path $OutputRoot "${mode}_mask")
        )
        if ($mode -eq 'A3') {
            $maskArguments += @(
                '--min_support', '2',
                '--max_dispersion', '2.0',
                '--alpha', '0.5',
                '--beta', '0',
                '--gamma', '0'
            )
        }
        Invoke-CheckedPython $maskArguments
    }
}
