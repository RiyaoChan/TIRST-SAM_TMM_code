param(
    [string]$ProjectDir = (Split-Path -Parent $PSScriptRoot),
    [Parameter(Mandatory = $true)]
    [string]$DataRoot,
    [string]$PythonBin = 'python',
    [int]$Epochs = 100,
    [int]$BatchSize = 4,
    [int]$Workers = 2,
    [string]$RunTag = '20260825'
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $ProjectDir

$featurePath = Join-Path $DataRoot 'gpt5p6_role_tokens_presence_count_null_control.pt'
$baseline = Join-Path $ProjectDir 'weights\efficient_sam_vitt.pt'
$outputRoot = Join-Path $ProjectDir "outputs\model1_matched_null_probe_$RunTag"
$dataset = Split-Path -Leaf $DataRoot
$experiment = "${dataset}_E3_matchedNull_rolePC_sparse2_probe${Epochs}_noGT"

foreach ($requiredPath in @($DataRoot, $featurePath, $baseline)) {
    if (-not (Test-Path -LiteralPath $requiredPath)) {
        throw "Required path not found: $requiredPath"
    }
}
New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null

$arguments = @(
    '-u', 'train_sirst_hq_ubuntu.py',
    '--data_root', $DataRoot,
    '--train_txt', '50_50/train.txt',
    '--val_txt', '50_50/test.txt',
    '--size', '256',
    '--keep_ratio_pad',
    '--batch_size', [string]$BatchSize,
    '--epochs', [string]$Epochs,
    '--workers', [string]$Workers,
    '--model', 'vitt',
    '--hq_warmup_epochs', '30',
    '--freeze_encoder_epochs', '60',
    '--sctransnet_preproc',
    '--sc_use_gamma',
    '--sc_pos_prob', '0.5',
    '--sc_eval_crop', 'resize',
    '--val_thr_search',
    '--pd_fa_dist', '3',
    '--init_from_baseline', $baseline,
    '--out_dir', $outputRoot,
    '--prompt_mode', 'assp_only',
    '--use_point_loss',
    '--point_loss_points', '4096',
    '--point_loss_weight', '0.3',
    '--use_mllm_prompt',
    '--disable_text_conditioner',
    '--mllm_features_path', $featurePath,
    '--use_text_sparse_prompt',
    '--text_sparse_num_tokens', '2',
    '--text_sparse_prompt_source', 'fused_tokens',
    '--exp_name', $experiment
)
if ($dataset -eq 'NUAA-SIRST') {
    $arguments += @(
        '--n_pos', '12',
        '--n_neg', '24',
        '--mask_suffix', '_pixels0',
        '--val_thr_min', '0.40',
        '--val_thr_max', '0.55',
        '--val_thr_step', '0.05'
    )
}

Write-Output "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] START $experiment"
Write-Output ("Resolved command: {0} {1}" -f $PythonBin, ($arguments -join ' '))
& $PythonBin @arguments
if ($LASTEXITCODE -ne 0) {
    throw "$experiment failed with exit code $LASTEXITCODE"
}
Write-Output "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] COMPLETE $experiment"
