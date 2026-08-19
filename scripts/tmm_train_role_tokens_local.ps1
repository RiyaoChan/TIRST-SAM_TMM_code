param(
    [string]$ProjectDir = (Split-Path -Parent $PSScriptRoot),
    [string]$DataBase = (Join-Path (Split-Path -Parent $PSScriptRoot) 'dataset'),
    [string]$PythonBin = 'python',
    [int]$Epochs = 1000,
    [int]$BatchSize = 4,
    [int]$Workers = 2,
    [string]$RunTag = '20260819'
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $ProjectDir

$outputRoot = Join-Path $ProjectDir 'outputs\model1_gpt5p6_rolepc_sparse2_formal'
$logRoot = Join-Path $ProjectDir "job_logs\model1_gpt5p6_rolepc_sparse2_$RunTag"
$baseline = Join-Path $ProjectDir 'weights\efficient_sam_vitt.pt'
New-Item -ItemType Directory -Force -Path $outputRoot, $logRoot | Out-Null

foreach ($dataset in @('IRSTD-1k', 'NUAA-SIRST', 'NUDT-SIRST')) {
    $dataRoot = Join-Path $DataBase $dataset
    $featurePath = Join-Path $dataRoot 'gpt5p6_role_tokens_presence_count.pt'
    $experiment = "${dataset}_Model1_GPT5p6_rolePC_sparse2_noGT_fromBaseline_split50_50"
    $logPath = Join-Path $logRoot "${dataset}.log"
    $commandPath = Join-Path $logRoot "${dataset}.cmd.txt"

    $arguments = @(
        '-u', 'train_sirst_hq_ubuntu.py',
        '--data_root', $dataRoot,
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

    $quoted = $arguments | ForEach-Object {
        if ($_ -match '[\s"]') { '"' + ($_ -replace '"', '\"') + '"' } else { $_ }
    }
    ($PythonBin + ' ' + ($quoted -join ' ')) | Set-Content -LiteralPath $commandPath -Encoding utf8
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] START $dataset" | Tee-Object -FilePath $logPath
    & $PythonBin @arguments 2>&1 | Tee-Object -FilePath $logPath -Append
    if ($LASTEXITCODE -ne 0) {
        throw "$dataset training failed with exit code $LASTEXITCODE"
    }
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] COMPLETE $dataset" | Tee-Object -FilePath $logPath -Append
}
