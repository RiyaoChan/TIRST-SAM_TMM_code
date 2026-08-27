param(
    [string]$Python = "python",
    [string]$DataRoot = "E:\code\SIRST-5K-main\SIRST-5K-main\dataset\IRSTD-1k",
    [ValidateSet("sequential", "parallel")][string]$Mode = "sequential",
    [int]$Epochs = 100
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$root = Join-Path $repo "outputs\microquery\end2end_full\IRSTD-1k"
$variants = @("c0_one_query", "c1_independent_aux", "f1_soft_gate", "f2_gate_token")
$directories = @("C0_one_query", "C1_independent_aux", "F1_soft_gate", "F2_gate_token")

& $Python (Join-Path $repo "scripts\train_microquery_end2end.py") --make_shared_init

if ($Mode -eq "parallel") {
    $processes = @()
    for ($index = 0; $index -lt $variants.Count; $index++) {
        $env:CUDA_VISIBLE_DEVICES = "$index"
        $log = Join-Path $root ("{0}\train.log" -f $directories[$index])
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $log) | Out-Null
        $arguments = @(
            (Join-Path $repo "scripts\train_microquery_end2end.py"),
            "--variant", $variants[$index], "--epochs", "$Epochs",
            "--batch_size", "4", "--gradient_accumulation", "1",
            "--data_root", $DataRoot
        )
        $processes += Start-Process -FilePath $Python -ArgumentList $arguments -WorkingDirectory $repo -WindowStyle Hidden -RedirectStandardOutput $log -RedirectStandardError ($log + ".err") -PassThru
    }
    $processes | Wait-Process
    Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
} else {
    for ($index = 0; $index -lt $variants.Count; $index++) {
        & $Python (Join-Path $repo "scripts\train_microquery_end2end.py") --variant $variants[$index] --epochs $Epochs --batch_size 4 --gradient_accumulation 1 --data_root $DataRoot
    }
}

for ($index = 0; $index -lt $variants.Count; $index++) {
    $run = Join-Path $root $directories[$index]
    & $Python (Join-Path $repo "scripts\eval_microquery_end2end.py") --checkpoint (Join-Path $run "best_fixed05_global_iou.pt") --data_root $DataRoot
}

$counterfactualRoot = Join-Path $root "counterfactuals"
foreach ($index in 2, 3) {
    $run = Join-Path $root $directories[$index]
    & $Python (Join-Path $repo "scripts\eval_microquery_counterfactuals.py") --checkpoint (Join-Path $run "best_fixed05_global_iou.pt") --main_evaluation_summary (Join-Path $run "evaluation_summary.json") --output_dir $counterfactualRoot --data_root $DataRoot
}
& $Python (Join-Path $repo "scripts\compare_microquery_end2end.py") --root $root --bootstrap_samples 2000

