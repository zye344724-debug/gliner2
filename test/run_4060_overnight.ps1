param(
    [string]$Python = "python",
    [int]$RawSamples = 10000,
    [int]$NerEpochs = 1,
    [int]$StructureEpochs = 3
)

# Windows PowerShell 5 wraps native stderr as NativeCommandError. Keep native
# process failures non-terminating and handle them via $LASTEXITCODE below.
$ErrorActionPreference = "Continue"
$TestDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PackageDir = Split-Path -Parent $TestDir
$DataDir = Join-Path $TestDir "data\full_4060"
$NerOut = Join-Path $TestDir "outputs\ner\full_4060"
$StructureOut = Join-Path $TestDir "outputs\structure\full_4060"
$LogDir = Join-Path $TestDir "logs"
$env:PYTHONPATH = "$PackageDir;$env:PYTHONPATH"
$env:TOKENIZERS_PARALLELISM = "false"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Invoke-Step {
    param(
        [string]$Name,
        [string[]]$Arguments
    )
    Write-Host "`n==> $Name" -ForegroundColor Cyan
    $LogName = ($Name -replace "[^a-zA-Z0-9_-]", "_").Trim("_") + ".log"
    $LogPath = Join-Path $LogDir $LogName
    & $Python @Arguments 2>&1 | Tee-Object -FilePath $LogPath
    $ExitCode = $LASTEXITCODE
    if ($ExitCode -ne 0) {
        throw "$Name failed with exit code $ExitCode. Full Python error: $LogPath"
    }
}

Invoke-Step -Name "check dependencies and CUDA" -Arguments @("-c", "import sys, torch, transformers, peft; print('Python:', sys.executable); print('torch:', torch.__version__); print('transformers:', transformers.__version__); print('peft:', peft.__version__); assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0)); print('VRAM GiB:', round(torch.cuda.get_device_properties(0).total_memory/2**30, 1))")

# One-deal sentences make the first overnight run fit the business metric. The
# fingerprint grouping in prepare_data keeps all pieces of one source in one split.
Invoke-Step -Name "prepare full-field single-deal data" -Arguments @(
    (Join-Path $TestDir "prepare_data.py"), "--schema-mode", "full",
    "--split-multi", "--variant", "full_4060", "--max-samples", "$RawSamples"
)
Invoke-Step -Name "validate prepared data" -Arguments @(
    (Join-Path $TestDir "validate_data.py"), "--data-dirs", $DataDir,
    "--out", (Join-Path $TestDir "logs\validate_full_4060.json")
)

$Common = @(
    "--schema-mode", "full", "--data-variant", "full_4060",
    "--batch-size", "2", "--eval-batch-size", "4", "--grad-accum", "8",
    "--max-len", "256", "--eval-steps", "100", "--fp16",
    "--gradient-checkpointing", "--early-stopping-patience", "3"
)

Invoke-Step -Name "stage 1 - NER warm-up" -Arguments (@(
    (Join-Path $TestDir "train_ner.py"), "--epochs", "$NerEpochs",
    "--output-dir", $NerOut
) + $Common)

Invoke-Step -Name "stage 2 - structure extraction" -Arguments (@(
    (Join-Path $TestDir "train_structure.py"), "--epochs", "$StructureEpochs",
    "--init-from", $NerOut, "--output-dir", $StructureOut
) + $Common)

Invoke-Step -Name "test sentence exact match" -Arguments @(
    (Join-Path $TestDir "evaluate_sentence_acc.py"), "--schema-mode", "full",
    "--data-variant", "full_4060", "--model-dir", $StructureOut,
    "--max-len", "256", "--batch-size", "4",
    "--tune-thresholds", "0.35,0.45,0.55,0.65", "--tune-limit", "300",
    "--out", (Join-Path $StructureOut "eval_sentence_acc.json")
)

Write-Host "`nFinished. Result: $StructureOut\eval_sentence_acc.json" -ForegroundColor Green
