param([string]$Python = "python")

# Windows PowerShell 5 wraps native stderr as NativeCommandError. Keep native
# process failures non-terminating so the full traceback reaches the log.
$ErrorActionPreference = "Continue"
$TestDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PackageDir = Split-Path -Parent $TestDir
$DataDir = Join-Path $TestDir "data\full_4060"
$OutDir = Join-Path $TestDir "outputs\ner\smoke_4060"
$LogPath = Join-Path $TestDir "logs\smoke_4060.log"
$env:PYTHONPATH = "$PackageDir;$env:PYTHONPATH"
$env:TOKENIZERS_PARALLELISM = "false"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
New-Item -ItemType Directory -Force -Path (Split-Path $LogPath) | Out-Null

$Probe = @"
import sys, torch, transformers, peft
print('python:', sys.executable)
print('torch:', torch.__version__)
print('torch_file:', torch.__file__)
print('transformers:', transformers.__version__)
print('peft:', peft.__version__)
assert torch.cuda.is_available(), 'CUDA unavailable'
print('gpu:', torch.cuda.get_device_name(0))
"@

& $Python -c $Probe 2>&1 | Tee-Object -FilePath $LogPath
if ($LASTEXITCODE -ne 0) { throw "dependency probe failed; see $LogPath" }

if (-not (Test-Path (Join-Path $DataDir "ner_train_clean.jsonl"))) {
    & $Python (Join-Path $TestDir "prepare_data.py") `
        --schema-mode full --split-multi --variant full_4060 --max-samples 3000 `
        2>&1 | Tee-Object -FilePath $LogPath -Append
    if ($LASTEXITCODE -ne 0) { throw "data preparation failed; see $LogPath" }
}

$TrainArgs = @(
    (Join-Path $TestDir "train_ner.py"),
    "--schema-mode", "full", "--data-variant", "full_4060",
    "--output-dir", $OutDir, "--epochs", "1", "--max-steps", "2",
    "--max-train-samples", "8", "--max-eval-samples", "8",
    "--batch-size", "1", "--eval-batch-size", "1", "--grad-accum", "1",
    "--max-len", "128", "--eval-steps", "1", "--fp16",
    "--gradient-checkpointing", "--allow-missing-field-labels"
)
& $Python @TrainArgs 2>&1 | Tee-Object -FilePath $LogPath -Append
$ExitCode = $LASTEXITCODE
if ($ExitCode -ne 0) { throw "NER smoke training failed with exit code $ExitCode; see $LogPath" }

Write-Host "Smoke training passed: two CUDA optimization steps completed." -ForegroundColor Green
Write-Host "Log: $LogPath"
