param(
    [string]$Python = "python",
    [int]$RawSamples = 4000,
    [int]$NerEpochs = 1,
    [int]$FocusStructureEpochs = 2,
    [int]$CalibrationEpochs = 1,
    [int]$RareFieldTarget = 200,
    [int]$FocusMaxRepeats = 2
)

# Windows PowerShell 5 wraps native stderr as NativeCommandError. Keep native
# process failures non-terminating and handle them via $LASTEXITCODE below.
$ErrorActionPreference = "Continue"
$TestDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PackageDir = Split-Path -Parent $TestDir
$Variant = "full_4060_6h"
$DataDir = Join-Path $TestDir "data\$Variant"
$NerOut = Join-Path $TestDir "outputs\ner\$Variant"
$FocusOut = Join-Path $TestDir "outputs\structure\${Variant}_focus"
$StructureOut = Join-Path $TestDir "outputs\structure\$Variant"
$LogDir = Join-Path $TestDir "logs"
$env:PYTHONPATH = "$PackageDir;$env:PYTHONPATH"
$env:TOKENIZERS_PARALLELISM = "false"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Invoke-Step {
    param(
        [string]$Name,
        [string[]]$Arguments,
        [switch]$InteractiveProgress
    )
    Write-Host "`n==> $Name" -ForegroundColor Cyan
    $LogName = ($Name -replace "[^a-zA-Z0-9_-]", "_").Trim("_") + ".log"
    $LogPath = Join-Path $LogDir $LogName
    if ($InteractiveProgress) {
        # Keep tqdm connected to the terminal. Tee-Object converts each
        # carriage-return refresh into a separate displayed/logged line.
        Write-Host "Progress is shown as one updating line; results are saved under outputs/." -ForegroundColor DarkGray
        & $Python @Arguments
    }
    else {
        & $Python @Arguments 2>&1 | Tee-Object -FilePath $LogPath
    }
    $ExitCode = $LASTEXITCODE
    if ($ExitCode -ne 0) {
        if ($InteractiveProgress) {
            throw "$Name failed with exit code $ExitCode. See the terminal output above."
        }
        throw "$Name failed with exit code $ExitCode. Full Python error: $LogPath"
    }
}

Invoke-Step -Name "check dependencies and CUDA" -Arguments @("-c", "import sys, torch, transformers, peft; print('Python:', sys.executable); print('torch:', torch.__version__); print('transformers:', transformers.__version__); print('peft:', peft.__version__); assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0)); print('VRAM GiB:', round(torch.cuda.get_device_properties(0).total_memory/2**30, 1))")

# Coverage-balanced sampling reads the complete source contract first, reserves
# rare-field rows, then fills the smaller budget. No business field is removed.
Invoke-Step -Name "prepare 6-hour full-field focus data" -Arguments @(
    (Join-Path $TestDir "prepare_data.py"), "--schema-mode", "full",
    "--split-multi", "--variant", $Variant, "--max-samples", "$RawSamples",
    "--focus-training", "--rare-field-target", "$RareFieldTarget",
    "--focus-max-repeats", "$FocusMaxRepeats"
)
Invoke-Step -Name "validate prepared data" -Arguments @(
    (Join-Path $TestDir "validate_data.py"), "--data-dirs", $DataDir,
    "--out", (Join-Path $TestDir "logs\validate_full_4060_6h.json")
)

$Common = @(
    "--schema-mode", "full", "--data-variant", $Variant,
    "--batch-size", "2", "--eval-batch-size", "2", "--grad-accum", "8",
    "--max-len", "256", "--eval-steps", "300", "--num-workers", "2",
    "--fp16", "--gradient-checkpointing", "--early-stopping-patience", "3"
)

Invoke-Step -Name "stage 1 - full-field NER focus warm-up" -Arguments (@(
    (Join-Path $TestDir "train_ner.py"), "--epochs", "$NerEpochs",
    "--output-dir", $NerOut,
    "--train-file", (Join-Path $DataDir "ner_train_balanced_clean.jsonl")
) + $Common) -InteractiveProgress

Invoke-Step -Name "stage 2 - difficult-field structure focus" -Arguments (@(
    (Join-Path $TestDir "train_structure.py"), "--epochs", "$FocusStructureEpochs",
    "--init-from", $NerOut, "--output-dir", $FocusOut,
    "--train-file", (Join-Path $DataDir "structure_train_balanced_clean.jsonl")
) + $Common) -InteractiveProgress

Invoke-Step -Name "stage 3 - full-schema calibration" -Arguments (@(
    (Join-Path $TestDir "train_structure.py"), "--epochs", "$CalibrationEpochs",
    "--init-from", $FocusOut, "--output-dir", $StructureOut,
    "--train-file", (Join-Path $DataDir "structure_train_clean.jsonl"),
    "--encoder-lr", "5e-6", "--task-lr", "1e-4"
) + $Common) -InteractiveProgress

Invoke-Step -Name "test sentence exact match" -Arguments @(
    (Join-Path $TestDir "evaluate_sentence_acc.py"), "--schema-mode", "full",
    "--data-variant", $Variant, "--model-dir", $StructureOut,
    "--max-len", "256", "--batch-size", "2",
    "--tune-thresholds", "0.35,0.45,0.55,0.65", "--tune-limit", "200",
    "--out", (Join-Path $StructureOut "eval_sentence_acc.json")
)

Write-Host "`nFinished. Result: $StructureOut\eval_sentence_acc.json" -ForegroundColor Green
