param([string]$Python = "python")

# GPU-only passability smoke for the complete 76-field three-stage route.
# This script is intentionally not executed on macOS.
$ErrorActionPreference = "Continue"
$TestDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PackageDir = Split-Path -Parent $TestDir
$Variant = "smoke_4060_full"
$DataDir = Join-Path $TestDir "data\$Variant"
$NerOut = Join-Path $TestDir "outputs\ner\$Variant"
$FocusOut = Join-Path $TestDir "outputs\structure\${Variant}_focus"
$FinalOut = Join-Path $TestDir "outputs\structure\$Variant"
$LogDir = Join-Path $TestDir "logs"
$LogPath = Join-Path $LogDir "smoke_4060_full.log"
$env:PYTHONPATH = "$PackageDir;$env:PYTHONPATH"
$env:TOKENIZERS_PARALLELISM = "false"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Invoke-SmokeStep {
    param(
        [string]$Name,
        [string[]]$Arguments,
        [switch]$InteractiveProgress
    )
    Write-Host "`n==> $Name" -ForegroundColor Cyan
    if ($InteractiveProgress) {
        & $Python @Arguments
    }
    else {
        & $Python @Arguments 2>&1 | Tee-Object -FilePath $LogPath -Append
    }
    $ExitCode = $LASTEXITCODE
    if ($ExitCode -ne 0) {
        throw "$Name failed with exit code $ExitCode"
    }
}

$DependencyProbe = @"
import sys, torch, transformers, peft
print('python:', sys.executable)
print('torch:', torch.__version__)
print('transformers:', transformers.__version__)
print('peft:', peft.__version__)
assert torch.cuda.is_available(), 'CUDA unavailable'
name = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_memory / 2**30
print('gpu:', name)
print('vram_gib:', round(vram, 1))
assert '4060' in name.upper(), f'Expected RTX 4060 smoke host, got {name}'
assert vram >= 7.0, f'Expected roughly 8 GiB VRAM, got {vram:.1f}'
"@
Invoke-SmokeStep -Name "dependency and RTX 4060 probe" -Arguments @("-c", $DependencyProbe)

Invoke-SmokeStep -Name "prepare coverage-balanced 76-field smoke data" -Arguments @(
    (Join-Path $TestDir "prepare_data.py"), "--schema-mode", "full",
    "--split-multi", "--variant", $Variant, "--max-samples", "500",
    "--retain-train-multi-max-deals", "5",
    "--focus-training", "--rare-field-target", "32", "--focus-max-repeats", "1"
)
Invoke-SmokeStep -Name "validate smoke data spans" -Arguments @(
    (Join-Path $TestDir "validate_data.py"), "--data-dirs", $DataDir,
    "--out", (Join-Path $LogDir "validate_smoke_4060_full.json")
)

$ContractProbe = @"
import json
from pathlib import Path
p = Path(r'$DataDir') / 'split_stats.json'
s = json.loads(p.read_text(encoding='utf-8'))
assert s['schema_mode'] == 'full'
assert s['field_contract']['schema_field_count'] == 76
assert s['field_contract']['all_fields_have_non_null_labels']
assert s['splits']['train']['fields_with_non_null_labels'] == 76
assert not s['splits']['train']['fields_without_non_null_labels']
assert s['focus_training']['n_balanced_structure_rows'] > 0
print('full-field contract passed: 76/76 fields, focus rows present')
"@
Invoke-SmokeStep -Name "assert complete field contract" -Arguments @("-c", $ContractProbe)

$Common = @(
    "--schema-mode", "full", "--data-variant", $Variant,
    "--epochs", "1", "--max-steps", "2",
    "--max-train-samples", "16", "--max-eval-samples", "8",
    "--batch-size", "1", "--eval-batch-size", "1", "--grad-accum", "1",
    "--max-len", "192", "--eval-steps", "1", "--num-workers", "0",
    "--fp16", "--gradient-checkpointing"
)

Invoke-SmokeStep -Name "NER two-step CUDA smoke" -Arguments (@(
    (Join-Path $TestDir "train_ner.py"), "--output-dir", $NerOut,
    "--train-file", (Join-Path $DataDir "ner_train_balanced_clean.jsonl")
) + $Common) -InteractiveProgress

Invoke-SmokeStep -Name "focused structure two-step CUDA smoke" -Arguments (@(
    (Join-Path $TestDir "train_structure.py"), "--init-from", $NerOut,
    "--output-dir", $FocusOut,
    "--train-file", (Join-Path $DataDir "structure_train_balanced_clean.jsonl")
) + $Common) -InteractiveProgress

Invoke-SmokeStep -Name "full-schema calibration two-step CUDA smoke" -Arguments (@(
    (Join-Path $TestDir "train_structure.py"), "--init-from", $FocusOut,
    "--output-dir", $FinalOut,
    "--train-file", (Join-Path $DataDir "structure_train_clean.jsonl"),
    "--encoder-lr", "5e-6", "--task-lr", "1e-4"
) + $Common) -InteractiveProgress

Invoke-SmokeStep -Name "full-schema inference smoke" -Arguments @(
    (Join-Path $TestDir "evaluate_sentence_acc.py"), "--schema-mode", "full",
    "--data-variant", $Variant, "--model-dir", $FinalOut,
    "--max-len", "192", "--batch-size", "1", "--limit", "8",
    "--threshold", "0.55", "--tune-field-thresholds", "0.35,0.55,0.75",
    "--tune-limit", "8",
    "--out", (Join-Path $FinalOut "eval_sentence_acc.json")
)

$ResultProbe = @"
import json
from pathlib import Path
roots = [Path(r'$NerOut'), Path(r'$FocusOut'), Path(r'$FinalOut')]
for root in roots:
    result = json.loads((root / 'training_result.json').read_text(encoding='utf-8'))
    assert result['total_steps'] == 2, (root, result.get('total_steps'))
report = json.loads((Path(r'$FinalOut') / 'eval_sentence_acc.json').read_text(encoding='utf-8'))
summary = report['summary']
assert summary['schema_mode'] == 'full'
assert summary['business_full_field_evaluation'] is True
assert summary['field_contract']['schema_field_count'] == 76
assert summary['n_samples'] == 8
print('PASS: data -> NER -> focused structure -> calibration -> 76-field inference')
"@
Invoke-SmokeStep -Name "assert smoke artifacts" -Arguments @("-c", $ResultProbe)

Write-Host "`nRTX 4060 full-field smoke passed." -ForegroundColor Green
Write-Host "Log: $LogPath"
