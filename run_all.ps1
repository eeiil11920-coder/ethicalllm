# =============================================================================
# run_all.ps1 — Orchestration of the reproducibility pipeline
# BUSI-D-26-01884 (Journal of Business Ethics, major revision)
# =============================================================================
# Usage:
#   powershell -ExecutionPolicy Bypass -File run_all.ps1            # dry-run mode
#   powershell -ExecutionPolicy Bypass -File run_all.ps1 -Real      # real wave-2 run
#   powershell -ExecutionPolicy Bypass -File run_all.ps1 -Real -Providers anthropic,deepseek
param(
    [switch]$Real,
    [string]$Providers = "",
    [string]$Python = "C:\Python313\python.exe"
)
$ErrorActionPreference = "Stop"
$env:USE_TF = "0"            # local TensorFlow install is broken; not needed
$repo = Split-Path -Parent $MyInvocation.MyCommand.Path
$analysis = (Resolve-Path (Join-Path $repo "..\..\..\analysis")).Path
$wave1xlsx = Join-Path $analysis "Cases_LLM_Ethics_2.xlsx"

function Step($n, $msg) { Write-Host "`n=== [$n] $msg ===" -ForegroundColor Cyan }

Step 1 "Build prompts/cases.json from 'Casos español.docx'"
& $Python -X utf8 (Join-Path $repo "prompts\build_cases.py")
if ($LASTEXITCODE -ne 0) { throw "build_cases failed" }

if ($Real) {
    Step 2 "Pre-check provider keys/credit (1-token requests)"
    $args = @((Join-Path $repo "collect.py"), "--check")
    if ($Providers) { $args += @("--providers", $Providers) }
    & $Python -X utf8 @args   # non-zero exit = some providers not ready (ok)

    Step 3 "Collect wave-2 dialogues (resumes automatically)"
    $args = @((Join-Path $repo "collect.py"), "--yes")
    if ($Providers) { $args += @("--providers", $Providers) }
    & $Python -X utf8 @args
    $collected = Join-Path $repo "data\wave2"
    $scoresOut = Join-Path $repo "results\wave2\auto_scores_wave2.csv"
} else {
    Step 2 "DRY RUN: 100 simulated dialogues (no API keys needed)"
    & $Python -X utf8 (Join-Path $repo "collect.py") --dry-run --overwrite
    if ($LASTEXITCODE -ne 0) { throw "collect dry-run failed" }
    $collected = Join-Path $repo "data\wave2_dryrun"
    $scoresOut = Join-Path $repo "results\dryrun\auto_scores_dryrun.csv"
}

Step 4 "Automated scoring of the collected dialogues"
if ($Real) {
    & $Python -X utf8 (Join-Path $repo "score.py") --wave2-dir $collected --out $scoresOut
} else {
    # heuristic backend keeps the smoke test fast; use NLI for real data
    & $Python -X utf8 (Join-Path $repo "score.py") --wave2-dir $collected --out $scoresOut --no-nli
}
if ($LASTEXITCODE -ne 0) { throw "score (wave2) failed" }

Step 5 "Automated scoring of wave 1 (NLI, GPU if available)"
& $Python -X utf8 (Join-Path $repo "score.py") --wave1 $wave1xlsx --out (Join-Path $repo "results\wave1\auto_scores_wave1.csv")
if ($LASTEXITCODE -ne 0) { throw "score (wave1) failed" }

Step 6 "Validate the NLI classifier against wave-1 human ratings"
& $Python -X utf8 (Join-Path $repo "validate_nli.py")
if ($LASTEXITCODE -ne 0) { throw "validate_nli failed" }

Step 7 "Statistical re-analysis (mixed-effects + sensitivity)"
& $Python -X utf8 (Join-Path $repo "stats.py")
if ($LASTEXITCODE -ne 0) { throw "stats failed" }

if ($Real) {
    Step 8 "Blinded expert sheets (only meaningful with real wave-2 data)"
    & $Python -X utf8 (Join-Path $repo "expert_sheets\make_sheets.py") --wave2-dir $collected
}

Write-Host "`nPipeline finished. See results\ for outputs." -ForegroundColor Green
