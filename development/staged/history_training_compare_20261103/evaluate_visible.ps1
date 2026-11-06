param(
    [int]$Episodes = 30,
    [string]$Source = 'reports\history_training_compare_20261102_234842_138252'
)
$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..\..\..')
$py = "$HOME\Documents\ranked-surfers-ai\.venv\Scripts\python.exe"
$runner = Join-Path $PSScriptRoot 'evaluate_model.py'
$batch = 'reports\history_visible_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff')
New-Item -ItemType Directory -Path $batch | Out-Null
$cases = @(
    @{ Layout = 'fixed4'; Folder = 'fixed4_train_w4' },
    @{ Layout = 'fixed5'; Folder = 'fixed5_train_w3' }
)
# Check both exact input contracts before opening either game.
foreach ($case in $cases) {
    & $py -B -s -u $runner --layout $case.Layout `
        --checkpoint "$Source\$($case.Folder)\latest.zip" `
        --output "$batch\$($case.Layout)" --episodes $Episodes `
        --visible-browser --check-only
    if ($LASTEXITCODE -ne 0) { throw 'Checkpoint preflight failed; no game started.' }
}
foreach ($case in $cases) {
    $layout = $case.Layout
    Write-Host "LIVE BROWSER: $layout - $Episodes episodes. Ctrl+C stops."
    # Windows PowerShell treats native stderr as ErrorRecords. Preserve the
    # full traceback instead of terminating its pipe at the first stderr line.
    $ErrorActionPreference = 'Continue'
    try {
        & $py -B -s -u $runner --layout $layout `
            --checkpoint "$Source\$($case.Folder)\latest.zip" `
            --output "$batch\$layout" --episodes $Episodes --minutes 60 `
            --visible-browser 2>&1 | Tee-Object -FilePath "$batch\$layout.log"
        $evalExit = $LASTEXITCODE
    } finally { $ErrorActionPreference = 'Stop' }
    if ($evalExit -ne 0) { throw "Evaluation incomplete (exit $evalExit). Preserve $batch." }
}
Write-Host "BOTH LIVE EVALUATIONS COMPLETE: $batch"
