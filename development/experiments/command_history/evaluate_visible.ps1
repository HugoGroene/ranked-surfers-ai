param(
    [ValidateRange(1, 10000)][int]$Episodes = 10,
    [string]$Checkpoint = 'reports\command_history_resume_20261104_093033\train\latest.zip',
    [switch]$CompareCapture
)
$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..\..\..')
$py = "$HOME\Documents\ranked-surfers-ai\.venv\Scripts\python.exe"
$runner = Join-Path $PSScriptRoot 'evaluate.py'
$batch = 'reports\visible_smoothness_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff')
$backends = @('desktop')
if ($CompareCapture) { $backends += 'page' }
$oldBackend = $env:RS_VISIBLE_CAPTURE
try {
    foreach ($backend in $backends) {
        $env:RS_VISIBLE_CAPTURE = $backend
        $output = "$batch\$backend"
        & $py -B -s -u $runner --layout fixed4 --checkpoint $Checkpoint `
            --output $output --episodes $Episodes --minutes 180 --visible-browser --check-only
        if ($LASTEXITCODE -ne 0) { throw 'Checkpoint preflight failed; no game started.' }
        Write-Host "VISIBLE EVAL: $backend; $Episodes episodes. Keep the game foreground and unobscured."
        $ErrorActionPreference = 'Continue'
        try {
            & $py -B -s -u $runner --layout fixed4 --checkpoint $Checkpoint `
                --output $output --episodes $Episodes --minutes 180 --visible-browser
            $evalExit = $LASTEXITCODE
        } finally { $ErrorActionPreference = 'Stop' }
        Write-Host "REPORT: $output\v61_eval_report.json"
        $renderReport = Join-Path $output 'electron\visible_render_timing.json'
        if (Test-Path -LiteralPath $renderReport) {
            Write-Host "RENDER TIMING: $renderReport"
            Get-Content -LiteralPath $renderReport
        }
        if ($evalExit -ne 0) { throw "Evaluation incomplete; preserve $output." }
    }
} finally {
    if ($null -eq $oldBackend) { Remove-Item Env:RS_VISIBLE_CAPTURE -ErrorAction SilentlyContinue }
    else { $env:RS_VISIBLE_CAPTURE = $oldBackend }
}
Write-Host "VISIBLE CHECKS COMPLETE: $batch"
