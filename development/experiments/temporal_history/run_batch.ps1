# Run AFTER other training stops. Sequential smoke gate, then the scratch batch.
param(
    [double]$TrainMinutes = 30,
    [int[]]$Seeds = @(123),
    [string]$Checkpoint = 'reports\v612_3workers_menu_recovery_24h_20261101_111341\latest.zip',
    [switch]$CaptureOnly,
    [switch]$SkipCapture,
    [switch]$CaptureFollowup,
    [string]$ContinueSmoke = '',
    [string]$ContinueBatch = ''
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location -LiteralPath $repo
$py = Join-Path $HOME 'Documents\ranked-surfers-ai\.venv\Scripts\python.exe'
$ErrorActionPreference = 'Continue'
& $py -B -s -m unittest discover -s $PSScriptRoot -p 'test_*.py' -q
$offlineExit = $LASTEXITCODE
$ErrorActionPreference = 'Stop'
if ($offlineExit -ne 0) { throw 'Offline tests failed; no live trial was started.' }
$captureRoot = Join-Path (Split-Path -Parent $PSScriptRoot) 'capture_pipeline'
$ErrorActionPreference = 'Continue'
& $py -B -s -m unittest discover -s $captureRoot -p 'test_*.py' -q
$captureOfflineExit = $LASTEXITCODE
$ErrorActionPreference = 'Stop'
if ($captureOfflineExit -ne 0) { throw 'Capture offline tests failed; no live trial was started.' }
try { $null = Invoke-WebRequest 'http://127.0.0.1:8000/' -UseBasicParsing -TimeoutSec 5 }
catch { throw 'Start the local server in another terminal: & $py -B -s server.py' }
if ($CaptureOnly -and $SkipCapture) { throw 'CaptureOnly and SkipCapture cannot be combined.' }
if ($CaptureFollowup -and $SkipCapture) { throw 'Use CaptureFollowup instead of SkipCapture for the targeted repeat.' }
if (-not $SkipCapture) {
    $ErrorActionPreference = 'Continue'
    if ($CaptureFollowup) {
        & $py -B -s -u (Join-Path $captureRoot 'batch.py') --checkpoint $Checkpoint `
            --arms bitmap_100 gpu_100 --workers 3 4 --repeats 2 --minutes 1 --execute
    } else {
        & $py -B -s -u (Join-Path $captureRoot 'batch.py') --checkpoint $Checkpoint --execute
    }
    $captureExit = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    if ($captureExit -ne 0) { throw 'Capture matrix failed; no temporal training started. Preserve the printed report.' }
}
if ($CaptureOnly) { exit 0 }
$entry = Join-Path $PSScriptRoot 'batch.py'
$ErrorActionPreference = 'Continue'
if ($ContinueSmoke) {
    & $py -B -s -u $entry --continue-smoke (Join-Path $ContinueSmoke 'plan.json') --execute
} else {
    & $py -B -s -u $entry `
        --train-minutes 1 --eval-episodes 3 --eval-max-minutes 3 --execute
}
$smokeExit = $LASTEXITCODE
$ErrorActionPreference = 'Stop'
if ($smokeExit -ne 0) { throw 'Smoke batch failed; the 30-minute trials were not started. Preserve the printed reports.' }
$ErrorActionPreference = 'Continue'
if ($ContinueBatch) {
    & $py -B -s -u $entry --continue-batch (Join-Path $ContinueBatch 'plan.json') --keep-going --execute
} else {
    & $py -B -s -u $entry --train-minutes $TrainMinutes --seeds @Seeds --keep-going --execute
}
$batchExit = $LASTEXITCODE
$ErrorActionPreference = 'Stop'
if ($batchExit -ne 0) { throw 'Experiment batch stopped. Preserve the printed reports; existing checkpoints remain available.' }
