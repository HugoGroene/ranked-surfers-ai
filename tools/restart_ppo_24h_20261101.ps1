# Run from a separate terminal. Never overwrites an existing run directory.
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$py = Join-Path $HOME 'Documents\ranked-surfers-ai\.venv\Scripts\python.exe'
$checkpoint = 'reports\v612_detectoroff_3workers_24h_20261101_095630\latest.zip'
if (-not (Test-Path -LiteralPath $checkpoint)) { throw "Missing checkpoint: $checkpoint" }
try { $null = Invoke-WebRequest 'http://127.0.0.1:8000/' -UseBasicParsing -TimeoutSec 5 }
catch { throw 'Start the local Practice server first: & $py -B -s server.py' }

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
foreach ($stage in @(@{ Name='menu_recovery_smoke'; Minutes=3; Plot='off' },
                      @{ Name='menu_recovery_24h'; Minutes=1440; Plot='show' })) {
    $run = "v612_3workers_$($stage.Name)_$stamp"
    $output = "reports\$run"
    $log = "ai_data\ppo_$run.log"
    if ((Test-Path -LiteralPath $output) -or (Test-Path -LiteralPath $log)) {
        throw "Output already exists: $output"
    }
    # Windows PowerShell wraps native stderr as ErrorRecord; preserve exit status.
    $ErrorActionPreference = 'Continue'
    & $py -B -s -u -m tools.train_ppo_parallel `
        --headless --headless-capture electron `
        --displays 1 2 3 --electron-transport shared_memory `
        --rollout-transport shared_memory --reuse-frame-state `
        --opencv-threads 1 --profile-reaction `
        --framing-profile canvas_v1 --canvas-policy-input --detector-mode off `
        --checkpoint $checkpoint --steps 12000000 `
        --max-minutes $stage.Minutes --save-every-minutes 10 --worker-timeout 160 `
        --telemetry --end-plot $stage.Plot --report $output `
        2>&1 | Tee-Object -FilePath $log
    $trainExit = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    Write-Host "CHECKPOINT: $output\latest.zip"
    Write-Host "REPORT: $output\parallel_training.json"
    Write-Host "LOG: $log"
    if ($trainExit -ne 0) { throw 'Training stopped. Preserve outputs; no next stage will start.' }
    $report = Get-Content -LiteralPath "$output\parallel_training.json" -Raw | ConvertFrom-Json
    if ($report.status -ne 'COMPLETE' -or @($report.audits).Count -ne 3 -or
        @($report.audits | Where-Object { $_.status -ne 'PASS_SETTLED_LEARNING' }).Count -ne 0) {
        throw 'Completion or learning audit failed; no next stage will start.'
    }
    $checkpoint = "$output\latest.zip"
    if (-not (Test-Path -LiteralPath $checkpoint)) { throw "Missing new checkpoint: $checkpoint" }
}
