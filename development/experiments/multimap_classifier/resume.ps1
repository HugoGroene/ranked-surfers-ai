param(
    [ValidateRange(0.1,48)][double]$Hours = 12,
    [ValidateRange(1,2880)][double]$Minutes = 720,
    [string]$Checkpoint = '',
    [ValidateRange(0,20)][int]$MaxRestarts = 3
)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path
$py = "$HOME\Documents\ranked-surfers-ai\.venv\Scripts\python.exe"
Set-Location -LiteralPath $repo
if (-not $Checkpoint) {
    $runs = Get-ChildItem -LiteralPath (Join-Path $repo 'reports') -Directory -Filter 'multimap_stability_*' |
        Sort-Object Name -Descending
    foreach ($run in $runs) {
        $planFile = Join-Path $run.FullName 'plan.json'
        if (Test-Path -LiteralPath $planFile) {
            $plan = Get-Content -LiteralPath $planFile -Raw | ConvertFrom-Json
            if ($plan.diagnostic_only -or $null -ne $plan.visible_worker) { continue }
        }
        $report = Join-Path $run.FullName 'stability/train/parallel_training.json'
        $candidate = Join-Path $run.FullName 'stability/train/latest.zip'
        $recoveryPath = Join-Path $run.FullName 'recovery.json'
        if (Test-Path -LiteralPath $recoveryPath) {
            $recovery = Get-Content -LiteralPath $recoveryPath -Raw | ConvertFrom-Json
            if ($recovery.status -ne 'COMPLETE') { continue }
            $candidate = $recovery.latest_checkpoint
            $report = Join-Path (Split-Path -Parent $candidate) 'parallel_training.json'
        }
        if ((Test-Path -LiteralPath $report) -and (Test-Path -LiteralPath $candidate)) {
            $result = Get-Content -LiteralPath $report -Raw | ConvertFrom-Json
            if ($result.status -eq 'COMPLETE' -and -not $result.error -and
                @($result.finalization_errors).Count -eq 0 -and
                $result.final_checkpoint_transitions -eq $result.optimized_transitions) {
                $Checkpoint = $candidate
                break
            }
        }
    }
    if (-not $Checkpoint) { throw 'No completed multimap training checkpoint found. Supply -Checkpoint explicitly.' }
}
$Checkpoint = (Resolve-Path -LiteralPath $Checkpoint).Path
# Refuse to compete for the training servers/dashboard. Do not stop other processes.
foreach ($port in @(8000,8010,8001)) {
    if (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) {
        throw "Port $port is in use. Wait for the current run/plot to close, or stop its server before continuing."
    }
}
if ($PSBoundParameters.ContainsKey('Hours') -and $PSBoundParameters.ContainsKey('Minutes')) {
    throw 'Use either -Minutes or -Hours, not both.'
}
if ($PSBoundParameters.ContainsKey('Hours')) { $Minutes = $Hours * 60 }
$duration = $Minutes.ToString([Globalization.CultureInfo]::InvariantCulture)
Write-Host "CONTINUING FROM: $Checkpoint"
Write-Host "TRAINING: $Minutes minutes, 4 workers, Halloween + Tokyo, stateful guard on, dashboard off; local servers start automatically."
& $py -B -s -u "$PSScriptRoot\stability.py" --workers 4 --minutes $duration --checkpoint $Checkpoint --stateful-guard --max-restarts $MaxRestarts
if ($LASTEXITCODE -ne 0) { throw 'Training stopped. Preserve the printed report; inspect the error before restarting.' }
