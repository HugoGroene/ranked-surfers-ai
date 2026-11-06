param([int]$Seconds=120)
$ErrorActionPreference='Stop'
$py = "$HOME\Documents\ranked-surfers-ai\.venv\Scripts\python.exe"
foreach ($split in @('train','validation')) {
    foreach ($map in @('halloween','tokyo')) {
        Read-Host "Next: $map / $split. Press Enter when ready to play manually"
        & $py -B -s -u "$PSScriptRoot\record.py" --map $map --split $split --seconds $Seconds
        if ($LASTEXITCODE -ne 0) { throw 'Recording incomplete. Saved frames are retained; resolve before continuing.' }
    }
}
Write-Host 'Recordings saved. Use each printed LABEL COMMAND to review; no classifier was replaced.'
