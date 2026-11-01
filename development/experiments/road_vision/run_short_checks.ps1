param([string]$Resume = '', [switch]$PlanOnly)
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location -LiteralPath $repo
$py=Join-Path $HOME 'Documents\ranked-surfers-ai\.venv\Scripts\python.exe'
$arguments=@('-B','-s','-u',(Join-Path $PSScriptRoot 'short_checks.py'))
if ($Resume) { $arguments+=@('--resume',$Resume) }
if (-not $PlanOnly) { $arguments+='--execute' }
$ErrorActionPreference='Continue'
& $py @arguments
if ($LASTEXITCODE -ne 0) { throw 'Short checks stopped or contain failures. Keep the printed report; completed stages are retained.' }
