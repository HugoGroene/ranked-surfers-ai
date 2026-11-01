param(
    [double]$TrainMinutes = 30,
    [int]$Frames = 4,
    [double]$SpanMs = 500,
    [double]$CaptureHz = 60,
    [double]$DecisionHz = 30,
    [ValidateSet('native_bitmap','bitmap_100','gpu_100')][string]$CaptureBackend = 'gpu_100',
    [string]$Resume = '',
    [switch]$Execute,
    [switch]$SmokeOnly
)
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location -LiteralPath $repo
$py=Join-Path $HOME 'Documents\ranked-surfers-ai\.venv\Scripts\python.exe'
$arguments=@('-B','-s','-u',(Join-Path $PSScriptRoot 'road_batch.py'),
    '--train-minutes',"$TrainMinutes",'--frames',"$Frames",'--span-ms',"$SpanMs",
    '--capture-hz',"$CaptureHz",'--decision-hz',"$DecisionHz",'--capture-backend',$CaptureBackend)
if ($Resume) { $arguments+=@('--resume',$Resume) }
if ($SmokeOnly) { $arguments+='--smoke-only' }
if ($Execute) {
    try { $null=Invoke-WebRequest 'http://127.0.0.1:8000/' -UseBasicParsing -TimeoutSec 5 }
    catch { throw 'Start the local game server first: & $py -B -s server.py' }
    $arguments+='--execute'
}
$ErrorActionPreference='Continue'
& $py @arguments
$code=$LASTEXITCODE
if ($code -ne 0) { throw 'Road experiment stopped or contains failed stages. Preserve printed output folders.' }
