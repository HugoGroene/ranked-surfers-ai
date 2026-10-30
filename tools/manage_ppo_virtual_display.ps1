# Only the identified Root\MttVDD device is eligible. Never touches NVIDIA/AMD.
[CmdletBinding(SupportsShouldProcess=$true)]
param(
    [ValidateSet('Status','Enable','Disable','SetCount')][string]$Action = 'Status',
    [ValidateRange(1,2)][int]$Count = 1
)
$ErrorActionPreference = 'Stop'
$devices = @(Get-PnpDevice -Class Display | Where-Object { $_.FriendlyName -eq 'Virtual Display Driver' })
if ($devices.Count -ne 1) { throw 'Expected exactly one installed Virtual Display Driver device.' }
$device = $devices[0]
$ids = @((Get-PnpDeviceProperty -InstanceId $device.InstanceId -KeyName 'DEVPKEY_Device_HardwareIds').Data)
if (-not ($ids | Where-Object { $_ -ieq 'Root\MttVDD' })) {
    throw 'Hardware identity is not Root\MttVDD. No device changes allowed.'
}
$settings = 'C:\VirtualDisplayDriver\vdd_settings.xml'
$text = [IO.File]::ReadAllText($settings)
$xml = [xml]$text
$oldCount = [int]$xml.SelectSingleNode('/vdd_settings/monitors/count').InnerText
if ($Action -eq 'Status') {
    [pscustomobject]@{Device=$device.InstanceId; Status=$device.Status; MonitorCount=$oldCount} | ConvertTo-Json
    return
}
$principal = [Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'This action needs an Administrator PowerShell. Status is read-only.'
}
if (-not $PSCmdlet.ShouldProcess($device.InstanceId, $Action)) { return }
$logName = 'vdd_' + $Action + '_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '_' + [guid]::NewGuid().ToString('N').Substring(0,6) + '.log'
$logPath = Join-Path (Join-Path (Split-Path -Parent $PSScriptRoot) 'reports') $logName
Start-Transcript -LiteralPath $logPath -NoClobber | Out-Null
try { switch ($Action) {
    'Enable' { Enable-PnpDevice -InstanceId $device.InstanceId -Confirm:$false }
    'Disable' { Disable-PnpDevice -InstanceId $device.InstanceId -Confirm:$false }
    'SetCount' {
        if ($oldCount -eq $Count) { Write-Output 'Monitor count is already correct.'; return }
        $matches = [regex]::Matches($text,'<count>\s*\d+\s*</count>')
        if ($matches.Count -ne 1) { throw 'Ambiguous settings file; no changes allowed.' }
        $backup = $settings + '.pre_rs_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.bak'
        Copy-Item -LiteralPath $settings -Destination $backup -ErrorAction Stop
        $updated = $text.Replace($matches[0].Value, "<count>$Count</count>")
        [IO.File]::WriteAllText($settings, $updated, (New-Object Text.UTF8Encoding($false)))
        Write-Output "Backup: $backup"
        & "$env:WINDIR\System32\pnputil.exe" /restart-device $device.InstanceId
        if ($LASTEXITCODE -ne 0) { throw "VDD reload failed ($LASTEXITCODE). Backup is preserved; inspect before retrying." }
    }
} } finally { Stop-Transcript | Out-Null }
Write-Output "VDD action complete: $Action"
