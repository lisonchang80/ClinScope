# Shared Cloudflare tunnel helper used by start_public.bat / restart_tunnel.bat.
#
# The tunnel is SHARED (gis / scopelab / harnesshub / tinghaochang.com, and clinscope
# when its server runs). It is normally started at logon by the Startup-folder script
#   %APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\cloudflared_tunnel.vbs
# ClinScope itself has had no logon autostart since 2026-08-07.
#
# Safety rules (review 2026-11-07 CS-3):
#  - Never "taskkill /IM cloudflared.exe": other cloudflared instances (e.g. a separate
#    media tunnel with its own --config) must survive. Only the process whose command
#    line contains THIS tunnel's UUID is touched.
#  - The UUID is read at run time from the "tunnel:" field of
#    %USERPROFILE%\.cloudflared\config.yml and is never stored in this (public) repo.
#    If it cannot be read, nothing is stopped or started.
#  - (Re)start hidden + detached (via the Startup VBS when present), so closing a
#    console window can never take the shared tunnel down.
#
# Keep this file PURE ASCII: Windows PowerShell 5.1 reads BOM-less .ps1 files as ANSI.
#
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File tunnel_ctl.ps1 <check|status|ensure|restart>
#   check   : exit 0 if the tunnel UUID can be read, else 2 (with an explanation)
#   status  : exit 0 if the tunnel process is running, 1 if not, 2 if UUID unreadable
#   ensure  : start the tunnel only if it is not already running
#   restart : stop only this tunnel's cloudflared, then start it again
param([ValidateSet('check', 'status', 'ensure', 'restart')][string]$Action = 'status')

$ErrorActionPreference = 'Stop'
$cfg = Join-Path $env:USERPROFILE '.cloudflared\config.yml'
$vbs = Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs\Startup\cloudflared_tunnel.vbs'
$uuidRe = '[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}'

function Get-TunnelId {
    if (-not (Test-Path -LiteralPath $cfg)) { return $null }
    foreach ($line in Get-Content -LiteralPath $cfg) {
        if ($line -match ('^\s*tunnel\s*:\s*["'']?(' + $uuidRe + ')["'']?\s*(#.*)?$')) { return $Matches[1] }
    }
    return $null
}

function Get-TunnelProcs([string]$id) {
    $needle = $id.ToLowerInvariant()
    @(Get-CimInstance Win32_Process -Filter "Name='cloudflared.exe'" |
        Where-Object { $_.CommandLine -and $_.CommandLine.ToLowerInvariant().Contains($needle) })
}

function Start-Tunnel([string]$id) {
    if (Test-Path -LiteralPath $vbs) {
        Write-Host "[tunnel] starting via Startup\cloudflared_tunnel.vbs (hidden, detached) ..."
        Start-Process -FilePath 'wscript.exe' -ArgumentList ('"' + $vbs + '"')
        $wait = 30
    } else {
        $exe = $null
        $cmd = Get-Command cloudflared.exe -ErrorAction SilentlyContinue
        if ($cmd) { $exe = $cmd.Source }
        if (-not $exe) {
            $guess = Join-Path ${env:ProgramFiles(x86)} 'cloudflared\cloudflared.exe'
            if (Test-Path -LiteralPath $guess) { $exe = $guess }
        }
        if (-not $exe) {
            Write-Host "[tunnel] Startup\cloudflared_tunnel.vbs is missing and cloudflared.exe was not found."
            Write-Host "         Nothing was started."
            exit 3
        }
        Write-Host "[tunnel] Startup VBS missing -> starting cloudflared directly (hidden, detached) ..."
        Start-Process -FilePath $exe -ArgumentList @('tunnel', 'run', $id) -WindowStyle Hidden
        $wait = 15
    }
    for ($i = 0; $i -lt $wait; $i++) {
        Start-Sleep -Seconds 1
        $p = Get-TunnelProcs $id
        if ($p.Count) { Write-Host ("[tunnel] running (PID " + (($p | ForEach-Object { $_.ProcessId }) -join ', ') + ")"); return }
    }
    Write-Host "[tunnel] WARNING: no cloudflared for this tunnel appeared within $wait s. Check it by hand."
    exit 4
}

$id = Get-TunnelId
if (-not $id) {
    Write-Host "[tunnel] Cannot read the tunnel UUID from the 'tunnel:' field of"
    Write-Host "           $cfg"
    Write-Host "         Nothing was stopped or started. The file must contain a line like:"
    Write-Host "           tunnel: <your-tunnel-UUID>"
    exit 2
}

$procs = Get-TunnelProcs $id
switch ($Action) {
    'check' {
        Write-Host "[tunnel] tunnel UUID found in config.yml"
        exit 0
    }
    'status' {
        if ($procs.Count) { Write-Host ("[tunnel] running (PID " + (($procs | ForEach-Object { $_.ProcessId }) -join ', ') + ")"); exit 0 }
        Write-Host "[tunnel] not running"
        exit 1
    }
    'ensure' {
        if ($procs.Count) {
            Write-Host ("[tunnel] already running (PID " + (($procs | ForEach-Object { $_.ProcessId }) -join ', ') + ") - not starting a second copy")
            exit 0
        }
        Start-Tunnel $id
        exit 0
    }
    'restart' {
        foreach ($p in $procs) {
            Write-Host ("[tunnel] stopping PID " + $p.ProcessId + " (this tunnel only; other cloudflared instances are left alone)")
            Stop-Process -Id $p.ProcessId -Force
        }
        if ($procs.Count) { Start-Sleep -Seconds 2 }
        Start-Tunnel $id
        exit 0
    }
}
