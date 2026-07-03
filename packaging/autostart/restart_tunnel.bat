@echo off
REM ============================================================
REM  Restart ONLY the Cloudflare tunnel.
REM  Use this when https://clinscope.tinghaochang.com shows a
REM  502 "Bad gateway / Host Error" AFTER you changed network
REM  (WiFi switch, hotspot <-> WiFi, sleep/wake, etc).
REM  The local GPU server on :7860 is left running untouched.
REM ============================================================
echo Stopping old cloudflared (stale network connections) ...
taskkill /IM cloudflared.exe /F >nul 2>&1
timeout /t 2 /nobreak >nul
echo Starting fresh tunnel on the current network ...
REM Run by UUID (not name) so renaming the tunnel can never break this script.
start "Shared Tunnel (tinghaochang-sites)" "C:\Program Files (x86)\cloudflared\cloudflared.exe" tunnel run e90e134d-5254-4976-8c7e-7e85a21ece98
echo.
echo Done. Wait ~10s, then reload https://clinscope.tinghaochang.com
echo (Keep this window open; closing it stops the tunnel.)
