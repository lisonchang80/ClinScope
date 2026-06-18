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
start "ClinScope Tunnel" "C:\Program Files (x86)\cloudflared\cloudflared.exe" tunnel run clinscope
echo.
echo Done. Wait ~10s, then reload https://clinscope.tinghaochang.com
echo (Keep this window open; closing it stops the tunnel.)
