@echo off
REM ============================================================
REM  ClinScope public site - manual launcher (visible windows)
REM  Use this to start the public site by hand, or to see logs.
REM  The silent logon-autostart equivalent lives in the Startup
REM  folder as clinscope_public.vbs (hidden, detached).
REM
REM  Public URL (Cloudflare Access protected):
REM      https://clinscope.tinghaochang.com
REM  Close the two spawned windows to STOP the public site.
REM ============================================================
chcp 65001 >nul
set GRADIO_SERVER_PORT=7860
set PYTHONIOENCODING=utf-8

echo [1/2] Starting GPU inference server on 127.0.0.1:7860 ...
start "ClinScope Server" "D:\Claude code\ClinScope\.venv\Scripts\python.exe" -u "D:\Claude code\ClinScope\app.py"

timeout /t 4 /nobreak >nul

echo [2/2] Starting Cloudflare Tunnel (tinghaochang-sites, shared) ...
REM Run by UUID (not name) so renaming the tunnel can never break this script.
start "Shared Tunnel (tinghaochang-sites)" "C:\Program Files (x86)\cloudflared\cloudflared.exe" tunnel run e90e134d-5254-4976-8c7e-7e85a21ece98

echo.
echo Done. Public URL: https://clinscope.tinghaochang.com
echo (Close the two new windows to stop the public site.)
