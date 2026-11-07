@echo off
REM ============================================================
REM  ClinScope public site - manual launcher (visible server window).
REM  ClinScope has NO logon autostart (retired 2026-08-07); start it
REM  by hand with this file (or restart_server.bat).
REM
REM  The Cloudflare tunnel is SHARED with the other sites and is
REM  normally already running (Startup\cloudflared_tunnel.vbs).
REM  This launcher only starts it when no cloudflared for this
REM  tunnel is running, and never starts a second copy. The tunnel
REM  UUID is read at run time from %USERPROFILE%\.cloudflared\
REM  config.yml (see tunnel_ctl.ps1); if it cannot be read, this
REM  launcher stops before starting anything.
REM
REM  Close the "ClinScope Server" window to STOP ClinScope only;
REM  the shared tunnel keeps serving the other sites.
REM ============================================================
chcp 65001 >nul
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tunnel_ctl.ps1" check
if errorlevel 1 (
    echo Stopped: fix the tunnel config above first. Nothing was started.
    pause
    exit /b 2
)
set GRADIO_SERVER_PORT=7860
set PYTHONIOENCODING=utf-8
REM Public site: browser-side upload instead of server-side native folder picker.
set CLINSCOPE_PUBLIC=1
REM Load in-app Google login secrets (git-ignored). No-op if the file is absent.
if exist "%~dp0env.local.bat" call "%~dp0env.local.bat"
set "CLINSCOPE_REPO=%~dp0..\.."

echo [1/2] Starting GPU inference server on 127.0.0.1:7860 ...
start "ClinScope Server" "%CLINSCOPE_REPO%\.venv\Scripts\python.exe" -u "%CLINSCOPE_REPO%\app.py"

timeout /t 4 /nobreak >nul

echo [2/2] Checking the shared Cloudflare tunnel ...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tunnel_ctl.ps1" ensure

echo.
echo Done. Close the "ClinScope Server" window to stop ClinScope
echo (the shared tunnel keeps running for the other sites).
