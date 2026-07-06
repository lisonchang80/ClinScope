@echo off
REM ============================================================
REM  Restart ONLY the public-site server (port 7860).
REM  Use this after you change app.py (UI / CSS / JS / models)
REM  and want it live on https://clinscope.tinghaochang.com .
REM  The Cloudflare tunnel is left running untouched.
REM    - Network-change 502 "Host Error"? -> use restart_tunnel.bat instead.
REM ============================================================
echo Stopping old server on :7860 ...
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":7860" ^| findstr LISTENING') do taskkill /F /PID %%p >nul 2>&1
timeout /t 1 /nobreak >nul
echo Starting fresh GPU server (venv) ...
set GRADIO_SERVER_PORT=7860
REM Public site: browser-side upload instead of server-side native folder picker.
set CLINSCOPE_PUBLIC=1
REM Load in-app Google login secrets (git-ignored). No-op if the file is absent.
if exist "%~dp0env.local.bat" call "%~dp0env.local.bat"
start "ClinScope Server" "D:\Claude code\ClinScope\.venv\Scripts\python.exe" -u "D:\Claude code\ClinScope\app.py"
echo.
echo Done. Wait ~20s for models to load, then reload the site.
echo (Keep this window open; closing it stops the server.)
