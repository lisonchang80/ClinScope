@echo off
REM ============================================================
REM  Restart ONLY the SHARED Cloudflare tunnel (tinghaochang-sites).
REM  Use this when the public sites show 502 "Bad gateway / Host
REM  Error" AFTER a network change (WiFi switch, hotspot <-> WiFi,
REM  sleep/wake) while the local servers are still fine.
REM
REM  The tunnel is shared: gis / scopelab / harnesshub / the home
REM  page (and clinscope when its server runs) all drop for ~10-30 s.
REM  It is normally started at logon by
REM    Startup\cloudflared_tunnel.vbs   (ClinScope has no autostart).
REM
REM  What this does (see tunnel_ctl.ps1):
REM   - reads the tunnel UUID at run time from the "tunnel:" field of
REM     %USERPROFILE%\.cloudflared\config.yml (never stored in git);
REM     if it cannot be read, NOTHING is stopped or started;
REM   - stops only the cloudflared whose command line has that UUID
REM     (other cloudflared instances keep running);
REM   - starts it again hidden + detached via the Startup VBS, so
REM     closing this window does NOT stop the tunnel.
REM  Local servers (e.g. ClinScope on :7860) are left untouched.
REM ============================================================
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tunnel_ctl.ps1" restart
echo.
echo Wait ~15 s, then reload the public site. You can close this window.
pause
