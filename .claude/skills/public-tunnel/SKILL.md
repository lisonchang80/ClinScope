---
name: public-tunnel
description: 維護/部署 ClinScope 公開網址 clinscope.tinghaochang.com(Cloudflare Tunnel;2026-07-06 起 Access 改 Bypass 公開 + 站內 Google 登入)。觸發詞:公開站連不上、502 Bad gateway、Host Error、重啟公開站/tunnel、開/關公開站、匿名瀏覽、站內 Google 登入/登出、站主 vs 訪客、誰能操作、CLINSCOPE_GOOGLE_CLIENT_ID/OWNER_EMAIL、env.local、模型篩選、本地模式、預設範例畫面、換網路後斷線、開機自啟、重新上鎖、把別的專案掛子網域。涉及 ~/.cloudflared/(config.yml)、packaging/autostart/(start_public.bat/restart_server.bat/restart_tunnel.bat/env.local.bat)、app.py launch_app()/身分與篩選、Cloudflare Zero Trust(one.dash.cloudflare.com)、GCP OAuth(plenary-treat-288305)。完整背景見 docs/DEPLOY_PLAN.md Track A + [[clinscope-project]] memory。
---

# Public Tunnel — ClinScope 公開站維運

把本機 GPU server(`127.0.0.1:7860`)經 Cloudflare Tunnel 開成 `https://clinscope.tinghaochang.com`。Cloudflare Access 對 clinscope 是 **Bypass(公開,匿名可唯讀)**,登入改由站內 Google Sign-In(見下方「現況關鍵值」)。
**tunnel 是共用的**(同一條還服務 gis / scopelab / harnesshub / tinghaochang.com 首頁),**不歸 ClinScope 管**:登入時由 Startup 的 `cloudflared_tunnel.vbs` 起;ClinScope **2026-08-07 起沒有開機自啟**,要開公開站得手動起 server。任何「重啟 / 停掉 tunnel」都會連帶影響其他網站。
**本機是真正在跑的地方,Cloudflare 只是轉送** → 電腦關機 / server 沒跑 / cloudflared 沒跑就連不上。
完整架構與決策見 `docs/DEPLOY_PLAN.md`(Track A);本 skill 給**維運速查 + 坑**。

## 現況關鍵值
- 網域 `tinghaochang.com`(Cloudflare Registrar);tunnel 名 `tinghaochang-sites`(原名 `clinscope`,2026-07-04 後台改名;UUID `e90e134d…` 見 `~/.cloudflared/config.yml`,自啟/重啟腳本一律用 UUID 跑、改名免疫)。
- **⚠️ 2026-07-06 認證大改**:clinscope 的 CF Access 已改 **Bypass/Everyone(公開,匿名可唯讀瀏覽)**,登入改由**站內 Google Sign-In**(app 層,見下方),不再靠 CF Access 擋。原 `任何登入者` Allow policy **仍留著只是被 Bypass 蓋過** → 想重新上鎖只要在 Zero Trust 移除那條 bypass policy(可逆)。scopelab/harnesshub 仍「任何登入者」未受影響(小心別動共用 policy)。
- 公開站跑 **GPU 開發版**(dev venv `python app.py`,非 CPU 安裝版),權重走本機 repo 免下載。
- **站內 Google 登入 + 站主控管(app 層)**:`app.py` 認證/身分靠環境變數 —— `CLINSCOPE_GOOGLE_CLIENT_ID`(GCP OAuth Web 用戶端「ClinScope」,授權來源 `https://clinscope.tinghaochang.com`,同 GIS/Portfolio 專案 `plenary-treat-288305`)+ `CLINSCOPE_SESSION_SECRET`,兩者放 **git-ignored `packaging/autostart/env.local.bat`**(三支 launcher 都會 `call` 它)。`OWNER_EMAIL` 預設 lisonchang80@gmail.com(可 `CLINSCOPE_OWNER_EMAIL` 覆寫)== 登入 email 才是站主(可操作+看到站主控制項),其餘=訪客(唯讀)。`AUTH_ENABLED = PUBLIC and GOOGLE_CLIENT_ID` → 有 client id 才走 `gr.mount_gradio_app(FastAPI+/auth/*)+uvicorn`、才顯示登入鈕;沒設則回退舊 `demo.launch()`。完整見 [[clinscope-project]] memory。
- 開機自啟:**ClinScope 沒有**(原 `clinscope_public.vbs` 2026-08-07 退役,Startup 資料夾裡已無此檔)。共用 tunnel 由 `%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\cloudflared_tunnel.vbs` 在登入時隱藏、detached 起(`tunnel run <UUID>`)。公開站 server 一律手動起:`packaging/autostart/start_public.bat` 或 `restart_server.bat`。
- 腳本找 tunnel 一律用 **UUID**,且 UUID **執行時才從 `%USERPROFILE%\.cloudflared\config.yml` 的 `tunnel:` 欄讀出**(`packaging/autostart/tunnel_ctl.ps1`,公開 repo 不寫死 UUID);讀不到就什麼都不停、不起,印出說明。這台機器上**還有別的 cloudflared**(例如另一個專案用自己 `--config` 起的 media tunnel)→ **永遠不要 `taskkill /IM cloudflared.exe`**,只結束命令列含該 UUID 的那一個。
- `cloudflared.exe` 在 `C:\Program Files (x86)\cloudflared\`(**無 admin → 不能裝 service**,用 Startup .vbs 取代)。
- Cloudflare 帳號 `Hao800922@gmail.com`、Zero Trust team `wandering-haze-5a99`。

## 維運速查
| 要做的事 | 怎麼做 |
|---|---|
| **502 Bad gateway / Host Error** | **先分清楚是哪個沒了**(「確認本機健康」那列)。① 7860 沒人聽 = ClinScope server 沒跑(2026-08-07 起沒有自啟,**這是常態**,不是 tunnel 壞)→ 要開公開站就雙擊 `start_public.bat`。② 7860 本機 200、而且 gis/scopelab 等**也** 502(多在換網路、睡醒後,cloudflared 連線卡舊網路)→ 雙擊 `packaging/autostart/restart_tunnel.bat`:只結束命令列含 config.yml 那個 UUID 的 cloudflared,再經 `Startup\cloudflared_tunnel.vbs` 隱藏、detached 重起(關掉視窗不會斷);**共用 tunnel → 其他網站會一起斷 10–30 秒**,其他 cloudflared 不受影響。 |
| **錯誤 1033 / Cloudflare 隧道錯誤頁**(≠502):DNS 指向 Cloudflare 但**沒有 cloudflared 連上** | 共用 tunnel 整個不在(所有子網域都會 1033)→ 跑 `restart_tunnel.bat`(沒有在跑的就直接起),或 `wscript "%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\cloudflared_tunnel.vbs"`。ClinScope server 另外用 `start_public.bat` 起(它會先檢查 tunnel、已在跑就不再起第二份);等 ~30s 模型載完 → 本機 200、公開站回 200 即恢復。常見於長閒置後被系統回收、或有人以前台方式起的 cloudflared 視窗被關掉。 |
| **改了 app.py 要套用到公開站**(版面/CSS/JS/模型) | 雙擊 `packaging/autostart/restart_server.bat`(只重啟 7860 server、不動 tunnel;已改成會 `call env.local.bat` 載入 Google 登入祕密)。**Claude 在 session 內重啟**:PowerShell 先靜默載入祕密再 detached 起 —— `$vars = cmd /c 'call "…\env.local.bat" >nul 2>&1 && set'; $vars \| %{ if($_ -match '^(CLINSCOPE_[^=]+)=(.*)$'){Set-Item "Env:\$($matches[1])" $matches[2]} }` → 設 `$env:CLINSCOPE_PUBLIC='1'`、`GRADIO_SERVER_PORT='7860'` → `Start-Process python -ArgumentList '-u','"D:\Claude code\ClinScope\app.py"' -WindowStyle Hidden`。**坑**:①殺舊 process 要用 `*ClinScope\app.py*`(帶反斜線)或**埠**判斷,`*ClinScope*app.py*` 會誤殺 **ScopeLab**(它用 ClinScope venv 跑 ScopeLab\app.py);②`Start-Process -ArgumentList` 陣列不替含空格路徑加引號 → 路徑元素要**自帶引號**;③`CLINSCOPE_FORCE_OWNER` **已移除**(舊版設了人人變站主;現在公開模式一律忽略並印 ⚠,站主只能經站內 Google 登入;本機要看站主畫面就不設 `CLINSCOPE_PUBLIC` 跑桌面模式);④啟動會多花 ~10s 預先分類範例夾。 |
| **手動開公開站** | 雙擊 `packaging/autostart/start_public.bat`(server 開可見視窗,好看 log;tunnel 已在跑就不再起,沒在跑才經 Startup VBS 隱藏起;讀不到 UUID 就整個停下) |
| **停掉公開站** | **只結束佔 7860 的 python**:關掉「ClinScope Server」視窗,或 PowerShell `Get-NetTCPConnection -LocalPort 7860 -State Listen \| %{ Stop-Process -Id $_.OwningProcess -Force }`。**不要結束 cloudflared**(共用 tunnel,gis/scopelab/harnesshub/首頁會一起斷);停掉後 clinscope 回 502 是正常的。 |
| **確認本機健康** | PowerShell:`Get-NetTCPConnection -LocalPort 7860 -State Listen`、`Invoke-WebRequest http://127.0.0.1:7860`(應 200);tunnel 用 `powershell -NoProfile -ExecutionPolicy Bypass -File packaging\autostart\tunnel_ctl.ps1 status`(只看這條 tunnel 的 cloudflared;`Get-Process cloudflared` 會連別的 tunnel 一起列出) |
| **確認公開站健康**(2026-07-06 起已公開) | `curl.exe -s -o NUL -w "%{http_code}" https://clinscope.tinghaochang.com` → **200**=正常(公開站匿名可看,不再導登入);**502**=回源斷(cloudflared 在跑但連不到 7860);**530 + 錯誤 1033**=cloudflared 沒連上(見上方 1033 列,多半 server+tunnel 都死)。⚠️改版前是「302=正常」,改成 Bypass 後變 200。 |
| **改誰是站主 / 誰能操作** | **app 層**,非 CF:改 `env.local.bat` 的 `CLINSCOPE_OWNER_EMAIL`(或程式預設 lisonchang80@gmail.com)。站主 = 站內用該 Google 帳號登入者;其餘匿名/他人 = 唯讀訪客。要驗實際收到的 email 看 `identity_debug.log`。 |
| **重新上鎖(改回要登入才能看)** | Zero Trust → Access controls → Applications → `clinscope` → 移除那條 Bypass/Everyone policy(原「任何登入者」Allow 還在,移除 bypass 即恢復強制登入)。 |
| **模型篩選 / 預設範例** | 站主登入後,模型列「⚙ 篩選」調分數門檻(預設 ≥85%)+ 手動隱藏 → 存 `model_filter.json`(git-ignored,套用到所有訪客)。預設畫面範例夾/模型:`CLINSCOPE_SAMPLE_DIR` / `CLINSCOPE_SAMPLE_MODEL`(預設 `06_真菌OpenFungi_fungi_50` + key `fungi`)。 |
| **檔案曝險(allowed_paths)** | Gradio 會把 `allowed_paths` 內**任何檔案**經 `/gradio_api/file=<絕對路徑>` 交給任何人。公開站(`CLINSCOPE_PUBLIC=1`)預設**只開桌面安裝檔**;`CLINSCOPE_ALLOWED_PATHS` 裡涵蓋/位於 repo 根或家目錄的項目會被剔除(啟動 log 印 ⚠)——**不要再設 `D:\`**(它涵蓋 repo,舊版因此可被抓走 `env.local.bat`)。範例夾/本地模式影像改由 `_servable` 複製進 Gradio 快取再顯示,匯出檔放系統暫存隨機子目錄,都不需 allowed_paths。桌面/dev 版維持原行為。 |
| **再掛別專案子網域** | `cloudflared tunnel route dns tinghaochang-sites x.tinghaochang.com` + `config.yml` ingress 加一條(或開新 tunnel) + **另建一個 Access app**(Access 綁單一主機名,不會自動涵蓋新子網域) |

## 坑(都實際踩過)
- **改 tunnel 顯示名(2026-07-04 實戰:clinscope→tinghaochang-sites)**:cloudflared CLI **沒有 rename**;後台對 locally-managed tunnel 只有 Configure=**Migrate(不可逆,勿按)**/Delete。正解=已登入 dash 分頁用 fetch 打 `PATCH /api/v4/accounts/<acct>/cfd_tunnel/<uuid>` body `{name}`,**必帶 header `x-cross-site-security: dash`**(不帶回 HTML 403)。一切(DNS/config.yml/憑證)綁 UUID→改名零斷線;**但 `tunnel run <名字>` 的腳本會壞**——改名前先把所有自啟/重啟腳本換成 `tunnel run <UUID>`(本 repo 已全改 UUID)。
- **無 admin → 不能 `cloudflared service install`**(裝 Windows 服務要系統管理員)→ 改 Startup 資料夾 `.vbs` 登入自啟(detached、隱藏視窗)。
- **換網路必斷**:WiFi 切換 / 熱點↔WiFi / 睡眠喚醒後,cloudflared 卡在舊網路連線 → 502(本機 server 其實沒事)。重啟 tunnel 即解;重開機走自啟是全新連線不受影響。這是「把服務掛在會換網路的筆電」的先天限制。
- **Access 只綁單一主機名**:`clinscope` app 只保護 `clinscope.tinghaochang.com`,apex 與其他子網域不受影響(要保護得另建 app,或用萬用 `*.tinghaochang.com`)。
- **Google OAuth IdP 設定**:Google Cloud → Google Auth Platform → OAuth client(網頁應用程式)+ consent screen External **且發布到 Production**(基本 scope email/profile 免 Google 審核、無 100 人上限)才讓「任何 Google 帳號」登入;測試模式只有 owner/測試者能進。redirect URI = `https://wandering-haze-5a99.cloudflareaccess.com/cdn-cgi/access/callback`。
- **登入 ≠ 不間斷**:固定網址只解決「找得到」,真正可連要本機開著 + server + cloudflared 都在跑。
- **路徑有空格**:一律用完整路徑字串 `"C:\Program Files (x86)\cloudflared\cloudflared.exe"`;啟動用 `wscript` 跑 `.vbs`(`WScript.Shell.Run ..., 0, False`)可得「隱藏 + detached、撐過 session/Claude 關閉」。
- **detached vs 背景任務**:用 run_in_background 起的 server/tunnel 會隨 session 結束被砍;要常駐請走 `.vbs`(wscript)那條。
- **公開站關著時要驗新版,不要用 7860**(2026-10-26):tunnel 的 ingress 固定把 clinscope.tinghaochang.com 轉到 `localhost:7860`,只要有東西聽 7860 就等於重新公開。驗證改用別的埠(例 7870)、只綁 127.0.0.1、`CLINSCOPE_PUBLIC=1` 起臨時實例,驗完關掉,最後確認公開站仍是 502。**2026-11-07 起**(審查 CS-1/GA-1):非公開模式(沒設 `CLINSCOPE_PUBLIC`)掛 `_LocalOnly` 本機限定中介層——帶 `cf-connecting-ip`/`cf-ray`/`cf-visitor`、`cdn-loop: cloudflare`,或 Host 不是 127.0.0.1/localhost/[::1] 一律 403(websocket 1008),即使誤起在 7860 也不會把整碟與站主權限公開;開發／桌面預設埠改 **7862**(`.claude/launch.json`、`run_desktop.py` 的 `CLINSCOPE_PORT` 同步),公開啟動器仍明確設 7860。驗公開模式行為仍要用 `CLINSCOPE_PUBLIC=1` 的臨時實例(非公開模式經 tunnel 一律 403)。
- **跨子網域防線(2026-11-07 審查 GA-3/GA-8/CS-2)**:SameSite=Lax 擋不住兄弟子網域,Gradio 內建 CORS 對任何 Origin 回 ACAO+credentials → app 層加了 `_SameOriginWriteGuard`(非 GET/HEAD/OPTIONS 要 `Sec-Fetch-Site: same-origin/none`,舊瀏覽器比 Origin 與 Host)、`_StripCrossOriginCors`(非同源拿掉 ACAO/ACAC)、`_FrameGuard`(`X-Frame-Options: SAMEORIGIN` + `frame-ancestors 'self'`);公開模式不替任何人下載 FileData URL。cookie 在 Secure 時改名 **`__Host-cs_session`**(本機 http 開發仍 `cs_session`)→ **新版上線後所有人要重登一次**。用 curl 驗寫入端點時不要帶別站的 `Origin`/`Sec-Fetch-Site`,否則是預期中的 403。
- **公開模式會剔除危險的 `CLINSCOPE_ALLOWED_PATHS`**(2026-10-26 安全審查後):設成磁碟根、repo 根或家目錄時啟動 log 印「⚠ 公開站忽略會曝露 repo/家目錄的…項目」並拿掉;對外檔案只剩 `ClinScope-Setup.exe`。舊 skill 建議過的 `D:\` 已無效,不用再設。
- **長閒置會整組掛掉(2026-06-19 實測)**:出現**錯誤 1033** 時查到 server(7860)+ cloudflared **兩個程序都不在**,且**自開機起未重開機**(排除 Windows Update)→ 長時間無人使用的空檔被系統回收。`502` = tunnel 在、但回源 7860 連不到(server 沒跑,或 tunnel 卡舊網路),`1033` = 沒有 cloudflared 連上——**先分清楚是哪個**(健康檢查那列),再決定只重 tunnel 還是也要起 server。(2026-08-07 前的「跑 Startup 的 `clinscope_public.vbs` 整組拉起」已不適用:那支已退役、檔案不在;現在 tunnel 走 `restart_tunnel.bat`/`cloudflared_tunnel.vbs`,server 走 `start_public.bat`。)
