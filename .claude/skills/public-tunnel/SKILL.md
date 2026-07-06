---
name: public-tunnel
description: 維護/部署 ClinScope 公開網址 clinscope.tinghaochang.com(Cloudflare Tunnel;2026-07-06 起 Access 改 Bypass 公開 + 站內 Google 登入)。觸發詞:公開站連不上、502 Bad gateway、Host Error、重啟公開站/tunnel、開/關公開站、匿名瀏覽、站內 Google 登入/登出、站主 vs 訪客、誰能操作、CLINSCOPE_GOOGLE_CLIENT_ID/OWNER_EMAIL、env.local、模型篩選、本地模式、預設範例畫面、換網路後斷線、開機自啟、重新上鎖、把別的專案掛子網域。涉及 ~/.cloudflared/(config.yml)、packaging/autostart/(start_public.bat/restart_server.bat/restart_tunnel.bat/env.local.bat)、app.py launch_app()/身分與篩選、Cloudflare Zero Trust(one.dash.cloudflare.com)、GCP OAuth(plenary-treat-288305)。完整背景見 docs/DEPLOY_PLAN.md Track A + [[clinscope-project]] memory。
---

# Public Tunnel — ClinScope 公開站維運

把本機 GPU server(`127.0.0.1:7860`)經 Cloudflare Tunnel 開成 `https://clinscope.tinghaochang.com`,前面用 Cloudflare Access(Google 登入)擋。
**本機是真正在跑的地方,Cloudflare 只是轉送** → 電腦關機 / server 沒跑 / cloudflared 沒跑就連不上。
完整架構與決策見 `docs/DEPLOY_PLAN.md`(Track A);本 skill 給**維運速查 + 坑**。

## 現況關鍵值
- 網域 `tinghaochang.com`(Cloudflare Registrar);tunnel 名 `tinghaochang-sites`(原名 `clinscope`,2026-07-04 後台改名;UUID `e90e134d…` 見 `~/.cloudflared/config.yml`,自啟/重啟腳本一律用 UUID 跑、改名免疫)。
- **⚠️ 2026-07-06 認證大改**:clinscope 的 CF Access 已改 **Bypass/Everyone(公開,匿名可唯讀瀏覽)**,登入改由**站內 Google Sign-In**(app 層,見下方),不再靠 CF Access 擋。原 `任何登入者` Allow policy **仍留著只是被 Bypass 蓋過** → 想重新上鎖只要在 Zero Trust 移除那條 bypass policy(可逆)。scopelab/harnesshub 仍「任何登入者」未受影響(小心別動共用 policy)。
- 公開站跑 **GPU 開發版**(dev venv `python app.py`,非 CPU 安裝版),權重走本機 repo 免下載。
- **站內 Google 登入 + 站主控管(app 層)**:`app.py` 認證/身分靠環境變數 —— `CLINSCOPE_GOOGLE_CLIENT_ID`(GCP OAuth Web 用戶端「ClinScope」,授權來源 `https://clinscope.tinghaochang.com`,同 GIS/Portfolio 專案 `plenary-treat-288305`)+ `CLINSCOPE_SESSION_SECRET`,兩者放 **git-ignored `packaging/autostart/env.local.bat`**(三支 launcher 都會 `call` 它)。`OWNER_EMAIL` 預設 lisonchang80@gmail.com(可 `CLINSCOPE_OWNER_EMAIL` 覆寫)== 登入 email 才是站主(可操作+看到站主控制項),其餘=訪客(唯讀)。`AUTH_ENABLED = PUBLIC and GOOGLE_CLIENT_ID` → 有 client id 才走 `gr.mount_gradio_app(FastAPI+/auth/*)+uvicorn`、才顯示登入鈕;沒設則回退舊 `demo.launch()`。完整見 [[clinscope-project]] memory。
- 開機自啟:`%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\clinscope_public.vbs`(隱藏、detached,起 server+tunnel)。
- `cloudflared.exe` 在 `C:\Program Files (x86)\cloudflared\`(**無 admin → 不能裝 service**,用 Startup .vbs 取代)。
- Cloudflare 帳號 `Hao800922@gmail.com`、Zero Trust team `wandering-haze-5a99`。

## 維運速查
| 要做的事 | 怎麼做 |
|---|---|
| **502 Bad gateway / Host Error**(多在換網路、睡醒後) | 雙擊 `packaging/autostart/restart_tunnel.bat`(只重啟 tunnel,不動 server)。根因:cloudflared 連線卡舊網路,origin `localhost:7860` 本機仍 200。 |
| **錯誤 1033 / Cloudflare 隧道錯誤頁**(≠502):DNS 指向 Cloudflare 但**沒有 cloudflared 連上** | 跟 502 不同——**通常 server(7860)+ tunnel 兩個程序都不在了**。先用「確認本機健康」那列看是哪個沒了;最穩是整組重啟:`wscript "%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\clinscope_public.vbs"`(detached 一次拉起 server+tunnel),等 ~30s 模型載完 → 本機 200、公開站導 Access 登入即恢復。常見於**長閒置後兩程序一起被系統回收**(經查可無重開機),或**訓練暫停 tunnel 後沒以 detached 正確重啟**。 |
| **改了 app.py 要套用到公開站**(版面/CSS/JS/模型) | 雙擊 `packaging/autostart/restart_server.bat`(只重啟 7860 server、不動 tunnel;已改成會 `call env.local.bat` 載入 Google 登入祕密)。**Claude 在 session 內重啟**:PowerShell 先靜默載入祕密再 detached 起 —— `$vars = cmd /c 'call "…\env.local.bat" >nul 2>&1 && set'; $vars \| %{ if($_ -match '^(CLINSCOPE_[^=]+)=(.*)$'){Set-Item "Env:\$($matches[1])" $matches[2]} }` → 設 `$env:CLINSCOPE_PUBLIC='1'`、`GRADIO_SERVER_PORT='7860'` → `Start-Process python -ArgumentList '-u','"D:\Claude code\ClinScope\app.py"' -WindowStyle Hidden`。**坑**:①殺舊 process 要用 `*ClinScope\app.py*`(帶反斜線)或**埠**判斷,`*ClinScope*app.py*` 會誤殺 **ScopeLab**(它用 ClinScope venv 跑 ScopeLab\app.py);②`Start-Process -ArgumentList` 陣列不替含空格路徑加引號 → 路徑元素要**自帶引號**;③**production 絕不可設 `CLINSCOPE_FORCE_OWNER`**(那是本地測站主用,設了人人變站主);④啟動會多花 ~10s 預先分類範例夾。 |
| **手動開公開站** | 雙擊 `packaging/autostart/start_public.bat`(可見視窗,好看 log) |
| **停掉公開站** | 工作管理員結束 `python.exe`(佔 7860 那個)+ `cloudflared.exe`;或刪 Startup 的 `.vbs` 取消開機自啟 |
| **確認本機健康** | PowerShell:`Get-NetTCPConnection -LocalPort 7860 -State Listen`、`Get-Process cloudflared`、`Invoke-WebRequest http://127.0.0.1:7860`(應 200) |
| **確認公開站健康**(2026-07-06 起已公開) | `curl.exe -s -o NUL -w "%{http_code}" https://clinscope.tinghaochang.com` → **200**=正常(公開站匿名可看,不再導登入);**502**=回源斷(cloudflared 在跑但連不到 7860);**530 + 錯誤 1033**=cloudflared 沒連上(見上方 1033 列,多半 server+tunnel 都死)。⚠️改版前是「302=正常」,改成 Bypass 後變 200。 |
| **改誰是站主 / 誰能操作** | **app 層**,非 CF:改 `env.local.bat` 的 `CLINSCOPE_OWNER_EMAIL`(或程式預設 lisonchang80@gmail.com)。站主 = 站內用該 Google 帳號登入者;其餘匿名/他人 = 唯讀訪客。要驗實際收到的 email 看 `identity_debug.log`。 |
| **重新上鎖(改回要登入才能看)** | Zero Trust → Access controls → Applications → `clinscope` → 移除那條 Bypass/Everyone policy(原「任何登入者」Allow 還在,移除 bypass 即恢復強制登入)。 |
| **模型篩選 / 預設範例** | 站主登入後,模型列「⚙ 篩選」調分數門檻(預設 ≥85%)+ 手動隱藏 → 存 `model_filter.json`(git-ignored,套用到所有訪客)。預設畫面範例夾/模型:`CLINSCOPE_SAMPLE_DIR` / `CLINSCOPE_SAMPLE_MODEL`(預設 `06_真菌OpenFungi_fungi_50` + key `fungi`)。 |
| **收緊檔案曝險** | 自啟腳本/啟動環境設 `CLINSCOPE_ALLOWED_PATHS=D:\`(只服務 D 槽、不曝 C:\Users);`app.py` `launch_app()` 已支援。亦可設 `CLINSCOPE_AUTH_USER/PASS` 改用 Gradio 密碼(但 Access 已是更強的邊緣驗證) |
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
- **長閒置會整組掛掉(2026-06-19 實測)**:出現**錯誤 1033** 時查到 server(7860)+ cloudflared **兩個程序都不在**,且**自開機起未重開機**(排除 Windows Update)→ 長時間無人使用的空檔被系統回收。`502` 只死 tunnel(server 還在),`1033` 是連 server 都沒了——**先分清楚是哪個**(健康檢查那列),再決定只重 tunnel 還是跑 Startup VBS 整組拉起。最穩的「整組常駐」恢復一律走 `wscript` 跑 Startup 的 `clinscope_public.vbs`。
