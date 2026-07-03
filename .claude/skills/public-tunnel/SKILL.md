---
name: public-tunnel
description: 維護/部署 ClinScope 公開網址 clinscope.tinghaochang.com(Cloudflare Tunnel + Access)。觸發詞:公開站連不上、502 Bad gateway、Host Error、重啟 tunnel、開/關公開站、改誰能登入、Access policy、IdP/Google 登入、換網路後斷線、開機自啟、把別的專案掛子網域。涉及 ~/.cloudflared/(config.yml/cert.pem/<UUID>.json)、packaging/autostart/(start_public.bat/restart_tunnel.bat)、app.py launch_app() 的 CLINSCOPE_AUTH_USER/PASS + CLINSCOPE_ALLOWED_PATHS、Cloudflare Zero Trust(one.dash.cloudflare.com)。完整背景見 docs/DEPLOY_PLAN.md Track A。
---

# Public Tunnel — ClinScope 公開站維運

把本機 GPU server(`127.0.0.1:7860`)經 Cloudflare Tunnel 開成 `https://clinscope.tinghaochang.com`,前面用 Cloudflare Access(Google 登入)擋。
**本機是真正在跑的地方,Cloudflare 只是轉送** → 電腦關機 / server 沒跑 / cloudflared 沒跑就連不上。
完整架構與決策見 `docs/DEPLOY_PLAN.md`(Track A);本 skill 給**維運速查 + 坑**。

## 現況關鍵值
- 網域 `tinghaochang.com`(Cloudflare Registrar);tunnel 名 `tinghaochang-sites`(原名 `clinscope`,2026-07-04 後台改名;UUID `e90e134d…` 見 `~/.cloudflared/config.yml`,自啟/重啟腳本一律用 UUID 跑、改名免疫)。
- Access app `clinscope`,destination=`clinscope.tinghaochang.com`(**單一主機名,非整域**),policy `任何登入者`=Everyone,登入=**Google OAuth IdP**(任何 Google 真人可進、擋 bot)。
- 公開站跑 **GPU 開發版**(dev venv `python app.py`,非 CPU 安裝版),權重走本機 repo 免下載。
- 開機自啟:`%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\clinscope_public.vbs`(隱藏、detached,起 server+tunnel)。
- `cloudflared.exe` 在 `C:\Program Files (x86)\cloudflared\`(**無 admin → 不能裝 service**,用 Startup .vbs 取代)。
- Cloudflare 帳號 `Hao800922@gmail.com`、Zero Trust team `wandering-haze-5a99`。

## 維運速查
| 要做的事 | 怎麼做 |
|---|---|
| **502 Bad gateway / Host Error**(多在換網路、睡醒後) | 雙擊 `packaging/autostart/restart_tunnel.bat`(只重啟 tunnel,不動 server)。根因:cloudflared 連線卡舊網路,origin `localhost:7860` 本機仍 200。 |
| **錯誤 1033 / Cloudflare 隧道錯誤頁**(≠502):DNS 指向 Cloudflare 但**沒有 cloudflared 連上** | 跟 502 不同——**通常 server(7860)+ tunnel 兩個程序都不在了**。先用「確認本機健康」那列看是哪個沒了;最穩是整組重啟:`wscript "%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\clinscope_public.vbs"`(detached 一次拉起 server+tunnel),等 ~30s 模型載完 → 本機 200、公開站導 Access 登入即恢復。常見於**長閒置後兩程序一起被系統回收**(經查可無重開機),或**訓練暫停 tunnel 後沒以 detached 正確重啟**。 |
| **改了 app.py 要套用到公開站**(版面/CSS/JS/模型) | 雙擊 `packaging/autostart/restart_server.bat`(只重啟 7860 server、不動 tunnel;新 server 起來 cloudflared 自動恢復轉送)。**在 session 內由 Claude 重啟需 detached**:wscript 跑「純 ASCII」VBS(`sh.Run "cmd /c set GRADIO_SERVER_PORT=7860&& ...python -u app.py", 0, False`)——VBS 含中文存成 UTF-8 會被 wscript 當 ANSI 解析失敗而靜默不啟動。 |
| **手動開公開站** | 雙擊 `packaging/autostart/start_public.bat`(可見視窗,好看 log) |
| **停掉公開站** | 工作管理員結束 `python.exe`(佔 7860 那個)+ `cloudflared.exe`;或刪 Startup 的 `.vbs` 取消開機自啟 |
| **確認本機健康** | PowerShell:`Get-NetTCPConnection -LocalPort 7860 -State Listen`、`Get-Process cloudflared`、`Invoke-WebRequest http://127.0.0.1:7860`(應 200) |
| **確認 Access 守門** | `curl.exe -s -o NUL -w "%{http_code}" https://clinscope.tinghaochang.com` → **302**=正常(導向 Access 登入頁);**200 直接進站**=Access 失效要查;**502**=回源斷(cloudflared 在跑但連不到 7860);**530 + 錯誤 1033**=cloudflared 沒連上(見上方 1033 列,多半 server+tunnel 都死) |
| **改誰能登入** | one.dash.cloudflare.com → Access controls → Policies → 編輯 `任何登入者`:`Everyone`=任何真人 /  改 `Emails` 白名單=限特定人 |
| **改登入方式** | Access controls → Applications → `clinscope` → Authentication 區(關/開 IdP;只勾 Google = 登入頁只剩 Google) |
| **收緊檔案曝險** | 自啟腳本/啟動環境設 `CLINSCOPE_ALLOWED_PATHS=D:\`(只服務 D 槽、不曝 C:\Users);`app.py` `launch_app()` 已支援。亦可設 `CLINSCOPE_AUTH_USER/PASS` 改用 Gradio 密碼(但 Access 已是更強的邊緣驗證) |
| **再掛別專案子網域** | `cloudflared tunnel route dns tinghaochang-sites x.tinghaochang.com` + `config.yml` ingress 加一條(或開新 tunnel) + **另建一個 Access app**(Access 綁單一主機名,不會自動涵蓋新子網域) |

## 坑(都實際踩過)
- **無 admin → 不能 `cloudflared service install`**(裝 Windows 服務要系統管理員)→ 改 Startup 資料夾 `.vbs` 登入自啟(detached、隱藏視窗)。
- **換網路必斷**:WiFi 切換 / 熱點↔WiFi / 睡眠喚醒後,cloudflared 卡在舊網路連線 → 502(本機 server 其實沒事)。重啟 tunnel 即解;重開機走自啟是全新連線不受影響。這是「把服務掛在會換網路的筆電」的先天限制。
- **Access 只綁單一主機名**:`clinscope` app 只保護 `clinscope.tinghaochang.com`,apex 與其他子網域不受影響(要保護得另建 app,或用萬用 `*.tinghaochang.com`)。
- **Google OAuth IdP 設定**:Google Cloud → Google Auth Platform → OAuth client(網頁應用程式)+ consent screen External **且發布到 Production**(基本 scope email/profile 免 Google 審核、無 100 人上限)才讓「任何 Google 帳號」登入;測試模式只有 owner/測試者能進。redirect URI = `https://wandering-haze-5a99.cloudflareaccess.com/cdn-cgi/access/callback`。
- **登入 ≠ 不間斷**:固定網址只解決「找得到」,真正可連要本機開著 + server + cloudflared 都在跑。
- **路徑有空格**:一律用完整路徑字串 `"C:\Program Files (x86)\cloudflared\cloudflared.exe"`;啟動用 `wscript` 跑 `.vbs`(`WScript.Shell.Run ..., 0, False`)可得「隱藏 + detached、撐過 session/Claude 關閉」。
- **detached vs 背景任務**:用 run_in_background 起的 server/tunnel 會隨 session 結束被砍;要常駐請走 `.vbs`(wscript)那條。
- **長閒置會整組掛掉(2026-06-19 實測)**:出現**錯誤 1033** 時查到 server(7860)+ cloudflared **兩個程序都不在**,且**自開機起未重開機**(排除 Windows Update)→ 長時間無人使用的空檔被系統回收。`502` 只死 tunnel(server 還在),`1033` 是連 server 都沒了——**先分清楚是哪個**(健康檢查那列),再決定只重 tunnel 還是跑 Startup VBS 整組拉起。最穩的「整組常駐」恢復一律走 `wscript` 跑 Startup 的 `clinscope_public.vbs`。
