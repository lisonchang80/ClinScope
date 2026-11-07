# ClinScope 部署計畫（前端公開網址 + 本地安裝程式）

兩條獨立軌道。**A** 把本機 app 開成固定公開網址；**B** 把 app 打包成 Windows 安裝程式，
權重「選了才下載」。權重統一放 **公開** HF Hub repo：`tinghaochang/clinscope-weights`。

---

## 軌道 A — 買網域 + Cloudflare Tunnel（固定網址）

| # | 步驟 | 誰 | 狀態 |
|---|---|---|---|
| A0 | 買網域（Cloudflare Registrar，DNS 自動在 CF） | 使用者 | ✅ 已購 **tinghaochang.com** |
| A1 | 裝 `cloudflared`(winget 2026.5.2)、`tunnel login`、建 named tunnel `clinscope` (UUID e90e134d…；2026-07-04 已改名 `tinghaochang-sites`，腳本一律用 UUID 跑) | Claude | ✅ |
| A2 | `clinscope.tinghaochang.com → http://localhost:7860`（`~/.cloudflared/config.yml` + DNS CNAME route） | Claude | ✅ |
| A3 | **登入自動啟動**（無 admin→不能用 service；改 Startup 資料夾 `.vbs`）。⚠️ **現況（2026-08-07 起）**：ClinScope 的 `clinscope_public.vbs` 已退役、**ClinScope 沒有自啟**；tunnel 改由 Startup 的 `cloudflared_tunnel.vbs` 起，**共用**給 gis / scopelab / harnesshub / tinghaochang.com | Claude | ✅→退役 |
| A4 | Cloudflare Access（Self-hosted app `clinscope`、policy `任何登入者`=Everyone、登入方式 = Google OAuth IdP → 任何 Google 真人可進、擋 bot；**僅保護 `clinscope` 單一子網域,非整個 tinghaochang.com**）。⚠️ **現況（2026-07-06 起）**：clinscope 改 **Bypass（公開，匿名唯讀）**，登入改站內 Google Sign-In（見 `.claude/skills/public-tunnel/SKILL.md`） | Claude/使用者 | ✅→Bypass |

> ✅ **2026-06-18 全鏈路上線並實證**：未登入連 `clinscope.tinghaochang.com` → HTTP 302 導向 `wandering-haze-5a99.cloudflareaccess.com`(Access 攔截)；用 Google 登入後進到 ClinScope。**登入方式 = Google OAuth**(OTP 已停用;policy=Everyone=任何 Google 真人,擋 bot)。Access 僅綁 `clinscope.tinghaochang.com` 單一主機名,其他子網域/apex 不受此規則影響。
> **公開站用 GPU 開發版**(dev venv cu124 `python app.py`，非 CPU 安裝版；徽章顯示 GPU、權重走本機 repo 免下載)。
> 固定網址 ≠ 不間斷：本機關機/睡眠/server 或 cloudflared 沒跑就連不上。**停掉公開站** = 只結束佔 7860 的 `python.exe`（關掉「ClinScope Server」視窗）。**不要結束 `cloudflared.exe`**：tunnel 是共用的，會連帶斷掉 gis / scopelab / harnesshub / 首頁；機器上也可能有別的 cloudflared（其他專案用自己的 `--config` 起）。ClinScope 2026-08-07 起沒有開機自啟，不必再刪 Startup 檔。
> ⚠️ **換網路(WiFi 切換 / 熱點↔WiFi / 睡眠喚醒)後常見 502「Host Error」**：cloudflared 連線卡在舊網路、沒乾淨重連(origin `localhost:7860` 本機仍 200)。修復=跑 `restart_tunnel.bat`：執行時從 `%USERPROFILE%\.cloudflared\config.yml` 的 `tunnel:` 欄讀 UUID（讀不到就什麼都不做），只結束命令列含該 UUID 的 cloudflared，再經 Startup 的 `cloudflared_tunnel.vbs` 隱藏、detached 重起（見 `packaging/autostart/tunnel_ctl.ps1`）。**不要用 `taskkill /IM cloudflared.exe`**。重開機走登入自啟則是全新連線、不受影響。2026-06-18 實證:熱點 172.20.10.4→WiFi 192.168.1.108 後 502,重啟 tunnel 立即恢復 302（當時 Access 還沒改 Bypass；現在正常是 200）。

### Track A 部署產物（已從程式碼/系統確認）
- `~/.cloudflared/`：`cert.pem`(origin 授權)、`<UUID>.json`(tunnel 憑證，**secret**)、`config.yml`(ingress)。
- Startup 自啟：ClinScope **沒有**（`clinscope_public.vbs` 2026-08-07 退役）；共用 tunnel = `%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\cloudflared_tunnel.vbs`(隱藏、detached，不在本 repo)。
- 手動啟動器：`packaging/autostart/start_public.bat`(server 開可見視窗，除錯/手動開用；tunnel 已在跑就不再起，沒在跑才經 Startup VBS 起；讀不到 UUID 就整個停下)。
- 換網路修復：`packaging/autostart/restart_tunnel.bat`(WiFi/熱點切換後出現 502 → 只重啟這條共用 tunnel，server 不動、其他 cloudflared 不動)。
- tunnel 小工具：`packaging/autostart/tunnel_ctl.ps1`（`check` / `status` / `ensure` / `restart`；UUID 執行時讀 config.yml，公開 repo 不寫死）。
- Cloudflare 帳號 `Hao800922@gmail.com`、Zero Trust team `wandering-haze-5a99`、Access app `clinscope`(destination=`clinscope.tinghaochang.com` **單一主機名,非整域**;policy `任何登入者`=Everyone)。
- 登入方式 = **Google OAuth IdP**:Google Cloud 專案 `plenary-treat-288305`(My First Project)→ OAuth client `ClinScope-Cloudflare`、consent screen External + **已發布 Production**(基本 scope email/profile 免審核、無 100 人上限)、redirect URI `https://wandering-haze-5a99.cloudflareaccess.com/cdn-cgi/access/callback`;該 app 已停用 One-time PIN(只留 Google)。要再保護別的子網域需「另建」Access app(或用萬用 `*.tinghaochang.com`)。

---

## 軌道 B — Inno Setup 安裝程式（選了才下載）

使用者拿到 `ClinScope-Setup.exe`（只含程式 + CPU torch，**不含權重**）→ 安裝 → 開 app →
模型下拉列全部 89 個 → 選到未下載者跳「⬇ 下載」進度條 → 抓自 HF Hub → 存快取 → 驗 sha → 載入。

| # | 里程碑 | 內容 | 狀態 |
|---|---|---|---|
| B1 | runtime 子集 + CPU torch | `packaging/requirements-runtime.txt` ✅；乾淨 CPU-only venv 凍結驗證留待 B4 | 🟢 大致完成 |
| B2 | HF 權重 + lazy download | resolver `src/weights.py` ✅ + `weights_manifest.json`(90 檔/6.25GB) ✅ + app.py 接線 ✅ + 端到端推論 ✅ + HF 上傳(90 檔 live、public 無 token 可下載驗過)✅ + **下載進度 UI**(選模型顯示未下載橫幅、計算/點圖時 `gr.Progress` 顯示「下載…」;缺席模擬已驗下載+sha)✅ | 🟢 完成 |
| B3 | pywebview 原生視窗 | `packaging/run_desktop.py`:背景起 Gradio→等埠就緒→pywebview 開窗;失敗退回瀏覽器;app.py 抽 `launch_app()` 共用。smoke 驗過(伺服器+HTTP 200);**2026-06-18 凍結+安裝後目視:pywebview 視窗正常 render(標題列、三欄工作台、CPU/89 模型徽章)** ✅ | 🟢 完成 |
| B4 | PyInstaller 凍結 | `clinscope.spec` + `build_exe.ps1` ✅;**已建出 `dist/ClinScope/ClinScope.exe`(onedir ~1.9GB,CPU torch)**;app.py/weights.py 加凍結守衛(`sys.frozen`→`_MEIPASS`);smoke 驗過:凍結 exe 起伺服器+HTTP 200、**載入 89 模型 + manifest 90 檔、lazy-download 路徑活**;**2026-06-18 安裝後實測:凍結內選未下載模型→批次計算→真的從 HF 抓 `pbc_resnet50.pt`(90MB)、sha256 與 manifest 相符、CPU 跑 ResNet-50 出分類** ✅ | 🟢 完成 |
| B5 | Inno Setup 安裝程式 | `clinscope.iss` ✅ + 已裝 Inno Setup 6.7.3(`%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe`)+ **已編出 `packaging/dist/ClinScope-Setup.exe`(321 MB,lzma2 從 1.9GB 壓縮)**。每使用者安裝免 admin、開始選單+桌面捷徑、解除安裝保留模型快取。**2026-06-18 全鏈路實證:`/VERYSILENT` 安裝→開始選單啟動→pywebview 視窗→選未下載模型→`gr.Progress` 顯示下載 %→權重落 `%LOCALAPPDATA%\ClinScope\models`(sha256 對)→出結果** ✅ | 🟢 完成 |

### 接線點（已從程式碼確認）
- `REGISTRY`（app.py:46）每條 `key/kind/label/path/blurb`，FM 另帶 `fm`(joblib)。
- `AVAILABLE`（app.py:411）原本 `.exists()` 過濾 → 改 `or W.is_downloadable(m)`。
- 載入：`load_cls/load_yolo/load_sahi(spec["path"])` → 包 `W.ensure_local(spec)`；`load_fm(rel)` → `W.ensure_file(rel)`。
- 權重快取：`%LOCALAPPDATA%\ClinScope\models\<repo-rel-path>`（裝/移除不洗掉）。

### 設計決策
- 權重 host：**公開** HF repo（下載免 token）。
- 預設 **CPU-only torch**（最易裝）；GPU 版列後續增補說明，不擋主線。
- det 權重原散在 `reports/yolo/<key>/weights/best.pt`，manifest 與快取沿用 repo-relative 路徑鏡像。
- ⚠️ 本機無 symlink 權限 → `hf_hub_download` 必用 `local_dir=` 強制複製，不可走預設 symlink 快取。

### 風險
- **B4 PyInstaller × torch/gradio/ultralytics 凍結最易爆**（hidden imports / 前端靜態檔 / dll）。備案：embeddable Python + 首次自建環境的半凍結。
- 安裝包仍偏大（CPU torch + gradio ~300–700 MB）。
- 未簽章 exe 觸發 SmartScreen/防毒（買簽章憑證可根治，可不做）。
- 對外曝險（A4 必做驗證）。

---

## 打包腳本（packaging/）
- `gen_weights_manifest.py` — 掃本機權重產 `weights_manifest.json`（rel → hf_path/sha256/size）。
- `upload_weights_hf.py` — 推權重 + manifest 到公開 HF repo（需 `--yes` 才真上傳；先查 `list_repo_files` **跳過 Hub 上已存在的檔 → 可續傳**，中斷重跑只補沒傳完的）。
- `requirements-runtime.txt` — runtime 子集（CPU torch wheel index）。
- `run_desktop.py` — 桌面啟動器(B3):背景起 Gradio→等埠→pywebview 開窗,退路瀏覽器;`CLINSCOPE_DESKTOP_SMOKE=1` 無頭驗證。
- `clinscope.spec` — PyInstaller 規格(B4):collect_all 各重套件 + 帶專案資料/模組。
- `build_exe.ps1` — B4 一鍵建置:衍生乾淨 CPU venv→裝 runtime+pyinstaller→`pyinstaller clinscope.spec`。
- `clinscope.iss` — Inno Setup 6 腳本(B5):打包 dist 成每使用者安裝程式 `ClinScope-Setup.exe`(需 ISCC.exe 編譯)。
