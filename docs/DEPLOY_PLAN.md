# ClinScope 部署計畫（前端公開網址 + 本地安裝程式）

兩條獨立軌道。**A** 把本機 app 開成固定公開網址；**B** 把 app 打包成 Windows 安裝程式，
權重「選了才下載」。權重統一放 **公開** HF Hub repo：`tinghaochang/clinscope-weights`。

---

## 軌道 A — 買網域 + Cloudflare Tunnel（固定網址）

| # | 步驟 | 誰 | 狀態 |
|---|---|---|---|
| A0 | 買網域 ~$10/年（建議 Cloudflare Registrar，DNS 自動在 CF） | 使用者 | ⏳ 待購 |
| A1 | 裝 `cloudflared`、`tunnel login`、建 named tunnel | Claude | — |
| A2 | `clinscope.<網域> → http://localhost:7860`（config + DNS route） | Claude | — |
| A3 | `cloudflared service install` 開機自動跑 | Claude | — |
| A4 | 安全層：Cloudflare Access（限 email）或基本驗證 | Claude | — |

> A 卡在 A0。固定網址 ≠ 不間斷：本機關機/睡眠/cloudflared 沒跑就連不上。

---

## 軌道 B — Inno Setup 安裝程式（選了才下載）

使用者拿到 `ClinScope-Setup.exe`（只含程式 + CPU torch，**不含權重**）→ 安裝 → 開 app →
模型下拉列全部 89 個 → 選到未下載者跳「⬇ 下載」進度條 → 抓自 HF Hub → 存快取 → 驗 sha → 載入。

| # | 里程碑 | 內容 | 狀態 |
|---|---|---|---|
| B1 | runtime 子集 + CPU torch | `packaging/requirements-runtime.txt` ✅；乾淨 CPU-only venv 凍結驗證留待 B4 | 🟢 大致完成 |
| B2 | HF 權重 + lazy download | resolver `src/weights.py` ✅ + `weights_manifest.json`(90 檔/6.25GB) ✅ + app.py 接線 ✅ + 端到端推論驗證 ✅ ｜ **待**:HF 上傳(`--yes`) + 下載進度 UI | 🟡 進行中 |
| B3 | pywebview 原生視窗 | 啟動後開桌面視窗指向 127.0.0.1:7860，失敗退回瀏覽器 | — |
| B4 | PyInstaller 凍結 | spec 收 gradio/ultralytics/transformers/torch；無 Python 環境雙擊驗證 | — |
| B5 | Inno Setup 安裝程式 | 包 dist、開始選單/桌面捷徑、解除安裝、快取目錄 → `ClinScope-Setup.exe` | — |

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
- `upload_weights_hf.py` — 推權重 + manifest 到公開 HF repo（需 `--yes` 才真上傳）。
- `requirements-runtime.txt` — runtime 子集（CPU torch wheel index）。
