---
name: repackage
description: 把 ClinScope 重新打包/發佈成桌面安裝程式 + 更新雲端權重。模型有變動(加/移除 track)、或改了 app.py/weights.py 後要重出 ClinScope-Setup.exe 時用。觸發詞:重新打包、重發佈、更新安裝程式、rebuild installer、repackage、出新版安裝檔、把新模型發佈出去、更新 HF 權重。涉及 packaging/(gen_weights_manifest/upload_weights_hf/clinscope.spec/build_exe.ps1/clinscope.iss)、app.py、src/weights.py。背景與決策見 docs/DEPLOY_PLAN.md(Track B)。
---

# Repackage — 重新打包 ClinScope 桌面發佈版

把目前的程式 + 模型清單,重新產出**雲端權重(HF)**與**本地安裝程式 `ClinScope-Setup.exe`**。
模型一變(`add-models`/`add-dataset-track` 跑完)或核心程式改動,安裝版與 HF 權重就過時 → 跑這條。

完整設計/決策(為何兩軌、為何 CPU torch、快取位置等)見 `docs/DEPLOY_PLAN.md`,本 skill 只給**有序步驟 + 坑**。

## 前置
- HF **Write** token 已登入(查:`python -m huggingface_hub.cli.hf auth whoami`;沒登入見下方坑)。
- 已有乾淨 build venv `.venv-build`(沒有就 `build_exe.ps1` 會自動由 dev venv 衍生)。
- Inno Setup 6 已裝(`%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe`;沒裝:`winget install JRSoftware.InnoSetup --silent`)。

## 步驟(有序)

> 只改了程式、模型沒變 → 跳 1、2,直接 3 起。模型有增減 → 1 起全跑。

1. **重生 manifest**(模型/權重有變動才需):
   ```
   .venv/Scripts/python.exe packaging/gen_weights_manifest.py
   ```
   產 `packaging/weights_manifest.json`(rel → hf_path/sha256/size_mb/bytes)。

2. **上傳新權重到 HF**(public repo `tinghaochang/clinscope-weights`):
   ```
   PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -u packaging/upload_weights_hf.py --yes
   ```
   會先 `list_repo_files` **跳過已存在檔 → 可續傳**;只補新模型,不重傳舊的。建議 `run_in_background`。

3. **PyInstaller 打包**(從**乾淨 CPU venv**,別用 dev cu124 venv):
   - 重建前先清乾淨,避免檔案鎖(見坑):kill 殘留 `ClinScope.exe`、手動刪 `packaging/dist/ClinScope`。
   - 一鍵:`powershell -ExecutionPolicy Bypass -File packaging\build_exe.ps1`
     或手動:`.venv-build/Scripts/python.exe -m PyInstaller --noconfirm packaging/clinscope.spec --distpath packaging/dist --workpath packaging/build`
   - 產 `packaging/dist/ClinScope/ClinScope.exe`(onedir ~1.9GB)。

4. **smoke 驗證凍結 exe**(務必做,否則只知「建得出來」不知「跑得起來」):
   ```
   cd packaging/dist/ClinScope
   CLINSCOPE_DESKTOP_SMOKE=1 CLINSCOPE_SMOKE_MARKER=<某路徑>/_mark.json ./ClinScope.exe
   cat <某路徑>/_mark.json
   ```
   marker 的 `available`/`registry` 要等於目前模型數、`manifest_files` 對、`sample_downloadable:true`、`root` 指向 `_internal`。不對 → 多半是 spec 的 datas 漏帶或 ROOT 守衛問題。

5. **編安裝程式**:
   ```
   cd packaging && "C:/Users/<user>/AppData/Local/Programs/Inno Setup 6/ISCC.exe" clinscope.iss
   ```
   產 `packaging/dist/ClinScope-Setup.exe`(lzma2,~321MB)。

6. **(可選)端到端實證**:`ClinScope-Setup.exe /VERYSILENT /DIR=<temp> /NOICONS` → 跑安裝後的 exe smoke → 讀 marker → `unins000.exe /VERYSILENT` 清掉。確認安裝鏈路真的可用。

## 坑(都實際踩過)
- **絕對從乾淨 CPU venv 建**:用 dev 的 `+cu124` venv 會把 ~2.5GB CUDA 打進包,文書機也用不到。`requirements-runtime.txt` 走 `download.pytorch.org/whl/cpu`。
- **重建 PermissionError WinError 32**:PyInstaller COLLECT 會 rmtree `dist/ClinScope`,但剛跑過的 exe / 防毒 / Explorer 會鎖住 → 先 `Stop-Process -Name ClinScope -Force` + 手動 `Remove-Item -Recurse -Force packaging\dist\ClinScope` 再建。
- **凍結路徑**:`app.py`/`weights.py` 已用 `sys.frozen`→`sys._MEIPASS` 算 ROOT;資料檔(`model_cards/`、`packaging/weights_manifest.json`、`src/*.py`、`app.py`)由 `clinscope.spec` 的 `datas` 帶進 `_internal/`。改了資料路徑要兩邊同步。
- **gradio 凍結**:spec 內 `module_collection_mode={"gradio":"py","gradio_client":"py"}` 必留(gradio 執行期讀自身原始碼)。
- **HF 登入 shim 壞**:改名 FungiScope→ClinScope 後 `huggingface-cli.exe`/`hf.exe` 寫死舊路徑會 `Fatal error in launcher` → 用 `python -m huggingface_hub.cli.hf auth login --force`(新版模組在 `huggingface_hub.cli.hf`)。Write token 才能上傳(Read 會 403)。
- **windowed exe 無 console**:`run_desktop.py` 已補 None-stdout 替身 + SMOKE marker 檔(windowed 看不到 stdout,靠 marker 驗證)。
- **dist/installer 大**:`packaging/dist/`、`ClinScope-Setup.exe`、`.venv-build/` 都已 gitignore,不進版控。
