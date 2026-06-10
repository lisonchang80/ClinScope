---
name: add-dataset-track
description: 在 FungiScope 新增一條資料集 track（下載 → ImageFolder 切分 → ResNet-50 訓練 → 推論）。觸發詞：新增資料集、加 track、train new dataset、跑 DeFungi/macro、補某菌種。涉及 src/prepare_data.py、src/train_resnet.py、src/predict.py。
---

# Add a dataset track to FungiScope

FungiScope 的核心重複動作：拿一個新真菌影像資料集，跑出「資料集內準確率 + 混淆矩陣」。
pipeline 已通用化，新 track 不必改程式，只要餵不同資料即可。

## 流程（4 步）

1. **下載資料集** → `data/<name>.zip`
   - Zenodo / 大檔：**一定用 aria2c 多連線**，不要用 curl（Zenodo 單連線限速 ~225KB/s）：
     ```powershell
     $aria = "C:\Users\hao80\AppData\Local\Microsoft\WinGet\Packages\aria2.aria2_*\aria2-*\aria2c.exe"
     & (Resolve-Path $aria) -x16 -s16 -k1M --file-allocation=none -d data -o <name>.zip <URL>
     ```
     16 連線可把 10h 壓到 ~30min。aria2c 已用 winget 裝（`winget install aria2.aria2`）。

2. **整理成 ImageFolder** → `data/<name>_split/{train,val}/<class>/`
   ```powershell
   .\.venv\Scripts\python.exe src\prepare_data.py --zip data\<name>.zip --subset <keyword> --out data\<name>_split
   ```
   - `--subset` 是關鍵字（micro/macro/...），自動找「名稱含該字、且底下有最多含圖子資料夾」的目錄當 class root
   - 結構不規則就用 `--src` 直接指定 class 母目錄
   - **務必傳絕對路徑**給 `--zip/--out`：本機 CWD 是 `D:\Claude code`（不是專案根），相對路徑會寫錯位置

3. **訓練 ResNet-50**
   ```powershell
   .\.venv\Scripts\python.exe src\train_resnet.py --data-dir data\<name>_split --epochs 30 `
     --out models\<name>_resnet50.pt --report-dir reports\<name>
   ```
   - two-phase：前 `--freeze-epochs`(預設5) 凍結 backbone 只訓 head，之後全網微調
   - 已內建 class-weighted loss（對付不平衡）、AMP（8GB VRAM 夠）
   - 產出：best 權重 + `classification_report.txt` + `confusion_matrix.png` + `summary.json`

4. **推論 / 跨域實測**
   ```powershell
   .\.venv\Scripts\python.exe src\predict.py --model models\<name>_resnet50.pt --input <資料夾或單張> --topk 3
   ```

## 環境與坑

- venv：`.venv\`（Python 3.11），torch 2.6.0+**cu124**（驅動支援到 CUDA 13.2）；裝法見 requirements.txt 開頭註解
- Windows DataLoader 多 worker 需 `if __name__=="__main__"` guard——train_resnet.py 已有，新腳本照辦
- 長時間的下載/解壓/訓練一律 `run_in_background: true`，靠完成通知接續，不要前景 sleep 輪詢

## 已驗證基準

- OpenFungi 顯微（571 張、6 類、嚴重不平衡）→ best val_acc **0.9825**（in-distribution）
- 注意：小驗證集（Rhizopus n=2 等）數字誤差大；in-distribution ≠ 你自己照片，跨域預期掉到 60–80%
