---
name: add-dataset-track
description: 在 ClinScope 新增一條臨床顯微辨識 track。分類走 ResNet（prepare_data → train_resnet → predict），偵測走 YOLO（prepare_yolo → train_yolo → predict_yolo），最後掛進 app.py 前端。觸發詞：新增資料集、加 track、train new dataset、加偵測模型、跑某資料集（DeFungi/PBC/HEp-2…）、補某菌種/某科別。涉及 src/ 六支腳本與 app.py。
---

# Add a dataset track to ClinScope

核心重複動作：拿一個新的臨床顯微影像資料集，跑出「資料集內成績 + 報表」，再掛進統一前端。
pipeline 已通用化，新 track 不必改核心程式，只要餵資料 + 在 app.py registry 加一條。

**先決定是哪一種 track：**
- **分類（classification）**：一張圖 = 一個目標要判類別 → ResNet。資料是 ImageFolder（`<class>/*.jpg`）。
- **偵測（detection）**：一張圖有多個目標要框出/計數 → YOLO。資料是框標註（VOC/COCO）。

## Step 0 — 找並「實測」下載源（最容易卡的一關）

公開資料的有無與可下載性才是真瓶頸。挑源時**一定先 HEAD 實測**，別假設連結活著：

```powershell
curl.exe -sIL --max-time 30 "<URL>" | Select-String -Pattern "HTTP/|Content-Length|Content-Type|location:"
```

實戰得到的源頭地圖：
- **Zenodo / 大檔**：單連線限速，用 aria2c 多連線（見下）。
- **UCI**：`https://archive.ics.uci.edu/static/public/<id>/<name>.zip`，純 zip、最可靠（DeFungi=773）。
- **Mendeley Data**：API 取直連 → `https://data.mendeley.com/public-api/datasets/<id>/files?folder_id=root&version=1`，解析 `content_details.download_url`（PBC=snkd93bnjr）。
- **Hugging Face dataset**：`https://huggingface.co/datasets/<owner>/<name>/resolve/main/<file>`，免註冊、會 302 到 CDN，aria2c 自動跟跳（HEp-2=Genius-Society/HEp2）。
- **憑證壞掉的學術站**：curl exit 35 → 加 `-k --tlsv1.2`；aria2c 加 `--check-certificate=false`（DIBaS 鏡像 doctoral.matinf.uj.edu.pl）。
- **要登入的牆**：IEEE Dataport、Heywhale、原 i3a 註冊站 → **跳過，換鏡像或別的資料集**，別硬鑽。
- **死掉/被搶註的網域**：原 DIBaS 站 misztal.edu.pl 已變博弈站 → 找學術鏡像（GitHub `gallardorafael/DIBaS-Dataset` 指到新 host）。

aria2c 多連線下載（10h→~30min）：
```powershell
$aria = (Resolve-Path "C:\Users\hao80\AppData\Local\Microsoft\WinGet\Packages\aria2.aria2_*\aria2-*\aria2c.exe").Path
& $aria -x16 -s16 -k1M --file-allocation=none --max-tries=5 --check-certificate=false -d "D:\Claude code\ClinScope\data" -o <name>.zip "<URL>"
```
多檔批次：把 URL 寫進一個清單檔，`-i list.txt -j4 -x8`。

## A. 分類 track（ResNet）

1. **切分** → `data/<name>_split/{train,val}/<class>/`
   ```powershell
   .\.venv\Scripts\python.exe src\prepare_data.py --src "D:\Claude code\ClinScope\data\<extracted_class_root>" --out "D:\Claude code\ClinScope\data\<name>_split" --val-frac 0.15
   ```
   - `--src` 直接指 class 母目錄（最穩）；或 `--zip`+`--subset <keyword>` 自動找含圖最多的子目錄。
   - **內建自動驗圖**（預設開）：開不了的壞檔會被剔除並印 `[verify] dropped N`。DIBaS、PBC 都靠它擋掉 corrupt 檔。要關用 `--no-verify`。
   - **務必絕對路徑**：CWD 是 `D:\Claude code` 不是專案根。

2. **訓練**
   ```powershell
   .\.venv\Scripts\python.exe -u src\train_resnet.py --data-dir "...\data\<name>_split" --epochs 15 --freeze-epochs 3 --batch-size 64 --workers 12 --out "...\models\<name>_resnet50.pt" --report-dir "...\reports\<name>" --title "<標題>"
   ```
   - two-phase 凍結→微調、class-weighted loss、AMP、`--title` 進混淆矩陣。
   - 大圖（如 DeFungi 500×500）解碼吃 CPU 會餓著 GPU → 開 `--workers 12`（i7-13700F 夠）。
   - epoch 數：大乾淨資料 12–15 夠；小/雜資料 20–40。

3. **推論 / 跨域實測**：`src\predict.py --model models\<name>_resnet50.pt --input <資料夾或單張> --topk 3`

## B. 偵測 track（YOLO，ultralytics 已裝）

1. **VOC→YOLO**（自動讀 ImageSets/Main 切分）：
   ```powershell
   .\.venv\Scripts\python.exe src\prepare_yolo.py --voc-root "...\data\<voc_root>" --classes RBC WBC Platelets --out "...\data\<name>_yolo"
   ```
   產出 `images/{train,val,test}/`、`labels/.../`、`data.yaml`。COCO 來源要先轉成這格式。
2. **訓練**：`src\train_yolo.py --data "...\data\<name>_yolo\data.yaml" --epochs 80 --batch 16 --name <name>` → `reports/yolo/<name>/weights/best.pt`，內建早停 + 全套圖。
3. **推論**：`src\predict_yolo.py --model reports\yolo\<name>\weights\best.pt --input <圖>` → 標註圖 + 每類計數。

## C. 掛進前端 app.py

在 `app.py` 的 `REGISTRY` 加一條（`kind`：`"cls"` 或 `"det"`），`path` 指向權重檔：
```python
{"key": "<key>", "kind": "cls", "label": "<emoji 中文 — 來源(染色)>",
 "path": ROOT / "models" / "<name>_resnet50.pt", "blurb": "<一句說明>"},
```
app 只列「權重檔存在」的模型。**改完要重啟 server 才生效**：先 `TaskStop` 舊的（或 port 7860 沒在聽就直接起），再 `python app.py`（背景），`curl http://127.0.0.1:7860/config` 驗 HTTP 200。
**注意**：別在某模型「訓練中」就重啟前端 —— 會去載被覆寫到一半的權重。

## 環境與坑（血淚）

- venv `.venv\`（Py 3.11）、torch 2.6.0+**cu124**、ultralytics、sklearn、gradio 6.x。
- **背景跑長任務**：下載/解壓/訓練一律 `run_in_background: true`，靠完成通知接續，不要前景 sleep 輪詢。
- **背景 python stdout 會 block-buffer**（看不到即時進度）→ 用 `python -u`。好處：即時可監控，且**程序若被砍 log 也留得住**。
- **電腦睡眠會把背景任務連同 gradio 一起砍掉**（曾靜默死掉一小時、0-byte log、無通知）。長跑前先 `powercfg /change standby-timeout-ac 0`，收工還原 `... 30`。
- Windows DataLoader 多 worker 需 `if __name__=="__main__"` guard——核心腳本都已有。
- Mac 來源的 zip 會夾 `__MACOSX/` 與 `._*` resource fork（副檔名像圖但 PIL 開不了）→ 解壓後刪掉，或交給 prepare_data 的驗圖剔除。

## 已驗證基準（in-distribution；跨到自己照片預期掉到 60–80%）

| track | 科別 | 引擎 | 資料 | 成績 |
|---|---|---|---|---|
| OpenFungi 真菌(LPCB) | 微生物 | ResNet | 571/6類 | val 0.9825（小 val） |
| DeFungi 真菌(KOH) | 微生物 | ResNet | 9114/5類 | val 0.905 |
| NIH Malaria | 寄生蟲 | ResNet | 27558/2類 | val 0.973 |
| PBC 白血球 | 血液 | ResNet | 17093/8類 | val 0.992 |
| DIBaS 細菌 | 微生物 | ResNet | 689/33類 | ⚠️ 小樣本 val=1.0 不可信 |
| HEp-2 ANA | 免疫 | ResNet | 13596/6類 | val 0.980 |
| BCCD 血球 | 血液 | YOLO | 364張 | mAP50 0.919 |

- 小驗證集（DIBaS 每類 4–5 張）數字噪音極大、且採集層級可能洩漏 → 別信單一 split，要 group-aware split / k-fold 才有意義。
