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
- **Hugging Face dataset（單檔 zip/檔案）**：`https://huggingface.co/datasets/<owner>/<name>/resolve/main/<file>`，免註冊、會 302 到 CDN，aria2c 自動跟跳（HEp-2=Genius-Society/HEp2）。
- **Hugging Face parquet 資料集（`format:parquet`、圖內嵌、非 zip）**：先 `/api/datasets/<id>` 確認 `gated:false`、再 `/api/datasets/<id>/tree/main?recursive=true` 列出 `data/*.parquet`,aria2c 抓那些 parquet（一檔一 process），用 `src/hf_parquet_to_imagefolder.py` 解碼成 ImageFolder（`pip install datasets pyarrow`;⚠️ 見下方 import 順序坑）。類別名串流偷看一次:`load_dataset(id, split='train', streaming=True).features['label'].names`,再以 `--class-names-json` 傳入,`--max-side 512` 縮圖省碟。Chula-ParasiteEgg-11=`Abdelkareem/parasitic-egg`（11 類蟲卵,1000/類）。**官方已切 train/val/test 的**(如 Matek 骨髓 `ekim15/bone_marrow_cell_dataset`,21 類、病人層級)→ 各 split 的 parquet 分別解到 `<out>/train`、`<out>/val` 子夾(直接餵 train_resnet，免 prepare_data 隨機切、避免病人洩漏)。⚠️ **這條會踩的 4 個坑**:(a) 有些 mirror 把 HF Image 欄**扁平成頂層 `bytes`/`path` 兩欄**,pyarrow 直讀的 `hf_parquet_to_imagefolder.py` 會 `KeyError 'label'/'bytes'` → 改用 **`src/hf_dataset_to_imagefolder.py`**(走 datasets 庫 `load_dataset`,什麼編碼都吃;MHIST `mamunrobi35/mhist_binary` 踩過,且它的 `validation` split 無 label→只解 train 自切);(b) **動態取檔名一定 `tr -d '\r'`** —— Windows venv python stdout 帶 `\r`,接到 resolve URL 尾變 `%0D`→404;(c) **`/tree/main` API 預設只回 ~1000 筆**,別用它估大小(甲狀腺 `cupofmocha/cytoal-thyroid-cytology-labeled` 看似 991 張、實際 27k);(d) 多軸類別 organ×label 用 `--group-col`+`--group-names-json` 還原(LC25000 → lung_n/lung_aca/lung_scc/colon_n/colon_aca)。
- **憑證壞掉的學術站**：curl exit 35 → 加 `-k --tlsv1.2`；aria2c 加 `--check-certificate=false`（DIBaS 鏡像 doctoral.matinf.uj.edu.pl）。
- **要登入的牆**：IEEE Dataport、Heywhale、原 i3a 註冊站 → **跳過，換鏡像或別的資料集**，別硬鑽。
- **Google Drive「散檔資料夾」(非單一 zip)**：`gdown --folder` 雖能列完上千檔(6.x 已無 50 檔上限),但**逐檔 auth 握手 ~0.3 檔/秒**,上萬個小檔要 9+ 小時 → 等同軟牆,當「要登入的牆」處理:找單一 zip 鏡像(Kaggle/Zenodo/HF)或換資料集,別硬抓(尿沉渣 GitHub `174614361` 的 Drive 散檔踩過,改做 Mendeley 的 SMIDS)。
- **死掉/被搶註的網域**：原 DIBaS 站 misztal.edu.pl 已變博弈站 → 找學術鏡像（GitHub `gallardorafael/DIBaS-Dataset` 指到新 host）。

aria2c 多連線下載（10h→~30min）：
```powershell
$aria = (Resolve-Path "C:\Users\hao80\AppData\Local\Microsoft\WinGet\Packages\aria2.aria2_*\aria2-*\aria2c.exe").Path
& $aria -x16 -s16 -k1M --file-allocation=none --max-tries=5 --check-certificate=false -d "D:\Claude code\ClinScope\data" -o <name>.zip "<URL>"
```
**多檔一定要「一檔一個 aria2c process」**（迴圈逐一呼叫）。不要把多個 URL 塞進同一次呼叫、也不要用 `-j` 併發 —— 同一 process 內 aria2c 會把**前一檔的 Content-Length 套到下一檔的 range request**，報 `Invalid range header`，只有第一檔成功（SIPaKMeD 5 檔踩過）。再加 `--max-tries=10 --retry-wait=10` 抗網路抖動；被中斷的檔會留 `.7z.aria2` control file → 用 `--continue=true` 續傳即可。

**長跑 / 會切網路時用 switch-proof 旗標**:`--max-tries=0`(無限重試) `--retry-wait=15 --timeout=60 --connect-timeout=60 --lowest-speed-limit=1K` —— 撐得過熱點↔家用 wifi 交接的斷網(停滯連線被丟棄重連、永不放棄),配 `--continue=true` 從 `.aria2` 控制檔續傳,不掉已下載 bytes。⚠️ **HF `snapshot_download`/`load_dataset` 在切網瞬間會卡死 socket 且不報錯**(progress 停住、`HF_HUB_DOWNLOAD_TIMEOUT` 沒救、也不會發完成通知),所以**會跨越切網的長下載一律改 aria2c 逐檔**(用 `/tree` API 拿確切檔名→逐檔 aria2c);只有小而穩的下載才用 snapshot_download。被卡死要 `TaskStop` + `taskkill //F //IM aria2c.exe`(清 orphan)再重啟。

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
   - **極端不平衡**(如 Matek 骨髓 21 類、最稀有類僅 6 張):預設反頻率 class-weight 會給稀有類 ~1000× 權重、把常見類拖垮 → 用 `--class-weight sqrt`(較溫和,1088×→33×)或 `none`;預設 `inverse` 維持其他 track 不變。啟動會印 `[loss] class-weight=... (min=.. max=..)` 可核對。

3. **推論 / 跨域實測**：`src\predict.py --model models\<name>_resnet50.pt --input <資料夾或單張> --topk 3`

## B. 偵測 track（YOLO，ultralytics 已裝）

1. **VOC→YOLO**（自動讀 ImageSets/Main 切分）：
   ```powershell
   .\.venv\Scripts\python.exe src\prepare_yolo.py --voc-root "...\data\<voc_root>" --classes RBC WBC Platelets --out "...\data\<name>_yolo"
   ```
   產出 `images/{train,val,test}/`、`labels/.../`、`data.yaml`。
   **COCO 來源**(Roboflow/RF100 等,每 split 一個 `_annotations.coco.json` + 圖)→ 用 **`src/coco_to_yolo.py --coco-root <dir> --out data\<name>_yolo --splits "train:train,valid:val"`**(自動跳過 RF100 的 id-0 placeholder supercategory、bbox→歸一化 YOLO、寫 data.yaml;mitosis `Francesco/mitosis-gjs3g` 用過)。tar 解壓注意下方 Windows 冒號坑。
2. **訓練**：`src\train_yolo.py --data "...\data\<name>_yolo\data.yaml" --epochs 80 --batch 16 --name <name>` → `reports/yolo/<name>/weights/best.pt`，內建早停 + 全套圖。⚠️ **ultralytics 把輸出存到 cached `runs_dir`** —— 改名專案後它可能殘留舊名(本機殘留 `FungiScope`,YOLO 權重跑到 `D:\Claude code\FungiScope\runs\detect\reports\yolo\<name>\`)。訓完**先確認 `reports/yolo/<name>/weights/best.pt` 真在那**,不在就從 `runs_dir` 搬回、並 `python -c "from ultralytics.utils import SETTINGS; SETTINGS.update({'runs_dir': r'D:/Claude code/ClinScope/runs'})"` 修掉。base 權重(`yolo*.pt`)會掉在 repo 根 → 已 gitignore。
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

- venv `.venv\`（Py 3.11）、torch 2.6.0+**cu124**、ultralytics、sklearn、gradio 6.x、datasets+pyarrow（HF parquet 用）。
- **⚠️ import 順序 segfault 地雷(2026-06 裝 `datasets` 後)**:`pip install datasets` 把 numpy 升到 2.x → **先 import torch、再 import sklearn/matplotlib 會 segfault(exit 139)**(原生 OpenMP/MKL 載入順序衝突;`--workers 0`、`KMP_DUPLICATE_LIB_OK=TRUE` 都救不了)。症狀:啟動 ~4s 內崩、stdout 幾乎沒印就只剩 bash 的 `Segmentation fault` 行。**修法:每支會同時用到 torch 和 sklearn/matplotlib 的腳本,把 matplotlib/sklearn 移到 torch 之前 import**(`train_resnet.py`/`eval_external.py` 已改成 sklearn-block-在前;`train_yolo.py` 在 `from ultralytics import YOLO` 前先 `import matplotlib`)。一行驗證:`python -c "import sklearn.metrics, matplotlib; import torch; print('ok')"`(sklearn 先=OK)對照反序(=139)。
- **背景跑長任務**：下載/解壓/訓練一律 `run_in_background: true`，靠完成通知接續，不要前景 sleep 輪詢。
- **背景 python stdout 會 block-buffer**（看不到即時進度）→ 用 `python -u`。好處：即時可監控，且**程序若被砍 log 也留得住**。
- **背景長任務會被靜默砍掉**（無完成/失敗通知）。2026-06 查 Kernel-Power event log，真兇是**重開機**而非睡眠：6/11 `TrustedInstaller`/`MoUsoCoreWorker` 的 Windows Update 自動升級、6/12 `StartMenuExperienceHost` 的「更新並重新啟動」（log 完全無「事件 42 進入睡眠」）。本機 AC 睡眠/休眠早就 0x0、無 Modern Standby，**關螢幕不殺進程**。對策：① 長跑前**暫停 Windows Update**（最關鍵）② 訓練中別點「更新並重新啟動」③ 永久擋自動重開：`HKLM\...\WindowsUpdate\AU\NoAutoRebootWithLoggedOnUsers=1`（需管理員）④ 防睡眠保險：訓練開頭 `ctypes.windll.kernel32.SetThreadExecutionState(0x80000000|0x1)`。**判死活：比對輸出檔 mtime vs 現在**（停在某 epoch 且 mtime 很久沒動 = 已死）。被砍時 `train_resnet` 每次 val 進步即存 best checkpoint → 模型不必重跑，用 `cross-domain-eval` / `eval_external.py` 在 val split 上補報表。
- Windows DataLoader 多 worker 需 `if __name__=="__main__"` guard——核心腳本都已有。
- Mac 來源的 zip 會夾 `__MACOSX/` 與 `._*` resource fork（副檔名像圖但 PIL 開不了）→ 解壓後刪掉，或交給 prepare_data 的驗圖剔除。
- **Windows `tar` 把 `D:/x.tar.gz` 的冒號當「遠端主機 D:」**（`tar: Cannot connect to D: resolve failed` / `gzip: unexpected end of file`）→ **`cd` 進該資料夾用相對路徑解壓**（`( cd "$ROOT/data" && tar xzf foo.tar.gz -C dest )`），或加 `--force-local`（mitosis COCO tar.gz 踩過）。
- **壓縮檔不是 zip**（`.7z`/`.rar`）→ `.7z` 用 `pip install py7zr`；**`.rar` 本機直接用 `"/c/Program Files/WinRAR/UnRAR.exe" x -o+ -y <a>.rar <dest>/`**（WinRAR 已裝、免 python module;HuSHeM `tt3yj2pf38` 用過）。解完用 `--src` 指 class 母目錄（找法:`CR=$(dirname "$(dirname "$(find <raw> -iname '*.bmp'|head -1)")")` 自動定位,免假設夾名）。解完**注意巢狀結構**：SIPaKMeD 每類同時有 cluster 全圖（`NNN.bmp`）+ `CROPPED/` 單細胞子夾，`collect()` 會 rglob 全抓 → 想要單一尺度就先把目標那組（如 CROPPED 4049 單細胞）複製成乾淨 ImageFolder 再切分。解完的中繼大檔（raw/extracted）記得清掉（SIPaKMeD raw+extracted 共 ~14GB）。
- **下載後先驗「真的有 class 標籤」再訓**（batch4 兩次教訓,各浪費一次下載）：標榜 cls 的 HF/zenodo 集可能 (a) 根本沒標籤——只是按片分的未標 patch（KidneyAI 只有 glomeruli patch、檔名=`slide-idx-bbox`）,或 (b) 只有自由文字 caption、無離散 label、甚至多器官混（`arch_gastric` features=image+text、sample 是骨肉瘤非胃）。**下載前/後先驗：資料夾是否 class 子夾 / HF `/info` features 是否含 `ClassLabel` / csv 是否有可離散化欄**;沒有就 skip 換候選。同理「YOLO 偵測」集可能是分類集 re-host（label 全是整圖退化框 `0 0.5 0.5 1 1`）→ 訓前 `cat` 一個 label 檔。**det/大檔 track 一律先 download+inspect（印 tree/副檔名/`unzip -l`/sample label）再寫 prep**,別盲寫。
- **自帶官方 k-fold / train-val-test 就直接用**：MK-11 `MK-11_5folds/fold_1/{train,val,test}`、urine/Tryp/BACH 都附官方 split → 省自切且多為病人級（無洩漏）。沒 metadata 也能病人級切——**檔名前綴 token 常就是 patient/slide id**。
- **無洩漏切分 helper（優先用,別預設隨機 split）**：`hf_parquet_group_split.py`(parquet 按 group 欄位病人/片) / `patchgastric_wsi_split.py`(扁平 patch+caption-csv→WSI 級,stream zip) / `bach_tile_group_split.py`(大圖 tile+影像級) / `ipdbrain_tile_split.py`(大圖 tile+病人級,檔名 token 分組) / `hf_parquet_to_imagefolder.py`(扁平 ImageFolder) / `hf_dataset_to_imagefolder.py`(datasets 庫救 flatten schema)。**組織分級類(PatchGastric 0.49 / IPD-Brain 0.53)在無洩漏切分下本就低分,是誠實非 bug**;塌成多數類(IPD-Brain→GBM)代表低倍 tile 失判別訊號。
- **多檔/動態檔名 URL 同中 `\r`→`%0D` 404**：從 Windows venv python 印出的檔名清單接到 URL 前一律 `tr -d '\r'`（MLL23 18 個 zenodo zip 清單踩過,同 HF 檔名坑）。
- **訓練被外部中止再現**（IPD-Brain ep10 無 traceback,疑 GPU TDR/重開）→ 判死活：**`tasklist|grep python` 無進程 + 輸出 log 凍結在某 epoch**。best.pt 逐 epoch 存→權重在,但 `classification_report` 最後才寫故**需重跑完整才有報表**(或用 eval_external 補)。

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
| Chula-ParasiteEgg-11 蟲卵 | 寄生蟲 | ResNet | 11000/11類 | val 0.995 |
| SIPaKMeD 子宮頸抹片 | 細胞學 | ResNet | 4049/5類 | val 0.970 |
| SMIDS 精液形態 | 男科 | ResNet | 3000/3類 | val 0.902 |
| Matek 骨髓細胞 | 血液 | ResNet | 137k+17k/21類 | val 0.865 weighted（macro-F1 0.71,極不平衡，用 `--class-weight sqrt`） |
| LC25000 肺/結腸 | 組織病理 | ResNet | 25000/5類 | val 1.00 ⚠️1250 原圖增強擴增、隨機 split 洩漏 |
| BreakHis 乳腺 | 組織病理 | ResNet | 7909/2類 | val 0.978 ⚠️40-400×混訓、隨機 split 病人洩漏 |
| NCT-CRC-HE-100K 結直腸 | 組織病理 | ResNet | 100k/7k官方 cohort/9類 | val 0.970（官方外部 val→誠實;STR↔MUS 互混） |
| PatchCamelyon 淋巴結 | 組織病理 | ResNet | 262k/官方 split/2類 | val 0.913（96px,binary 轉移） |
| MHIST 大腸息肉 | 組織病理 | ResNet | 3152/2類 | val 0.922（HP vs SSA 診斷）⚠️同片洩漏 |
| 甲狀腺 FNA | 細胞學 | ResNet | 27496/6類 | val 0.947（1-6 疑 Bethesda）⚠️同片洩漏 |
| Mitosis 偵測 | 組織病理 | YOLO | 274/1類 | mAP50 0.941（COCO→YOLO,小資料 demo） |
| HuSHeM 精子頭 | 男科 | ResNet | 216/4類 | val 0.94 ⚠️極小 val(每類 8)不可信 |

> 此表為早期 19-track 快照。**batch3/4 另加 16 條,完整 35 模型以 `app.py` REGISTRY 為準**:血液細分(RBC 11類異形 / erythrocytesIDB 鐮刀 / UNIVALI 14類 / MLL23 18類含毛細胞白血病 / MK-11 巨核 11亞型 / rouleaux 串錢狀)、白血病 C-NMC、瘧種 MP-IDB、錐蟲 Tryp(det)/血液原蟲 3類、尿沉渣 urine(det,🆕第8科尿液分析)、核偵測 PanNuke(det)、組織病理(BACH 乳腺4級 / PatchGastric 胃4級 / IPD-Brain 腦膠質瘤 / PathoTiroid 甲狀腺)。**其中 6 條無洩漏(C-NMC/BACH/PathoTiroid/MK-11/PatchGastric/IPD-Brain);PatchGastric 0.49 / IPD-Brain 0.53 是誠實低分(無洩漏+硬任務)。**

- 小驗證集（DIBaS 每類 4–5 張、HuSHeM 每類 8 張）數字噪音極大、且採集層級可能洩漏 → 別信單一 split，要 group-aware split / k-fold 才有意義。
- **組織病理/細胞學的洩漏特別常見**:同一 WSI / 同片抹片切出多 tile-crop,隨機 split 會把同源 tile 分到 train/val → val 偏高(LC25000 增強擴增是極端案例 val=1.0)。誠實作法是 patient/slide-level split;沒做就在 blurb 掛 caveat。
