# ClinScope

臨床顯微影像辨識平台 —— 把多種醫檢顯微辨識任務收進**同一個前端**：32 個模型、跨醫檢 8 科、12 種成像模態、**ResNet 分類 + YOLO 偵測雙引擎**。
自學 / 作品集性質，**非臨床診斷工具**。（原名 FungiScope，從純真菌 PoC 擴充而來。）

## 現實定位（先讀這段）

這個專案誠實的價值在「**流程與模型能力的展示**」，不是臨床對齊。幾個必須先知道的限制：

- **跨域落差最大**：每個模型只在它自己資料集的分布內準確。換成你自己顯微鏡拍的照片（不同染色法 / 相機 / 倍率），準確率預期掉到 **60–80%**（同 in-distribution vs out-of-distribution 的老坑）。
- **無可下載的臨床「菌種級」模型**：市面上沒有現成可下載的臨床菌種辨識模型，只能自訓；而 LPCB 種級（如皮癬菌 Trichophyton）的公開資料缺口是**結構性的**（鎖在醫院）。
- **成像模態不可混訓**：LPCB 培養 / KOH 直接鏡檢 / Giemsa 血抹片 / Gram 染色 / 免疫螢光（IIF）各自是獨立 track，不能混在一起訓練。
- **小驗證集數字不可信**：DIBaS 每類僅 ~20 張，val=1.0 是採集洩漏 + 噪音造成的假象，需要 group-aware split / k-fold 才有意義。

## 已收錄模型（皆 in-distribution 成績）

| 科別 | Track | 模態 / 染色 | 引擎 | 資料 | 成績 |
|---|---|---|---|---|---|
| 微生物（真菌） | OpenFungi | LPCB 培養 | ResNet-50 | 571 張 / 6 屬 | val 0.98 ⚠️ 小 val |
| 微生物（真菌） | DeFungi | KOH 直接鏡檢 | ResNet-50 | 9,114 張 / 5 類 | val 0.905 |
| 微生物（細菌） | DIBaS | Gram 染色 | ResNet-50 | 689 張 / 33 種 | ⚠️ val=1.0 小樣本不可信 |
| 寄生蟲 | NIH Malaria | Giemsa 血抹片 | ResNet-50 | 27,558 張 / 2 類 | val 0.973 |
| 寄生蟲 | Chula-ParasiteEgg-11 | 糞便濕抹片 | ResNet-50 | 11,000 張 / 11 類 | val 0.995 |
| 血液 | PBC 白血球分類 | Wright/Giemsa | ResNet-50 | 17,093 張 / 8 類 | val 0.992 |
| 血液 | BCCD 血球偵測 | 血抹片 | YOLOv8 | 364 張 | mAP50 0.919 |
| 免疫 | HEp-2 ANA 型態 | 免疫螢光 IIF | ResNet-50 | 13,596 張 / 6 型態 | val 0.980 |
| 細胞學 | SIPaKMeD 子宮頸抹片 | Pap 染色（單細胞） | ResNet-50 | 4,049 細胞 / 5 類 | val 0.970 |
| 男科（精液分析） | SMIDS 精液形態 | 精液抹片（亮場） | ResNet-50 | 3,000 張 / 3 類 | val 0.902 |
| 血液（骨髓細胞學） | Matek 骨髓細胞 | May-Grünwald-Giemsa（骨髓抹片） | ResNet-50 | 154,239 張 / 21 類 | val 0.865 weighted ⚠️ macro-F1 0.71 |
| 組織病理（肺/結腸） | LC25000 | H&E 組織切片 | ResNet-50 | 25,000 張 / 5 類 | val 1.00 ⚠️ 增強擴增、隨機 split 洩漏 |
| 組織病理（乳腺） | BreakHis | H&E 組織切片（40–400×） | ResNet-50 | 7,909 張 / 2 類 | val 0.978 ⚠️ 隨機 split 病人洩漏 |
| 組織病理（結直腸） | NCT-CRC-HE-100K | H&E 組織切片（224px） | ResNet-50 | 100,000 張 / 9 類 | val 0.970（官方外部 7K cohort） |
| 組織病理（淋巴結） | PatchCamelyon | H&E 組織切片（96px patch） | ResNet-50 | 262,144 張 / 2 類 | val 0.913（官方 val 32k） |
| 組織病理（大腸息肉） | MHIST | H&E 組織切片 | ResNet-50 | 3,152 張 / 2 類 | val 0.922（HP vs SSA 診斷）|
| 細胞學（甲狀腺 FNA） | CytoAtlas 甲狀腺 | 細胞學抹片（FNA） | ResNet-50 | 27,496 張 / 6 類 | val 0.947 ⚠️ 同片洩漏 |
| 組織病理（乳腺 mitosis） | Mitosis 偵測 | H&E 組織切片 | YOLOv8 | 274 張 / 1 類 | mAP50 0.941 |
| 男科（精子頭形態） | HuSHeM | Diff-Quik 精子頭 | ResNet-50 | 216 張 / 4 類 | val 0.94 ⚠️ 極小 val 不可信 |
| 血液（紅血球形態） | RBC morphology | 血液抹片（單顆 RBC） | ResNet-50 | 6,240 張 / 11 類 | val 0.978、macro-F1 0.972 |
| 血液腫瘤（白血病） | C-NMC 2019 | 血液抹片（單細胞） | ResNet-50 | 10,311 張 / 2 類 | val 0.853 ✅ 病人級切分、無洩漏 |
| 寄生蟲（瘧原蟲種級） | MP-IDB | 薄血膜 Giemsa | ResNet-50 | 1,571 張 / 4 種 | val 0.997 ⚠️ 不平衡 + img/crop 同視野洩漏 |
| 血液（鐮刀型紅血球） | erythrocytesIDB | 血液抹片（單顆 RBC） | ResNet-50 | 626 張 / 3 類 | val 0.984、macro-F1 0.984 ⚠️ val 偏小 |
| 組織病理（細胞核偵測） | PanNuke | H&E 組織切片 | YOLOv8 | 7,901 圖 / 5 核類 | mAP50 0.654（val 6.7 萬框） |
| 血液（白血球擴充） | UNIVALI | 周邊血抹片 | ResNet-50 | 10,697 張 / 14 類 | val 0.905、macro-F1 0.816 ⚠️ 未成熟稀有類弱 |
| 組織病理（乳腺嚴重度） | BACH ICIAR2018 | H&E 組織切片（tile） | ResNet-50 | 400 圖→4,800 tile / 4 級 | val 0.873（影像級切分、無 tile 洩漏） |
| 組織病理（甲狀腺 PTC） | PathoTiroid PTIC | H&E 組織切片 | ResNet-50 | 1,006 張 / 2 類 | val 0.994 ✅ 病人級切分（⚠️ 僅 5 病人） |
| 尿液分析（尿沉渣） | Urine particles | 明場尿沉渣 | YOLOv8 | 4,250 圖 / 7 類 | mAP50 0.896 |
| 血液（單細胞 18 類） | MLL23 | Giemsa 單 WBC | ResNet-50 | 41,621 張 / 18 類 | val 0.948、macro-F1 0.861（含毛細胞白血病） |
| 血液（RBC 排列） | Rouleaux | Giemsa 全視野薄血膜 | ResNet-50 | 1,544 張 / 2 類 | val 1.0 ⚠️ 任務易+val 小，恐受試者洩漏 |
| 血液/骨髓（巨核細胞） | MK-11 | Wright-Giemsa 單細胞 | ResNet-50 | 7,204 張 / 11 亞型 | val 0.810 ✅ 病人級切分 |
| 寄生蟲（錐蟲） | Tryp | 未染色厚血膜 | YOLOv8 | 3,166 圖 / 1 類 | mAP50 0.807 |

> 分類模型回傳 top-k 機率；偵測模型回傳標註圖 + 每類計數。`app.py` 只會列出**權重檔存在**的模型，所以即使某些 track 還沒訓練好，前端也能正常啟動。

## 架構

```
分類 track：prepare_data.py → train_resnet.py → predict.py
偵測 track：prepare_yolo.py → train_yolo.py → predict_yolo.py
統一前端  ：app.py（Gradio, 127.0.0.1:7860）
```

pipeline 已通用化，新增一條 track 不必改核心程式 —— 只要備好資料、訓練、在 `app.py` 的 `REGISTRY` 加一條。完整流程與踩過的坑見 `.claude/skills/add-dataset-track/`。

## 環境

- Python 3.11 venv（`.venv/`），不用 conda
- GPU：RTX 4060 Ti 8GB，PyTorch **CUDA 12.4** wheel
- 主要套件：torch / torchvision、ultralytics（YOLO）、scikit-learn、gradio、matplotlib、pillow、tqdm

```powershell
# torch/torchvision 先從 cu124 index 裝，其餘走 requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

執行一律用 `.\.venv\Scripts\python.exe`，並傳**絕對路徑**（本機 CWD 是 `D:\Claude code` 不是專案根）。

## 執行

```powershell
# 啟動統一前端（上傳圖 → 選模型 → 辨識）
.\.venv\Scripts\python.exe app.py
# 然後開 http://127.0.0.1:7860

# 訓練一條分類 track（範例）
.\.venv\Scripts\python.exe -u src\train_resnet.py --data-dir data\<name>_split --epochs 15 --title "<標題>"

# 對單張或整夾推論
.\.venv\Scripts\python.exe src\predict.py --model models\<name>_resnet50.pt --input <圖或資料夾> --topk 3
```

## 目錄

- `app.py` — Gradio 統一前端，含 11 條 track 的 model registry
- `src/prepare_data.py` — 解壓 / 切分 ImageFolder（內建自動驗圖剔除壞檔）
- `src/train_resnet.py` — ResNet-50 transfer learning（two-phase 凍結→微調、class-weighted loss、AMP）
- `src/predict.py` — 分類推論 / 跨域實測
- `src/prepare_yolo.py` — Pascal VOC → YOLO 格式轉換
- `src/train_yolo.py` — YOLOv8 偵測訓練（ultralytics，內建早停）
- `src/predict_yolo.py` — YOLO 推論：標註圖 + 每類計數
- `data/`、`models/`、`reports/`、`.venv/` — 已 gitignore，不進版控

## 資料來源

| Track | 來源 |
|---|---|
| OpenFungi | MDPI *Life* 2025（DOI 10.3390/life15071132），Zenodo 10.5281/zenodo.15692069 |
| DeFungi | UCI Machine Learning Repository（id 773） |
| NIH Malaria | NIH / NLM Cell Images for Malaria Detection |
| Chula-ParasiteEgg-11 | Hugging Face `Abdelkareem/parasitic-egg`（parquet 內嵌圖，11 類腸道蟲卵） |
| PBC（白血球） | Mendeley Data `snkd93bnjr` |
| DIBaS | 33 菌種 Gram 染色，學術鏡像 doctoral.matinf.uj.edu.pl |
| HEp-2 | Hugging Face `Genius-Society/HEp2` |
| BCCD | Blood Cell Count and Detection（Pascal VOC） |
| SIPaKMeD | Univ. of Ioannina `cs.uoi.gr/~marina/SIPAKMED`（5× `.7z`，單細胞 CROPPED 子集） |
| SMIDS | Mendeley Data `6xvdhc9fyb`（精液形態 3 類:正常/異常/非精子,3,000 張亮場 BMP） |
| Matek 骨髓 | Hugging Face `ekim15/bone_marrow_cell_dataset`（parquet 內嵌圖,21 類骨髓細胞,官方病人層級 train/val/test 切分） |
| LC25000 | Hugging Face `1aurent/LC25000`（parquet,organ×label 還原肺/結腸 5 類 H&E） |
| BreakHis | Mendeley Data `jxwvdwhpc2`（乳腺腫瘤 H&E,40–400× 放大倍率混訓的二元良/惡） |
| NCT-CRC-HE-100K | Zenodo 10.5281/zenodo.1214456（train 100K + 官方獨立 cohort CRC-VAL-HE-7K 當 val） |
| PatchCamelyon | Hugging Face `1aurent/PatchCamelyon`（parquet,官方 train/val split,淋巴結轉移 binary） |
| MHIST | Hugging Face `mamunrobi35/mhist_binary`（大腸息肉 HP vs SSA;mirror 為 flatten bytes/path schema） |
| 甲狀腺細胞學 | Hugging Face `cupofmocha/cytoal-thyroid-cytology-labeled`（imagefolder,6 類 FNA,cc-by-nc-4.0） |
| Mitosis 偵測 | Hugging Face `Francesco/mitosis-gjs3g`（Roboflow RF100 COCO,COCO→YOLO 轉換） |
| HuSHeM | Mendeley Data `tt3yj2pf38`（精子頭形態 4 類 `.rar`,Diff-Quik 染色） |
| RBC morphology | Zenodo `10.5281/zenodo.14936017`（紅血球異形 poikilocytosis 11 類 `.rar`,CC-BY-4.0;血液抹片單顆紅血球） |
| C-NMC leukemia | Hugging Face `dwb2023/cnmc-leukemia-2019`（B-ALL blast vs 正常,parquet 含 subject_id/fold,MIT;以 subject_id 做病人級切分避洩漏） |
| MP-IDB malaria | GitHub `andrealoddo/MP-IDB-The-Malaria-Parasite-Image-Database`（瘧原蟲 4 種,全視野 img + 單寄生蟲 crops + mask;codeload zip） |
| erythrocytesIDB | Zenodo `10.5281/zenodo.18299473`（鐮刀型紅血球 3 類,UIB;用 erythrocytesIDB1 individual cells 標註子集） |
| PanNuke | Hugging Face `RationAI/PanNuke`（H&E 核 instance mask;自寫 `src/pannuke_to_yolo.py` 轉 mask→YOLO bbox） |
| UNIVALI leukocyte | Zenodo `10.5281/zenodo.17743609`（白血球 14 類 ViT class-folder export,Roboflow one-hot CSV→ImageFolder,UNIVALI 巴西） |
| BACH | Hugging Face `1aurent/BACH`（ICIAR2018 乳腺 4 級大圖;自寫 `src/bach_tile_group_split.py` 切 512 tile + 影像級切分） |
| PathoTiroid | Zenodo `10.5281/zenodo.18827019`（甲狀腺 PTC vs 非PTC,24 病人 PTIC;以檔名 patient token 做病人級切分） |
| Urine sediment | Hugging Face `dikaizm/urine-particle-image-dataset`（尿沉渣 7 類 Roboflow COCO,官方 split;`coco_to_yolo` 轉 YOLO） |
| MLL23 | Zenodo `14277609`（周邊血 18 類 per-class zip,含毛細胞白血病/芽細胞/抹片細胞;MLL 慕尼黑） |
| Rouleaux | Zenodo `13763939`（串錢狀 vs 正常 RBC 薄血膜,100 受試者;用 original-form 全視野） |
| MK-11 | figshare `29264819`（骨髓巨核細胞 11 亞型,class 子資料夾 + 官方 5-fold 病人級切分） |
| Tryp | figshare `32397798`（錐蟲未染色厚血膜偵測,YOLO 標註 v3 + 陰性背景;Sci Data 2023） |

## 免責

研究 / 作品集用途，**非臨床診斷**。所有成績皆為資料集分布內（in-distribution）數字，不代表臨床或真實採檢條件下的表現。
