---
name: fix-leaky-split
description: 把 ClinScope 既有「隨機 split → 採集洩漏」的 track 重切成 group-disjoint(病人/片/視野級)並重訓,用誠實的 leakage-free 數字取代灌水的舊分數。觸發詞:修洩漏、補洩漏切分、de-leak、group-disjoint 重切、patient/slide-level split 重訓、舊 track 隨機切分修正、洩漏太樂觀重評。涉及 src/imagefolder_group_split.py + train_resnet.py + app.py + README。
---

# Fix a leaky (random-split) track → honest group-disjoint number

既有 track 若用**隨機 split**,而資料是「同病人多倍率 / 同片多 tile / 同視野多 crop」,val 會被採集洩漏灌高(看似 0.95+ 其實在背片不是學病灶)。本 skill 把它重切成 **group-disjoint**(同一 group 整組只進 train 或 val),拿真實泛化數字。**分數掉就是目的**,不是 bug;誠實 > 好看。單條新增看 [[add-dataset-track]]。

## 何時做 / 找候選

掃 app.py blurb、README 成績欄、[[clinscope-project]] memory 裡帶 `⚠️ … 洩漏`(隨機 split 病人/同片/同視野/增強)的 track。本批已修:BreakHis 0.978→0.839、MP-IDB 0.997→0.509、thyroid 0.947→0.505(見下「成果」)。

## 可行性閘門(動工前先過,別硬幹)

1. **既有 split 的檔名 token 已洗掉 → 一律要重抓 raw。** `prepare_data.py` 切分時做 `shutil.copy2(f, dst/f"{cls}_{i:04d}{suf}")`(line ~121)、**不存對照表** → `<class>_0000.png` 無法回推原始病人/片 id。所以重切**必須重抓原始來源**(保留原檔名/分組欄的那份)。
2. **分組 token 真的可還原嗎?** 沒有就**誠實標明不可修、別假裝**:
   - ✅ 可:檔名編碼病人/片(BreakHis `SOB_B_A-14-22549AB-…`→病人 `14-22549`、去字母後綴合併同病人多片;thyroid `1020130_row_…_col_…`→片 `1020130`;MP-IDB 全視野 `1305121398-0001-…`→血片前綴);HF parquet 有 `subject_id`/slide 欄(用 `hf_parquet_group_split.py`);自帶官方 k-fold/患者切分。
   - ❌ 不可:**增強擴增**且原圖對照已失(LC25000:1250 原圖→augmented,parquet 無 original-id);**太小**(HuSHeM 216 張);檔名 subject 編碼亂無法 disjoint(rouleaux);**無任何分組欄**(blood-protozoa ClassLabel parquet)。
3. **group 數量閘門:** 某類只有 **2–3 個 group** → group-disjoint 後該類 val = 單一 group、數字無統計意義(MP-IDB Malariae 僅 1 訓練片 → F1 0.00)。仍可做但**重 caveat**,點名哪些類不可信。
4. **先驗來源真有分組/partition 再投入。** 別信檔名/檔案數的表面:MHIST mirror `mamunrobi35/mhist_binary` 看似 `train-*.parquet`+`validation-*.parquet`,實際全資料塞在 train、validation 是 1 列 `MHIST_dev.png` 空殼、檔名是流水號(`1.png`)無片 id → **無官方 partition 可取 → 放棄**(且 MHIST 本就一區一 tile、隨機切無嚴重洩漏,原 caveat 過重)。

## 流程(逐條,GPU 序列 × 重抓 I/O 可與訓練重疊)

1. **重抓 raw**(保留原檔名那份):
   - 單一壓縮檔(BreakHis UFPR `https://www.inf.ufpr.br/vri/databases/BreaKHis_v1.tar.gz` 4.3GB、MP-IDB codeload zip)→ aria2c switch-proof(見 [[add-dataset-track]] Step 0)。
   - **HF imagefolder 上萬個小檔 → 不要逐檔 aria2c**:`resolve/main/<path>` 逐檔下載會被 HF **靜默限流**(exit 0 但只下一部分、整類缺、不報錯,thyroid 27k 檔停在 9484)。改 **`git clone --depth 1 https://huggingface.co/datasets/<repo> <dir>` 再 `cd <dir> && git lfs pull`**(一次 pack/LFS batch,繞過逐檔節流,快又穩)。⚠️ HF 的 `.gitattributes` 常含 `*.png filter=lfs`(看 `.gitattributes` 要看全、別只看前 15 行)→ **`GIT_LFS_SKIP_SMUDGE=1` clone 會留 130-byte pointer**(`Image.open` 報 cannot identify),要 `git lfs pull` 才換真內容。
2. **group-disjoint 切分** → `src/imagefolder_group_split.py`(本 skill 核心、本批新寫的通用 helper):
   ```bash
   python -u src/imagefolder_group_split.py --src <raw 影像樹> --out data/<name>_xx_split \
     --label-regex="<類別 regex,作用在 POSIX 相對路徑、group(1)=class>" \
     --group-regex="<分組 regex,作用在檔名 stem、group(1)=group id>" \
     --val-frac 0.2 --max-side <0=copy 原圖 / N=長邊縮 N 並轉 JPG> [--dry-run]
   ```
   - **先 `--dry-run` 看 group 數 / 每類 val-train group 分布**(過閘門 3)。
   - **⚠️ regex 以 `-` 開頭(如 `-(\d+-\d+)`)一定用 `--group-regex="..."` 等號式**,空格式會被 argparse 當 flag(`expected one argument`)。
   - label-regex 範例:BreakHis `"(benign|malignant)"`、thyroid `"^([1-6])/"`、MP-IDB(只取全視野)`"(Falciparum|Malariae|Ovale|Vivax)/img/"`(crops/gt 路徑不含 `/img/` 自動跳過)。group-regex 範例:BreakHis `"-(\d+-\d+)"`、thyroid/通用 `"^([^_]+)_"`(吃 `1020130_`、`I 413_`、`VI 406_`)、MP-IDB `"^(\d+)-"`。
   - 跨類的 group(少數 tile 標到兩類)→ helper 取 majority class 做分層、但每張仍按自身 label 路由,leakage-safe。
3. **重訓到新檔名**(別覆蓋舊 leaky 權重,先比對):
   `python -u src/train_resnet.py --data-dir data/<name>_xx_split --out models/<name>_xx_resnet50.pt --report-dir reports/<name>_xx --epochs 15 --title "..." [--class-weight sqrt]`
4. **比對 + 品質政策**:讀 `reports/<name>_xx/classification_report.txt`。val > 隨機(1/C)且 macro-F1 > 0.30 → **保留 + 誠實 caveat**(掉多少正是重點);否則才考慮移除。逐類點名失敗類(如 group 太少塌掉的)。
5. **取代 + 更新前端**:`cp models/<name>_xx_resnet50.pt models/<原註冊檔>.pt`(覆蓋舊 leaky)→ `torch.load` 驗 classes/val_acc → 改 **app.py blurb**(把 `⚠️…洩漏` 換成 `✅ <病人/片/視野>級切分 + 誠實 val + macro-F1 + 對比舊分數掉幾 pt + 哪些類不可信`)+ **README 成績列**。整理 `reports/`(rm 舊、mv 新→原名)、清重抓的 raw(GB 級,如 BreakHis tar+解壓 ~8GB)。
6. **驗 + commit + push**:`python -c "import app; print(len(app.REGISTRY), [m['key'] for m in app.REGISTRY if not m['path'].exists()])"` → `fix(<track>): <group>-level split replaces leaky random split (val A to B)`,結尾 `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`,push 已預先授權(失敗先 `git pull --rebase`,不強推)。memory 不入 repo。**留新 split 夾(可重訓)、別留臨時 `_xx` 權重**。

## 本批成果(2026-06-13)

| track | 舊(leaky) | 新(誠實) | 落差 | group | 重點 |
|---|---|---|---|---|---|
| BreakHis 乳腺良惡 | 0.978 | 0.839 | −14pt | 病人(70) | macro-F1 0.829 穩;benign recall 0.69 偏判 malignant |
| MP-IDB 瘧 4 種 | 0.997 | 0.509 | −49pt | 血片(17) | macro-F1 0.348;Malariae 0.00(僅 1 訓練片=結構性極限) |
| thyroid FNA 6 類 | 0.947 | 0.505 | −44pt | 片(118) | 洩漏最重;class 1/6 能學、中間 Bethesda 2–5 難 |

通則:**每 group 的影像/tile 數越多 → 洩漏越誇張**(thyroid 每片數百 tile→掉 44pt)。無洩漏 track 由 6 增為 9。仍待修(已查不可修原因見閘門 2):LC25000 / HuSHeM / rouleaux / blood-protozoa。
