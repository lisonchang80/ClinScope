---
name: cross-domain-eval
description: 量化 ClinScope 某分類模型「跨域 / OOD」掉多少 —— 拿一批外部顯微圖（使用者自己拍的、或另一個資料集）對已訓練模型評分，輸出 per-class 準確率、混淆矩陣、並對照訓練時的 in-distribution val_acc 算出落差。觸發詞：跨域實測、OOD 測試、外部圖評估、模型在自己照片上準不準、cross-domain、eval external、丟外部圖看崩多少、out-of-distribution。用 src/eval_external.py。
---

# Cross-domain / OOD evaluation

ClinScope 每個模型的 in-distribution val_acc 都很漂亮（0.9+），但那是**同一資料集分布內**的數字。
這條 skill 回答真正重要的問題：**換成不同來源的圖，掉多少？** 這是本專案最有說服力的一張圖，
也是 roadmap 第一順位。用 `src/eval_external.py`，不必改任何訓練碼。

## 何時用

- 使用者拿自己顯微鏡拍的圖、或另一個公開資料集，想知道某個已訓練模型還準不準。
- 要為作品集產出「in-dist vs OOD 落差」的證據（混淆矩陣 + 掉幅）。
- 注意：這條是**分類模型**（ResNet，`models/*_resnet50.pt`）專用。YOLO 偵測的跨域評估走 ultralytics 自己的 `val`，不在這裡。

## 兩種模式（自動偵測，可用 `--mode` 強制）

- **labeled**：外部資料夾是 ImageFolder（`<類別>/*.jpg`），資料夾名對得上模型類別 → 算 per-class 準確率、分類報告、混淆矩陣，並對照 checkpoint 內存的 `val_acc` 印出**落差**。
- **unlabeled**：外部資料夾是一坨圖（沒有類別子夾）→ 沒有 ground truth，改印「預測類別分布 + 平均 top-1 信心 + 低信心比例」。**自信地錯**（信心高但分布全偏一類）是 OOD 崩掉的典型訊號。

## 跑法（一律走 Bash tool + 絕對路徑 venv python）

labeled（資料夾名 = 模型類別）：
```bash
"D:/Claude code/ClinScope/.venv/Scripts/python.exe" "D:/Claude code/ClinScope/src/eval_external.py" \
  --model "D:/Claude code/ClinScope/models/pbc_resnet50.pt" \
  --data-dir "D:/Claude code/ClinScope/data/<external_imagefolder>"
```

外部資料夾名跟模型類別不同時，給對應表（外部名 → 模型類別名）：
```bash
... --class-map '{"infected": "Parasitized", "healthy": "Uninfected"}'
```

unlabeled（只有一坨圖、沒標籤）：
```bash
... --model "...\\models\\dibas_resnet50.pt" --data-dir "...\\my_gram_photos" --mode unlabeled
```

## 怎麼讀結果

- 重點不是 external_acc 本身，而是 **`drop = val_acc - external_acc`** 那一行（含相對掉幅 %）。預期落到 in-dist 的 60–80%。
- 產出寫到 `reports/<model-stem>_xdomain/`：`external_report.txt`、`external_confusion_matrix.png`、`external_summary.json`（`reports/` 已 gitignore，不進版控）。
- 混淆矩陣若整欄塌到某一類 → 模型在新分布下退化成「只會猜多數類」，比單看 accuracy 更能說明問題。

## 坑

- **類別名要對得上**：對不上的資料夾會被跳過並印 `[warn] folders not matched`，這時補 `--class-map`。
- **labeled 偵測規則**：`--data-dir` 底下只要有「含圖的子資料夾」就判 labeled；圖直接放在 `--data-dir` 下才判 unlabeled。混在一起時用 `--mode` 明確指定。
- 壞檔會被容錯跳過（`LOAD_TRUNCATED_IMAGES`），不會中斷整批。
- 大批圖（上千張）是逐張推論、會慢但會跑完；跨域實測通常只丟幾十張精選圖即可。
- 找外部資料來源（不知道去哪抓）→ 交給 `dataset-scout` agent。新增整條 track → 見 `add-dataset-track` skill。
