---
name: add-models
description: ClinScope 一次自動連加 N 個新模型的排程器。GPU 訓練序列化、下載/切分與找源跟訓練重疊(prefetch),逐模型 commit、可續跑抗重開機。觸發詞:增加 N 個模型、加 N 個模型、連加幾個模型、add N models、批次加 track、自動續加模型。上層呼叫 dataset-scout(找源) + add-dataset-track(單模型 recipe);維護 data/_pipeline/queue.json 帳本。
---

# Add N models to ClinScope (orchestrator)

使用者下「**增加 N 個模型**」時,自動連加 N 條新 track,讓 **GPU 永不閒置**:訓練第 i 個的同時,下載+切分第 i+1 個的資料、並持續評估排序更後面的候選。

這是**上層排程器**,不重造輪子:

```
add-models (本 skill) ── 排程 / ledger / 兩 lane / 停-續-失敗-品質政策 ──┐
   ├─ 呼叫 ► dataset-scout (agent)        找源 + HEAD 驗證(只找不下載)   │
   └─ 呼叫 ► add-dataset-track (skill)    單模型:下載→切分→訓練→掛 app.py
```
單模型怎麼做一律照 `add-dataset-track`;本 skill 只管「N 個怎麼排、怎麼重疊、怎麼續跑」。

## 核心洞察:GPU 序列 + I/O 預取

本機瓶頸是**單顆 GPU(RTX 4060 Ti)** → 訓練一次只能一個、嚴格序列。但「找源 / 下載 / 解壓切分」是 I/O+CPU,**可與 GPU 訓練重疊**。所以排程目標 = GPU 永遠有「已備好待命」的資料,不空轉。

```
時間 ───────────────────────────────────────────────────────►
GPU lane :   [ 訓練 #1 ============ ][ 訓練 #2 ============ ][ 訓練 #3 ===…
I/O lane : [下載+切分#1][ 下載+切分#2 …… ][ 下載+切分#3 …… ][ 下載+切分#4…
scout    :[排候選清單]…… backlog<2 就補查 …………………………………………………
```

- **預取深度 = 1**(同時只保 1 個「prepared 待命」):下載+切分通常比訓練短,深度 1 就足以餵飽 GPU;深度 2 只在慢源時有用,而慢源直接跳過(見失敗政策),不值得多佔碟。
- **最多兩條背景 lane**:1 訓練 + 1 下載。scout 是短任務,backlog 低時臨時補。

## Ledger 帳本(單一真相、可續跑、抗重開機)

路徑 `data/_pipeline/queue.json`(gitignore,屬 data/ 不進版控)。這是整條流程唯一真相來源:

```jsonc
{
  "target_remaining": 5,          // 還要成功加幾個
  "registered_this_run": 0,
  "candidates": [
    {
      "id": "urine-sediment",
      "name": "...", "discipline": "...", "modality": "...",
      "source": {"kind": "hf_parquet|mendeley|zenodo|uci|hf_file",
                 "ref": "...", "verified": true, "approx_gb": 1.8,
                 "cmd": "<aria2c / resolve URL>", "notes": "..."},
      "status": "scouted",        // 狀態機見下
      "weights": "models/xxx_resnet50.pt",
      "kind_engine": "cls|det",
      "val_acc": null, "macro_f1": null, "caveats": [], "fail_reason": null
    }
  ]
}
```

狀態機(每個候選):

```
scouted ──► downloading ──► prepared ──► training ──► trained ──► registered ✓
                  │              │            │
                  ▼              ▼            ▼
            failed(牆/需登入) / skipped(重複/磁碟滿)  → scout 補一個替補
```

**為什麼非要 ledger**:本機有 Windows Update 自動重開機前科(見 [[reference-background-sleep]])。連加數個模型必然跨越「可能被重開機」的時間窗。有帳本 + **逐模型 commit**,重開機最多損失「當下正在訓練的那一個」,不是全軍覆沒;重啟後對帳(掃哪些 weight 檔/report 已存在 → 回填 status)就能接著跑。

## 啟動流程(收到「增加 N 個模型」)

1. **建/讀帳本**:`target_remaining = N`。若帳本已存在(續跑)→ 先對帳:掃 `models/`、`reports/`、背景任務輸出,回填每個候選的 status,跳過已 registered 的。
2. **scout 補 backlog**:跑 `dataset-scout` agent,目標湊到 **≥ max(N, 2)** 個「已 HEAD 驗證可下載」候選。對既有 11 條 track + 帳本去重;排序偏好:小而乾淨 > 大而有牆、官方已切 split > 需自己切、新科別/新模態 > 重複科別。
3. **貼 shortlist 當 FYI 後直接跑**(已定案的自主程度):把排好序的 N 個候選列成一句一個(科別/模態/來源/可下載性)貼給使用者**當參考**,**不等確認**就進控制迴圈。使用者可隨時喊停或改順序。

## 控制迴圈(事件驅動,被背景任務完成通知喚醒)

```
每次醒來:
  1. 對帳:把剛完成的背景任務結果寫回 ledger
  2. 若有「訓練」剛結束:
       a. predict mini-check + 讀 classification_report
       b. 套用品質政策(見下) → 收錄 / 標 caveat / 或丟棄換候選
       c. 收錄者:app.py REGISTRY 加一條 + README 表格加一列 + 更新 memory
       d. git commit + push(這一個模型獨立一個 commit)     ← 逐模型 checkpoint
       e. status=registered;target_remaining -= 1
  3. 若 target_remaining==0 → 收尾總結、停止
  4. 若 GPU 閒置 且 有 candidate.status==prepared → 啟動訓練(GPU lane,背景)
  5. 若 沒有下載在跑 且 預取槽空(無 prepared 待命) 且 有 status==scouted
       → 啟動下載+切分(I/O lane,背景)
         成功 → status=prepared(順手清掉 raw 壓縮檔/parquet,只留 split)
         遇牆/需登入/慢源 → status=failed + 原因,觸發 scout 補替補
  6. 若 已驗證 backlog < 2 → 跑 dataset-scout 補查(去重)
  7. 等下一個背景任務完成(靠完成通知,不主動 sleep 輪詢)
```

所有背景任務一律 `run_in_background: true` + `python -u`(即時進度、被砍也留 log),靠完成通知接續。單模型的下載/切分/訓練指令細節全照 `add-dataset-track`。

## 品質政策(已定案:全收錄 + 自動標 caveat)

訓出來一律收進前端,但**自動偵測並掛 caveat**,符合本專案「誠實展示流程、非臨床對齊」的定位:

- **自動 caveat 觸發條件**(寫進 ledger `caveats[]` + README 成績欄 + blurb):
  - 總 val < ~300 張 或 任一類 val support < ~15 → `⚠️ 小 val 不可信`(同 DIBaS)
  - macro-F1 < 0.60,或 macro 比 weighted 低 > 0.10 → `⚠️ 類別不平衡,稀有類偏弱`(同骨髓 21 類)
  - 任一類 support < 10 → 點名該類「學不起來」
  - 單一採集來源/可能洩漏 → 註記「需 group-aware split 才可信」
- **只有明顯沒學起來才丟棄**:val_acc 沒顯著高於亂猜 1/C,或 macro-F1 < 0.30 → status=failed(不收錄),換 backlog 下一個候選。
- **絕不靜默**:被丟棄/降級的候選都要在最終總結講原因。

## 失敗 / 磁碟 / 停止 政策

| 面向 | 策略 |
|---|---|
| 下載軟牆(Google Drive 散檔 ~0.3 檔/秒)/ 需 Kaggle/IEEE 登入 | 不硬幹,標 failed+原因,換 backlog 下一個(見 add-dataset-track Step 0) |
| 磁碟保護 | 切分後立刻清 raw(壓縮檔/parquet);可用空間 < ~15GB 就暫停預取,先消化 GPU 佇列 |
| 重複/已有 track | scout 階段就 dedup,直接 skip,不佔名額 |
| 連續失敗 | 同類失敗 ≥ 3 或 backlog 補不到新候選 → 停止,回報「實際加了 M/N 個 + 為何停」 |
| **下載卡死無通知** | HF `snapshot_download`/`load_dataset` 切網時會**卡死 socket 不報錯、不發完成通知** → 事件驅動迴圈會空等。**長下載一律 aria2c switch-proof 逐檔**(`--max-tries=0`+stall 偵測,撐過熱點↔wifi);疑似卡住就主動比對 raw 檔 mtime,`TaskStop`+`taskkill //F //IM aria2c.exe` 後重啟續傳(見 add-dataset-track) |
| prep 腳本即 ledger | 每個候選一支 `data/_pipeline/prep_<id>.sh`(gitignore),內含「下載→解壓/解碼→切分→清 raw→印 PREPARED_OK」;當背景任務直接跑、可續跑可重啟。prep bug(非源頭牆)就地修腳本重跑、不算入連續失敗 |
| **資料集無可用標籤** | 標榜 cls 的 HF/zenodo 集可能 (a) 只有未標 patch(KidneyAI 只有 glomeruli patch、檔名=slide-bbox)或 (b) 只有自由文字 caption、無離散 label、甚至多器官混(arch_gastric)→ **下載後先驗 label/features(class 子夾 / HF `ClassLabel` / csv 可離散欄)才訓,沒有就 skip 換 backlog**。「YOLO 偵測」集可能是分類集 re-host(label 全整圖退化框 `0 0.5 0.5 1 1`)→ 訓前 `cat` 一個 label。det/大檔一律先 download+inspect 再寫 prep(見 add-dataset-track) |
| 訓練被外部中止 | 背景訓練可能無 traceback、無完成通知就死(Windows Update 重開 / GPU TDR;IPD-Brain ep10 中過)。判死活:`tasklist\|grep python` 無進程 + 輸出 log 凍結在某 epoch。best.pt 逐 epoch 存→權重在,但 `classification_report` 最後才寫故**重跑完整才有報表**。長批前暫停 Windows Update |
| 無洩漏切分優先 | 有官方 k-fold/患者 id 就用病人/片/WSI-disjoint(helper: `hf_parquet_group_split`/`patchgastric_wsi_split`/`bach_tile_group_split`/`ipdbrain_tile_split`)。組織分級在無洩漏下本就低分(PatchGastric 0.49/IPD-Brain 0.53)= 誠實,非丟棄條件(仍 >隨機、macro>0.30) |
| 不靜默截斷 | 「想加 N 只加到 M」一定明講原因(牆/磁碟/品質),不假裝全做完 |

## 逐模型 checkpoint(每完成一個就 commit)

每收錄一個模型 → 一個獨立 commit(app.py + README + 該 track 報表索引;**data/models 已 gitignore 不進版控**),訊息結尾固定:
```
Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
```
push 已預先授權直接推;失敗(remote 領先)先 `git pull --rebase` 再 push,衝突停下回報不強推。memory 不入 repo。

## 重開機防護(長跑前必做)

連加多個模型 = 數小時長跑,務必:
1. **長跑前暫停 Windows Update**(最關鍵)、訓練中別點「更新並重新啟動」(見 [[reference-background-sleep]])。
2. 訓練腳本開頭已 `keep_awake()` 防睡眠(擋不了 Update 重開,只擋睡眠)。
3. 真被重開 → 靠 ledger + best-checkpoint 對帳續跑,別重跑已完成的。

## 收尾總結

停止時輸出一張表:每個目標候選的最終 status(registered / failed / skipped)、成績、掛上的 caveat、失敗原因;以及「N 目標 → 實際 registered M、前端現有模型總數」。
