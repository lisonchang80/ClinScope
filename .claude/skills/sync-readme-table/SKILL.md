---
name: sync-readme-table
description: 從 model_cards/*.json 重生 ClinScope README 的模型表 + 資料來源表 + 科目統計,並對帳卡片數 vs app.py REGISTRY。每跑完一個 batch 用來更新 README 那張大表、補新列、查數量是否同步。觸發詞:更新 README 表、同步模型表、重生科目表、README 對帳、補 batch 新列、sync readme、科目統計。涉及 .claude/skills/sync-readme-table/gen_readme_table.py + model_cards/ + README.md。
disable-model-invocation: true
---

# sync-readme-table — 重生 README 模型 / 來源表

每加完一批模型,README 的大表、header 數字、資料來源表都要手改,易漏、易和 REGISTRY 不同步。本 skill 用 `model_cards/*.json`(由 `src/gen_model_cards.py` 自動生)當真相,機械重生這些表。

## 前置:先把卡片更新到最新
剛加完模型要先重生卡片,否則表會落後:
```
.venv\Scripts\python.exe src\gen_model_cards.py
```

## 用法
腳本 `.claude/skills/sync-readme-table/gen_readme_table.py`(純標準庫,全域 python 即可)。

1. **對帳數量**(最快,先做):
   ```
   python .claude\skills\sync-readme-table\gen_readme_table.py --check
   ```
   印「app.py REGISTRY N 條 vs model_cards M 張」。不一致 → 先補卡片或查 REGISTRY,別動 README。

2. **只取剛加那批的新列**(增量更新最實用):
   ```
   python .claude\skills\sync-readme-table\gen_readme_table.py --batch "batch-9 (2026-06-15)"
   ```
   貼進 README 模型表;順手把 README header 的模型總數改成 `--check` 給的數字。

3. **重生整張表 + 來源表 + 科目統計**(全面校時用):
   ```
   python .claude\skills\sync-readme-table\gen_readme_table.py
   ```
   stdout 是兩段帶 `<!-- MODELS:START/END -->`、`<!-- SOURCES:START/END -->` 標記的表;stderr 是科目統計 + 對帳。

## 重要原則
- **成績欄是「規範化字串」**(`val 0.98 / F1 .. ⚠️`),不還原 README 手寫的敘事成績(如「0.679（FM 救回 LGSC）」)。所以**全表重生會洗掉手寫敘事**——預設走 `--batch` 增量,只在校正漂移時才全表重生,且重生後人工掃一遍把重要敘事補回。
- 引擎欄 FM / SAHI 兩個特例靠腳本 `ENGINE_OVERRIDE` 補(卡片 model_type 不分 FM);新增 FM / SAHI track 要在腳本補一筆。
- 科別取自卡片 `discipline`(較粗,如「組織病理」);README 若要細到「組織病理（乳腺）」需人工加括號。
- 此 skill 會改 README(有副作用)→ user-only(`disable-model-invocation: true`),只由你 `/sync-readme-table` 觸發。
