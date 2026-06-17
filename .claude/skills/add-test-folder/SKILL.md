---
name: add-test-folder
description: 依既有慣例幫 ClinScope 某個（或多個）模型建立前端「選擇資料夾」用的測試資料夾——從該模型的 split 取樣影像、依命名慣例平鋪、附 _README.txt，並用 predict 驗證會分對。觸發詞：建測試資料夾、加測試資料夾、做某模型的 demo 資料夾、ClinScope 測試圖、前端載入用的範例資料夾、幫某模型準備測試影像。涉及 model_cards/<key>.json、data/<key>_split/、src/predict.py、測試根目錄 D:\ClinScope_測試\。
---

# 幫模型建前端測試資料夾

使用者要「給某模型一個能在前端 📁 選擇資料夾 載入、跑批次推論的測試資料夾」時用。
資料夾放在 **`D:\ClinScope_測試\`**（非 repo、非 gitignore 範圍內的 data/，是獨立目錄），由 `launch(allowed_paths=['C:\\','D:\\'])` 開放讀取。

## 慣例（務必照抄既有 4–6 個資料夾）

- **資料夾名**：`NN_中文簡名_<key>_<張數>`，NN 為兩位流水號接續現有最大號（例：`05_真菌DeFungi_fungi_koh_50`）。`<key>` = app.py REGISTRY／model_cards 的模型鍵。
- **檔名**：`<NN序號2位>_<真實類別>_<原始檔名>`（序號跨整個資料夾連續，類別取自來源 split 的類別子資料夾名）。內嵌真實類別 = 載入後可肉眼核對模型分對沒。
- **`_README.txt`**：單行 `搭配模型:<app.py REGISTRY 的 label，含 emoji>`（例：`搭配模型:🍄 真菌(KOH 直接鏡檢)— DeFungi`）。label 直接從 app.py REGISTRY 抄，不要自己編。
- **張數**：每資料夾約 50 張。cls 模型 → 各類別均衡取（val 夠就每類 ~10；val 太小就 **val 優先、再從 train 補**到每類目標數，湊滿涵蓋所有類別）。

## 步驟

1. **定位模型與資料**：`model_cards/<key>.json` 取 `task`(cls/det)、`classes`、`weights`；對應 split 在 `data/<key>_split/{train,val}/<class>/*`（黴菌例：defungi_split / openfungi_split）。app.py REGISTRY 取 `label`（含 emoji，給 README）。
2. **取樣 + 命名 + 寫 README**：用 `data/_pipeline/make_*_testfolders.py` 風格的小腳本（既有範例 `make_fungi_testfolders.py`）：每類 val→train 收齊、`shutil.copy2` 成 `NN_類別_原名`、寫 `_README.txt`。det 模型同理（類別取 YOLO label 或直接混合一批代表圖）。
3. **驗證會分對**：reuse `src/predict.py` 的 `load_model`/`predict_one`，對新資料夾跑 top-1，用「檔名內嵌的真實類別」算準確率（檔名去掉 `NN_` 前綴後 `startswith(class+'_')` 比對）。val 取樣的應接近 model_card 的 val_acc；明顯偏低代表取錯類/搬錯圖。
4. **回報**：列每個資料夾張數、各類分佈、top-1 準確率。

## 坑

- **gr.Image 純輸出元件 headless 無法載圖**、preview 的 Fullscreen API 被擋 → 別想用 preview 截圖驗證「載入後畫面」，改用上面的 predict 後端驗證即可。
- 取樣腳本放 `data/_pipeline/`（屬 data/，gitignore，不進版控）——一次性 helper，不必 commit。
- OpenFungi 這類 val 極不平衡（某類僅 2–3 張）→ 一定要從 train 補，否則資料夾涵蓋不到所有類別。
- 黴菌兩模型對照：`fungi`(OpenFungi 6 屬, resnet50_fungi.pt, openfungi_split) vs `fungi_koh`(DeFungi H1/H2/H3/H5/H6, defungi_resnet50.pt, defungi_split)——名稱易混，認權重檔與類別。

進度與環境見 [[clinscope-repo]]。
