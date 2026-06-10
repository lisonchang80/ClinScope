# FungiScope

真菌顯微影像辨識 PoC。先在公開資料集 **OpenFungi** 上以 ResNet-50 transfer learning 跑通流程，
驗證「資料集內 ~97%」可重現，再評估是否投入收集自己的顯微鏡照片做臨床對齊。

## 現實定位（先讀這段）

- **目標情境**：臨床真菌顯微辨識（醫院 AI 對應 Aspergillus / Penicillium / 皮癬菌 Trichophyton 等）。
- **OpenFungi 涵蓋**：Aspergillus（Flavi + Nigri）、Penicillium、Rhizopus、Alternaria、Fusarium + mixed。
  ✅ 有 Aspergillus / Penicillium，❌ **沒有 Trichophyton（皮癬菌）**；Rhizopus/Alternaria/Fusarium 偏食品/環境/植物黴菌。
- 因此本 PoC 證明的是「**流程與模型能力**」，不是臨床皮癬菌對齊。
- **跨域落差**：資料集內 val acc 預期 90–97%；換成你自己顯微鏡拍的照片，預期掉到 60–80%
  （同 PoseLab 的 in-distribution vs out-of-distribution 坑）。資料集只有 ~571 張顯微影像、類別不平衡，更會放大這點。

## 資料集

- 來源：OpenFungi (MDPI *Life* 2025, DOI 10.3390/life15071132)
- 下載：Zenodo `https://doi.org/10.5281/zenodo.15692069`（`openfungi.zip` ≈ 8.0 GB）
- 顯微 571 張（2560×1920）／巨觀 678 張（4624×3472）
- 程式碼參考：github.com/rolandbolboaca/CNN_Fungi_classification

## 環境

- Python 3.11 venv（`.venv/`），不用 conda
- GPU：RTX 4060 Ti 8GB，PyTorch CUDA 12.4 wheel
- 安裝：見 `requirements.txt`

## 執行步驟

```powershell
# 1. 整理資料（解壓 + 顯微子集 stratified 8:2 切分）
.\.venv\Scripts\python.exe src\prepare_data.py --zip data\openfungi.zip --subset micro

# 2. 訓練 ResNet-50
.\.venv\Scripts\python.exe src\train_resnet.py --data-dir data\micro_split --epochs 30

# 產出：models\resnet50_fungi.pt、reports\classification_report.txt、reports\confusion_matrix.png
```

## 目錄

- `src/prepare_data.py` — 解壓 + 自動定位顯微子集 + train/val 切分
- `src/train_resnet.py` — ResNet-50 transfer learning（two-phase 凍結→微調、class-weighted loss、AMP）
- `data/` — 資料集與切分（不進 git）
- `models/` — 訓練好的權重
- `reports/` — 分類報告與混淆矩陣
