---
name: pick-model-arch
description: 當 ClinScope 某 track 訓練後分數差(cls val 偏低 / det mAP 偏低)、或要評估換掉預設的 ResNet-50 / YOLOv8s 時,依「失敗模式」對症選替代架構。涵蓋 timm 現代骨幹、病理/生醫基礎模型(UNI/Phikon-v2/CONCH/BiomedCLIP)、細胞核分割專用(StarDist/Cellpose/HoVer-Net/InstanSeg)、與 SAHI 大圖切片偵測。Trigger: 換骨幹/換模型/架構選擇/分數差/成績差/foundation model/基礎模型/MIL/SAHI/StarDist/Cellpose/HoVer-Net/細胞核偵測/密集小核/WSI 亞型/ordinal 分級/QWK。涉及 src/train_resnet.py, src/train_yolo.py。
---

# 為某個 track 選對的模型架構

ClinScope 預設兩引擎:**cls=ResNet-50**(`src/train_resnet.py`)、**det=YOLOv8s**(`src/train_yolo.py`)。兩者是穩健的通用 baseline,80+ track 多數夠用。但**特定失敗模式下換架構/加技巧能顯著拉分**。本 skill 是「分數差時換什麼」的決策表 + 最小整合配方。

> 硬體前提:單顆 **RTX 4060 Ti 8GB VRAM**。所有建議都以 8GB 可行為前提。
> 定位:本平台 self-study/portfolio → **非商用 OK**(可用 research-only 權重);但授權欄仍標清楚,未來若要商用是硬限制。

## ✅ 本專案實測驗證(2026-06-15 PoC, 兩支柱都坐實)

| PoC | 原 baseline | 換架構後 | 提升 | 成本 |
|---|---|---|---|---|
| **cls: Phikon-v2 frozen + LogReg @ ubc_ocean** | ResNet-50 val 0.543 / macro-F1 0.436(LGSC F1 0) | **val 0.679 / macro-F1 0.593(LGSC 0.222)** | **+13.6 acc / +15.7 macro-F1**,救回全崩的稀有類 | 只訓 1 個 logistic regression,骨幹凍結;3×3 tile mean-pool 1024 維特徵 |
| **det: SAHI 切片 vs 整圖 @ puma(密集小核)** | 整圖 mAP50 0.200 | **SAHI 切片 mAP50 0.276** | **+37.6% 相對**,框數 12k→26k 找回小核 | **零重訓**,只切片推論(slice 640/overlap 0.2) |

cls 支柱穩贏。但 det 端 **SAHI 不是通解** —— 對 8 條 det track 做 sweep(`sahi_sweep.py`)後**只有 puma 受益**:

| track | 訓練影像尺寸 | 整圖 | SAHI 切片 | Δ |
|---|---|---|---|---|
| **puma** | 1024 | 0.200 | **0.276** | **+37.6%** ✅ 整合 |
| pannuke / nuinsseg / cryonuseg | 256 / 512 / 512 | 0.69 / 0.60 / 0.47 | ≈同 | -0~2%(圖≤640,根本沒切到) |
| oral_cyto | 2048 | 0.450 | 0.334 | **-25.8%** ❌ |
| tvmi3k | 3088 | 0.542 | 0.393 | **-27.6%** ❌ |

> **關鍵修正(重要,別再誤用)**:SAHI **zero-shot 推論只在「模型訓練尺度 ≈ slice 尺度」時有益**。puma 訓練 1024→640(接近 640 slice)故大賺;但 oral_cyto/tvmi3k 訓練時把 2048/3088 大幅縮到 640,模型學的是「縮小後的小目標」,改用 640 native slice 推論 = **訓練/推論尺度錯配 → 反而大跌 25~28%**。圖 ≤640 的 track 切片不發生 → 中性。
> **正解**:(1) SAHI 論文「+12~14% AP」是**重訓 on slices** 的數字,非 zero-shot;(2) 要讓 oral_cyto/tvmi3k 這類大圖享受 SAHI,得**用 SAHI 切片資料重新訓練/fine-tune**(slice 與推論同尺度),而非直接套切片推論;(3) zero-shot 套用前**務必先 sweep 驗證**(同模型同評估器比整圖 vs 切片),別假設一定贏。
> 本專案落地:cls「病理 FM frozen+linear」穩贏;det **只有 puma 在 app.py 啟用 `"sahi": True`**(REGISTRY flag → `run()` 走 `load_sahi`+`get_sliced_prediction`)。PoC/sweep 腳本 `data/_pipeline/{phikon_poc,sahi_poc,sahi_sweep}.py`(gitignore)。

## 先診斷:這是「模型問題」還是別的問題?

換架構前先排除**不是模型能救的**(換了也沒用,別浪費 GPU):

| 症狀 | 真因 | 對策(非換模型) |
|---|---|---|
| det mAP 極低(如 riva 0.146) + 標註是「關鍵點轉合成框」 | **標籤是假的** | 取得真 bbox 或改成 cls/計數任務;見 [[fix-leaky-split]] 思路 |
| val 只有數張 / 稀有類 support<15(如 cryonuseg val=10) | **val 不可信**,非真的學不起來 | 掛 caveat,別據此換架構;補資料才有意義 |
| 隨機 split 高分、group-disjoint 暴跌 | **洩漏**,高分是假象 | 已用 [[fix-leaky-split]] 修正;低分是誠實值 |
| 分級任務 val~0.5(Gleason/Bethesda/分化) | **任務上限低**(判讀者間變異大) | 換架構只救一部分,主要靠 ordinal head + QWK 評估(見下) |

**模型真能救的**只有:表徵不足(小資料+弱骨幹)、解析度錯配(大圖小物件被縮掉)、範式不對(用 bbox 數密集核)。

## 失敗模式 → 換什麼(對症表)

| 失敗模式 | ClinScope 例子 | 換什麼(優先序) |
|---|---|---|
| **WSI 亞型/降採縮圖分類,資料小** | ubc_ocean(0.54)、ipdbrain(0.53)、patchgastric(0.49) | ①病理 FM **frozen 特徵 + linear/MIL head** ②tile-and-aggregate 別硬 resize 成 224 |
| **資料極小(數十~數百張)** | kidney(78圖)、cryonuseg(30圖) | ①FM frozen + **linear probe**(數十張唯一不崩) ②zero/few-shot VLM(CONCH/BiomedCLIP) ③EfficientNet-B0/ConvNeXtV2-Nano(參數少抗過擬合) |
| **大圖密集小核偵測** | puma(0.19)、riva、cryonuseg det、kidney det | ①**SAHI 切片**(外掛現有 YOLO,最低成本)**但僅當訓練尺度≈slice 才賺(實測 puma +37.6%、oral_cyto/tvmi3k 反跌 26~28%),否則要 slice 重訓** ②**StarDist** H&E 核開箱權重當第二引擎 ③NMS-free(RT-DETR/YOLOv10)解 NMS 爆炸 |
| **稀疏微小寄生蟲偵測** | tcruzi(0.26)、leishmania(0.27 mAP50-95)、pf_vivax(0.40) | **YOLO + SAHI**(切片讓小蟲體相對變大);⚠️先 sweep 驗證(實測 tcruzi 套切片反略跌,大幅降採訓練者需 slice 重訓);不要用 StarDist(那是數核工具) |
| **連續分級(有序譜)** | thyroid(0.50)、sicap Gleason(G5 弱)、patchgastric | **ordinal head**(CORAL/CORN)取代 N 類 softmax + **label smoothing** + 評估用 **QWK** 而非 acc;搭 FM 特徵 |

## cls 替代架構表(取代/補強 ResNet-50)

| 模型 | 類型/參數 | 授權 | 8GB 可行性 | 何時用 |
|---|---|---|---|---|
| ResNet-50(基準) | CNN 25.6M | 可商用 | 全量微調輕鬆 | 通用 baseline |
| EfficientNetV2-S | CNN 21.5M | 可商用 | 全量微調 | ResNet 無痛升級、小資料性價比高 |
| ConvNeXt-Tiny / V2-Nano | 現代 CNN 28.6M/3.7M | 可商用 | Tiny 全量微調順 | 遷移強、小資料常勝 ResNet;極小資料用 V2-Nano |
| Swin-Tiny | Transformer 28M | 可商用 | 全量微調可 | 需較多資料才贏 CNN,數百例不一定優 |
| ViT-B/16 | Transformer 86M | 可商用 | frozen/LoRA(全微調吃力) | **小資料別從頭微調**,易輸 |
| **Phikon(v1)** | 病理 ViT-B 85.8M | Owkin 非商用 | frozen 很輕,LoRA 可 | **最易上手病理 FM:免申請、HF 直抓** |
| **Phikon-v2** ⭐ | 病理 ViT-L 0.3B | Owkin 非商用 | **frozen 抽特徵**(微調吃力) | **最強又免機構信箱**;WSI 亞型/小資料首選 |
| UNI | 病理 ViT-L 307M | CC-BY-NC-ND | frozen(gated 個人帳號通常可過) | 同級 Phikon-v2;UNI2-h(681M)明文擋 gmail |
| Virchow2 | 病理 ViT-H 632M | CC-BY-NC-ND gated | frozen bs 小 | 頂級特徵但門檻/體量高 |
| CONCH | 病理 VLM ViT-B+text | CC-BY-NC-ND gated | frozen 很輕 | **zero/few-shot 強**,極小資料 |
| **BiomedCLIP** | 生醫 VLM ViT-B/16 | **MIT 可商用** ✅ | frozen/微調都順 | **唯一可商用 FM**、跨多科別廣度,本平台 ~18 科很契合 |
| CTransPath / RetCCL | 輕量病理 FM 27M/25.6M | GPLv3 非商用 | 全程無壓力 | 資源吃緊時;RetCCL=RN50 體量最無痛接現有管線 |

> 重點:病理 FM 的用法是 **frozen 抽一次特徵存 .npy → 只訓小 head**,GPU 只在抽特徵時 inference(bs 小),8GB 完全夠;訓 head 幾乎不吃 VRAM。**不要全量微調 ViT-L 級**(8GB 不切實際)。

## det 替代架構表(取代/補強 YOLOv8s)

| 模型/技巧 | 任務 | 授權 | 8GB | 密集小核適配 |
|---|---|---|---|---|
| YOLOv8s(基準) | bbox/seg | AGPL-3.0 | 訓推皆可 | 弱:@640 核縮成 ~8px |
| YOLOv10 / RT-DETR(原版) | bbox(NMS-free) | AGPL-3.0 / **Apache-2.0** | 同/較重 | 解「val NMS 偶發爆炸」;小物件仍需 tiling |
| Faster/Cascade R-CNN(Detectron2) | bbox | **Apache-2.0** | batch 小可 | 兩階段對小物件較友善,仍需 tiling |
| **StarDist** ⭐ | instance-seg(star-convex) | **BSD-3 可商用** | 輕鬆 | **最強**:專為密集貼合核;**開箱 2D_versatile_he H&E 權重** |
| Cellpose v3 / SAM | instance-seg | code BSD-3 / **權重 CC-BY-NC** | 推論可 | 通用 robust,稀疏佳、極密不如 StarDist;**權重禁商用** |
| HoVer-Net | seg+**核分類** | code MIT / PanNuke 權重 CC-BY-NC-SA | 訓重推慢 | H&E 核分割+分類一把抓,但慢、商用受限 |
| InstanSeg | instance-seg | **Apache-2.0 含模型** | 輕量 | 商用友善核分割,QuPath 內建 |
| **SAHI** ⭐ | 切片推論框架(套任一偵測器) | **MIT** | 推論記憶體友善 | 原解析度切片;**論文 +12.7~14.5% 是 fine-tune 數字**;zero-shot **僅訓練尺度≈slice 才賺**(本專案實測:puma +37.6%,但大幅降採訓練的 oral_cyto/tvmi3k 反跌)→套用前先 sweep |

> imgsz 硬拉 1280 在 8GB **會 OOM**。要提解析度走 **tiling 訓練(切 640/768) + SAHI 切片推論**,不要單張拉大。

## 最小整合配方(8GB)

**A. 病理 FM frozen 特徵 + linear/MIL(救 WSI 亞型/小資料 cls)**
1. `pip install timm`(UNI/Phikon 走 timm 或 transformers);Phikon-v2 HF 直抓免申請。
2. 凍結 FM,對每張(或每 tile)抽特徵存 `.npy`(GPU inference, bs 8~16)。
3. 2964px 縮圖**別 resize 224**:切多個 224/256 tile → 各抽特徵 → mean-pool 或 attention-MIL 聚合。
4. 只訓 logistic regression / 1-2 層 MLP head(CPU/GPU 都極快)。
5. 新引擎落 `src/train_fm_probe.py`,app.py 照舊掛權重/特徵+head。

**B. SAHI 外掛(救大圖密集/稀疏小物件 det,不用重訓、不動授權)**
1. `pip install sahi`。
2. 用既有 YOLO best.pt,`get_sliced_prediction(slice=640, overlap=0.2)`。
3. 要更好可用 SAHI 切片把訓練圖也切片後 fine-tune。

**C. StarDist 第二引擎(救 H&E 密集核 det)**
1. `pip install stardist csbdeep`。
2. `StarDist2D.from_pretrained('2D_versatile_he')` 開箱推論,多半不訓練就堪用。
3. 當 det track 的替代輸出(instance mask → 計數/bbox)。

## 不要做的事
- 不要對 ViT-L 級 FM 在 8GB 全量微調(不切實際)→ 一律 frozen + head。
- 不要 imgsz=1280 訓練(OOM)→ 用 tiling + SAHI。
- 不要把 StarDist/HoVer-Net 用在「非核」的稀疏寄生蟲/細胞學偵測(那是數核工具)→ 用 YOLO+SAHI。
- 不要為了好看把 group-disjoint 改回隨機 split 來「提分」(見 [[fix-leaky-split]])。
- 標籤假/val 太小/任務上限低 → 換架構不會救,先誠實掛 caveat。
- 商用紅線:病理 FM 幾乎全 non-commercial、ultralytics 全 AGPL;要商用只剩 BiomedCLIP(MIT) / StarDist / InstanSeg / Detectron2 / SAHI。
