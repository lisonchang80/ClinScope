"""
ClinScope — unified clinical-microscopy recognition demo (Gradio).

Upload a microscopy image, pick the matching model:
  - classification models (ResNet-50)  -> top-k species/class probabilities
  - detection model (YOLO)             -> annotated image + per-class counts

Only models whose weight files exist are shown, so the app degrades gracefully
while individual tracks are still training.

Run:
    .\.venv\Scripts\python.exe app.py
then open http://127.0.0.1:7860
"""
from pathlib import Path
import csv
import html as _html
import json
import os
import re
import secrets
import shutil
import tempfile
from collections import Counter
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image, ImageFile
import gradio as gr

# cp950 console 下 print 中文/emoji 會 UnicodeEncodeError;腳本自我 reconfigure 為 utf-8
# (不依賴啟動端帶 PYTHONIOENCODING,任何 console 直接 `python app.py` 都不崩)
import sys as _sys
for _s in (_sys.stdout, _sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ImageFile.LOAD_TRUNCATED_IMAGES = True
if getattr(_sys, "frozen", False):     # PyInstaller 凍結:資料/模組都在 _MEIPASS(_internal)
    ROOT = Path(getattr(_sys, "_MEIPASS", Path(_sys.executable).resolve().parent))
else:
    ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_sys.path.insert(0, str(ROOT / "src"))
import weights as W          # 權重 lazy-download resolver(本機有→passthrough;缺→抓 HF) 見 docs/DEPLOY_PLAN.md

# --- model registry: one entry per track -----------------------------------
REGISTRY = [
    {"key": "fungi", "kind": "cls",
     "label": "🍄 真菌 Fungi — OpenFungi (LPCB 培養)",
     "path": ROOT / "models" / "resnet50_fungi.pt",
     "blurb": "環境/食品黴菌屬級分類(Aspergillus、Penicillium…)。LPCB 染色、培養後影像。"},
    {"key": "fungi_koh", "kind": "cls",
     "label": "🍄 真菌(KOH 直接鏡檢)— DeFungi",
     "path": ROOT / "models" / "defungi_resnet50.pt",
     "blurb": "KOH 直接鏡檢真菌 5 類(原始標註 H1–H6 編碼,含皮癬菌屬);對照 OpenFungi 的 LPCB 培養模態。"},
    {"key": "malaria", "kind": "cls",
     "label": "🦟 瘧原蟲 Malaria — NIH (Giemsa 血抹片)",
     "path": ROOT / "models" / "malaria_resnet50.pt",
     "blurb": "已切好的單一血球:受感染 Parasitized vs 未感染 Uninfected。"},
    {"key": "parasite_egg", "kind": "cls",
     "label": "🪱 腸道蟲卵 Parasite eggs — Chula-ParasiteEgg-11 (糞便鏡檢)",
     "path": ROOT / "models" / "parasite_egg_resnet50.pt",
     "blurb": "糞便寄生蟲卵 11 類:蛔蟲、鞭蟲、鉤蟲、蟯蟲、薑片蟲、肝吸蟲、肺吸蟲、絛蟲、縮小/微小膜殼絛蟲、毛細線蟲。對照瘧疾的 Giemsa 血抹片,屬糞便濕抹片模態。"},
    {"key": "wbc", "kind": "cls",
     "label": "🩸 白血球分類 WBC differential — PBC (Wright/Giemsa)",
     "path": ROOT / "models" / "pbc_resnet50.pt",
     "blurb": "周邊血液白血球 8 類:嗜中性/嗜酸/嗜鹼性球、淋巴球、單核球、未成熟顆粒球(IG)、紅血球母細胞、血小板。"},
    {"key": "bonemarrow", "kind": "cls",
     "label": "🦴 骨髓細胞 Bone marrow — Matek BM (21 類)",
     "path": ROOT / "models" / "bonemarrow_resnet50.pt",
     "blurb": "骨髓抹片細胞形態 21 類(Matek):芽細胞 BLA、前驅紅血球 PEB/EBO、各階段顆粒球(PMO/MYB/MMZ/NGB/NGS)、嗜酸/嗜鹼球、淋巴球 LYT/LYI、漿細胞 PLM、單核球 MON 等。May-Grünwald-Giemsa 染色,對照 PBC 周邊血;類別重度不平衡,稀有類成績較低。"},
    {"key": "bacteria", "kind": "cls",
     "label": "🧫 細菌 Bacteria — DIBaS (Gram 染色)",
     "path": ROOT / "models" / "dibas_resnet50.pt",
     "blurb": "33 種菌種分類(Gram 染色)。每類僅 ~20 張的小樣本,demo 性質。"},
    {"key": "ana", "kind": "cls",
     "label": "🟢 ANA 抗核抗體型態 — HEp-2 (免疫螢光 IIF)",
     "path": ROOT / "models" / "hep2_resnet50.pt",
     "blurb": "自體免疫篩檢:HEp-2 細胞 6 種螢光型態(均質/斑點/核仁/著絲點/核膜/高基氏體)。"},
    {"key": "pap", "kind": "cls",
     "label": "🔬 子宮頸抹片 Pap — SIPaKMeD (5 類細胞)",
     "path": ROOT / "models" / "sipakmed_resnet50.pt",
     "blurb": "細胞學:子宮頸抹片單顆細胞 5 類 — 正常(表層-中層/旁基底)、異常(挖空細胞 koilocyte/角化不良)、化生 metaplastic。Pap 染色明場。"},
    {"key": "sperm", "kind": "cls",
     "label": "🔬 精液形態 Sperm morphology — SMIDS (精液抹片)",
     "path": ROOT / "models" / "smids_resnet50.pt",
     "blurb": "男科/精液分析:精子形態 3 類 — 正常精子 Normal、異常精子 Abnormal、非精子物質 Non-sperm。亮場精液抹片,對應 WHO 精液分析的形態判讀。"},
    {"key": "bloodcells", "kind": "det",
     "label": "🩸 血球偵測 Blood cells — BCCD (YOLO)",
     "path": ROOT / "reports" / "yolo" / "bccd" / "weights" / "best.pt",
     "blurb": "整片血抹片偵測 + 計數:RBC / WBC / Platelets。"},
    # --- 組織病理 Histopathology (H&E 組織切片) ---
    {"key": "lc25000", "kind": "cls",
     "label": "🧬 肺/結腸組織病理 LC25000 (H&E, 5 類)",
     "path": ROOT / "models" / "lc25000_resnet50.pt",
     "blurb": "組織病理 H&E 切片 5 類:結腸腺癌/正常、肺腺癌/正常/鱗狀細胞癌。本專案首條組織病理 track。⚠️ LC25000 由 1250 原圖增強擴增成 25000 張,隨機 split 會把同源增強圖分到 train/val → val 1.0 偏樂觀,需 image-source-level split 才可信。"},
    {"key": "breakhis", "kind": "cls",
     "label": "🧬 乳腺腫瘤組織病理 BreakHis (H&E, 良性/惡性)",
     "path": ROOT / "models" / "breakhis_resnet50.pt",
     "blurb": "組織病理 H&E 乳腺腫瘤二元分類:良性 benign vs 惡性 malignant(40–400× 放大倍率混訓,magnification-independent)。**病人級切分**(70 病人,val 18 人 disjoint、同病人不跨 train/val)val 0.839、macro-F1 0.829。對比舊隨機 split 的 0.978 掉約 14pt —— 那是同病人不同倍率洩漏灌出的假象,0.839 才是誠實泛化;benign recall 偏低 0.69(模型偏判 malignant)。"},
    {"key": "nct_crc", "kind": "cls",
     "label": "🧬 結直腸組織 9 類 NCT-CRC-HE-100K (H&E)",
     "path": ROOT / "models" / "nct_crc_resnet50.pt",
     "blurb": "組織病理 H&E 結直腸組織 9 類:腫瘤上皮 TUM、基質 STR、淋巴球 LYM、平滑肌 MUS、黏液 MUC、正常黏膜 NORM、脂肪 ADI、碎屑 DEB、背景 BACK。在官方獨立 cohort(CRC-VAL-HE-7K)驗證 val 0.970 —— 跨 cohort、較誠實;STR 基質↔MUS 肌肉最易混。"},
    {"key": "pcam", "kind": "cls",
     "label": "🧬 淋巴結轉移 PatchCamelyon (H&E, binary)",
     "path": ROOT / "models" / "pcam_resnet50.pt",
     "blurb": "組織病理 H&E 淋巴結 96×96 patch 二元分類:有無轉移腫瘤 metastasis(no_tumor / tumor)。知名 PatchCamelyon benchmark;在官方 32,768 張驗證集 val 0.913、兩類 F1 ~0.91。"},
    {"key": "mhist", "kind": "cls",
     "label": "🔬 大腸息肉 MHIST (H&E, HP vs SSA)",
     "path": ROOT / "models" / "mhist_resnet50.pt",
     "blurb": "消化道病理 H&E 大腸息肉二元『診斷』:增生性息肉 Hyperplastic vs 無柄鋸齒狀腺瘤 Sessile-serrated adenoma(臨床上易混、判讀者間一致性低)。val 0.922。與其他 H&E track 不同 —— 這是診斷鑑別、非組織分型。⚠️ 隨機 split 可能同片洩漏(同一 WSI 多 tile)。"},
    {"key": "thyroid_cyto", "kind": "cls",
     "label": "🔬 甲狀腺細胞學 Thyroid FNA (6 級)",
     "path": ROOT / "models" / "thyroid_cyto_resnet50.pt",
     "blurb": "細胞學:甲狀腺細針穿刺 FNA 6 類(原始標註 1–6,疑對應 Bethesda I–VI,資料卡未明確定義)。**slide(片)級切分**(118 片,每類留 3–6 片 disjoint、同片不跨 train/val)誠實 val 0.505、macro-F1 0.518。⚠️ 對比舊隨機 split 的 0.947 暴跌約 44pt —— 三條修正中洩漏最嚴重,0.947 幾乎全是在背同一抹片的外觀(每片數百 tile)。slide-disjoint 下:形態明確的 class 1(F1 0.79)、class 6(0.89)能學,中間級 class 2–5(0.16–0.50)很難——與臨床上 Bethesda 中間類判讀者間變異大一致。對照子宮頸 Pap,屬不同部位細胞學。"},
    {"key": "mitosis", "kind": "det",
     "label": "🔬 有絲分裂偵測 Mitosis — 乳腺 H&E (YOLO)",
     "path": ROOT / "reports" / "yolo" / "mitosis" / "weights" / "best.pt",
     "blurb": "乳腺組織病理 H&E 有絲分裂象 mitotic figure 偵測 + 計數(每高倍視野有絲分裂數是乳癌分級指標)。YOLOv8,mAP50 0.941 / mAP50-95 0.651。第 2 條偵測 track(對照 BCCD 血球)。小資料 274 圖 demo。"},
    {"key": "husem", "kind": "cls",
     "label": "🔬 精子頭形態 HuSHeM (4 類)",
     "path": ROOT / "models" / "husem_resnet50.pt",
     "blurb": "男科:人類精子頭部形態 4 類 — 正常 Normal / 漸細 Tapered / 梨形 Pyriform / 無定形 Amorphous(Diff-Quik 染色頭部裁切)。對照 SMIDS 全細胞 3 類,本 track 聚焦頭部。⚠️ 極小資料(216 張、val 每類僅 8 張),val 0.94 噪音極大、不可信。"},
    # --- batch 3 新增 (紅血球形態 / 白血病 / 瘧種 / 細胞核偵測 / 口腔) ---
    {"key": "rbc_morphology", "kind": "cls",
     "label": "🩸 紅血球形態 RBC morphology (11 類)",
     "path": ROOT / "models" / "rbc_morphology_resnet50.pt",
     "blurb": "血液抹片紅血球形態 11 類:正常 normal、標靶 Target cell、淚滴 teardrop、口形 Stomatocyte、球形 Spherocyte、橢圓 Ovalocyte、雪茄/鉛筆 cigar、棘 Acanthocyte、刺 Burr、裂片 Schistocyte、低色素 Hypochromia。poikilocytosis 異形紅血球形態學。val 0.978、macro-F1 0.972(sqrt 類別權重平衡不平衡)。⚠️ Schistocyte val 僅 12 張(<15)、單細胞裁切隨機 split 同抹片細胞可能跨 train/val,嚴格需 smear-level split。"},
    {"key": "cnmc_leukemia", "kind": "cls",
     "label": "🩸 急性淋巴芽細胞白血病 C-NMC (blast vs 正常)",
     "path": ROOT / "models" / "cnmc_leukemia_resnet50.pt",
     "blurb": "血液腫瘤:單細胞顯微下急性淋巴芽細胞白血病 B-ALL 芽細胞 cancer (blast) vs 正常 B 淋巴系 healthy (HEM) 二元分類(C-NMC 2019 ISBI challenge)。**病人級(subject_id disjoint)切分的誠實 val**:val_acc 0.853、macro-F1 0.824、weighted-F1 0.851 —— 同一病人不跨 train/val,無洩漏(對照本專案多數隨機 split track 的樂觀數字)。blast 與正常淋巴細胞形態高度相似為本任務難點。"},
    {"key": "mpidb_species", "kind": "cls",
     "label": "🦟 瘧原蟲蟲種 MP-IDB (4 種)",
     "path": ROOT / "models" / "mpidb_species_resnet50.pt",
     "blurb": "寄生蟲種級:瘧原蟲 4 蟲種分類 — 惡性瘧 Falciparum / 三日瘧 Malariae / 卵形瘧 Ovale / 間日瘧 Vivax(薄血膜 Giemsa,**全視野影像**、非單一寄生蟲裁切)。**slide(血片)級切分**(17 片,5 片 disjoint 留 val、同片不跨 train/val)誠實 val 0.509、macro-F1 0.348 —— 對比舊隨機 split 的 0.997 崩落,正是 img/crops 同視野洩漏灌出的假象。⚠️ 但資料集結構性受限:Falciparum 10 片、其餘僅 2–3 片 → Malariae 完全失敗(F1 0.00、僅 1 片可訓)、Ovale 0.18,只有 Falciparum(0.53)/Vivax(0.68)有意義;且全視野於 224px 下寄生蟲過小,模型多依染片外觀而非蟲體形態。"},
    {"key": "erythrocytes_idb", "kind": "cls",
     "label": "🩸 鐮刀型紅血球 erythrocytesIDB (3 類)",
     "path": ROOT / "models" / "erythrocytes_idb_resnet50.pt",
     "blurb": "血液抹片紅血球形狀 3 類:正常圓形 circular / 鐮刀變形 elongated(鐮刀型貧血 sickle cell)/ 其他 other。單顆紅血球裁切。val 0.984、macro-F1 0.984(三類均衡)。對照 RBC morphology 的 11 類異形,本 track 聚焦鐮刀型鑑別。⚠️ val 僅 125 張偏小、單細胞裁自 196 source 圖,隨機 split 可能同源跨組 → 嚴格需 group-aware split。"},
    {"key": "pannuke", "kind": "det",
     "label": "🔬 細胞核偵測 PanNuke — H&E (5 類, YOLO)",
     "path": ROOT / "reports" / "yolo" / "pannuke" / "weights" / "best.pt",
     "blurb": "泛組織病理 H&E 細胞核偵測 + 分類 5 類:腫瘤 neoplastic / 發炎 inflammatory / 結締組織 connective / 壞死 dead / 上皮 epithelial。YOLOv8s,平均 23.8 核/圖、密集小目標(val 共 6.7 萬框)。val mAP50 0.654 / mAP50-95 0.405(neoplastic·epithelial ~0.79 最佳;dead 0.31 稀少最弱)。第 3 條偵測 track(對照 BCCD 血球、mitosis 有絲分裂)。mask→YOLO bbox 由 src/pannuke_to_yolo.py 轉換。"},
    {"key": "univali_leukocyte", "kind": "cls",
     "label": "🩸 白血球 14 類 UNIVALI (含未成熟顆粒球/芽細胞)",
     "path": ROOT / "models" / "univali_leukocyte_resnet50.pt",
     "blurb": "周邊血抹片白血球 14 類 —— 在 PBC 8 類基礎上擴入臨床更難的未成熟系列:芽細胞 Blast、前/中/晚幼顆粒球 Promyelocyte/Myelocyte/Metamyelocyte、桿狀 Band、紅血球母細胞 Erythroblast、非典型淋巴球 Atypical lymphocyte,以及人工物/碎屑 Artifact/Debris。val_acc 0.905、weighted-F1 0.90、macro-F1 0.816(成熟系強,未成熟稀有類弱:Metamyelocyte F1 0.43)。對照 PBC 周邊血 8 類,本 track 細到未成熟分化階段。⚠️ 官方 val 小(432)、多稀有類 val<15(Basophil/Erythroblast/Artifact 等)、augment ×2.6 可能增強同源洩漏。"},
    {"key": "bach", "kind": "cls",
     "label": "🧬 乳腺組織 4 級 BACH (H&E, 嚴重度梯度)",
     "path": ROOT / "models" / "bach_resnet50.pt",
     "blurb": "組織病理 H&E 乳腺顯微 4 級嚴重度分類:正常 Normal / 良性 Benign / 原位癌 InSitu / 浸潤癌 Invasive(ICIAR 2018 BACH)。400 張大圖(2048×1536)切成 512 tile;**影像級切分**(同一張圖的 tile 不跨 train/val,無 tile 洩漏)。val(tile 級、來自 80 張獨立來源圖)acc 0.873、macro-F1 0.873;Invasive 最佳 0.93,Benign↔Normal 較易混(臨床合理)。別於 BreakHis 良惡二元,本 track 是 4 級嚴重度梯度。"},
    {"key": "pathotiroid", "kind": "cls",
     "label": "🧬 甲狀腺乳突癌組織 PathoTiroid (H&E, PTC vs 非PTC)",
     "path": ROOT / "models" / "pathotiroid_resnet50.pt",
     "blurb": "組織病理 H&E 甲狀腺二元:乳突癌樣 PTC-like vs 非乳突癌 Non-PTC(PTIC 資料集)。**病人級(patient-disjoint)切分** —— 24 病人留 5 人當 val,同一病人不跨 train/val,無洩漏。val_acc 0.994、macro-F1 0.992(PTC 核特徵 Orphan-Annie 核/核溝辨識度高)。對照甲狀腺 FNA 細胞學 track(細胞抹片),本 track 是組織切片。⚠️ val 僅 5 病人 → 小 cohort、高變異,0.994 換新中心/掃描儀未必維持。"},
    {"key": "urine", "kind": "det",
     "label": "🔬 尿沉渣顆粒 Urine sediment (7 類, YOLO)",
     "path": ROOT / "reports" / "yolo" / "urine" / "weights" / "best.pt",
     "blurb": "尿液分析:明場尿沉渣顆粒偵測 + 分類 7 類 —— 管型 cast、結晶 cryst、上皮細胞 epith/epithn、紅血球 eryth、白血球 leuko、黴菌/酵母 mycete。YOLOv8s,官方 train/valid split。val mAP50 0.896 / mAP50-95 0.545(紅血球 eryth 0.957、白血球 leuko 0.942 最佳;管型 cast 0.753 稀少較弱)。第 4 條偵測 track,新增尿液分析 site。"},
    {"key": "mll23", "kind": "cls",
     "label": "🩸 周邊血 18 類 MLL23 (含毛細胞白血病/芽細胞)",
     "path": ROOT / "models" / "mll23_resnet50.pt",
     "blurb": "周邊血單細胞 18 類 —— 比 PBC-8/UNIVALI-14 更細的診斷導向分類:含毛細胞白血病 hairy cell、芽細胞 myeloblast、非典型前髓細胞 promyelocyte atypical、抹片細胞 smudge cell、漿細胞 plasma cell、反應性/腫瘤性淋巴球、紅血球母細胞 normoblast 及各階段顆粒球。MLL 大型血液病理庫。val_acc 0.948、weighted-F1 0.948、macro-F1 0.861(hairy cell 0.99、myeloblast 0.98 等診斷關鍵類佳;過渡型 band/metamyelocyte 較弱)。⚠️ 長尾,lymphocyte_reactive val 僅 5 張(F1 0.33 不可信)。"},
    {"key": "rouleaux", "kind": "cls",
     "label": "🩸 紅血球串錢狀 Rouleaux (正常 vs rouleaux)",
     "path": ROOT / "models" / "rouleaux_resnet50.pt",
     "blurb": "血液抹片紅血球排列 二元:正常分散 vs 串錢狀聚集 rouleaux(RBC 像錢幣堆疊,見於高球蛋白血症/發炎/多發性骨髓瘤)。用全視野影像(rouleaux 是排列現象、需看整片,非單細胞)。val 1.0 —— ⚠️ 此任務視覺差異極大、本就易;val 僅 232 張、檔名受試者編碼不一致無法保證 subject-disjoint,1.0 偏樂觀,需 subject-level split 才可信(同 DIBaS/LC25000 型)。對照鐮刀/異形單細胞 track,本 track 看 RBC 整體排列。"},
    {"key": "mk11", "kind": "cls",
     "label": "🦴 骨髓巨核細胞 MK-11 (11 亞型)",
     "path": ROOT / "models" / "mk11_resnet50.pt",
     "blurb": "骨髓巨核細胞 megakaryocyte 形態 11 亞型(MK-blast 芽細胞、各成熟階段 Pro/N/PP/MMK、低分葉 HL-MK、微小巨核 MN-MK、裸核 NN-MK、細胞質異常 CA-MK 等)。Wright-Giemsa 單細胞。**官方 5-fold 病人級切分(無洩漏誠實 val)**:val_acc 0.810、macro-F1 0.758(MK-blast 0.98 等診斷關鍵類佳;稀有 Pro-MK/MN-MK 較弱)。新增巨核細胞譜系(對照 Matek 骨髓全細胞、PBC/MLL23 周邊血)。⚠️ 長尾,數個稀有亞型 val<20。"},
    {"key": "tryp", "kind": "det",
     "label": "🦟 錐蟲偵測 Trypanosome — 厚血膜 (YOLO)",
     "path": ROOT / "reports" / "yolo" / "tryp" / "weights" / "best.pt",
     "blurb": "血液寄生蟲:未染色厚血膜錐蟲 trypanosome 偵測(非洲昏睡病 T. brucei / 美洲查加斯病 T. cruzi)。YOLOv8s 單類偵測(含陰性背景圖降假陽性),官方 v3 split。val mAP50 0.807 / mAP50-95 0.439。第 5 條偵測 track,新增血液寄生蟲(對照瘧原蟲血抹片分類)。"},
    {"key": "patchgastric", "kind": "cls",
     "label": "🧬 胃腺癌分化分級 PatchGastric (H&E, 4 級)",
     "path": ROOT / "models" / "patchgastric_resnet50.pt",
     "blurb": "組織病理 H&E 胃腺癌分化分級 4 類:高分化 Well / 中分化 Moderate / 低分化 Poor / 乳突狀 Papillary(PatchGastricADC22,WSI patch、每片 cap 40)。**WSI 級切分(同片 patch 不跨 train/val,無洩漏)**。val_acc 0.489、macro-F1 0.477 —— ⚠️ 分化分級是連續譜、判讀者間變異大,WSI 級無洩漏下本就難(隨機 0.25、本模型 0.49,有學到但偏弱);此 track 正是『誠實切分 vs 洩漏樂觀數字』的對照。新增胃組織病理器官。"},
    {"key": "ipdbrain", "kind": "cls",
     "label": "🧠 腦膠質瘤亞型 IPD-Brain (H&E, 3 類)",
     "path": ROOT / "models" / "ipdbrain_resnet50.pt",
     "blurb": "組織病理 H&E 腦膠質瘤亞型 3 類:膠質母細胞瘤 Glioblastoma / 星形細胞瘤 Astrocytoma / 寡突膠質瘤 Oligodendroglioma(IPD-Brain,大圖切 512 tile)。**病人級(Case Number disjoint)切分、無洩漏**。val_acc 0.529、macro-F1 0.332 —— ⚠️ **誠實的弱結果**:模型幾乎只猜多數類 GBM(F1 0.71),星形/寡突召回極低(0.15/0.06),0.529 ≈ GBM 佔比基線;膠質瘤亞型需 H&E 形態 + 分子標記(IDH/1p19q),低倍 tile 已失判別訊號 → 非臨床可用,示範 WSI 降採樣的極限。新增腦組織器官。"},
    {"key": "blood_protozoa", "kind": "cls",
     "label": "🦠 血液原蟲 3 類 (Babesia/Toxoplasma/Trypanosoma)",
     "path": ROOT / "models" / "blood_protozoa_resnet50.pt",
     "blurb": "血液/組織原蟲分類 3 類:焦蟲 Babesia(紅血球內環狀)/ 弓形蟲 Toxoplasma(新月形速殖子)/ 錐蟲 Trypanosoma(鞭毛蟲體)。Giemsa 顯微。val 1.0 —— ⚠️ 三種形態差異極大、本就易;val 僅 158 張、隨機 split 可能同源洩漏,1.0 偏樂觀。Babesia/Toxoplasma 為新增蟲種;對照 Tryp 偵測 track,本 track 是多蟲種分類。"},
    # --- batch 5 新增 (新科別/新模態: 細胞遺傳 / IHC / 腸道原蟲 / 瘧分期 / 革蘭氏痰 / 菌落 / 細胞學) ---
    {"key": "chromosome", "kind": "cls",
     "label": "🧬 染色體核型分類 — G-band 24 類 (CIR-Net)",
     "path": ROOT / "models" / "chromosome_resnet50.pt",
     "blurb": "細胞遺傳學(本平台首條 cytogenetics track):G-band(Giemsa 帶型、灰階)單條染色體 24 類 — 第 1–22 號體染色體 + X + Y(類別碼 00–23 對應 CIR-Net 原始標註,22=X、23=Y)。val_acc 0.886、macro-F1 0.889,24 類全部 F1≥0.75。⚠️ 無官方 split、檔名無法還原 karyotype/病人 id → 隨機 split;同一 metaphase 影像批次外觀可能略微灌高 val(惟染色體 cls 的批次洩漏遠輕於同片 tile-leak)。class 23(Y)僅 6 val,F1 1.0 不可盡信。"},
    {"key": "her2_ihc", "kind": "cls",
     "label": "🟤 HER2 免疫組化評分 — IHC (0/1+/2+/3+)",
     "path": ROOT / "models" / "her2ihc_resnet50.pt",
     "blurb": "免疫組化 IHC(本平台首條 IHC track、DAB 棕色顯色,別於所有 H&E 組織 track):乳腺癌 HER2 細胞膜染色強度 4 級評分 0 / 1+ / 2+ / 3+(對應 HER2 標靶治療資格判讀)。官方 patch 級切分(split-before-extract,WSI/case 不跨 train/val → 低洩漏)val_acc 0.985、macro-F1 0.982,4 類 F1≥0.97(含最少的 2+,150 val)。HER2 分數本質是膜染色強度梯度、視覺可分性高,故高準確率屬合理而非洩漏。"},
    {"key": "parasitobank", "kind": "det",
     "label": "🦠 腸道寄生蟲偵測 ParasitoBank (糞便濕抹片, 13 類, YOLO)",
     "path": ROOT / "reports" / "yolo" / "parasitobank" / "weights" / "best.pt",
     "blurb": "寄生蟲學(本平台首條腸道原蟲 track):糞便濕抹片(saline + Lugol 碘染)腸道寄生蟲偵測 13 類 — 腸道原蟲(梨形鞭毛蟲 Giardia / 大腸阿米巴 E. coli / 哈氏阿米巴 E. hartmanni / 內阿米巴複合 / 微小內蜒阿米巴 Endolimax nana / 碘阿米巴 Iodamoeba / 唇鞭毛蟲 Chilomastix / 人芽囊原蟲 Blastocystis / 球蟲 coccidia)+ 蠕蟲卵(蛔蟲 Ascaris / 絛蟲 Taenia / 鞭蟲 Trichuris / 糞線蟲 Strongyloides)。YOLOv8s,val mAP50 0.752 / mAP50-95 0.456。別於既有 Chula 蟲卵 track —— 本 track 聚焦腸道原蟲、糞便濕抹片不同染色。⚠️ 長尾:最常見的人芽囊原蟲 mAP50 0.43(形態變異大)反而最難;稀有蠕蟲(taenia/tricocefalo 等 val 僅 1–2 框)mAP 雖高但統計不可信。"},
    {"key": "bbbc041_stage", "kind": "det",
     "label": "🦟 瘧原蟲生活史分期偵測 BBBC041 (P. vivax, YOLO)",
     "path": ROOT / "reports" / "yolo" / "bbbc041" / "weights" / "best.pt",
     "blurb": "寄生蟲分期(新切角,別於既有瘧種 MP-IDB / 感染細胞 NIH track):間日瘧原蟲 P. vivax 薄血膜 Giemsa 全視野,偵測生活史 4 期 ring 環狀體 / trophozoite 滋養體 / schizont 裂殖體 / gametocyte 配子體 + leukocyte 白血球(5 類,RBC 紅血球背景未標為目標)。官方 train/test split(1208/120)。YOLOv8s val mAP50 0.395 / mAP50-95 0.306 —— 誠實的中等結果:trophozoite(mAP50 0.59)、ring(0.43)可偵測,期別形態重疊 + 蟲體於全視野偏小使整體偏低。⚠️ schizont/gametocyte val 僅 11/12 框、leukocyte val=0 無法驗證,稀有期不可信;單一蟲種、非商用授權。"},
    {"key": "mrose_gram", "kind": "det",
     "label": "🦠 革蘭氏染色痰抹片偵測 M-ROSE (G+/G-, YOLO)",
     "path": ROOT / "reports" / "yolo" / "mrose" / "weights" / "best.pt",
     "blurb": "微生物快速現場評估 M-ROSE(本平台首條革蘭氏染色抹片 track,別於 DIBaS 菌落 cls):下呼吸道痰抹片 Gram 染色,偵測菌體並依染色反應分 2 類 — 革蘭氏陽性 G+(紫)/ 革蘭氏陰性 G-(紅)。源圖切 640 tile、source-disjoint 切分(1798/449 源圖,同源 tile 不跨 train/val、無洩漏)。YOLOv8s val mAP50 0.908 / mAP50-95 0.667(G- 0.937、G+ 0.879)。Gram 反應是臨床微生物的第一線判讀。⚠️ 資料集另有 4 類(染色×菌形)變體,但無類名檔故本 track 採語意明確的 2 類 Gram 反應;G+ 為少數類(train 2045 vs G- 6989)。"},
    {"key": "dataset19_colony", "kind": "det",
     "label": "🧫 培養盤菌落偵測 Dataset19 (19 菌種, YOLO)",
     "path": ROOT / "reports" / "yolo" / "dataset19" / "weights" / "best.pt",
     "blurb": "臨床微生物培養(本平台首條巨觀培養盤 track、全新模態 — 白光巨觀攝影而非顯微):瓊脂培養盤上 19 種臨床細菌的菌落偵測 + 菌種辨識。類別沿用資料集原始 3-字母代碼(aba/eco/kpn/sau/ses/sma/sep/efa/stm/eca/enc/cst/mmo/bce/spy/sag/ppu/spn/bcp;涵蓋 Acinetobacter / E. coli / Klebsiella / S. aureus / Enterococcus / Streptococcus 等)。YOLOv8s val mAP50 0.970 / mAP50-95 0.764,19 種 mAP50 0.93–0.99 均勻優異。菌落判讀是臨床微生物鑑定的核心。⚠️ stratified-by-species 切分(每種 40 train/10 val 圖);若同株多影像為同盤裁切可能輕微洩漏。對照 DIBaS(顯微 Gram 菌落 cls),本 track 是整盤巨觀偵測。"},
    {"key": "leishmania", "kind": "det",
     "label": "🦠 皮膚利什曼原蟲偵測 Leishmania (Giemsa 皮膚抹片, YOLO)",
     "path": ROOT / "reports" / "yolo" / "leishmania" / "weights" / "best.pt",
     "blurb": "寄生蟲(本平台首條利什曼原蟲 track,皮膚黏膜利什曼病):Giemsa 皮膚抹片偵測利什曼原蟲無鞭毛體 amastigote(單類 parasite)。官方 slide-disjoint 切分(德文 geteilte Slides,合併兩子資料集、同片不跨 train/val、無洩漏)、含兩種顯微鏡來源(含可攜式)增加域變異。YOLOv8s val mAP50 0.651 / mAP50-95 0.265 —— amastigote 微小、皮膚抹片背景雜,屬合理中等偵測;定位精度(mAP50-95)偏低。對照血液原蟲分類 track(Babesia/Toxoplasma/Trypanosoma),本 track 是皮膚利什曼的偵測。AGPL-3.0 授權。"},
    {"key": "giardia_crypto", "kind": "det",
     "label": "🦠 Giardia/Cryptosporidium 囊體偵測 (明場顯微, YOLO)",
     "path": ROOT / "reports" / "yolo" / "giardia" / "weights" / "best.pt",
     "blurb": "寄生蟲(水源性原蟲囊體偵測,新增 Cryptosporidium):明場顯微偵測 2 類 — 梨形鞭毛蟲囊體 Giardia cyst / 隱孢子蟲卵囊 Cryptosporidium oocyst(水質/食媒寄生蟲監測核心)。合併 brightfield reference + sample 兩子集(同明場模態)、各用官方 fold_1 split。YOLOv8s val mAP50 0.895 / mAP50-95 0.463(Giardia 0.947、Crypto 0.844)。⚠️ 來源為水/蔬菜環境樣本、非臨床糞便檢體(囊體/卵囊形態可遷移,但臨床糞便基質更雜);對照 ParasitoBank(臨床糞便腸道原蟲),本 track 聚焦環境水源、且新增 Cryptosporidium。"},
    # --- batch 6 (2026-06-14) ---
    {"key": "kupbc13", "kind": "cls",
     "label": "🩸 白血球+血小板 13 類 KU-optofil (May-Grünwald-Giemsa)",
     "path": ROOT / "models" / "kupbc13_resnet50.pt",
     "blurb": "血液(13 類單顆細胞,新增血小板形態):Sysmex DI-60 自動數位顯微、May-Grünwald-Giemsa 周邊血。11 種白血球(節狀/桿狀嗜中性球、嗜酸/嗜鹼球、淋巴球、反應性淋巴球、單核球、後髓/髓/晚幼粒細胞、芽細胞、紅血球母細胞)+ 2 種血小板形態(巨大血小板 Giant Platelet / 血小板聚集 Platelet Cluster)。官方 patient-level split(無洩漏)。ResNet-50 val_acc 0.961 / macro-F1 0.845。血小板形態 2 類為既有 WBC track(PBC/UNIVALI/MLL23)所無;其餘細胞與之重疊。⚠️ 類別不平衡:Segmented Neutrophil 佔多數;Myelocyte(val 僅 9 張)F1 0.36、Band Neutrophil/Reactive Lymphocyte 偏弱,macro(0.84) vs weighted(0.96) 落差反映稀有類較弱。"},
    {"key": "mdsbm27", "kind": "cls",
     "label": "🦴 骨髓 MDS 細胞 29 類 (May-Grünwald-Giemsa)",
     "path": ROOT / "models" / "mdsbm27_resnet50.pt",
     "blurb": "血液/骨髓(本平台首條 MDS 骨髓異常增生 track,別於既有健康/AML 骨髓 Matek/MK-11):骨髓抹片 May-Grünwald-Giemsa,29 類含 MDS 特徵性發育不良細胞(發育不良紅血球母細胞/顆粒球、假 Pelger 等)與正常各階段。125 名 MDS 病人,合併 main+add 子集。ResNet-50 val_acc 0.773 / macro-F1 0.590。⚠️ 隨機 split(資料集無病人 id 可分組,同片細胞可能輕微洩漏);類別重度不平衡,稀有發育不良類偏弱;已剔除 <10 張無法切分的 3 類(原 32→29)。"},
    {"key": "sicap", "kind": "cls",
     "label": "🔬 攝護腺 Gleason 分級 — SICAPv2 (H&E, NC/G3/G4/G5)",
     "path": ROOT / "models" / "sicap_resnet50.pt",
     "blurb": "組織病理(本平台首條攝護腺 track):攝護腺切片 H&E patch,Gleason 分級 4 類 — 非癌 NC / Gleason pattern 3 / 4 / 5(SICAPv2 正規化子集,G4 cribriform 併入 G4)。官方 slide-disjoint split(train 7472 / val 2487,同片不跨、無洩漏)。ResNet-50 val_acc 0.721 / macro-F1 0.630。Gleason 分級本就困難、形態連續。⚠️ G5(高惡性度、較稀)F1 0.28 偏弱;NC 最佳 0.84。"},
    {"key": "sbsd", "kind": "det",
     "label": "🩸 血抹片細胞偵測 SBSD (含裂紅細胞 Schistocyte, 6 類, YOLO)",
     "path": ROOT / "reports" / "yolo" / "sbsd" / "weights" / "best.pt",
     "blurb": "血液(本平台首條含裂紅細胞 schistocyte 的偵測 track):周邊血抹片 Giemsa,偵測 6 類 — RBC / 裂紅細胞 Schistocyte(微血管病性溶血、TTP/HUS 標誌)/ 血小板 PLT / 顆粒球 GRAN / 單核球 MONO / 淋巴球 LYM。YOLOv8s val mAP50 0.895 / mAP50-95 0.713。裂紅細胞為既有血液 track 所無的關鍵形態。⚠️ 影像級隨機 split(檔名僅流水號、無 slide id 可分組,可能輕微洩漏)。"},
    {"key": "evisan", "kind": "det",
     "label": "🔬 精子偵測 EVISAN (多精子偵測, YOLO)",
     "path": ROOT / "reports" / "yolo" / "evisan" / "weights" / "best.pt",
     "blurb": "男科(精子偵測,新任務 — 別於既有精子形態分類 SMIDS/HuSHeM):顯微多精子偵測,2 類依精子頭外觀 black/white(沿用資料集原始標註碼)。源為連續影片幀(200X+600X),採 video-block-disjoint 切分(整段不跨 train/val、避免相鄰幀洩漏)。YOLOv8s val mAP50 0.436 / mAP50-95 0.253 —— 誠實中等:video-disjoint + 200X/600X 倍率分布差異使 val 偏難(比隨機幀切誠實)。對應電腦輔助精液分析 CASA 的精子定位/計數。"},
    {"key": "bcg", "kind": "det",
     "label": "🦠 血培養革蘭氏微生物偵測 (Gram 染色, YOLO)",
     "path": ROOT / "reports" / "yolo" / "bcg" / "weights" / "best.pt",
     "blurb": "微生物(本平台首條陽性血培養 track,菌血症/敗血症關鍵):陽性血培養抹片 Gram 染色(×100 油鏡)偵測微生物 1 類 microorganism(涵蓋革蘭氏陽/陰球菌、桿菌、酵母菌等多種形態,57 菌種來源)。YOLOv8s val mAP50 0.737 / mAP50-95 0.420。血培養陽性的快速形態判讀直接影響經驗性抗生素選擇。官方 DoubleCheck 共識標註。⚠️ 影像級隨機 split(505 圖各為一名病人陽性血培養,image-level≈patient-level、洩漏低);僅偵測「有無微生物」不分菌種(菌種仍需培養/質譜)。"},
    {"key": "puma", "kind": "det", "sahi": True,
     "label": "🔬 黑色素瘤組織核偵測 PUMA (H&E, 10 核類, YOLO+SAHI)",
     "path": ROOT / "reports" / "yolo" / "puma" / "weights" / "best.pt",
     "blurb": "組織病理(本平台首條皮膚黑色素瘤 track):原發+轉移黑色素瘤 H&E ROI,偵測 10 種細胞核 — 腫瘤核/淋巴球/基質/內皮/組織球/嗜黑素細胞 melanophage/漿細胞/嗜中性球/凋亡核/上皮。YOLOv8s 整圖 val mAP50 0.193;**本 track 前端啟用 SAHI 切片推論(640/overlap 0.2)→ mAP50 0.200→0.276(+37.6%,零重訓)**——1024px 訓練尺度≈slice 尺度,密集小核切片後不被縮掉、找回約 2× 框。10 類極不平衡(腫瘤核 57k vs 嗜中性 366),稀有核型偏弱。含黑色素瘤特有 melanophage 核型,為既有 PanNuke/NuInsSeg 所無。ROI 級切分,CC0 授權。"},
    {"key": "tb_det", "kind": "det",
     "label": "🦠 結核抗酸桿菌偵測 TB-ZN (Ziehl-Neelsen 痰抹片, YOLO)",
     "path": ROOT / "reports" / "yolo" / "tb_det" / "weights" / "best.pt",
     "blurb": "微生物(本平台首條結核 track,填補全球結核負擔的關鍵 ★ 缺口):ZN 抗酸染色痰抹片(手機接顯微鏡,Makerere/Mulago 烏干達臨床)偵測抗酸桿菌 1 類 TBbacillus。YOLOv8s val mAP50 0.840 / mAP50-95 0.414(P 0.78 / R 0.77),抗酸桿菌偵測表現良好。ZN 鏡檢是高負擔地區結核診斷第一線。⚠️ 影像級隨機切分(phonecamera 連號視野、無 slide id,相鄰視野可能輕微洩漏)。"},
    {"key": "tb_cls", "kind": "cls",
     "label": "🦠 結核塗片陽/陰性 TB-ZN (Ziehl-Neelsen, ResNet)",
     "path": ROOT / "models" / "tb_cls_resnet50.pt",
     "blurb": "微生物(結核塗片判讀,搭配 tb_det 偵測):ZN 痰抹片整張陽性(含抗酸桿菌)/陰性二分類。ResNet-50 val_acc 0.99 / macro-F1 0.972(Negative 0.994、Positive 0.95)。⚠️ Positive val 僅 20 張、影像級隨機切分(同片視野可能輕微洩漏);ZN 陽陰視覺可分性高,高分屬合理但仍需片級切分驗證。Mendeley 來源(培養確診)。"},
    {"key": "nuinsseg", "kind": "det",
     "label": "🔬 多器官細胞核偵測 NuInsSeg (H&E 31 器官, YOLO)",
     "path": ROOT / "reports" / "yolo" / "nuinsseg" / "weights" / "best.pt",
     "blurb": "組織病理(多器官泛化,補既有 PanNuke 單一來源):31 種人體器官 H&E 的細胞核偵測(1 類 nucleus),由 instance mask 連通標記轉 bbox。YOLOv8s val mAP50 0.618 / mAP50-95 0.334。涵蓋膀胱/腦/腎/肝/小腸/子宮頸…等 31 器官,器官分層切分;對照 PanNuke(H&E 核 5 類),本 track 強調跨器官泛化的單類核偵測。"},
    {"key": "effusion", "kind": "cls",
     "label": "🔬 積液細胞學 良性/惡性 (體腔積液, ResNet)",
     "path": ROOT / "models" / "effusion_resnet50.pt",
     "blurb": "細胞學(本平台首條體腔積液 track):肋膜/腹腔/心包積液離心抹片,良性 vs 惡性細胞團二分類。ResNet-50 在小驗證集達 val 1.0。⚠️ **影像級隨機切分、693 圖疑似同案例多團塊分散到 train/val 造成洩漏,1.0 不代表真實表現**,僅供流程 demo;臨床判積液良惡性很常見,但此數字需病例級切分才可信。授權 © 原作者(Parikshit Sanyal),僅研究/作品集用。"},
    # --- batch 6 續 (2026-06-14) ---
    {"key": "msi", "kind": "cls",
     "label": "🔬 大腸癌微衛星不穩定 MSI/MSS (H&E, 分子病理, ResNet)",
     "path": ROOT / "models" / "msi_resnet50.pt",
     "blurb": "組織病理／分子病理(本平台首條「形態預測分子狀態」track):大腸直腸癌 H&E patch 預測微衛星不穩定狀態 — 高度不穩定 MSImut(MSI-H,對免疫檢查點治療反應佳、與 Lynch 症候群相關)vs 穩定 MSS 二分類(Kather CRC-MSI,TCGA 來源)。ResNet-50 val_acc 0.753 / macro-F1 0.629(n_val 98904)。MSI 狀態關乎免疫治療適應症與遺傳性篩查。⚠️ MSImut(臨床關鍵少數類)recall 僅 0.31、0.75 acc 僅略高於多數類基線 0.71(MSS 佔 71%);patch 級隨機切分(TCGA 同病人/片可能跨 train/val 洩漏),數字需 patient-level 切分才可信。"},
    {"key": "orchid", "kind": "cls",
     "label": "🔬 口腔鱗癌分化分級 ORCHID (H&E, 5 類含 OSMF 癌前)",
     "path": ROOT / "models" / "orchid_resnet50.pt",
     "blurb": "組織病理(本平台首條口腔/頭頸癌 track):口腔鱗狀細胞癌 OSCC 組織 H&E 5 類 — 正常 normal / 口腔黏膜下纖維化 OSMF(癌前病變)/ 高分化 wdOSCC / 中分化 mdOSCC / 低分化 pdOSCC。ResNet-50 val_acc 0.881 / macro-F1 0.894(類別均衡、表現佳),涵蓋癌前→分化分級全譜。⚠️ 完整 train.zip 達 39GB,改以官方 val 當 train、test 當 val(規模較小但 case 不跨、無洩漏);來源 ORCHID。"},
    {"key": "kidney", "kind": "det",
     "label": "🔬 腎絲球偵測 — 正常/硬化 (H&E/PAS, YOLO)",
     "path": ROOT / "reports" / "yolo" / "kidney" / "weights" / "best.pt",
     "blurb": "組織病理(本平台首條腎臟病理 track):腎切片腎絲球偵測 2 類 — 正常 normal / 硬化 sclerosed glomerulus(腎絲球硬化為慢性腎病、腎絲球腎炎的關鍵病變)。YOLOv8s val mAP50 0.465 / mAP50-95 0.330(P 0.62 / R 0.40),slide-disjoint 切分(檔名含機構/片號)。⚠️ 僅 78 影像極小資料、recall 偏低(0.40),數字不穩、僅示流程;硬化類較稀。"},
    {"key": "mihic", "kind": "cls",
     "label": "🔬 肺癌多重免疫組化組織 7 類 MIHIC (multiplex-IHC, ResNet)",
     "path": ROOT / "models" / "mihic_resnet50.pt",
     "blurb": "免疫組化／組織病理(搭配既有 HER2 IHC,本平台首條多重 IHC 組織分型):肺癌多重免疫組化(multiplex-IHC)組織 patch 7 類 — 腫瘤 Tumor / 基質 Stroma / 免疫細胞 Immune cells / 壞死 Necrosis / 肺泡 alveoli / 背景 background / 其他 Other。ResNet-50 val_acc 0.838 / macro-F1 0.836(類別均衡)。可用於量化腫瘤微環境組成。⚠️ 原始 train ~19.5 萬張過大,每類抽樣(train≤5000、val≤1500)後訓練;來源 MIHIC。"},
    {"key": "tvmi3k", "kind": "det",
     "label": "🦠 陰道滴蟲偵測 TVMI3K (濕抹片, YOLO)",
     "path": ROOT / "reports" / "yolo" / "tvmi3k" / "weights" / "best.pt",
     "blurb": "寄生蟲(本平台首條陰道滴蟲 track):陰道分泌物濕抹片陰道滴蟲 Trichomonas vaginalis 偵測 1 類(TVMI3K,官方 Train/Test 切分,由 mask 連通標記轉 bbox)。YOLOv8s val mAP50 0.417 / mAP50-95 0.240 —— 誠實中等:滴蟲與背景上皮/白血球形態相近、為小目標偵測難題。對應臨床陰道炎濕抹片鏡檢的滴蟲辨識。"},
    {"key": "koh", "kind": "det",
     "label": "🍄 皮癬菌絲 KOH 鏡檢偵測 (皮屑/指甲, YOLO)",
     "path": ROOT / "reports" / "yolo" / "koh" / "weights" / "best.pt",
     "blurb": "微生物／黴菌(本平台首條皮癬菌絲「直接鏡檢偵測」track,填 ★★ KOH 缺口,補既有培養型態 OpenFungi/DeFungi):皮屑/指甲 10% KOH 製片菌絲 hyphae 偵測 1 類(40×+100× 兩倍率合併,含背景負樣本降假陽)。YOLOv8s val mAP50 0.865 / mAP50-95 0.513(P 0.84 / R 0.80)—— 表現優異。對應皮癬菌病 tinea 第一線 KOH 鏡檢。影像級切分;來源 Figshare。"},
    {"key": "tcga", "kind": "cls",
     "label": "🔬 腫瘤浸潤淋巴球 TIL ± (多癌種 H&E, 免疫腫瘤, ResNet)",
     "path": ROOT / "models" / "tcga_resnet50.pt",
     "blurb": "組織病理／免疫腫瘤(本平台首條腫瘤浸潤淋巴球 TIL track):多癌種 H&E patch 預測腫瘤浸潤淋巴球密度 til-positive / til-negative 二分類(TCGA-TILs,pool 多種癌別)。ResNet-50 val_acc 0.896 / macro-F1 0.896(類別均衡)。**patient-disjoint 切分(依檔名 TCGA barcode 分組、去重 hardlink,無洩漏,數字可信)**。TIL 密度為免疫檢查點治療反應與預後的重要生物標記。"},
    {"key": "placenta", "kind": "cls",
     "label": "🔬 胎盤絨毛細胞分型 HAPPY (H&E, 9 類)",
     "path": ROOT / "models" / "placenta_resnet50.pt",
     "blurb": "組織病理(本平台首條胎盤病理 track):胎盤絨毛組織細胞分型 9 類 — 纖維母 FIB / 融合滋養層 SYN / 血管內皮 VEN / 血管肌 VMY / 白血球 WBC / 融合結 KNT / 絨毛外滋養層 EVT / 間質 MES / 細胞滋養層 CYT。ResNet-50 val_acc 0.886 / macro-F1 0.839(官方 split,HAPPY 資料集)。⚠️ CYT(val 僅 20 張)等稀有類較弱,macro<weighted 反映類別不平衡。"},
    {"key": "tanzania", "kind": "cls",
     "label": "🦠 瘧疾厚/薄抹片 4 類 (Giemsa 血膜, ResNet)",
     "path": ROOT / "models" / "tanzania_resnet50.pt",
     "blurb": "寄生蟲(瘧疾厚抹片新模態,補既有薄抹片細胞級 malaria/MP-IDB/BBBC041):坦尚尼亞瘧疾血膜 4 類 — 厚抹片感染/未感染、薄抹片感染/未感染。厚抹片(thick smear)為 WHO 瘧疾鏡檢初篩標準,本平台首條納入。ResNet-50 val_acc 0.998 / macro-F1 0.997。⚠️ **影像級隨機切分:同一抹片連續視野形態近似、可能分散到 train/val 造成洩漏,0.998 疑高估**,需病人/抹片級切分才可信,僅示流程。"},
    {"key": "riva", "kind": "det",
     "label": "🔬 子宮頸 Bethesda 細胞偵測 RIVA (8 類, YOLO)",
     "path": ROOT / "reports" / "yolo" / "riva" / "weights" / "best.pt",
     "blurb": "細胞學(子宮頸抹片病變細胞偵測,補既有 SIPaKMeD 分類所無的「定位+分級」任務):子宮頸抹片 Bethesda 系統 8 類細胞偵測 — NILM 正常 / INFL 發炎 / ASCUS / ASCH / LSIL / HSIL / SCC / ENDO。YOLOv8s val mAP50 0.146 / mAP50-95 0.067 —— **誠實偏低**:原始為關鍵點標註,本 track 以固定框(影像邊長 7%)近似合成 bbox、8 類重度不平衡(NILM/INFL 佔多數、稀有病變類近乎學不起來),僅示流程;case-disjoint 切分,來源 RIVA。"},
    # --- batch 6 run-3 (round-3 預抓資料, 2026-06-14) ---
    {"key": "csf", "kind": "cls",
     "label": "🔬 腦脊髓液細胞學 16 類 CSF (體液細胞學, ResNet)",
     "path": ROOT / "models" / "csf_resnet50.pt",
     "blurb": "細胞學(本平台首條腦脊髓液 CSF 體液 track):CSF 離心抹片細胞學 16 類 — 嗜中性/嗜酸性顆粒球、(活化)淋巴球、(活化)單核球、巨噬細胞、紅血球/吞紅細胞、含鐵血黃素吞噬細胞、Hämatoidin、漿細胞、分裂相、核影、人造假影、腫瘤細胞 Tumor。ResNet-50 val_acc 0.829 / macro-F1 0.686。**patient-disjoint 切分(檔名 Liquor_<病人> 分組,無洩漏)**,大類別抽樣上限 3000。CSF 中腫瘤細胞偵測為腦膜轉移/癌病關鍵。⚠️ 類別重度不平衡,稀有類(Plasmazelle/eosinophil/Artifizielle val 僅 10–24 張)F1 偏低、數字不穩。來源 Zenodo 6543147。"},
    {"key": "ivcm", "kind": "cls",
     "label": "👁️ 角膜共軛焦顯微 感染性角膜炎 4 類 IVCM (ResNet)",
     "path": ROOT / "models" / "ivcm_resnet50.pt",
     "blurb": "眼科(本平台首條角膜活體共軛焦顯微 IVCM track):感染性角膜炎共軛焦顯微 4 類 — 棘阿米巴 Acanthamoeba / 真菌 Fungal / 非特異性 NonSpecific / 健康 Healthy。ResNet-50 val_acc 0.981 / macro-F1 0.965。角膜共軛焦可即時偵測棘阿米巴囊體與真菌菌絲、指導感染性角膜炎治療。⚠️ 小資料(~3000 圖)、影像級切分:同一眼連續共軛焦影像形態近似、可能分散 train/val 造成輕微洩漏,高分需病人級切分驗證。來源 Figshare 19838083(IVCM-Keratitis)。"},
    {"key": "cryonuseg", "kind": "det",
     "label": "🧊 凍結切片 H&E 細胞核偵測 CryoNuSeg (10 器官, YOLO)",
     "path": ROOT / "reports" / "yolo" / "cryonuseg" / "weights" / "best.pt",
     "blurb": "組織病理(本平台首條「凍結切片 cryosection」模態,補既有全 FFPE 核資料如 PanNuke/NuInsSeg):術中冷凍切片 H&E 細胞核偵測 1 類(10 器官 × 3 = 30 張,器官分層 20 train / 10 val,每核 instance mask→bbox)。YOLOv8s val mAP50 0.466 / mAP50-95 0.265(P 0.72 / R 0.50)。冷凍切片有凍結假影、形態與 FFPE 差異大,是術中快速診斷的核偵測難點。⚠️ 資料極小(val 僅 10 張),成績僅供參考、數字不穩;組織級切分(同器官不同張分 train/val)。來源 Kaggle ipateam(CryoNuSeg, Mahbod et al.)。"},
    {"key": "blastocyst", "kind": "cls",
     "label": "🫧 IVF 囊胚 Gardner 擴張度分級 5 類 Blastocyst (ResNet)",
     "path": ROOT / "models" / "blastocyst_resnet50.pt",
     "blurb": "生殖醫學/胚胎學(本平台首條 IVF 試管嬰兒胚胎 track):人類囊胚明場影像 Gardner 擴張度(expansion)分級 5 類 EXP0–EXP4(早期囊胚腔→完全擴張/孵化)。ResNet-50 val_acc 0.775 / macro-F1 0.686 / weighted-F1 0.793(官方 train(silver)/test(gold) embryo-disjoint 切分)。囊胚擴張度是 IVF 植入潛能與胚胎品質評分的關鍵軸(另有 ICM/TE 兩軸,因含 ND 不可判類與稀有類未取)。⚠️ train 為 silver-standard(半自動標),EXP4 完全擴張僅 5 張 val(F1 0.30、數字不穩)。來源 Figshare 20123153(Gomez et al. human blastocyst benchmark)。"},
    {"key": "thalassemia", "kind": "cls",
     "label": "🩸 紅血球形態 9 類 (地中海貧血/貧血篩檢, ResNet)",
     "path": ROOT / "models" / "thalassemia_resnet50.pt",
     "blurb": "血液(延伸既有 RBC 形態 track,聚焦地中海貧血/貧血篩檢形態):周邊血單顆紅血球形態 9 類 — 正常 / 標靶細胞 target / 淚滴 teardrop / 橢圓 elliptocyte / 鉛筆 pencil / 棘狀 acanthocyte / 低色素 hypochromic / 球形 spherocyte / 口形 stomatocyte。ResNet-50 val_acc 0.952 / macro-F1 0.871 / weighted-F1 0.955。標靶/淚滴/低色素是地中海貧血與缺鐵性貧血抹片的關鍵形態。⚠️ pencil 類僅 4 張 val(F1 0.33、不可信);隨機切分(無病人分組,可能輕微洩漏);與既有 rbc_morphology/erythrocytes_idb 形態軸部分重疊。來源 Mendeley rfdz6wfzn4(RBCdataset)。"},
    {"key": "osteosarcoma", "kind": "cls",
     "label": "🦴 骨肉瘤組織存活度分型 3 類 Osteosarcoma (H&E, ResNet)",
     "path": ROOT / "models" / "osteosarcoma_resnet50.pt",
     "blurb": "組織病理／骨與軟組織肉瘤(本平台首條肌肉骨骼腫瘤 track):骨肉瘤 H&E tile 3 類 — 非腫瘤 Non-Tumor / 壞死腫瘤 Non-Viable-Tumor / 存活腫瘤 Viable。ResNet-50 val_acc 0.991 / macro-F1 0.991(官方 train/validate 切分)。腫瘤壞死率(化療後存活腫瘤比例)是骨肉瘤化療反應評估與預後的關鍵指標。來源 Kaggle(TCIA Osteosarcoma-Tumor-Assessment, UT Southwestern/UT Dallas)。"},
    {"key": "breast_idc", "kind": "cls",
     "label": "🎀 乳腺 IDC 組織分級 1/2/3 (H&E 40×, ResNet)",
     "path": ROOT / "models" / "breast_idc_resnet50.pt",
     "blurb": "組織病理(延伸既有乳腺 BreakHis/BACH,新增 Nottingham 組織分級任務):浸潤性乳管癌 IDC 組織分級 3 類 Grade 1/2/3(40× 高倍)。ResNet-50 val_acc 0.691 / macro-F1 0.695。腫瘤分級(腺管形成/核多形性/有絲分裂)決定預後與治療強度。⚠️ 資料小(40× 僅 454 張)、隨機切分(無病人分組可能洩漏);分級本就主觀且難,成績僅中等、相鄰級易混。來源 Mendeley w7jjcx7gj6(BreastCancer IDC Grade)。"},
    {"key": "sperm_dfi", "kind": "cls",
     "label": "🧬 精子 DNA 碎片化 SCD halo 分級 4 類 (男科, ResNet)",
     "path": ROOT / "models" / "sperm_dfi_resnet50.pt",
     "blurb": "男科／生殖(延伸既有精子形態 SMIDS/HuSHeM,新增 DNA 完整性軸):精子染色質擴散試驗 SCD halo 大小 4 類 — 大暈 Large / 中暈 Medium / 小暈 Small / 無暈 Without halo。ResNet-50 val_acc 0.983 / macro-F1 0.983(類別均衡)。halo 大小反映 DNA 碎片化程度(無暈/小暈=高碎片化),與男性不孕、受精失敗相關。來源 Figshare 30120811(expert-annotated sperm SCD)。"},
    {"key": "ebhi_seg", "kind": "cls",
     "label": "🔬 大腸組織分化 6 階段 EBHI-SEG (H&E, ResNet)",
     "path": ROOT / "models" / "ebhi_seg_resnet50.pt",
     "blurb": "組織病理(延伸既有大腸 NCT-CRC/MHIST,新增完整分化譜分級):大腸 H&E 組織分化 6 階段 — 正常 Normal / 息肉 Polyp / 低度上皮內瘤變 Low-grade IN / 高度上皮內瘤變 High-grade IN / 腺癌 Adenocarcinoma / 鋸齒狀腺瘤 Serrated adenoma。ResNet-50 val_acc 0.949 / macro-F1 0.923。涵蓋正常→癌前→癌的完整大腸病變光譜(含 EBHI-SEG 官方 mask,本 track 取 cls)。⚠️ Normal(65)/Serrated(49) 類偏少。來源 Figshare 21540159(EBHI-SEG)。"},
    {"key": "autokary2022", "kind": "det",
     "label": "🧬 染色體核型偵測 24 類 AutoKary2022 (中期分裂全圖, YOLO)",
     "path": ROOT / "reports" / "yolo" / "autokary2022" / "weights" / "best.pt",
     "blurb": "細胞遺傳學(全新任務,補既有 chromosome「單條 24 類分類」之不足):中期分裂染色體**全展開圖**偵測+分類 24 類(編號 1–22 + X/Y)，即核型分析的「找出並分類每一條染色體」。YOLOv8s val mAP50 0.851 / mAP50-95 0.663(P 0.86 / R 0.76)—— 表現優異。polygon instance seg 標註轉 bbox 訓練。⚠️ val 僅 30 張全圖(框數多但圖少),數字宜參考。來源 Roboflow karyotypezhongxin/autokary2022(AutoKary2022, ICME2023)。"},
    {"key": "tcruzi_chagas", "kind": "det",
     "label": "🦟 克氏錐蟲 T. cruzi (Chagas) 偵測 (Giemsa 血抹片, YOLO)",
     "path": ROOT / "reports" / "yolo" / "tcruzi_chagas" / "weights" / "best.pt",
     "blurb": "寄生蟲(全新病原,vs 既有 tryp=布氏錐蟲 T. brucei 為不同蟲種):克氏錐蟲 Trypanosoma cruzi 偵測 1 類(美洲錐蟲病 Chagas 病原),Giemsa 血抹片 + CSF。YOLOv8s val mAP50 0.257 / mAP50-95 0.101(P 0.26 / R 0.37)。**case-disjoint 切分(依 caseId 分組,無洩漏)**,但 val 蟲體稀疏(69 圖僅 59 框、多為陰性背景)使 mAP 偏低、數字不穩。⚠️ 成績僅參考;CC-BY-NC(非商用)。來源 Zenodo 15007339(PLOS NTD 2026)。"},
    {"key": "ubc_ocean", "kind": "cls", "fm": "models/ubc_ocean_fmprobe.joblib",
     "label": "♀️ 卵巢癌組織亞型 5 類 UBC-OCEAN (H&E, Phikon-v2 FM)",
     "path": ROOT / "models" / "ubc_ocean_resnet50.pt",
     "blurb": "組織病理／婦科腫瘤(本平台首條卵巢癌 track):卵巢癌組織亞型 5 類 — 高度漿液性 HGSC / 子宮內膜樣 EC / 亮細胞 CC / 低度漿液性 LGSC / 黏液性 MC。**本 track 改用病理基礎模型引擎:Phikon-v2(ViT-L, frozen)特徵 + LogReg → val_acc 0.679 / macro-F1 0.604**,顯著勝過 ResNet-50 從頭訓(0.543 / 0.436),且救回 ResNet 全崩的稀有類 LGSC(F1 0→0.22)——小資料 WSI 亞型用 frozen 病理 FM 特徵+線性 head 比從頭微調更穩(見 skill pick-model-arch)。⚠️ 用 2964px 降採縮圖(3×3 tile mean-pool)、資料小(538 例)、隨機切分,卵巢亞型本就難。Phikon-v2 為 Owkin 非商用授權。來源 Kaggle 競賽 UBC-OCEAN(標籤)+ 社群縮圖鏡像 utm529fg/ubc-reduced-png-2964x2964。"},
    {"key": "bracs", "kind": "cls",
     "label": "🎀 乳腺病變 7 類細分 BRACS (H&E, ResNet)",
     "path": ROOT / "models" / "bracs_resnet50.pt",
     "blurb": "組織病理(延伸既有乳腺 BreakHis/BACH/IDC,最細分 7 類病變光譜):乳腺 H&E 7 類 — 正常 Normal / 良性 PB / 一般型增生 UDH / 平坦上皮非典型 FEA / 非典型導管增生 ADH / 原位癌 DCIS / 侵襲癌 IC(官方 train/val/test 切分)。ResNet-50 val_acc 0.659 / macro-F1 0.641。涵蓋良性→非典型→原位→侵襲完整光譜,是乳腺最細粒度公開分型。⚠️ 7 類細粒度分類本就難(相鄰類如 UDH/FEA/ADH 易混)、用 299px 縮圖版。來源 Kaggle 鏡像 assismathias/bracs-original-299(BRACS, bracs.icar.cnr.it)。"},
    {"key": "segpc", "kind": "det",
     "label": "🩸 多發性骨髓瘤漿細胞偵測 SegPC-2021 (骨髓抹片, YOLO)",
     "path": ROOT / "reports" / "yolo" / "segpc" / "weights" / "best.pt",
     "blurb": "血液腫瘤(本平台首條漿細胞偵測 track):多發性骨髓瘤骨髓抹片漿細胞 plasma cell 偵測 1 類(SegPC-2021 challenge,每細胞含細胞質+細胞核 instance mask→整 cell bbox)。YOLOv8s val mAP50 0.833 / mAP50-95 0.735(P 0.83 / R 0.83)—— 表現優異。骨髓漿細胞比例是多發性骨髓瘤診斷與分期關鍵。官方 train(298)/validation(200) 切分。來源 Kaggle sbilab/segpc2021dataset(作者 SBILab, TCIA SegPC-2021)。"},
    {"key": "hmu_gc", "kind": "cls",
     "label": "🔬 胃癌腫瘤微環境 8 組織類 HMU-GC (H&E, ResNet)",
     "path": ROOT / "models" / "hmu_gc_resnet50.pt",
     "blurb": "組織病理(延伸既有胃 PatchGastric「分化分級」,新增組織成分分型,類比 NCT-CRC 於胃):胃癌 H&E patch 腫瘤微環境 8 類 — 脂肪 ADI / 碎屑 DEB / 淋巴球 LYM / 黏液 MUC / 肌肉 MUS / 正常黏膜 NOR / 基質 STR / 腫瘤上皮 TUM。ResNet-50 val_acc 0.746 / macro-F1 0.747(8 類均衡各 ~3300)。可作胃癌 WSI 的組織成分量化基石。⚠️ patch 級隨機切分(無 WSI/病人分組,可能洩漏,數字偏樂觀)。來源 Figshare 25954813(HMU-GC-HE-30K)。"},
    {"key": "glomeruli_char", "kind": "cls",
     "label": "🔬 腎絲球 正常/硬化 分類 (PAS, ResNet)",
     "path": ROOT / "models" / "glomeruli_char_resnet50.pt",
     "blurb": "組織病理／腎臟(延伸既有 kidney「腎絲球偵測」,新增分類任務):腎絲球 patch 二分類 — 正常 normal / 硬化 sclerosed。ResNet-50 val_acc 0.997 / macro-F1 0.997。硬化腎絲球比例是慢性腎病進展與預後指標。⚠️ 成績極高需保留:patch 級隨機切分(無 WSI 分組)+ 正常 vs 硬化形態差異極大,高分含洩漏成分、非臨床對齊。來源 Mendeley k7nvtgn2x6(DATASET_B)。"},
    {"key": "vet_egg", "kind": "cls",
     "label": "🐄 獸醫家畜糞檢蟲卵 6 類 (光學顯微, ResNet)",
     "path": ROOT / "models" / "vet_egg_resnet50.pt",
     "blurb": "獸醫寄生蟲學(本平台首條獸醫 track,與既有人類糞便蟲卵不同物種):反芻家畜糞檢蟲卵/囊體 6 類 — 蛔蟲 Ascaris / 結腸小袋纖毛蟲 Balantidium coli / 肝吸蟲 Fasciola / 同盤吸蟲 Paramphistomum / 胃線蟲 Stomach worm / 陰性對照。ResNet-50 val_acc 0.893 / macro-F1 0.785。展示跨物種(獸醫)泛化。⚠️ 資料小(546 圖、稀有類 val 7-8 張)、Negative 類佔多;隨機切分。來源 Mendeley wbvb4whbks。"},
    {"key": "openfungi", "kind": "cls",
     "label": "🍄 絲狀黴菌 6 屬顯微辨識 OpenFungi (乳酸酚棉藍, ResNet)",
     "path": ROOT / "models" / "openfungi_resnet50.pt",
     "blurb": "微生物學/黴菌學(延伸既有 DeFungi,不同真菌軸):絲狀黴菌屬級顯微 6 類 — 鏈格孢 Alternaria / 麴菌黃麴組 Aspergillus(Flavi) / 黑麴組 Aspergillus(Nigri) / 鐮刀菌 Fusarium / 青黴 Penicillium / 根黴 Rhizopus(乳酸酚棉藍+培養鏡檢,僅取顯微 micro 子集、排除菌落 macro 照)。ResNet-50 val_acc 0.989 / macro-F1 0.958。絲狀黴菌形態鑑別是臨床黴菌學與環境/食品分離株判讀的基本功。⚠️ val 極小(87 張,Rhizopus 僅 2、Penicillium 4、Alternaria 3→數字不穩);類別重度不平衡(Aspergillus-Flavi 384 vs Rhizopus 12);含食品/環境分離株、非純臨床。來源 Zenodo 15692069(OpenFungi)。"},
    {"key": "ufsc", "kind": "det",
     "label": "🔬 子宮頸/口腔多染色細胞核偵測 22 類 UFSC-MultiCyto (Pap/AgNOR/Feulgen, YOLO)",
     "path": ROOT / "reports" / "yolo" / "ufsc" / "weights" / "best.pt",
     "blurb": "細胞學(本平台首條 AgNOR/Feulgen 多染色核偵測,補既有 SIPaKMeD/RIVA 僅 Pap 之缺):多染色細胞學細胞核偵測 22 類 — Pap 染(正常/異常/失焦/血性/分裂/反應性等 7)+ AgNOR 染(核/簇/衛星/重疊核/白血球核等 7)+ Feulgen 染(改變/中間/老化/重疊/嗜中性等 8)。YOLOv8s val mAP50 0.497 / mAP50-95 0.375(ep70 早停)。AgNOR(核仁組成區)與 Feulgen(DNA 定量)染色的核形態量化是子宮頸/口腔癌風險評估研究工具。官方 train 12502 / val 2448。⚠️ 22 類含多染色×核態、長尾稀有類偏弱;器官(子宮頸/口腔)與既有 track 重疊,新增值在多染色模態。來源 Mendeley b8bygdxr7r(UFSC Multi-Cyto, CC-BY-4.0)。"},
    {"key": "parasites", "kind": "cls",
     "label": "🦠 血液/體液寄生蟲 8 類面板 (Giemsa, ResNet)",
     "path": ROOT / "models" / "parasites_resnet50.pt",
     "blurb": "寄生蟲學(統一多蟲種面板,新增 Babesia/Toxoplasma 病原):血液/體液寄生蟲與宿主細胞 8 類 — 巴貝斯蟲 Babesia / 利什曼原蟲 Leishmania / 瘧原蟲 Plasmodium / 弓形蟲 Toxoplasma / 陰道滴蟲 Trichomonad / 錐蟲 Trypanosome + 宿主紅血球 RBC / 白血球 Leukocyte。ResNet-50 val_acc 0.998 / macro-F1 0.995。Babesia(蜱媒)、Toxoplasma 為既有單蟲種 track 所無;統一面板可一次判別蟲種 vs 宿主細胞。⚠️ **成績過高、強烈疑洩漏**:隨機切分、每蟲種多為單一採集來源→模型可能在背染片/來源外觀而非泛化形態(同 tanzania 0.998 caveat);Plasmodium/Trichomonad/Trypanosome/Leishmania 與既有 track 部分同源。需 group-aware split 才可信。來源 Mendeley 38jtn4nzs6。"},
    {"key": "oral_cyto", "kind": "det",
     "label": "🦷 口腔脫落細胞學核偵測 4 類 (Pap, YOLO)",
     "path": ROOT / "reports" / "yolo" / "oral_cyto" / "weights" / "best.pt",
     "blurb": "細胞學(本平台首條口腔脫落細胞學「核偵測」任務,別於既有 ORCHID 口腔組織分類):口腔刷取細胞學 Pap 抹片細胞核偵測,依 geojson 核標註分 4 級 I/II/III/IV(類 Bethesda 口腔分級)。YOLOv8s val mAP50 0.486 / mAP50-95 0.356。2048px 大圖、每圖 21–136 個密集小核。口腔細胞學是口腔癌篩檢的非侵入工具。⚠️ 密集小核 @640 縮放後極小(~8px)偵測難;分級類別不平衡;2048px+密集核使 batch 需降載避免驗證 NMS/VRAM 爆炸(SAHI 切片可改善,見 skill pick-model-arch)。來源 Kaggle abhijeetptl5/oral-cytology-dataset。"},
    {"key": "uti_urine", "kind": "det",
     "label": "💧 尿沉渣 UTI 細胞偵測 7 類 (未染色明場, YOLO)",
     "path": ROOT / "reports" / "yolo" / "uti_urine" / "weights" / "best.pt",
     "blurb": "微生物/泌尿(UTI 特異尿沉渣,別於既有 Urine-particles 一般尿沉渣 track):未染色明場尿沉渣 7 類 — 桿菌 Rod / 紅白血球 RBC-WBC / 酵母菌 Yeast / 雜項 Miscellaneous / 單一上皮細胞 Single-EPC / 小上皮細胞片 Small-EPC-sheet / 大上皮細胞片 Large-EPC-sheet。YOLOv8s val mAP50 0.402 / mAP50-95 0.286(RBC-WBC 0.80、Single-EPC 0.87 表現佳)。UTI 可由尿沉渣顯微鏡細胞標記診斷,本資料集走簡易成像普及化路線(x20 物鏡、未染色明場)。⚠️ 與既有 urine(Urine-particles 一般尿沉渣 det, mAP50 0.896)同屬尿沉渣偵測;本 track 區別在 UTI 臨床情境、未染色、UTI 特異類別(桿菌/酵母/上皮細胞片)、語意遮罩來源。⚠️ 小資料(官方 train 97 / val 98 圖, 共 300);標註為**語意分割遮罩→連通元件 bbox**,同類相鄰細胞會併成一框(尤其密集桿菌 Rod mAP50 僅 0.06);稀有類 Yeast(val 20 框)/Small-EPC-sheet(16)/Large-EPC-sheet(2) 樣本太少不可信。來源 RODARE 10.14278/rodare.2473(CC-BY-4.0, 倫敦 LUTS 門診)。"},
    {"key": "balf", "kind": "det",
     "label": "🫁 支氣管肺泡灌洗液 BALF 細胞偵測 7 類 (Diff-Quik/Gram, YOLO)",
     "path": ROOT / "reports" / "yolo" / "balf" / "weights" / "best.pt",
     "blurb": "細胞學(本平台首條支氣管肺泡灌洗液 BALF 體液細胞學偵測):BAL 灌洗液 7 類細胞 — 紅血球 erythrocyte / 淋巴球 lymphocyte / 纖毛柱狀上皮 ciliated-columnar / 鱗狀上皮 squamous / 巨噬細胞 macrophage / 嗜酸性球 eosinophil / 嗜中性球 neutrophil。YOLOv8s val mAP50 0.810 / mAP50-95 0.592(macrophage 0.93、lymphocyte/neutrophil 0.85 佳,七類皆學起來無弱類)。BALF 細胞分類計數(巨噬/淋巴/嗜中性/嗜酸比例)是間質性肺病與肺部感染診斷的臨床依據。官方 train 1895 / val 210 圖(共 2105 圖/13263 細胞, 中國 PLA 總醫院)。⚠️ class-id→名稱對應為依各類豐度+BAL 臨床比例推定(資料集/論文未附機器可讀 legend;id4=macrophage 最多、id1=lymphocyte 次多已由 val 豐度佐證,其餘 4 類沿論文列舉序推定);標註多邊形→bbox;少量重複框由 YOLO 自動移除。來源 Zenodo 14871206(PW-BALFC, Sci Data 2025, CC-BY-4.0)。"},
    {"key": "midog", "kind": "det",
     "label": "🔬 跨掃描儀有絲分裂偵測 MIDOG 2021 (H&E, YOLO, scanner-disjoint)",
     "path": ROOT / "reports" / "yolo" / "midog" / "weights" / "best.pt",
     "blurb": "組織病理(跨掃描儀 domain-generalization 示範,別於既有 mitosis 乳腺隨機切分 in-distribution 版):MIDOG 2021 乳腺癌 H&E ROI 有絲分裂偵測 2 類 — 有絲分裂 mitotic-figure / 形似非分裂 imposter。YOLOv8s **scanner-disjoint** val(訓練 Hamamatsu XR+S360 兩台掃描儀、驗證從未見過的 Aperio CS2 第三台) mAP50 0.362 / mitotic-figure 單類 0.560。跨掃描儀泛化是數位病理核心難題,本 track 刻意以掃描儀分層切分展示 out-of-domain 掉分。150 張大圖(7215×5412)tile 成 1329 train / 653 val patch。⚠️ mAP 偏低是**設計使然**:(1)scanner-disjoint 跨域驗證本就比 in-distribution 難(同 README『跨域掉到 60-80%』老坑的極端示範);(2)imposter 類對抗性強、mAP50 僅 0.163;(3)mitosis 物件小、mAP50-95 0.099(tile 框緊度有限)。與既有 mitosis(乳腺隨機切分)同任務但 distinct 在跨掃描儀 domain-gen。來源 Zenodo 4643381(MIDOG 2021, CC-BY)。"},
    {"key": "caishi", "kind": "cls",
     "label": "🔬 子宮頸腺癌原位 AIS 組織病理 2 類 (H&E, ResNet)",
     "path": ROOT / "models" / "caishi_resnet50.pt",
     "blurb": "組織病理(子宮頸腺體 AIS,別於既有 pap/RIVA 子宮頸「細胞學」):子宮頸腺癌原位 H&E 組織病理二分類 — 正常腺體 Normal / 腺癌原位 Abnormal(AIS)。ResNet-50 val_acc 0.982 / macro-F1 0.982(Abnormal precision 1.00、recall 0.97;Normal recall 1.00)。AIS 是子宮頸腺癌前驅病變,組織學上易與良性腺體混淆,自動篩檢有臨床意義。train 1905 / val 337(Normal 1011 + Abnormal 1231 圖)。⚠️ 子宮頸器官與既有 pap/riva 重疊,但本 track 是腺體「組織病理」AIS(任務/模態不同於細胞學抹片);2 類、單中心來源,跨域(他院切片/染色)恐顯著掉分(同 README 跨域 60-80% 老坑)。來源 figshare 24548953(CASIHI / MIaMIA-Open-Data, CC-BY-4.0)。"},
    {"key": "cats_dogs_gi_parasites", "kind": "det",
     "label": "🐕 犬貓糞檢腸道寄生蟲偵測 9 類 (TF-Test VetPet 明場, YOLO)",
     "path": ROOT / "reports" / "yolo" / "cats_dogs_gi_parasites" / "weights" / "best.pt",
     "blurb": "獸醫寄生蟲(本平台首條獸醫腸道寄生蟲『偵測』track,別於既有蟲卵分類):犬貓糞便抹片(TF-Test VetPet 暫時染色)腸道寄生蟲偵測 9 類 — 貓 3 類(cat-p1~3)＋犬 6 類(dog-p1~6)。⚠️ 來源僅公開遮罩、未公開物種拉丁名與『像素值↔物種』對應,且貓/犬為兩個各自編號的資料集 → 為免把不同種誤併,採來源代碼分命名空間(論文稱貓 3 種/犬 5 種＋糞便雜質)。由語意分割遮罩連通元件轉 bbox、官方 train/test 切分。YOLOv8s val mAP50 0.869 / mAP50-95 0.763(P 0.94 / R 0.84;cat-p1~3 與 dog-p1/p4/p5 達 0.95~0.99)。⚠️ dog-p6 最稀有(僅 30 框)mAP50 0.13 學不起、dog-p3 mAP50-95 0.49 偏低;同類相鄰遮罩可能併框。對應犬貓糞檢自動化寄生蟲篩檢。來源 GitHub LIDS-Datasets(Comput Biol Med 2023, #107203)。"},
    {"key": "ude_diatoms", "kind": "cls",
     "label": "🔬 淡水矽藻種別 101 類 (法醫／環境, ResNet)",
     "path": ROOT / "models" / "ude_diatoms_resnet50.pt",
     "blurb": "法醫／環境(本平台首條矽藻 track,別於既有臨床醫檢):淡水矽藻 diatom 物種細粒度分類 101 種(UDE Diatoms in the Wild,focus-stacked 光學顯微)。應用於溺水鑑定(肺/骨髓矽藻檢出)與水質生態監測。ResNet-50 val_acc 0.728 / weighted-F1 0.725 / **macro-F1 0.520**。⚠️ 重度長尾(Achnanthidium minutissimum ~15.5k 獨大、用 sqrt class-weight)→ macro 比 weighted 低 0.20、稀有與形態相近種偏弱,12 類 val<5 樣本指標不可信;101 細粒度種本就難(發表 SOTA 亦未飽和)。**玻片級分組 split(slide-disjoint)無洩漏**:train 61143 / val 6451、span 319 玻片,全 101 類 train/val 皆覆蓋。來源 Zenodo 10410655(CC0)。"},
    {"key": "pollen73s", "kind": "cls",
     "label": "🌼 花粉粒種別 73 類 (孢粉學/法醫, ResNet)",
     "path": ROOT / "models" / "pollen73s_resnet50.pt",
     "blurb": "法醫／環境(本平台首條花粉/孢粉學 track):花粉粒 pollen grain 物種分類 73 種(POLLEN73S,巴西 Cerrado 植物相,光學顯微)。應用於法醫孢粉學(物證地緣連結)、空氣過敏原監測、蜂蜜植物溯源(melissopalynology)。ResNet-50 val_acc 0.950 / macro-F1 0.949(資料平衡、macro≈weighted)。⚠️ 小資料(~35 張/類、val 僅 ~5/類)→ 單類指標噪音大、跨採集泛化未驗;隨機 split(花粉粒影像無玻片/來源 id 可供分組)。來源 figshare 23307950(CC-BY-4.0)。"},
]

AVAILABLE = [m for m in REGISTRY if m["path"].exists() or W.is_downloadable(m)]
BY_LABEL = {m["label"]: m for m in AVAILABLE}

# --- model cards: discipline / classes / metrics / caveats -----------------
CARDS: dict = {}
_cards_dir = ROOT / "model_cards"
if _cards_dir.is_dir():
    for _p in _cards_dir.glob("*.json"):
        if _p.name == "index.json":
            continue
        try:
            CARDS[_p.stem] = json.loads(_p.read_text(encoding="utf-8"))
        except Exception:
            pass

# 少數卡片 discipline 為 null,補上對應科別;未知歸「其他」
_DISC_OVERRIDE = {
    "fungi_koh": "黴菌", "malaria": "寄生蟲", "bonemarrow": "血液/骨髓",
    "bacteria": "微生物", "mhist": "組織病理", "univali_leukocyte": "血液",
    "mll23": "血液", "mk11": "血液/骨髓", "bloodcells": "血液",
}
_DISC_ORDER = ["血液", "血液/骨髓", "血液腫瘤", "微生物", "黴菌", "寄生蟲",
               "組織病理", "免疫組化", "免疫", "細胞學", "男科", "生殖醫學",
               "尿液分析", "細胞遺傳學", "眼科", "獸醫", "其他"]


def disc_of(key: str) -> str:
    c = CARDS.get(key) or {}
    return c.get("discipline") or _DISC_OVERRIDE.get(key) or "其他"


_DISC_PRESENT = {disc_of(m["key"]) for m in AVAILABLE}
DISCIPLINES = [d for d in _DISC_ORDER if d in _DISC_PRESENT] + \
              sorted(d for d in _DISC_PRESENT if d not in _DISC_ORDER)


def _model_score(key: str):
    c = CARDS.get(key) or {}
    m = c.get("metrics") or {}
    return m.get("val_acc") if c.get("task") == "classification" else m.get("mAP50")


def _score_tier(key: str) -> str:
    """成績分級色點:🟢≥.85 🟡≥.70 🟠≥.50 🔴<.50 ⚪無數據。"""
    s = _model_score(key)
    if s is None:
        return "⚪"
    return "🟢" if s >= 0.85 else "🟡" if s >= 0.70 else "🟠" if s >= 0.50 else "🔴"


# ── 模型篩選(公開站站主可調,持久化 → 套用到所有訪客):分數門檻 + 手動隱藏清單 ──
# 預設只顯示成績 ≥ 0.85 的模型。桌面/dev(非 PUBLIC)不套用篩選,一律全部顯示。
MODEL_FILTER_FILE = ROOT / "model_filter.json"


def _load_model_filter():
    d = {"min_score": 0.85, "hidden": []}
    try:
        if MODEL_FILTER_FILE.exists():
            j = json.loads(MODEL_FILTER_FILE.read_text(encoding="utf-8"))
            if "min_score" in j and (j["min_score"] is None or isinstance(j["min_score"], (int, float))):
                d["min_score"] = j["min_score"]
            if isinstance(j.get("hidden"), list):
                d["hidden"] = [str(x) for x in j["hidden"]]
    except Exception:
        pass
    return d


MODEL_FILTER = _load_model_filter()


def _save_model_filter():
    try:
        MODEL_FILTER_FILE.write_text(json.dumps(MODEL_FILTER, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _visible_model(key: str) -> bool:
    if not PUBLIC:                                   # 桌面/dev:不篩選,全顯示
        return True
    if key in MODEL_FILTER.get("hidden", []):        # 站主手動隱藏
        return False
    ms = MODEL_FILTER.get("min_score")
    if not ms:                                       # None / 0 = 全部顯示
        return True
    s = _model_score(key)
    return True if s is None else (s >= ms)          # 無分數(⚪)不因門檻被藏,交由手動隱藏


def labels_for(disc: str):
    return [m["label"] for m in AVAILABLE if disc_of(m["key"]) == disc and _visible_model(m["key"])]


def visible_disciplines():
    return [d for d in DISCIPLINES
            if any(disc_of(m["key"]) == d and _visible_model(m["key"]) for m in AVAILABLE)]


def model_choices(disc: str):
    """模型下拉:(顯示, 值) 元組;顯示前綴成績色點,值為原始 label(BY_LABEL 鍵)。"""
    return [(f"{_score_tier(m['key'])} {m['label']}", m["label"])
            for m in AVAILABLE if disc_of(m["key"]) == disc and _visible_model(m["key"])]

_cls_cache: dict = {}
_yolo_cache: dict = {}
_sahi_cache: dict = {}
_fm_cache: dict = {}


def load_fm(rel, progress=None):
    """載入 FM-probe 產物(Phikon-v2 frozen 特徵 + LogReg head),見 src/train_fm_probe.py。"""
    import joblib
    p = str(W.ensure_file(rel, progress=progress))     # 本機有→原路徑;缺→抓 HF 到快取(可帶進度)
    if p not in _fm_cache:
        _fm_cache[p] = joblib.load(p)
    return _fm_cache[p]


def prettify(name: str) -> str:
    return name.replace("_", " ").replace(".", " ").strip()


def load_cls(path: Path):
    key = str(path)
    if key not in _cls_cache:
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        classes = ckpt["classes"]
        img_size = ckpt.get("img_size", 224)
        model = models.resnet50(weights=None)
        model.fc = nn.Sequential(nn.Dropout(0.3),
                                 nn.Linear(model.fc.in_features, len(classes)))
        model.load_state_dict(ckpt["state_dict"])
        model.to(DEVICE).eval()
        tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        _cls_cache[key] = (model, classes, tf)
    return _cls_cache[key]


def load_yolo(path: Path):
    key = str(path)
    if key not in _yolo_cache:
        from ultralytics import YOLO
        _yolo_cache[key] = YOLO(str(path))
    return _yolo_cache[key]


def load_sahi(path: Path):
    """SAHI 切片推論模型(大圖密集/稀疏小目標 track 用,見 skill pick-model-arch)。"""
    key = str(path)
    if key not in _sahi_cache:
        from sahi import AutoDetectionModel
        dev = "cuda:0" if DEVICE.type == "cuda" else "cpu"
        try:
            m = AutoDetectionModel.from_pretrained(model_type="ultralytics", model_path=str(path), device=dev)
        except Exception:
            m = AutoDetectionModel.from_pretrained(model_type="yolov8", model_path=str(path), device=dev)
        _sahi_cache[key] = m
    return _sahi_cache[key]


# ============================================================================
#  推論核心(可單張、可批次)
#    分類 cls_rank → 全類別機率排序;偵測 det_infer → 計數 + 標註圖
# ============================================================================
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif"}
TOTAL = "📋 總表"
NULLPAGE = "Null"
ADDPAGE = "__cs_add__"          # 「＋ 新增分類」分頁的內部值;點它(前端攔截 prompt)→ 建自訂分類分頁
SW_PLACEHOLDER = "⇄ 切換分類"   # 切換分類下拉未選檔時顯示此提示(並虛化)
# 預設範例:公開站啟動時載入這夾影像、用此模型預先分類,當作「有內容」的預設畫面(空狀態→現成 demo)。
SAMPLE_DIR = Path(os.environ.get("CLINSCOPE_SAMPLE_DIR", r"D:\ClinScope_測試\06_真菌OpenFungi_fungi_50"))
SAMPLE_MODEL_KEY = os.environ.get("CLINSCOPE_SAMPLE_MODEL", "fungi")   # 黴菌 → 真菌 Fungi — OpenFungi (LPCB 培養)
SAMPLE = {"label": None, "files": [], "results": {}}   # 啟動時由 _precompute_sample 填
# 公開站(CLINSCOPE_PUBLIC=1):server 跑在遠端本機,伺服器端原生選夾(tkinter)會彈在 server 那台、
# 訪客瀏覽器看不到且阻塞請求 → 改用瀏覽器端「分批上傳」(繞過 Cloudflare 單次 ~100MB 上限)。
# 本機桌面/dev 不設此旗標,保留原生選夾。
PUBLIC = bool(os.environ.get("CLINSCOPE_PUBLIC", "").strip())
# Gradio 上傳暫存根目錄:分批上傳後,前端回傳的伺服器路徑必須落在此目錄內才接受(擋路徑注入)。
_GRADIO_TMP = Path(os.environ.get("GRADIO_TEMP_DIR") or (Path(tempfile.gettempdir()) / "gradio")).resolve()
# 公開站身分:僅登入 email == OWNER_EMAIL 為「站主」(可操作),其餘(含匿名)為訪客(唯讀)。
OWNER_EMAIL = os.environ.get("CLINSCOPE_OWNER_EMAIL", "lisonchang80@gmail.com").strip().lower()
_ID_LOG = Path(__file__).resolve().parent / "identity_debug.log"   # 記錄真流量看到的 email,便於確認(可日後移除)

# ── 站內 Google 登入(取代 CF Access,讓路人可匿名唯讀瀏覽)────────────────────────
# 前端 Google Sign-In → ID token(JWT)→ 後端驗簽/audience/issuer → 簽 itsdangerous cookie 記 email。
# 需設 CLINSCOPE_GOOGLE_CLIENT_ID 才啟用;未設時公開站沒有站主(全員唯讀),除非明確信任 CF Access 標頭。
GOOGLE_CLIENT_ID = os.environ.get("CLINSCOPE_GOOGLE_CLIENT_ID", "").strip()
COOKIE_NAME = "cs_session"
SESSION_MAX_AGE = 60 * 60 * 24 * 30      # 30 天
AUTH_ENABLED = bool(PUBLIC and GOOGLE_CLIENT_ID)   # 站內 Google 登入是否啟用(啟用才 mount 路由 + 顯示登入鈕)
_PUBLIC_DEFAULT_SECRET = "dev-insecure-change-me"   # 舊版預設值,寫在公開 repo 裡 → 絕不可拿來簽 cookie


def _session_secret(raw):
    """cookie 簽章金鑰。未設 / 沿用公開預設值 / 短於 32 字 → 改用本次啟動的隨機金鑰(fail-closed:
    重啟後需重新登入,但沒人能自簽站主 cookie)。正式站請在 env.local.bat 設 CLINSCOPE_SESSION_SECRET。"""
    raw = (raw or "").strip()
    if len(raw) >= 32 and raw != _PUBLIC_DEFAULT_SECRET:
        return raw
    if AUTH_ENABLED:
        print("[ClinScope] ⚠ CLINSCOPE_SESSION_SECRET 未設或太弱(需 ≥32 字)→ 改用隨機金鑰,重啟後需重新登入。")
    return secrets.token_urlsafe(48)


SESSION_SECRET = _session_secret(os.environ.get("CLINSCOPE_SESSION_SECRET"))
# CF Access 身分標頭只有在 Access 真的擋在前面(非 Bypass)時才由 Cloudflare 填入;Access 改 Bypass 後
# 它就只是普通請求標頭,任何人都能自帶冒充站主 → 預設不信任。僅「未啟用站內登入」且明確設 =1 時採用。
TRUST_CF_ACCESS_HEADER = os.environ.get("CLINSCOPE_TRUST_CF_ACCESS_HEADER", "").strip() == "1"
_SER = None


def _serializer():
    global _SER
    if _SER is None:
        from itsdangerous import URLSafeTimedSerializer
        _SER = URLSafeTimedSerializer(SESSION_SECRET, salt="clinscope-session")
    return _SER


def make_session(email):
    return _serializer().dumps({"email": email})


def read_session(token):
    try:
        data = _serializer().loads(token, max_age=SESSION_MAX_AGE)
        return (data.get("email") or "").strip().lower() or None
    except Exception:
        return None


def verify_google_credential(credential):
    """驗證 Google Sign-In 回傳的 ID token,回傳 claims(email/sub/...)。無 client id 則報錯。"""
    if not GOOGLE_CLIENT_ID:
        raise ValueError("CLINSCOPE_GOOGLE_CLIENT_ID 未設定")
    from google.auth.transport import requests as g_requests
    from google.oauth2 import id_token as gid
    info = gid.verify_oauth2_token(credential, g_requests.Request(), GOOGLE_CLIENT_ID,
                                   clock_skew_in_seconds=30)   # 容忍時鐘差,免「used too early」
    if info.get("iss") not in ("accounts.google.com", "https://accounts.google.com"):
        raise ValueError("invalid issuer")
    return info


def _cookie_email(request):
    """從簽章 cookie 讀回登入 email(站內 Google 登入);無/失效回 None。"""
    try:
        cookies = getattr(request, "cookies", None)
        if not cookies:
            raw = getattr(getattr(request, "request", None), "cookies", None)
            cookies = raw or {}
        if not cookies:                    # 後備:自己解析 Cookie 標頭
            h = getattr(request, "headers", None)
            ck = h.get("cookie") if (h and hasattr(h, "get")) else None
            cookies = {}
            if ck:
                for part in ck.split(";"):
                    if "=" in part:
                        k, v = part.strip().split("=", 1)
                        cookies[k] = v
        tok = cookies.get(COOKIE_NAME) if cookies else None
        return read_session(tok) if tok else None
    except Exception:
        return None


def _req_email(request):
    """讀 Cloudflare Access 認證後注入的登入 email 標頭(過渡/後備用);無則 None。"""
    try:
        h = getattr(request, "headers", None)
        if h is None:
            return None
        v = None
        if hasattr(h, "get"):
            v = h.get("cf-access-authenticated-user-email") or h.get("Cf-Access-Authenticated-User-Email")
        if not v and hasattr(h, "items"):
            for k, vv in h.items():
                if str(k).lower() == "cf-access-authenticated-user-email":
                    v = vv
                    break
        return v.strip().lower() if v else None
    except Exception:
        return None


def _identity_email(request):
    """目前登入 email:站內 Google cookie;CF Access 標頭只在明確信任時採用(見 TRUST_CF_ACCESS_HEADER);皆無=匿名。"""
    email = _cookie_email(request)
    if email:
        return email
    if TRUST_CF_ACCESS_HEADER and not AUTH_ENABLED:
        return _req_email(request)
    return None


def _is_owner(request):
    """公開站:僅登入 email == OWNER_EMAIL 才是站主(fail-closed —— 匿名/不符一律訪客)。
    桌面/dev(非 PUBLIC,單人本機)一律站主。公開模式沒有任何放行捷徑(舊 CLINSCOPE_FORCE_OWNER 已移除)。"""
    if not PUBLIC:
        return True
    return _identity_email(request) == OWNER_EMAIL


def _add_auth_routes(fastapi_app):
    """掛在 Gradio 底層 FastAPI 上的登入/登出端點(站內 Google 登入)。"""
    from fastapi import Request
    from fastapi.responses import JSONResponse
    cookie_secure = os.environ.get("CLINSCOPE_COOKIE_SECURE", "1") == "1"

    @fastapi_app.post("/auth/google")
    async def _auth_google(request: Request):
        try:
            body = await request.json()
        except Exception:
            body = {}
        try:
            info = verify_google_credential((body or {}).get("credential", ""))
        except Exception as e:
            return JSONResponse({"error": f"invalid credential: {e}"}, status_code=401)
        email = (info.get("email") or "").strip().lower()
        resp = JSONResponse({"email": email, "owner": email == OWNER_EMAIL})
        resp.set_cookie(COOKIE_NAME, make_session(email), max_age=SESSION_MAX_AGE,
                        httponly=True, samesite="lax", secure=cookie_secure, path="/")
        return resp

    @fastapi_app.post("/auth/logout")
    async def _auth_logout():
        resp = JSONResponse({"ok": True})
        resp.delete_cookie(COOKIE_NAME, path="/")
        return resp


def _guard_owner(request):
    """後端縱深防禦:非站主直接呼叫寫入/計算端點時擋下(UI 虛化只是第一層)。"""
    if not _is_owner(request):
        raise gr.Error("僅站主可執行此操作（您目前為訪客／唯讀）。")


def on_identity(request: gr.Request):
    """載入時判定身分 → 身分列(含登入/登出鈕)+ 開/關(虛化)可操作按鈕。公開站專用。"""
    owner = _is_owner(request)
    email = _identity_email(request)
    try:
        with open(_ID_LOG, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S}\temail={email}\towner={owner}\tauth={AUTH_ENABLED}\n")
    except Exception:
        pass
    logout_btn = '<button class="cs-authbtn" id="cs-logout">登出</button>' if AUTH_ENABLED else ''
    if owner:
        bar = ('<div class="cs-idbar cs-owner">✅ 已登入站主 <b>' + _html.escape(email or OWNER_EMAIL)
               + '</b> · 可完整操作' + logout_btn + '</div>')
    elif email:      # 已登入但非站主
        bar = ('<div class="cs-idbar cs-guest">👁 訪客（唯讀）· 目前登入 <b>' + _html.escape(email)
               + '</b>，非站主' + logout_btn + '</div>')
    else:            # 匿名
        login = '<span class="cs-gbtn" id="cs-gbtn"></span>' if AUTH_ENABLED else ''
        tail = ('　站主登入 ' + login) if AUTH_ENABLED else ''
        bar = ('<div class="cs-idbar cs-guest">👁 訪客模式（唯讀）· 僅站主可操作按鈕' + tail + '</div>')
    en = gr.update(interactive=owner)
    vis = gr.update(visible=owner)
    # 對應輸出:identity_html, btn_folder, btn_batch, btn_single, btn_apply, btn_csv, btn_save, mode_chk, btn_filter
    return bar, en, en, en, en, en, en, vis, vis


def _open_rgb(path):
    return Image.open(path).convert("RGB")


def scan_folder(folder):
    p = Path(folder) if folder else None
    if not p or not p.is_dir():
        return []
    return sorted(str(f) for f in p.iterdir()
                  if f.is_file() and f.suffix.lower() in IMG_EXTS)


def model_classes(spec):
    """cls 類別名(prettify 過,與推論輸出一致)。先讀 model card,否則讀權重。"""
    c = CARDS.get(spec["key"]) or {}
    raw = c.get("classes")
    if not raw:
        raw = load_fm(spec["fm"])["classes"] if spec.get("fm") else load_cls(W.ensure_local(spec))[1]
    return [prettify(x) for x in raw]


@torch.no_grad()
def cls_rank(image, spec, progress=None):
    """回傳 [(類別, 機率)],含全部類別、由高到低。"""
    if spec.get("fm"):
        import sys as _sys
        _sys.path.insert(0, str(ROOT / "src"))
        import fm_features
        art = load_fm(spec["fm"], progress)
        feat = fm_features.extract_feature(image, DEVICE, grid=art["grid"]).reshape(1, -1)
        probs = art["clf"].predict_proba(art["scaler"].transform(feat))[0]
        classes = art["classes"]
    else:
        model, classes, tf = load_cls(W.ensure_local(spec, progress=progress))
        x = tf(image).unsqueeze(0).to(DEVICE)
        probs = F.softmax(model(x), dim=1)[0].cpu().numpy()
    order = list(np.argsort(probs)[::-1])
    return [(prettify(classes[i]), float(probs[i])) for i in order]


@torch.no_grad()
def det_infer(image, spec, conf, progress=None):
    """回傳 (counts dict, 標註圖 RGB ndarray, 物件總數)。"""
    counts: dict = {}
    if spec.get("sahi"):
        from sahi.predict import get_sliced_prediction
        from sahi.utils.cv import visualize_object_predictions
        sm = load_sahi(W.ensure_local(spec, progress=progress))
        sm.confidence_threshold = float(conf)
        img_np = np.array(image)
        result = get_sliced_prediction(
            img_np, sm, slice_height=640, slice_width=640,
            overlap_height_ratio=0.2, overlap_width_ratio=0.2, verbose=0)
        opl = result.object_prediction_list
        annotated = visualize_object_predictions(img_np.copy(), object_prediction_list=opl)["image"]
        for o in opl:
            counts[o.category.name] = counts.get(o.category.name, 0) + 1
    else:
        model = load_yolo(W.ensure_local(spec, progress=progress))
        res = model.predict(source=image, conf=float(conf), verbose=False)[0]
        annotated = res.plot()[:, :, ::-1]  # BGR -> RGB
        for c in res.boxes.cls.tolist():
            name = res.names[int(c)]
            counts[name] = counts.get(name, 0) + 1
    return counts, annotated, int(sum(counts.values()))


# ============================================================================
#  工作台狀態 + 檢視渲染
# ============================================================================
def new_state(label):
    spec = BY_LABEL[label]
    return {"label": label, "kind": spec["kind"], "files": [], "results": {},
            "page": TOTAL, "view": "詳細資料", "page_files": [], "selected": None,
            "custom_pages": [], "view_open": False, "switch_open": False}


def all_classes(S):
    """分類型可指派的分頁:模型內建類別 + Null + 使用者自訂分類。"""
    return model_classes(BY_LABEL[S["label"]]) + [NULLPAGE] + S.get("custom_pages", [])


def switch_choices(S):
    """工具列「⇄ 切換分類」下拉選項(含自訂分類)。"""
    if S["kind"] != "cls":
        return [SW_PLACEHOLDER]
    return [SW_PLACEHOLDER] + [f"⇄ {c}" for c in all_classes(S)]


def page_choices(S):
    """回傳 [(顯示, 值)] —— 分類後分頁標籤帶數量,如『neutrophil（12）』;值為乾淨頁鍵。"""
    files, res, n = S["files"], S["results"], len(S["files"])
    out = [(f"{TOTAL}（{n}）" if files else TOTAL, TOTAL)]
    if S["kind"] == "cls":
        cnt = Counter(r["page"] for r in res.values()) if res else None
        for k in all_classes(S):
            out.append((f"{k}（{cnt[k]}）", k) if cnt is not None else (k, k))
        out.append(("＋", ADDPAGE))   # 「新增自訂分類」分頁(前端攔截點擊→prompt)
        return out
    if res:
        for c in sorted({c for r in res.values() for c in r["counts"]}):
            out.append((f"{c}（{sum(1 for r in res.values() if c in r['counts'])}）", c))
    return out


def page_files(S):
    files, res, page = S["files"], S["results"], S["page"]
    if page in (TOTAL, ADDPAGE):
        return files
    if S["kind"] == "cls":
        return [p for p in files if res.get(p, {}).get("page") == page]
    return [p for p in files if page in (res.get(p, {}).get("counts") or {})]


def gallery_items(S, pf):
    res, kind, out = S["results"], S["kind"], []
    for p in pf:
        r = res.get(p)
        if r is None:
            cap = Path(p).name
        elif kind == "cls":
            cap = f"{r['page']} · {r['prob'] * 100:.0f}%"
        else:
            cap = f"{r['total']} 物件"
        out.append((_servable(p), cap))
    return out


def _servable(p):
    """公開站圖示檢視:原檔(範例夾/站主本地模式資料夾)不在 allowed_paths 內,先複製進 Gradio 快取
    (以內容雜湊命名,與 Gradio 自己的做法相同)再交給 Gallery;原資料夾因此不必對外開放。桌面版直接用原路徑。"""
    if not PUBLIC:
        return p
    try:
        if Path(p).resolve().is_relative_to(_GRADIO_TMP):
            return p
        from gradio.processing_utils import save_file_to_cache
        return save_file_to_cache(p, str(_GRADIO_TMP))
    except Exception:
        return p          # 交回原路徑 → Gradio 自身的路徑檢查會擋(fail-closed,不會因此對外提供)


def _file_meta(p):
    """回傳 (日期, 類型, 大小) 字串。"""
    pp = Path(p)
    try:
        st = pp.stat()
        date = datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d")
        sz = f"{st.st_size / 1024:.0f} KB" if st.st_size < 1048576 else f"{st.st_size / 1048576:.1f} MB"
    except Exception:
        date, sz = "—", "—"
    return date, (pp.suffix.lstrip(".").upper() or "—"), sz


def _cls_select_html(classes, selected, idx):
    """分類欄的內嵌下拉選單(<select>);change 由前端 JS 透過 bridge 回傳給 on_bridge。"""
    opts = "".join(
        f'<option value="{_html.escape(c)}"{" selected" if c == selected else ""}>{_html.escape(c)}</option>'
        for c in classes)
    return f'<select class="cs-cls" data-i="{idx}">{opts}</select>'


def _dt_row(idx, name, cells, selected=False):
    tds = f'<td class="cs-name" title="{name}">{name}</td>' + "".join(f"<td>{c}</td>" for c in cells)
    sel = " cs-sel-row" if selected else ""
    return f'<tr class="cs-row{sel}" data-i="{idx}">{tds}</tr>'


def detail_html_str(S):
    """詳細資料改用自訂 HTML 表格 —— 分類欄為真正的內嵌 <select> 下拉。"""
    kind, res, pf = S["kind"], S["results"], S["page_files"]
    sel_p = S.get("selected")
    if not S["files"]:   # 尚未選資料夾 → 友善空狀態(避免一開啟整片空白)
        return ('<div class="cs-empty-state"><div class="cs-es-ic">📂</div>'
                '<div class="cs-es-t">尚未載入影像</div>'
                f'<div class="cs-es-s">點上方 <b>📁 {"上傳影像" if PUBLIC else "選擇資料夾"}</b> 載入影像，再按 '
                '<b>▶ 批次計算</b> 開始分類；可用 <b>＋</b> 分頁自訂分類。</div></div>')
    if kind == "cls":
        headers = ["檔名", "分類", "信心", "日期", "類型", "大小"]
        classes = all_classes(S)
        rows = []
        for i, p in enumerate(pf):
            r = res.get(p)
            d, t, z = _file_meta(p)
            if r:
                conf = "手動" if r.get("manual") else f"{r['prob'] * 100:.1f}%"
                cls_cell = _cls_select_html(classes, r["page"], i)
            else:
                conf, cls_cell = "—", "—"
            rows.append(_dt_row(i, _html.escape(Path(p).name), [cls_cell, conf, d, t, z], selected=(p == sel_p)))
    else:
        headers = ["檔名", "物件總數", "主要類別", "日期", "類型", "大小"]
        rows = []
        for i, p in enumerate(pf):
            r = res.get(p)
            d, t, z = _file_meta(p)
            if r and r["counts"]:
                cells = [str(r["total"]), _html.escape(max(r["counts"], key=r["counts"].get)), d, t, z]
            elif r:
                cells = ["0", "—", d, t, z]
            else:
                cells = ["—", "—", d, t, z]
            rows.append(_dt_row(i, _html.escape(Path(p).name), cells, selected=(p == sel_p)))
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join(rows) or '<tr><td colspan="6" class="cs-empty">（此分頁沒有檔案）</td></tr>'
    cls_attr = "cs-dt cls" if kind == "cls" else "cs-dt det"
    return (f'<div class="cs-dt-wrap"><table class="{cls_attr}">'
            f'<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>')


def count_summary_md(S):
    n, res = len(S["files"]), S["results"]
    if not S["files"]:
        return "📂 尚未選擇資料夾。"
    if not res:
        return f"📂 共 **{n}** 張影像 · 尚未計算（按 ▶ 批次計算）。"
    if S["kind"] == "cls":
        pages = Counter(r["page"] for r in res.values())
        nullc = pages.get(NULLPAGE, 0)
        body = " ｜ ".join(f"{k} **{v}**" for k, v in pages.most_common(8) if k != NULLPAGE)
        return f"已分類 **{len(res)}/{n}** · 待確認 Null **{nullc}**　{('｜ ' + body) if body else ''}"
    agg = Counter()
    for r in res.values():
        for k, v in r["counts"].items():
            agg[k] += v
    body = " ｜ ".join(f"{k} **{v}**" for k, v in agg.most_common(12))
    return f"已計算 **{len(res)}/{n}** 張 · 物件總計 **{sum(agg.values())}**　{('｜ ' + body) if body else ''}"


def render(S):
    """回傳 (page_radio, gallery, detail_html, count_md) 四個 update。"""
    pages = page_choices(S)
    keys = [p[1] for p in pages]
    if S["page"] == ADDPAGE or S["page"] not in keys:  # 「＋」永不是真正選中的分頁
        S["page"] = TOTAL
    pf = page_files(S)
    S["page_files"] = pf
    if S["view"] == "詳細資料":
        gal = gr.update(visible=False)
        dh = gr.update(visible=True, value=detail_html_str(S))
    else:  # 圖示(已移除超大圖示)
        gal = gr.update(visible=True, columns=8, value=gallery_items(S, pf))
        dh = gr.update(visible=False)
    pr = gr.update(choices=pages, value=S["page"])
    cs = count_summary_md(S)
    return pr, gal, dh, cs


# ============================================================================
#  模型卡 / 影像詮釋文字
# ============================================================================
def engine_of(spec, card):
    if spec.get("fm"):
        return "🧬 Phikon-v2 FM + LogReg"
    if spec.get("sahi"):
        return "🔪 YOLOv8s + SAHI（切片推論）"
    return card.get("model_type") or ("YOLOv8s" if spec["kind"] == "det" else "ResNet-50")


def card_text(label):
    spec = BY_LABEL[label]
    card = CARDS.get(spec["key"]) or {}
    kind = spec["kind"]
    rows = [f"### {label}"]
    tag = "偵測 · 計數型" if kind == "det" else "分類型"
    rows.append(f"`{tag}` ｜ 引擎 **{engine_of(spec, card)}** ｜ 科別 **{disc_of(spec['key'])}**")
    m = card.get("metrics") or {}
    if kind == "cls":
        if card.get("num_classes"):
            rows.append(f"類別數 **{card['num_classes']}**")
        seg = []
        if m.get("val_acc") is not None:
            seg.append(f"val acc **{m['val_acc']:.3f}**")
        if m.get("macro_f1") is not None:
            seg.append(f"macro-F1 **{m['macro_f1']:.3f}**")
        if m.get("n_train") and m.get("n_val"):
            seg.append(f"train/val {m['n_train']}/{m['n_val']}")
        if seg:
            rows.append(" ｜ ".join(seg))
    else:
        seg = []
        if m.get("mAP50") is not None:
            seg.append(f"mAP50 **{m['mAP50']:.3f}**")
        if m.get("mAP50_95") is not None:
            seg.append(f"mAP50-95 **{m['mAP50_95']:.3f}**")
        if m.get("precision") is not None:
            seg.append(f"P {m['precision']:.2f} / R {m['recall']:.2f}")
        if seg:
            rows.append(" ｜ ".join(seg))
    meta_seg = [x for x in [(f"來源 {card.get('source')}" if card.get("source") else None),
                            (f"授權 {card.get('license')}" if card.get("license") else None),
                            (f"切分 {card.get('split_method')}" if card.get("split_method") else None)] if x]
    if meta_seg:
        rows.append("<small>" + " ｜ ".join(meta_seg) + "</small>")
    blurb = card.get("experiment_summary") or spec.get("blurb") or ""
    if blurb:
        rows.append(blurb)
    cav = card.get("caveats") or []
    if cav:
        rows.append("**⚠️ 注意**\n" + "\n".join(f"- {c}" for c in cav))
    return "\n\n".join(rows)


def meta_text_cls(p, r):
    name = Path(p).name
    if not r:
        return f"**{name}**\n\n尚未計算此影像（按 ◉ 單張計算）。"
    lines = [f"**{name}**", f"指派分類　**{r['page']}**" + ("　*（手動）*" if r.get("manual") else "")]
    if r.get("prob") is not None:
        lines.append(f"Top-1 信心　**{r['prob'] * 100:.1f}%**")
    if r.get("ranked"):
        lines.append("")
        lines += [f"- {n} · {pb * 100:.1f}%" for n, pb in r["ranked"][:5]]
    return "\n".join(lines)


def meta_text_det(p, r):
    name = Path(p).name
    if not r:
        return f"**{name}**\n\n尚未計算此影像（按 ◉ 單張計算）。"
    body = " ｜ ".join(f"{k} **{v}**" for k, v in sorted(r["counts"].items())) or "（未偵測到物件）"
    return f"**{name}**\n\n物件總數　**{r['total']}**\n\n{body}"


# ============================================================================
#  事件處理
# ============================================================================
def _thr_update(kind):
    if kind == "cls":
        return gr.update(minimum=0.0, maximum=1.0, value=0.5, step=0.01,
                         label="信心門檻（分類）",
                         info="Top-1 機率低於此值 → 歸入 Null 待確認")
    return gr.update(minimum=0.05, maximum=0.9, value=0.25, step=0.05,
                     label="偵測門檻（計數）", info="低於此信心的偵測框不計入")


def _dl_banner(spec) -> str:
    """模型權重未下載時的提示橫幅(packaged 安裝才會出現;dev 權重齊全→空字串)。"""
    mb = W.pending_size_mb(spec)
    if mb <= 0:
        return ""
    return (f"> ⬇ **此模型權重尚未下載（約 {mb:g} MB）。** "
            f"按 **▶批次計算** 或點選影像時會自動從雲端下載並快取，之後免再下載。\n\n")


def on_disc(disc):
    labels = labels_for(disc)
    return gr.update(choices=model_choices(disc), value=(labels[0] if labels else None))


def on_model(label, S):
    S["label"] = label
    spec = BY_LABEL[label]
    S["kind"] = spec["kind"]
    S["results"] = {}
    S["page"] = TOTAL
    S["selected"] = None
    S["custom_pages"] = []   # 換模型 = 換分類體系,清掉上一個模型的自訂分頁
    S["view_open"] = False
    S["switch_open"] = False
    pr, gal, df, cs = render(S)
    # 換模型:重設 ⇄ 分類選項(僅分類型可用,偵測型停用)
    return (S, _dl_banner(spec) + card_text(label), _thr_update(spec["kind"]),
            pr, gal, df, cs,
            gr.update(value=SW_PLACEHOLDER, interactive=False,  # switch_dd:未選檔虛化、顯示「切換分類」
                      choices=switch_choices(S)),
            None, "選一張影像檢視大圖。")


def _native_dir():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askdirectory(title="選擇影像資料夾")
        try:
            root.destroy()
        except Exception:
            pass
        return path or ""
    except Exception:
        return ""


def on_load_folder(path, S):
    files = scan_folder(path)
    S["files"], S["results"], S["page"], S["selected"] = files, {}, TOTAL, None
    pr, gal, df, cs = render(S)
    msg = (f"📂 已載入 `{path}` 共 **{len(files)}** 張影像。" if files
           else f"⚠️ `{path}` 找不到影像檔（支援 jpg/png/tif/bmp/webp/gif）。")
    return S, pr, gal, df, cs, None, msg


def on_pick_folder(S):
    path = _native_dir()
    if not path:
        pr, gal, df, cs = render(S)
        return S, pr, gal, df, cs, gr.update(), "已取消（或系統無法開啟選擇視窗）。"
    return on_load_folder(path, S)


def on_uploaded_paths(paths_json, S, request: gr.Request = None):
    """公開站分批上傳:前端已把整夾影像切成多個 <90MB 批次逐批上傳到 Gradio 暫存夾,
    再把「已上傳的伺服器路徑」JSON 陣列送來這裡。**只接受落在 Gradio 上傳暫存根內、且真的是檔案的
    影像路徑**(擋路徑注入 —— 否則訪客可塞任意 server 路徑,經 allowed_paths 各磁碟根被讀出)。"""
    _guard_owner(request)   # 縱深防禦:訪客即使繞過虛化按鈕也不能上傳
    try:
        raw = json.loads(paths_json) if paths_json else []
    except Exception:
        raw = []
    imgs = []
    for p in raw:
        if not isinstance(p, str):
            continue
        try:
            rp = Path(p).resolve()
        except Exception:
            continue
        if rp.suffix.lower() not in IMG_EXTS:
            continue
        try:
            inside = rp.is_relative_to(_GRADIO_TMP)
        except Exception:
            inside = False
        if inside and rp.is_file():
            imgs.append(str(rp))
    imgs = sorted(imgs)
    S["files"], S["results"], S["page"], S["selected"] = imgs, {}, TOTAL, None
    S["upseq"] = S.get("upseq", 0) + 1          # 單調遞增:前端上傳進度層據此判定「已完成」並收起
    pr, gal, df, cs = render(S)
    msg = (f"📂 已載入 **{len(imgs)}** 張影像。按 ▶ 批次計算 開始分類。" if imgs
           else "⚠️ 這批檔案裡找不到有效影像（支援 jpg/png/tif/bmp/webp/gif）。")
    return S, pr, gal, df, cs, None, msg, f"done:{S['upseq']}:{len(imgs)}"


def _precompute_sample():
    """啟動時把預設範例夾影像用 SAMPLE_MODEL_KEY 模型預先分類好,存進全域 SAMPLE。
    只在公開站 + 資料夾存在 + 模型可用時做;失敗不影響啟動(退回空狀態)。"""
    try:
        if not (PUBLIC and SAMPLE_DIR.is_dir()):
            return
        spec = next((m for m in AVAILABLE if m["key"] == SAMPLE_MODEL_KEY and m.get("kind") == "cls"), None)
        files = scan_folder(str(SAMPLE_DIR))
        if not spec or not files:
            return
        for rel in W.pending_rels(spec):
            W.ensure_file(rel)
        res = {}
        for p in files:
            try:
                ranked = cls_rank(_open_rgb(p), spec)
                top, prob = ranked[0]
                res[p] = {"page": top if prob >= 0.5 else NULLPAGE,
                          "prob": prob, "ranked": ranked[:6], "manual": False}
            except Exception:
                continue
        SAMPLE["label"], SAMPLE["files"], SAMPLE["results"] = spec["label"], files, res
        print(f"[ClinScope] sample preloaded: {len(res)}/{len(files)} imgs via {spec['key']}")
    except Exception as e:
        print(f"[ClinScope] sample preload skipped: {e}")


def on_load_sample(S):
    """把預先分類好的範例載入目前 session(當作預設畫面)。無範例→維持空狀態。"""
    if not SAMPLE["files"]:
        return (S,) + render(S) + (None, gr.update())
    S["files"] = list(SAMPLE["files"])
    S["results"] = {k: dict(v) for k, v in SAMPLE["results"].items()}
    S["page"], S["selected"] = TOTAL, None
    pr, gal, df, cs = render(S)
    return S, pr, gal, df, cs, None, "📂 範例:OpenFungi 絲狀黴菌 50 張(已預先分類,可直接瀏覽)。"


def on_view(view, S):
    S["view"] = (view or "👁 圖示").replace("👁 ", "")
    return (S,) + render(S)


def on_page(page, S):
    if page == ADDPAGE:        # 「＋」由前端 prompt 處理;後端收到就維持原分頁(不切換)
        pr, gal, df, cs = render(S)
        return S, pr, gal, df, cs, gr.update(), gr.update()
    S["page"], S["selected"] = page, None
    pr, gal, df, cs = render(S)
    return S, pr, gal, df, cs, None, "選一張影像檢視大圖。"


def on_batch(label, thr, S, progress=gr.Progress(), request: gr.Request = None):
    _guard_owner(request)
    spec = BY_LABEL[label]
    files = S["files"]
    if not files:
        raise gr.Error("請先選擇含影像的資料夾。")
    for rel in W.pending_rels(spec):    # 首次用此模型:先下載權重(顯示進度),再開始計算
        W.ensure_file(rel, progress=progress)
    res = {}
    for i, p in enumerate(files):
        progress((i + 1) / len(files), desc=f"計算中 {i + 1}/{len(files)}")
        try:
            img = _open_rgb(p)
        except Exception:
            continue
        if spec["kind"] == "cls":
            ranked = cls_rank(img, spec)
            top, prob = ranked[0]
            res[p] = {"page": top if prob >= float(thr) else NULLPAGE,
                      "prob": prob, "ranked": ranked[:6], "manual": False}
        else:
            counts, _, total = det_infer(img, spec, float(thr))
            res[p] = {"counts": counts, "total": total}
    S["results"], S["page"], S["selected"] = res, TOTAL, None
    pr, gal, df, cs = render(S)
    return S, pr, gal, df, cs, None, f"✅ 批次完成 **{len(res)}** 張。"


def on_single(label, thr, S, progress=gr.Progress(), request: gr.Request = None):
    _guard_owner(request)
    sel = S["selected"]
    if not sel:
        raise gr.Error("請先在下方點選一張影像（單張計算）。")
    spec = BY_LABEL[label]
    for rel in W.pending_rels(spec):    # 首次用此模型:先下載權重(顯示進度)
        W.ensure_file(rel, progress=progress)
    img = _open_rgb(sel)
    if spec["kind"] == "cls":
        ranked = cls_rank(img, spec)
        top, prob = ranked[0]
        S["results"][sel] = {"page": top if prob >= float(thr) else NULLPAGE,
                             "prob": prob, "ranked": ranked[:6], "manual": False}
        pv, meta = img, meta_text_cls(sel, S["results"][sel])
    else:
        counts, annotated, total = det_infer(img, spec, float(thr))
        S["results"][sel] = {"counts": counts, "total": total}
        pv, meta = annotated, meta_text_det(sel, S["results"][sel])
    pr, gal, df, cs = render(S)
    return S, pr, gal, df, cs, pv, meta


def on_switch(newpage, S):
    sel = S["selected"]
    page = newpage.replace("⇄ ", "") if newpage else None
    if sel and sel in S["results"] and page and page != "切換分類":
        S["results"][sel]["page"] = page
        S["results"][sel]["manual"] = True
    return (S,) + render(S)


def on_apply_thr(thr, S):
    if S["kind"] == "cls" and S["results"]:
        for r in S["results"].values():
            if not r.get("manual") and r.get("ranked"):
                top, prob = r["ranked"][0]
                r["page"] = top if prob >= float(thr) else NULLPAGE
    return (S,) + render(S)


def _select_core(S, label, thr, idx, progress=None):
    pf = S["page_files"]
    if not isinstance(idx, int) or idx < 0 or idx >= len(pf):
        return S, gr.update(), gr.update(), gr.update()
    p = pf[idx]
    S["selected"] = p
    spec = BY_LABEL[label]
    r = S["results"].get(p)
    if spec["kind"] == "cls":
        sw = gr.update(interactive=(r is not None),
                       value=(f"⇄ {r['page']}" if r else SW_PLACEHOLDER))
        return S, sw, _open_rgb(p), meta_text_cls(p, r)
    # det:點選即以目前門檻產生標註圖(首次用此模型先下載權重)
    for rel in W.pending_rels(spec):
        W.ensure_file(rel, progress=progress)
    counts, annotated, total = det_infer(_open_rgb(p), spec, float(thr))
    S["results"][p] = {"counts": counts, "total": total}
    return S, gr.update(), annotated, meta_text_det(p, S["results"][p])


def on_select(evt: gr.SelectData, S, label, thr, progress=gr.Progress()):
    return _select_core(S, label, thr, evt.index, progress=progress)


def on_bridge(cmd, S, label, thr, progress=gr.Progress()):
    """詳細資料自訂 HTML 表的前端事件橋接:
       {"a":"sel","i":idx} 選列(更新大圖) / {"a":"cls","i":idx,"p":類別} 內嵌下拉改分類 /
       {"a":"addpage","name":名稱} 「＋」分頁新增自訂分類。
       輸出順序 = [S, detail_html, gallery, page_radio, count_md, switch_dd, preview_img, preview_meta]。"""
    noop = (S,) + (gr.update(),) * 7
    if not cmd:
        return noop
    try:
        obj = json.loads(cmd)
    except Exception:
        return noop
    act = obj.get("a")
    if act == "addpage":   # 「＋」分頁:新增一個自訂分類分頁並切換過去
        name = (obj.get("name") or "").strip()
        cp = S.setdefault("custom_pages", [])
        reserved = set(model_classes(BY_LABEL[S["label"]])) | {NULLPAGE, TOTAL, ADDPAGE, "＋"}
        if name and name not in cp and name not in reserved:
            cp.append(name)
            S["page"], S["selected"] = name, None
        pr, gal, dh, cs = render(S)
        sw = gr.update(choices=switch_choices(S), value=SW_PLACEHOLDER, interactive=False)
        return (S, dh, gal, pr, cs, sw, gr.update(), gr.update())
    idx = obj.get("i")
    pf = S["page_files"]
    if not isinstance(idx, int) or idx < 0 or idx >= len(pf):
        return noop
    p = pf[idx]
    if act == "sel":
        # 選列:不重繪表格(避免方向鍵連按時閃爍),只更新大圖 + 工具列切換下拉
        _, sw, pv, meta = _select_core(S, label, thr, idx, progress=progress)
        return (S, gr.update(), gr.update(), gr.update(), gr.update(), sw, pv, meta)
    if act == "cls":
        page = obj.get("p")
        S["selected"] = p
        if page and p in S["results"]:
            S["results"][p]["page"] = page
            S["results"][p]["manual"] = True
        pr, gal, dh, cs = render(S)
        sw = gr.update(value=f"⇄ {page}", interactive=True)
        return (S, dh, gal, pr, cs, sw, gr.update(), gr.update())
    return noop


def _safe(name):
    return re.sub(r'[\\/:*?"<>|\s]+', "_", str(name)).strip("_") or "x"


def _export_dir():
    """匯出檔(CSV/ZIP)存放處。桌面版沿用 repo/exports;公開站改放系統暫存下的隨機子目錄 ——
    不在 repo、不在 allowed_paths、也不在 Gradio 上傳暫存根 → 原檔無法被 /gradio_api/file= 猜路徑抓走,
    只透過 Gradio 以內容雜湊命名的快取副本下載。順手清掉一天前的舊匯出。"""
    if not PUBLIC:
        exp = ROOT / "exports"
        exp.mkdir(exist_ok=True)
        return exp
    base = Path(tempfile.gettempdir()) / "clinscope_exports"
    base.mkdir(parents=True, exist_ok=True)
    cutoff = datetime.now().timestamp() - 86400
    for old in base.iterdir():
        try:
            if old.stat().st_mtime < cutoff:
                shutil.rmtree(old, ignore_errors=True)
        except OSError:
            pass
    return Path(tempfile.mkdtemp(dir=base))


def on_csv(label, S, request: gr.Request = None):
    _guard_owner(request)
    res = S["results"]
    if not res:
        raise gr.Error("尚無計算結果可匯出。")
    spec = BY_LABEL[label]
    exp = _export_dir()
    fn = exp / f"clinscope_{spec['key']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(fn, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if spec["kind"] == "cls":
            w.writerow(["檔案", "指派分類", "Top-1信心(%)", "手動"])
            for p, r in res.items():
                w.writerow([Path(p).name, r["page"],
                            f"{r.get('prob', 0) * 100:.2f}", "Y" if r.get("manual") else ""])
        else:
            allc = sorted({c for r in res.values() for c in r["counts"]})
            w.writerow(["檔案", "物件總數"] + allc)
            for p, r in res.items():
                w.writerow([Path(p).name, r["total"]] + [r["counts"].get(c, 0) for c in allc])
    return gr.update(value=str(fn), visible=True)


def _format_name(mode, cls, p):
    """依使用者選的命名格式組出新檔名(分類已 _safe,原檔主檔名保留)。"""
    stem, ext, c = p.stem, p.suffix, _safe(cls)
    if mode == "n_c":
        return f"{stem}_{c}{ext}"         # 檔名_分類
    if mode == "n_pc":
        return f"{stem} ({c}){ext}"       # 檔名 (分類)
    return f"{c}_{stem}{ext}"             # 分類_檔名(預設)


def on_saveas(S):
    """按『另存資料夾』:先跳出檔名格式對話框(實際複製在 on_saveas_confirm)。"""
    if not S["page_files"]:
        raise gr.Error("目前分頁沒有檔案可另存。")
    return gr.update(visible=True)


def on_saveas_confirm(mode, S):
    """選好檔名格式後:開資料夾選擇框 → 依格式命名複製;回傳關閉對話框的 update。"""
    page, pf = S["page"], S["page_files"]
    if not pf:
        gr.Info("目前分頁沒有檔案可另存。")
        return gr.update(visible=False)
    dest = _native_dir()
    if not dest:
        gr.Info("已取消另存。")
        return gr.update(visible=False)
    res, kind, n = S["results"], S["kind"], 0
    for p in pf:
        cls = page if page != TOTAL else ((res.get(p, {}).get("page") or "未分類") if kind == "cls" else "全部")
        try:
            shutil.copy2(p, Path(dest) / _format_name(mode, cls, Path(p)))
            n += 1
        except Exception:
            pass
    label = {"n_c": "檔名_分類", "n_pc": "檔名 (分類)"}.get(mode, "分類_檔名")
    gr.Info(f"✅ 已另存 {n} 個檔案到 {dest}（{label}）")
    return gr.update(visible=False)


def on_saveas_confirm_zip(mode, S, request: gr.Request = None):
    """公開站:選好檔名格式後 → 依格式命名打包成 zip 供瀏覽器下載(取代寫入 server 硬碟,
    因為遠端訪客本來就不該、也無法寫檔到 server)。回傳(關對話框, 顯示可下載的 zip)。"""
    if not _is_owner(request):
        gr.Info("僅站主可打包下載。")
        return gr.update(visible=False), gr.update()
    import zipfile
    page, pf = S["page"], S["page_files"]
    if not pf:
        gr.Info("目前分頁沒有檔案可下載。")
        return gr.update(visible=False), gr.update()
    res, kind = S["results"], S["kind"]
    zpath = _export_dir() / f"clinscope_{_safe(page)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    used, n = {}, 0
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in pf:
            cls = page if page != TOTAL else ((res.get(p, {}).get("page") or "未分類") if kind == "cls" else "全部")
            arc = _format_name(mode, cls, Path(p))
            k = arc.lower()
            if k in used:                       # 同名衝突:加序號避免壓縮包內互相覆蓋
                used[k] += 1
                arc = f"{Path(arc).stem}_{used[k]}{Path(arc).suffix}"
            else:
                used[k] = 0
            try:
                z.write(p, arcname=arc)
                n += 1
            except Exception:
                pass
    label = {"n_c": "檔名_分類", "n_pc": "檔名 (分類)"}.get(mode, "分類_檔名")
    gr.Info(f"✅ 已打包 {n} 個檔案（{label}），開始下載。")
    return gr.update(visible=False), gr.update(value=str(zpath), visible=True)


# ── 站主「本地模式」切換:公開站跑在站主本機,站主人在機器旁時可改用伺服器端原生選夾/另存 ──
def _save_modal_title(is_local):
    t = "📁 另存資料夾" if is_local else "📦 打包下載 ZIP"
    verb = "複製" if is_local else "打包"
    return (f'<div class="cs-modal-title">{t} — 檔名格式</div>'
            f'<div class="cs-modal-sub">選擇{verb}後的檔名命名方式（分類取自目前分頁／各檔指派）</div>')


def on_mode_change(is_local, S, request: gr.Request = None):
    """站主切公開/本地模式 → 改資料夾鈕與另存鈕的標籤 + 前端旗標(#cs-mode) + 對話框標題。"""
    if not _is_owner(request):
        return S, gr.update(), gr.update(), gr.update(), gr.update()
    S["localmode"] = bool(is_local)
    folder_lbl = "📁 選擇資料夾" if is_local else "📁 上傳影像"
    save_lbl = "💾 另存資料夾" if is_local else "💾 打包下載 ZIP"
    flag = "local" if is_local else "public"
    return S, gr.update(value=folder_lbl), gr.update(value=save_lbl), flag, _save_modal_title(is_local)


def on_pick_folder_pub(S, request: gr.Request = None):
    """公開站的資料夾鈕:僅『站主 + 本地模式』走伺服器端原生選夾;否則 no-op
    (公開模式的上傳由前端 UPLOAD_JS 攔截處理,不會走到這裡)。"""
    if PUBLIC and S.get("localmode") and _is_owner(request):
        return on_pick_folder(S)
    pr, gal, df, cs = render(S)
    return S, pr, gal, df, cs, gr.update(), gr.update()


def on_saveas_confirm_dispatch(fmt, S, request: gr.Request = None):
    """公開站確認另存:本地模式(站主)→ 原生另存資料夾;否則 → 打包下載 ZIP。"""
    if S.get("localmode") and _is_owner(request):
        return on_saveas_confirm(fmt, S), gr.update()   # 原生另存回單值,補 zip_out 佔位
    return on_saveas_confirm_zip(fmt, S, request)


def on_open_filter(request: gr.Request = None):
    """站主點篩選 icon → 開篩選對話框(訪客不會有此鈕,仍加身分保險)。"""
    return gr.update(visible=_is_owner(request))


def on_save_filter(min_score, hidden_keys, S, request: gr.Request = None):
    """站主儲存模型篩選:寫入持久化設定 → 重建科別/模型下拉(套用到所有訪客);回傳關對話框+更新兩下拉。"""
    if not _is_owner(request):
        gr.Info("僅站主可調整模型篩選。")
        return gr.update(visible=False), gr.update(), gr.update()
    pct = float(min_score or 0)                              # slider 0–100(%)
    MODEL_FILTER["min_score"] = None if pct <= 0 else pct / 100.0
    MODEL_FILTER["hidden"] = [str(k) for k in (hidden_keys or [])]
    _save_model_filter()
    vd = visible_disciplines() or DISCIPLINES
    disc0 = vd[0]
    labels = labels_for(disc0) or [m["label"] for m in AVAILABLE if disc_of(m["key"]) == disc0]
    gr.Info(f"✅ 已更新篩選:門檻 {('≥ ' + str(int(MODEL_FILTER['min_score']*100)) + '%') if MODEL_FILTER['min_score'] else '全部'}"
            f"、隱藏 {len(MODEL_FILTER['hidden'])} 個。")
    return (gr.update(visible=False),
            gr.update(choices=vd, value=disc0),
            gr.update(choices=model_choices(disc0), value=(labels[0] if labels else None)))


# ============================================================================
#  介面外觀:海軍藍 / 白 / 檸檬黃
# ============================================================================
THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("IBM Plex Sans"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"],
)

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&display=swap');
:root{
  --navy-900:#0B1F3A; --navy-700:#163C6B; --navy-500:#22568F; --navy-100:#E5EDF7;
  --paper:#F4F8FD; --surface:#FFFFFF; --ink:#0B1F3A; --muted:#5B708A;
  --lemon:#FFE14D; --lemon-soft:#FFF7C4; --line:#D8E3F0;
}
/* 滿版、隨視窗縮放;窄視窗時欄位自動換行堆疊 */
.gradio-container{background:var(--paper)!important; color:var(--ink);
  max-width:100%!important; width:100%!important; padding:8px 16px!important;
  height:100vh!important; overflow:hidden!important;}  /* 整頁鎖在視窗高度,捲動移進面板內部 */
/* Gradio .main.fillable 預設 max-width:1536px → 解除才能真正貼齊視窗 */
.gradio-container .main, .gradio-container .main.fillable{
  max-width:100%!important; width:100%!important; margin:0!important; padding:0!important;
  height:100%!important; min-height:0!important; overflow:hidden!important;}
/* 把視窗高度一路傳到內容欄,讓編輯區可 flex 撐滿、放到最大時不出現瀏覽器卷軸 */
.main.fillable > .wrap,
.main.fillable > .wrap > .contain,
.main.fillable > .wrap > .contain > .column{flex:1 1 auto!important; min-height:0!important; overflow:hidden!important;}
.main.fillable > .wrap > .contain > .column{gap:10px!important;}
.contain > .column > .block{flex:0 0 auto!important; margin:0!important;}  /* 品牌列 / 頁尾不伸縮 */
.toprow{flex:0 1 auto!important; min-height:0!important;}                  /* 上排(模型/顯示)依內容、可壓縮 */
.editrow{flex:1 1 0!important; min-height:0!important; align-items:stretch!important;}  /* 下排吃滿剩餘高度;stretch 讓面板撐滿(否則 unequal-height 只取內容高) */
.editrow > .column, .editrow .panel{min-height:0!important; height:100%!important;}
footer{display:none!important;}
/* 公開站身分列:站主(綠)/訪客(琥珀) */
#cs-identity:empty{display:none!important;}
.cs-idbar{font-size:12.5px;font-weight:600;padding:5px 12px;border-radius:8px;margin:2px 0 0;
  display:inline-block;border:1px solid var(--line)}
.cs-idbar.cs-owner{background:#E7F6EC;color:#1B7A3D;border-color:#B7E3C6}
.cs-idbar.cs-guest{background:var(--lemon-soft);color:#7A5B00;border-color:#F0DC8A}
.cs-authbtn{margin-left:10px;padding:2px 12px;border-radius:7px;border:1px solid currentColor;
  background:transparent;color:inherit;font-weight:700;font-size:12px;cursor:pointer;vertical-align:middle}
.cs-authbtn:hover{background:rgba(0,0,0,.06)}
.cs-gbtn{display:inline-block;vertical-align:middle;margin-left:4px}

/* 頂部品牌列 — 外層 gr.HTML(.brandwrap)的 .prose 內距歸零,讓品牌列填滿欄寬、
   與下方面板列等寬(不再靠負邊距硬撐,手機/桌機都對齊) */
.brandwrap, .brandwrap.prose, .brandwrap .html-container{padding:0!important; margin:0!important;
  border:none!important; background:transparent!important;}
.brandbar{display:flex; justify-content:space-between; align-items:center;
  background:linear-gradient(100deg,var(--navy-900),var(--navy-700)); color:#fff;
  padding:14px 22px; border-radius:14px; margin:0 0 14px 0;
  box-shadow:0 6px 22px rgba(11,31,58,.20);}
/* 右側:下載鈕 + 徽章一組(可換行) */
.brandbar .bmeta{display:flex; align-items:center; gap:8px; flex-wrap:wrap; justify-content:flex-end;}
.brandbar .bmeta .chip{margin:0;}
.brandbar .dlbtn{display:inline-flex; align-items:center; gap:5px; text-decoration:none; cursor:pointer;
  background:var(--lemon); color:var(--navy-900)!important; border:none; border-radius:999px;
  padding:6px 14px; font-size:13px; font-weight:700; font-family:'Space Grotesk',sans-serif;
  box-shadow:0 2px 8px rgba(255,205,0,.35); transition:transform .08s,box-shadow .12s;}
.brandbar .dlbtn:hover{transform:translateY(-1px); box-shadow:0 5px 16px rgba(255,205,0,.55);}
.brandbar .brand{display:flex; align-items:center; gap:14px;}
.brandbar .logo{color:var(--lemon); font-size:28px;}
.brandbar .btext{display:inline-flex; flex-direction:column; align-items:stretch;}
.brandbar .bt{font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:36px; letter-spacing:3px;
  color:var(--lemon)!important; text-shadow:0 1px 3px rgba(0,0,0,.30); line-height:1.05;}
.brandbar .bsub{color:#E3EEFB; font-size:15px; letter-spacing:0;
  text-align:justify; text-align-last:justify; margin-top:2px;}
.brandbar .chip{background:rgba(255,255,255,.12); border:1px solid rgba(255,255,255,.22);
  padding:4px 12px; border-radius:999px; font-size:12px; margin-left:8px; font-family:'IBM Plex Mono',monospace;}
.brandbar .chip.lemon{background:var(--lemon); color:var(--navy-900); border:none; font-weight:600;}

/* 三大視窗面板:明確外框 + 海軍藍標題列 */
.panel{background:var(--surface)!important; border:2px solid var(--navy-500)!important;
  border-radius:14px!important; padding:10px 14px 12px!important;
  box-shadow:0 6px 20px rgba(11,31,58,.13); display:flex!important; flex-direction:column!important; min-height:0!important;}
.phead{display:flex; justify-content:space-between; align-items:center; flex:0 0 auto;
  background:linear-gradient(95deg,var(--navy-900),var(--navy-700));
  margin:-10px -14px 10px -14px; padding:7px 14px;
  border-radius:11px 11px 0 0; border-bottom:3px solid var(--lemon);}
.phead .ph-tag{font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:19px; color:var(--lemon)!important;}
.phead .ph-en{color:#9FC0E6; font-size:11px; letter-spacing:2px; margin-left:8px; font-family:'IBM Plex Mono',monospace;}
.phead .ph-sub{color:#CFE0F2; font-size:11.5px;}
/* 上排短下拉(檢視 / 切換分類)— 與按鈕同列、寬度短 */
.shortdd{min-width:104px!important; max-width:150px!important; flex:0 0 auto!important;}
.shortdd .wrap, .shortdd input{font-size:13px!important;}
.shortdd input:disabled, .shortdd input[disabled]{opacity:.5!important; cursor:not-allowed;}
/* 分類 + 模型 同一排;壓掉外框 padding 讓上排更矮、把高度讓給編輯區 */
.modelrow{gap:10px!important; margin-bottom:0!important;}
.modelrow .block{padding:0!important; min-width:0!important;}
/* 閾值 slider 壓縮垂直高度 */
.thr{padding:2px 0!important; margin-top:2px!important;}
.thr .head{margin-bottom:2px!important;}
/* 分頁欄與影像格/表頭上下緊鄰:父層 column 有 16px flex gap,用負 margin 抵銷讓分頁與表頭零間隙 */
.tabstrip{margin-bottom:0!important;}
.countbar{margin:2px 0 4px!important;}
.grid{margin-top:-16px!important;}
.detail{margin-top:-16px!important;}

/* 模型卡 — Gradio 把 elem_classes 同時掛在外層 block 與內層 .prose 上,
   兩者都吃到 max-height+overflow 會出現「雙卷軸」→ 只讓外層 block 捲動,內層放開 */
.card{background:#F8FBFF!important; border:1px solid var(--line); border-radius:12px;
  padding:10px 13px!important; margin-top:4px; font-size:12.5px; max-height:13vh; overflow:auto;}
.card.prose{max-height:none!important; overflow:visible!important; padding:0!important; border:none!important;
  background:transparent!important; margin-top:0!important; border-radius:0!important;}
.card h3{font-family:'Space Grotesk',sans-serif; font-size:15px; margin:.1em 0 .5em; color:var(--navy-900);}
/* 卡片內文用墨色、粗體海軍藍 — Gradio prose 預設偏淡(手機上幾乎看不見),強制提高對比 */
.card.prose, .card.prose p, .card.prose li{color:var(--ink)!important;}
.card.prose strong, .card.prose b{color:var(--navy-700)!important; font-weight:700;}
.card code{background:var(--navy-100); color:var(--navy-700); padding:1px 7px; border-radius:6px; font-size:12px;}
.card small{color:var(--muted);}

/* 顯示面板大圖 */
.preview{border-radius:12px!important; overflow:hidden; background:#fff; border:1px dashed var(--line)!important;
  height:29vh!important; min-height:140px!important;}
.preview img{object-fit:contain!important;}
/* 右上角「放大」(Fullscreen API)時填滿整個視窗 — 對 image-container / image-button / 整個 .preview 三種全螢幕目標都成立 */
:fullscreen{width:100vw!important; height:100vh!important; max-height:100vh!important;
  background:#0B1F3A!important; padding:0!important; border:none!important; border-radius:0!important;}
:fullscreen .image-button, :fullscreen .image-frame{width:100%!important; height:100%!important; max-height:100vh!important;}
:fullscreen img{width:100%!important; height:100%!important; max-width:100vw!important; max-height:100vh!important;
  object-fit:contain!important;}
/* 自製放大 lightbox(取代原生 fullscreen,保證布滿整頁) */
.preview img{cursor:zoom-in;}
.cs-lb{position:fixed!important; inset:0!important; z-index:100000!important; background:rgba(11,31,58,.96)!important;
  display:none!important; align-items:center!important; justify-content:center!important; cursor:zoom-out!important;}
.cs-lb.on{display:flex!important;}
.cs-lb img{max-width:97vw!important; max-height:97vh!important; width:auto!important; height:auto!important;
  object-fit:contain!important; border-radius:8px; box-shadow:0 12px 50px rgba(0,0,0,.65);}
.cs-lb .cs-lb-hint{position:fixed; top:14px; right:18px; color:#cfe0f2; font-size:13px; opacity:.8;}
.pmeta{font-size:13px; color:var(--ink); margin-top:8px; min-height:24px;}
.pmeta strong{color:var(--navy-700);}

/* 工具列 — ICON 緊鄰、全部靠左貼合 */
.toolbar{gap:5px!important; flex-wrap:wrap; align-items:center; margin-bottom:4px;
  justify-content:flex-start!important;}
.toolbar > *{flex:0 0 auto!important; margin:0!important;}  /* 去掉 Gradio auto-margin,才能真正靠左緊鄰 */
.tbtn{min-width:0!important; flex:0 0 auto!important; border-radius:10px!important;
  border:1px solid var(--line)!important; background:#fff!important; color:var(--navy-700)!important;
  font-weight:600!important; box-shadow:none!important; transition:transform .08s,box-shadow .12s;}
.tbtn:hover{transform:translateY(-1px); box-shadow:0 4px 12px rgba(11,31,58,.12); border-color:var(--navy-500)!important;}
.tbtn.primary{background:var(--navy-700)!important; color:var(--lemon)!important; border-color:var(--navy-700)!important;}
.tbtn.primary:hover{background:var(--navy-900)!important;}
.tbtn.accent{background:var(--lemon)!important; color:var(--navy-900)!important; border-color:var(--lemon)!important;}
.tbtn.accent:hover{box-shadow:0 4px 14px rgba(255,205,0,.45);}
.tbtn.navy2{background:var(--navy-500)!important; color:#fff!important; border-color:var(--navy-500)!important;}
.tbtn.navy2:hover{background:var(--navy-700)!important;}
.switch{min-width:200px!important; flex:0 0 auto!important;}

.subbar{gap:10px!important; align-items:center; margin:4px 0;}
.status{font-size:12px; color:var(--muted); min-height:18px;}
.countbar{font-size:12.5px; color:var(--navy-700); background:#F2F7FE; border:1px solid var(--line);
  border-radius:8px; padding:6px 12px!important; margin:6px 0;}

/* 分頁(分類)— 仿頁籤 */
.tabstrip{overflow-x:auto; white-space:nowrap; border-bottom:2px solid var(--line);}
.tabstrip .wrap{flex-wrap:nowrap!important; gap:2px!important;}
.tabstrip label{border:1.5px solid var(--navy-500)!important; background:#fff!important; border-radius:8px 8px 0 0!important;
  padding:7px 14px!important; color:var(--navy-700)!important; font-weight:600!important; cursor:pointer;
  box-shadow:none!important; margin:0 3px 0 0!important;}
.tabstrip label input{display:none!important;}
.tabstrip label:hover{background:var(--navy-100)!important; color:var(--navy-700)!important; border-color:var(--navy-700)!important;}
.tabstrip label:has(input:checked){background:var(--navy-700)!important; color:#fff!important;
  border-color:var(--navy-700)!important; box-shadow:inset 0 -3px 0 var(--lemon)!important;}
/* 「＋」自訂分類分頁:字放大一點 + 內容置中 */
.tabstrip label:has(input[value="__cs_add__"]){font-size:20px!important; font-weight:700!important; line-height:1!important;
  padding:5px 13px!important; display:inline-flex!important; align-items:center!important; justify-content:center!important;}

/* 檢視切換 */
.viewseg .wrap{gap:0!important;}
.viewseg label{border:1px solid var(--line)!important; background:#fff!important; padding:6px 12px!important;
  color:var(--navy-700)!important; font-size:12.5px!important; box-shadow:none!important; margin:0 -1px 0 0!important;}
.viewseg label:has(input:checked){background:var(--navy-700)!important; color:#fff!important; border-color:var(--navy-700)!important;}
.viewseg label input{display:none!important;}

/* 影像格 / 詳細表 — 撐滿編輯區剩餘高度,捲動發生在格內(整頁不出卷軸) */
.grid{background:#fff!important; border:1px solid var(--line)!important; border-radius:12px!important;
  padding:6px!important; flex:1 1 0!important; min-height:0!important; height:auto!important; overflow:auto!important;}
.grid .grid-wrap, .grid .thumbnails{min-height:0!important; height:100%!important;}
.grid .thumbnail-item{border-radius:10px!important; overflow:hidden;}
.grid .thumbnail-item.selected, .grid .selected{outline:3px solid var(--lemon)!important; outline-offset:-1px;}
.grid .caption, .grid .caption-label{font-family:'IBM Plex Mono',monospace!important; font-size:11px!important;
  background:var(--navy-900)!important; color:#fff!important;}
/* 詳細資料 = 自訂 HTML 表;撐滿編輯區剩餘高度,表身內部捲動、表頭 sticky 貼住分頁 */
.detail{flex:1 1 0!important; min-height:0!important; overflow:hidden!important; padding:0!important;
  border:1px solid var(--line)!important; border-radius:12px!important; background:#fff!important;}
.cs-dt-wrap{height:100%; overflow:auto; border-radius:12px;}
.cs-dt{width:100%; border-collapse:collapse; table-layout:fixed; font-size:13px;}
.cs-dt col, .cs-dt th, .cs-dt td{box-sizing:border-box;}
.cs-dt thead th{position:sticky; top:0; z-index:3; background:var(--navy-100); color:var(--navy-900);
  text-align:left; padding:7px 10px; border-bottom:2px solid var(--navy-500); font-weight:700; white-space:nowrap;}
.cs-dt th:nth-child(1), .cs-dt td:nth-child(1){width:30%;}
.cs-dt th:nth-child(2), .cs-dt td:nth-child(2){width:20%;}
.cs-dt th:nth-child(3), .cs-dt td:nth-child(3){width:12%;}
.cs-dt th:nth-child(4), .cs-dt td:nth-child(4){width:15%;}
.cs-dt th:nth-child(5), .cs-dt td:nth-child(5){width:10%;}
.cs-dt th:nth-child(6), .cs-dt td:nth-child(6){width:13%;}
.cs-dt td{padding:3px 10px; border-bottom:1px solid var(--line); color:var(--ink); white-space:nowrap;}
.cs-dt td.cs-name{overflow:hidden; text-overflow:ellipsis; font-family:'IBM Plex Mono',monospace; font-size:12px;}
.cs-row{cursor:pointer;}
.cs-row:hover{background:var(--navy-100);}
.cs-row.cs-sel-row{background:var(--lemon-soft)!important; box-shadow:inset 3px 0 0 var(--lemon);}
.cs-empty{text-align:center; color:var(--muted); padding:18px!important;}
/* 空狀態(尚未載入資料夾)— 置中、撐滿、給明確操作指引,避免一開啟整片空白 */
.cs-empty-state{height:100%; min-height:120px; display:flex; flex-direction:column;
  align-items:center; justify-content:center; text-align:center; gap:8px; padding:20px; box-sizing:border-box;}
.cs-empty-state .cs-es-ic{font-size:46px; line-height:1; opacity:.85;}
.cs-empty-state .cs-es-t{font-family:'Space Grotesk',sans-serif; font-size:19px; font-weight:700; color:var(--navy-700);}
.cs-empty-state .cs-es-s{font-size:13.5px; color:var(--muted); max-width:440px; line-height:1.6;}
.cs-empty-state .cs-es-s b{color:var(--navy-700);}
.cs-cls{width:100%; max-width:100%; font-size:12.5px; padding:2px 4px; cursor:pointer;
  border:1.5px solid var(--navy-500); border-radius:6px; background:#fff; color:var(--navy-700); font-weight:600;}
.cs-cls:hover{border-color:var(--navy-700); background:var(--navy-100);}
/* 隱藏 bridge 文字框 + 觸發鈕完全不佔位 */
#cs-bridge, #cs-trigger, #cs-upstatus, #cs-uppaths, #cs-uppaths-btn, #cs-mode{display:none!important;}
/* 站主本地模式切換:絕對定位到「推論」標題列右側(標題副標讓出右側空間,不被蓋到) */
.cs-infer{position:relative!important;}
.cs-infer .phead .ph-sub{padding-right:132px!important;}
.cs-infer .cs-modechk{position:absolute!important;top:6px;right:16px;z-index:6;
  width:auto!important;min-width:0!important;flex:0 0 auto!important;margin:0!important;}
.cs-infer .cs-modechk label{gap:5px!important;white-space:nowrap;font-size:12px!important;color:var(--lemon)!important;}
/* 模型篩選 icon:絕對定位到「模型」標題列右側(成績圖例讓出右側空間) */
.cs-model{position:relative!important;}
.cs-model .phead .ph-sub{padding-right:82px!important;}
.cs-model .cs-filterbtn{position:absolute!important;top:5px;right:14px;z-index:6;width:auto!important;
  min-width:0!important;flex:0 0 auto!important;margin:0!important;padding:2px 10px!important;
  font-size:12px!important;background:var(--lemon)!important;color:var(--navy-900)!important;
  border:none!important;border-radius:7px!important;font-weight:700!important;}

/* 另存資料夾:檔名格式對話框(全頁遮罩 + 置中卡片) */
.cs-modal-backdrop{position:fixed!important; inset:0!important; z-index:9999!important;
  background:rgba(11,31,58,.45)!important; display:flex!important; align-items:center!important;
  justify-content:center!important; padding:0!important; margin:0!important;}
/* 尊重 Gradio visible=False 的 .hide(否則 display:flex!important 會讓遮罩關不掉) */
.cs-modal-backdrop.hide{display:none!important;}
.cs-modal-box{flex:0 0 auto!important; width:380px!important; max-width:92vw!important;
  background:#fff!important; border:2px solid var(--navy-500)!important; border-radius:14px!important;
  padding:20px 22px!important; gap:14px!important; box-shadow:0 18px 60px rgba(11,31,58,.45)!important;}
.cs-modal-title{font-family:'Space Grotesk',sans-serif; font-size:17px; font-weight:700; color:var(--navy-900);}
.cs-modal-sub{font-size:12.5px; color:var(--muted); margin-top:2px;}
.cs-fmt .wrap{flex-direction:column!important; gap:7px!important; flex-wrap:nowrap!important;}
.cs-fmt label{border:1.5px solid var(--line)!important; background:#fff!important; border-radius:9px!important;
  padding:9px 12px!important; color:var(--navy-700)!important; font-weight:600!important; cursor:pointer;
  display:flex!important; align-items:center; gap:8px; box-shadow:none!important; margin:0!important;}
.cs-fmt label:hover{background:var(--navy-100)!important; border-color:var(--navy-500)!important;}
.cs-fmt label:has(input:checked){background:var(--lemon-soft)!important; border-color:var(--navy-500)!important;
  box-shadow:inset 3px 0 0 var(--lemon)!important;}
.cs-modal-actions{gap:10px!important; justify-content:flex-end!important;}

/* 閾值 slider 檸檬黃 */
.thr input[type=range]{accent-color:var(--lemon);}

.disc-foot{text-align:center; color:var(--muted); font-size:11px; margin-top:4px;
  border-top:1px solid var(--line); padding-top:5px; line-height:1.3;}
.disc-foot b{color:var(--navy-700);}

@media (prefers-reduced-motion: reduce){ .tbtn{transition:none;} }

/* ── 手機 / 窄視窗(≤768px):桌機是「鎖視窗高度、捲動移進面板內」,手機放不下 →
      釋放高度鎖,回到正常整頁捲動;品牌列換行避免徽章壓字;固定 vh 高度全部解開 ── */
@media (max-width:768px){
  /* 解開整頁 100vh + overflow:hidden,改回文件自然捲動 */
  .gradio-container{height:auto!important; overflow:visible!important; padding:6px 10px!important;}
  .gradio-container .main, .gradio-container .main.fillable{height:auto!important; min-height:0!important; overflow:visible!important;}
  .main.fillable > .wrap,
  .main.fillable > .wrap > .contain,
  .main.fillable > .wrap > .contain > .column{height:auto!important; overflow:visible!important; flex:0 0 auto!important;}
  .main.fillable > .wrap > .contain > .column{gap:12px!important;}
  .toprow, .editrow{flex:0 0 auto!important;}
  .editrow > .column, .editrow .panel{height:auto!important; min-height:0!important;}

  /* 品牌列:標題縮小、允許換行,徽章落到下一行不再壓住下方面板;去掉負 margin 免水平溢出 */
  .brandbar{flex-wrap:wrap; gap:8px; padding:12px 16px; margin:0 0 12px 0;}
  .brandbar .bt{font-size:25px; letter-spacing:2px;}
  .brandbar .bsub{font-size:12px;}
  .brandbar .chip{margin:0 6px 0 0;}

  /* 模型卡不再被 13vh 裁切 → 整段資訊可見;預覽改固定高、整頁捲動帶過 */
  .card{max-height:none!important; overflow:visible!important;}
  .preview{height:240px!important; min-height:180px!important;}

  /* 影像格 / 詳細表:不再吃滿剩餘高度(手機沒有「剩餘高度」),改依內容、上限 60vh 內捲 */
  .grid, .detail{flex:0 0 auto!important; height:auto!important; max-height:60vh!important; overflow:auto!important;}
  .cs-dt-wrap{max-height:58vh;}
  .grid{margin-top:-12px!important;}
  .detail{margin-top:-12px!important;}

  /* 工具列:窄螢幕按鈕換行、平均分配寬度 */
  .toolbar{gap:6px!important;}
  .toolbar .tbtn{flex:1 1 auto!important; justify-content:center;}
}
"""


def panel_head(tag, en, sub):
    return (f'<div class="phead"><div><span class="ph-tag">{tag}</span>'
            f'<span class="ph-en">{en}</span></div><span class="ph-sub">{sub}</span></div>')


# 桌面安裝版下載:經本機 server 自己的 /gradio_api/file= 路由提供(同網域、走 tunnel、Access 後可下載),
# 不另外上傳外部。錨定 __file__ 不依賴 CWD;URL-encode 空白;檔案不存在(如尚未打包)則不顯示按鈕。
from urllib.parse import quote as _urlquote
_SETUP_EXE = Path(__file__).resolve().parent / "packaging" / "dist" / "ClinScope-Setup.exe"
SETUP_DL_URL = ("/gradio_api/file=" + _urlquote(_SETUP_EXE.as_posix())) if _SETUP_EXE.exists() else None

_DL_BTN = (f'<a id="cs-dlbtn" class="dlbtn" href="{SETUP_DL_URL}" download="ClinScope-Setup.exe" '
           f'title="下載 Windows 桌面安裝版(約 320 MB;安裝後選模型才下載權重)">⬇ 下載桌面版</a>'
           ) if SETUP_DL_URL else ''
BRAND_HTML = (
    '<div class="brandbar"><div class="brand"><span class="logo">◉</span>'
    '<span class="btext"><span class="bt">ClinScope</span>'
    '<span class="bsub">臨床顯微辨識工作台</span></span></div>'
    f'<div class="bmeta">{_DL_BTN}'
    f'<span class="chip">⚙ {DEVICE.type.upper()}</span>'
    f'<span class="chip lemon">{len(AVAILABLE)} 模型</span></div></div>')

FOOTER_HTML = (
    '<div class="disc-foot">⚠️ 研究／作品集用途,<b>非臨床診斷</b>。'
    '每個模型僅在其資料集分布內準確;跨到不同染色法／相機／倍率的影像,準確率預期下降。</div>')

# ── WebTour 首次進站導覽(spotlight 遮罩;引擎複製自 ~/.claude/skills/web-tour) ──
# 引擎單檔 static/tour.js,去掉結尾兩行 ES export 後內嵌進 Gradio 的 js= hook(比照 ARROW_JS)。
# selector 全用穩定 elem_id(#cs-*);Gradio 節點 Svelte 動態掛載,引擎 waitFor 每步輪詢等待,
# 找不到就 warn+跳步不中斷。只在公開站注入(訪客視角文案),見 build() 的 if PUBLIC 區塊。
_TOUR_ENGINE_SRC = (Path(__file__).resolve().parent / "static" / "tour.js").read_text(encoding="utf-8")
_TOUR_ENGINE_SRC = _TOUR_ENGINE_SRC[: _TOUR_ENGINE_SRC.index("export {")]   # 切掉 ES export 兩行
_TOUR_STEPS = [
    {"title": "歡迎使用 ClinScope",
     "body": "臨床顯微影像辨識工作台,內建 90+ 個模型。你現在是訪客唯讀模式:可瀏覽範例、切換模型看結果,操作類按鈕會虛化。用 ← → 鍵或點「下一步」,Esc 隨時關閉。"},
    {"selector": "#cs-model-panel", "title": "選辨識模型", "placement": "right", "interactive": True,
     "body": "先選科別,再選模型;下方卡片會說明這個模型辨識什麼、準確率多少。"},
    {"selector": "#cs-modelcard", "title": "模型卡", "placement": "bottom",
     "body": "圖示是準確率分級:🟢≥.85 🟡≥.70 🟠≥.50 🔴<.50,內文說明資料來源與適用情境。"},
    {"selector": "#cs-results", "title": "辨識結果", "placement": "top", "interactive": True,
     "body": "每一列是一張圖的分類與信心;上方分頁依類別分組,也可切「圖示」看縮圖牆。"},
    {"selector": "#cs-preview", "title": "點圖看詳情", "placement": "left", "interactive": True,
     "body": "點任一張縮圖,這裡會放大顯示辨識結果 —— 分類機率或偵測標註框。"},
    {"selector": "#cs-upload", "title": "上傳與計算", "placement": "bottom",
     "body": "上傳影像、批次／單張計算、匯出結果都在這一排。這些是站主功能,訪客看得到但不能點。"},
    {"selector": "#cs-identity", "title": "站主登入", "placement": "bottom",
     "body": "你是站主的話,在這裡用 Google 登入即可解鎖上傳、計算、匯出等操作。"},
    {"selector": "#cs-dlbtn", "title": "下載桌面版", "placement": "bottom",
     "body": "想離線、不受訪客限制,可下載 Windows 桌面版,本機執行沒有站主限制。"},
]
# 組成 Gradio js= 用的箭頭函式:引擎源碼 + 步驟 + 啟動(輪詢等 Gradio 框架掛載再播)。
WEBTOUR_JS = (
    "() => {\n"
    "  if (window.__cs_tour) return;\n"          # Gradio 有時重連會重觸 demo.load;去重
    "  window.__cs_tour = true;\n"
    + _TOUR_ENGINE_SRC + "\n"
    "  const STEPS = " + json.dumps(_TOUR_STEPS, ensure_ascii=False) + ";\n"
    # 下載鈕是靜態 HTML(未打包則不存在),boot 當下即可判定 → 不存在就抽掉該步免空等;
    # 其餘動態節點交給引擎 waitFor。
    "  const steps = STEPS.filter(s => s.selector !== '#cs-dlbtn' || document.querySelector('#cs-dlbtn'));\n"
    "  const opts = { key: 'clinscope', version: 1, accent: '#2f6df6' };\n"
    "  let tries = 0;\n"
    "  const boot = () => {\n"
    "    if (!window.WebTour) return;\n"
    "    const framed = document.querySelector('.brandbar') && document.querySelector('#cs-model-panel');\n"
    "    if (!framed && tries++ < 40) { setTimeout(boot, 200); return; }\n"   # 等 Gradio 掛載,最多 8s
    "    window.WebTour.attachHelpButton(steps, opts);\n"
    "    window.WebTour.autoStartOnce(steps, opts);\n"
    "  };\n"
    "  boot();\n"
    "}\n"
)

# 前端互動橋接(詳細資料改成自訂 HTML 表):
#  - 選列(點/方向鍵)→ 經隱藏 textbox #cs-bridge 送 {"a":"sel","i":idx} 給 on_bridge 更新大圖
#  - 分類欄內嵌 <select> change → 送 {"a":"cls","i":idx,"p":類別} 改分類
#  - 圖示檢視維持原本點 thumbnail 的方式
ARROW_JS = """
() => {
  if (window.__cs_keys) return;
  window.__cs_keys = true;
  window.__cs_idx = 0;
  const detailMode = () => { const d = document.querySelector('.detail'); return d && d.offsetParent !== null; };
  const galItems = () => {
    const g = document.querySelector('.grid');
    if (g && g.offsetParent !== null) {
      const t = Array.from(g.querySelectorAll('button.thumbnail-item, .thumbnail-item'));
      if (t.length) return t;
    }
    return null;
  };
  const detailRows = () => Array.from(document.querySelectorAll('.detail .cs-row'));
  const bridgeEl = () => document.querySelector('#cs-bridge textarea, #cs-bridge input');
  const setNative = (el, val) => {
    const proto = el.tagName === 'TEXTAREA' ? window.HTMLTextAreaElement.prototype : window.HTMLInputElement.prototype;
    const d = Object.getOwnPropertyDescriptor(proto, 'value');
    if (d && d.set) d.set.call(el, val); else el.value = val;
    el.dispatchEvent(new Event('input', {bubbles: true}));
  };
  const send = (obj) => {
    const el = bridgeEl(); if (!el) return;
    window.__cs_n = (window.__cs_n || 0) + 1; obj.n = window.__cs_n;
    setNative(el, JSON.stringify(obj));  // 更新 textbox 值(Svelte store)
    const btn = document.querySelector('#cs-trigger button') || document.getElementById('cs-trigger');
    if (btn) btn.click();                // 點隱藏鈕才會真正送後端事件
  };
  const highlight = (row) => {
    document.querySelectorAll('.detail .cs-row.cs-sel-row').forEach((r) => r.classList.remove('cs-sel-row'));
    if (row) row.classList.add('cs-sel-row');
  };
  const selDetail = (row) => {
    if (!row) return;
    const i = parseInt(row.dataset.i, 10);
    window.__cs_idx = i; highlight(row); row.scrollIntoView({block: 'nearest'});
    send({a: 'sel', i: i});
  };
  // 點選:詳細資料列(分類下拉本身除外)→ 選列;圖示 → 記錄索引
  document.addEventListener('click', (e) => {
    // 「＋」分頁:攔截(preventDefault 阻止 radio 選中)→ prompt 取名 → 經 bridge 新增自訂分類
    const tab = e.target.closest ? e.target.closest('.tabstrip label') : null;
    if (tab && tab.textContent.trim() === '＋') {
      e.preventDefault(); e.stopPropagation();
      const name = (window.prompt('輸入新分類名稱（自訂分頁）') || '').trim();
      if (name) send({a: 'addpage', name: name});
      return;
    }
    if (e.target.closest && e.target.closest('.detail .cs-cls')) return;
    const row = e.target.closest ? e.target.closest('.detail .cs-row') : null;
    if (row) { selDetail(row); return; }
    if (!detailMode()) {
      const items = galItems() || [];
      const i = items.findIndex((it) => it.contains(e.target));
      if (i >= 0) window.__cs_idx = i;
    }
  }, true);
  // 原生 <select> 預設「原地再點一下不收回」→ 用 mousedown 旗標自行切換:第二下收回
  document.addEventListener('mousedown', (e) => {
    const s = e.target.closest ? e.target.closest('.detail .cs-cls') : null;
    if (!s) { window.__cs_openSel = null; return; }
    if (window.__cs_openSel === s) { e.preventDefault(); s.blur(); window.__cs_openSel = null; }
    else { window.__cs_openSel = s; }
  }, true);
  document.addEventListener('blur', (e) => {
    if (e.target && e.target.closest && e.target.closest('.detail .cs-cls')) window.__cs_openSel = null;
  }, true);
  // 分類欄內嵌下拉改值
  document.addEventListener('change', (e) => {
    const sel = e.target.closest ? e.target.closest('.detail .cs-cls') : null;
    if (sel) { window.__cs_openSel = null; const i = parseInt(sel.dataset.i, 10); window.__cs_idx = i; send({a: 'cls', i: i, p: sel.value}); }
  });
  // 方向鍵切換檔案(聚焦在輸入框/下拉時不攔截)
  document.addEventListener('keydown', (e) => {
    if (!['ArrowRight', 'ArrowLeft', 'ArrowUp', 'ArrowDown'].includes(e.key)) return;
    const t = (document.activeElement || {}).tagName || '';
    if (t === 'INPUT' || t === 'TEXTAREA' || t === 'SELECT') return;
    const fwd = (e.key === 'ArrowRight' || e.key === 'ArrowDown');
    if (detailMode()) {
      const rows = detailRows();
      if (!rows.length) return;
      const i = Math.max(0, Math.min(rows.length - 1, (window.__cs_idx || 0) + (fwd ? 1 : -1)));
      selDetail(rows[i]);
      e.preventDefault();
      return;
    }
    const items = galItems();
    if (!items || !items.length) return;
    const i = Math.max(0, Math.min(items.length - 1, (window.__cs_idx || 0) + (fwd ? 1 : -1)));
    window.__cs_idx = i;
    items[i].click();
    items[i].scrollIntoView({block: 'nearest'});
    e.preventDefault();
  });
  // ===== 圖片放大 lightbox(取代原生 fullscreen,確保布滿整頁) =====
  const lbEl = () => {
    let el = document.querySelector('.cs-lb');
    if (!el) {
      el = document.createElement('div');
      el.className = 'cs-lb';
      el.innerHTML = '<span class="cs-lb-hint">點任意處 / ESC 關閉</span><img alt="">';
      document.body.appendChild(el);
      el.addEventListener('click', () => el.classList.remove('on'));
    }
    return el;
  };
  window.csOpenLB = (src) => { if (!src) return; const el = lbEl(); el.querySelector('img').src = src; el.classList.add('on'); };
  // 攔截「點圖本身」與原生「右上角放大鈕」(aria-label Fullscreen),一律改開自製 lightbox
  // —— 在 capture 階段攔住,不讓 Gradio 的 requestFullscreen 跑(它在某些瀏覽器會失敗→只半放大卡在上方)
  document.addEventListener('click', (e) => {
    if (!e.target.closest) return;
    const btn = e.target.closest('button, a');
    if (btn && btn.closest('.preview') && /ullscreen/i.test(btn.getAttribute('aria-label') || btn.getAttribute('title') || '')) {
      e.preventDefault(); e.stopPropagation();
      const img = document.querySelector('.preview img');
      if (img && img.src) window.csOpenLB(img.src);
      return;
    }
    const img = e.target.closest('.preview img');
    if (img && img.src) { e.preventDefault(); e.stopPropagation(); window.csOpenLB(img.src); }
  }, true);
  // 後備:萬一原生全螢幕真的進去了,也退出改用 lightbox
  document.addEventListener('fullscreenchange', () => {
    const fe = document.fullscreenElement;
    if (fe && fe.closest && fe.closest('.preview')) {
      const img = fe.querySelector('img') || document.querySelector('.preview img');
      const src = img && img.src;
      if (document.exitFullscreen) document.exitFullscreen().catch(() => {});
      if (src) window.csOpenLB(src);
    }
  });
  // ESC 關閉
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') { const el = document.querySelector('.cs-lb.on'); if (el) el.classList.remove('on'); }
  });
  // ===== 手機 / 窄視窗:解除整頁 100vh 高度鎖,回到自然整頁捲動 =====
  // 桌機鎖在 root container(height:100vh; overflow:hidden),捲動移進面板內;窄視窗放不下。
  // Gradio 6 會把使用者 CSS 選擇器加上 `.contain ` 前綴 → @media 改不到 root container/main/wrap/contain
  // (它們是 .contain 的祖先),只能用 JS 直接設 inline style;setProperty 帶 'important' 才蓋得過 stylesheet !important。
  const csResponsive = () => {
    const narrow = window.matchMedia('(max-width:768px)').matches;
    const chain = ['.gradio-container', '.main', '.wrap', '.contain', '.contain > .column'];
    const props = ['height', 'max-height', 'min-height', 'overflow'];
    chain.forEach((sel) => document.querySelectorAll(sel).forEach((el) => {
      if (narrow) {
        el.style.setProperty('height', 'auto', 'important');
        el.style.setProperty('max-height', 'none', 'important');
        el.style.setProperty('min-height', '0', 'important');
        el.style.setProperty('overflow', 'visible', 'important');
      } else {
        props.forEach((p) => el.style.removeProperty(p));  // 還原桌機鎖(交回 stylesheet)
      }
    }));
    [document.documentElement, document.body].forEach((el) => {
      if (narrow) { el.style.setProperty('height', 'auto', 'important'); el.style.setProperty('overflow-y', 'auto', 'important'); }
      else { el.style.removeProperty('height'); el.style.removeProperty('overflow-y'); }
    });
  };
  csResponsive();
  window.addEventListener('resize', () => { clearTimeout(window.__cs_rsz); window.__cs_rsz = setTimeout(csResponsive, 150); });
}
"""


# 公開站「自動分批上傳」:純前端把整夾影像切成多個 < BATCH_MB 的批次,逐批以 XHR 上傳到
# Gradio 暫存夾(每批各自是一個 < ~100MB 的請求 → 繞過 Cloudflare 單次上限),XHR upload.onprogress
# 給「真實百分比」進度條;全部上傳完把伺服器路徑 JSON 寫進 #cs-uppaths、點 #cs-uppaths-btn 交後端;
# 後端寫回 #cs-upstatus 的 token 變化 → 收起。任何批次失敗/中斷都轉成可關閉的提示,絕不靜默卡死。
BATCH_MB = 90    # 每批上限(< Cloudflare 單次 ~100MB,留邊際)
BATCH_N = 300    # 每批檔數上限(避免單一 multipart 過大/過慢)
UPLOAD_JS = ("""
() => {
  if (window.__cs_up_init) return;
  window.__cs_up_init = true;
  const BATCH_MB = __BATCH_MB__, BATCH_N = __BATCH_N__;
  const IMG_RE = /\\.(jpe?g|png|bmp|tiff?|webp|gif)$/i;
  const st = document.createElement('style');
  st.textContent = `
    #cs-uploading{position:fixed;inset:0;z-index:99999;display:none;align-items:center;justify-content:center;background:rgba(11,31,58,.5);backdrop-filter:blur(2px)}
    #cs-uploading.on{display:flex}
    #cs-uploading .cs-up-card{background:#fff;border:2px solid #163C6B;border-radius:14px;padding:22px 28px;min-width:320px;max-width:86vw;text-align:center;box-shadow:0 12px 44px rgba(0,0,0,.32)}
    #cs-uploading.err .cs-up-card{border-color:#C0392B}
    #cs-uploading .cs-up-t{font-size:15px;font-weight:700;color:#0B1F3A;margin-bottom:14px}
    #cs-uploading .cs-up-track{height:10px;border-radius:6px;background:#E5EDF7;overflow:hidden;position:relative}
    #cs-uploading.err .cs-up-track{display:none}
    #cs-uploading .cs-up-bar{position:absolute;left:-42%;width:42%;height:100%;border-radius:6px;background:linear-gradient(90deg,#22568F,#FFE14D);animation:cs-up-slide 1.1s ease-in-out infinite}
    #cs-uploading.det .cs-up-bar{left:0;animation:none;transition:width .2s ease}
    @keyframes cs-up-slide{0%{left:-42%}100%{left:100%}}
    #cs-uploading .cs-up-sub{font-size:12px;color:#5B708A;margin-top:12px;line-height:1.55}
    #cs-uploading .cs-up-x{display:none;margin-top:14px;font-size:12px;font-weight:700;color:#163C6B;cursor:pointer;text-decoration:underline}
    #cs-uploading.err .cs-up-x{display:inline-block}
  `;
  document.head.appendChild(st);
  const ov = document.createElement('div');
  ov.id = 'cs-uploading';
  ov.innerHTML = '<div class="cs-up-card"><div class="cs-up-t" id="cs-up-t">⏳ 上傳影像中…</div>'
    + '<div class="cs-up-track"><div class="cs-up-bar" id="cs-up-bar"></div></div>'
    + '<div class="cs-up-sub" id="cs-up-sub">準備上傳…</div>'
    + '<div class="cs-up-x" id="cs-up-x">關閉</div></div>';
  document.body.appendChild(ov);
  const titleEl = () => document.getElementById('cs-up-t');
  const barEl = () => document.getElementById('cs-up-bar');
  const sub = () => document.getElementById('cs-up-sub');
  const statusEl = () => document.querySelector('#cs-upstatus textarea, #cs-upstatus input');
  let base = '', poll = null, busy = false;
  const setNative = (el, val) => {
    const proto = el.tagName === 'TEXTAREA' ? window.HTMLTextAreaElement.prototype : window.HTMLInputElement.prototype;
    const d = Object.getOwnPropertyDescriptor(proto, 'value');
    if (d && d.set) d.set.call(el, val); else el.value = val;
    el.dispatchEvent(new Event('input', { bubbles: true }));
  };
  const hide = () => { ov.classList.remove('on', 'err', 'det'); clearInterval(poll); poll = null; busy = false; };
  document.getElementById('cs-up-x').addEventListener('click', hide);
  const fail = (msg) => {            // 失敗/中斷 → 可關閉提示,絕不靜默卡住
    clearInterval(poll); poll = null; busy = false;
    ov.classList.add('on', 'err'); ov.classList.remove('det');
    if (titleEl()) titleEl().textContent = '⚠️ 上傳未完成';
    if (sub()) sub().innerHTML = msg;
  };
  const showPrep = (n, mb) => {      // 選好夾:先出遮罩(不定量動畫),消除靜默空窗
    ov.classList.remove('err', 'det');
    if (titleEl()) titleEl().textContent = '⏳ 上傳影像中…';
    if (sub()) sub().textContent = '準備上傳 ' + n + ' 張影像（約 ' + mb.toFixed(0) + ' MB）…';
    ov.classList.add('on');
  };
  const setProgress = (frac, bi, nb) => {   // 真實百分比
    ov.classList.add('det');
    const pct = Math.max(1, Math.min(99, Math.round(frac * 100)));
    if (barEl()) barEl().style.width = pct + '%';
    if (sub()) sub().textContent = '上傳中 ' + pct + '%（第 ' + bi + '/' + nb + ' 批）…雲端上行可能較慢，請稍候';
  };
  // 單批:XHR multipart POST 到 Gradio 上傳端點(相對路徑 → 經 tunnel + Access + 帶 cookie)
  const xhrUpload = (files, onprog) => new Promise((resolve, reject) => {
    const fd = new FormData();
    for (const f of files) fd.append('files', f, f.webkitRelativePath || f.name);
    const x = new XMLHttpRequest();
    x.open('POST', '/gradio_api/upload');
    x.upload.onprogress = (e) => { if (e.lengthComputable) onprog(e.loaded); };
    x.onload = () => {
      if (x.status >= 200 && x.status < 300) { try { resolve(JSON.parse(x.responseText)); } catch (_) { resolve([]); } }
      else reject({ status: x.status });
    };
    x.onerror = () => reject({ status: 0 });
    x.send(fd);
  });
  const startUpload = async (allFiles) => {
    if (busy) return; busy = true;
    const imgs = allFiles.filter((f) => IMG_RE.test(f.name));
    const total = imgs.reduce((s, f) => s + (f.size || 0), 0);
    showPrep(imgs.length, total / 1048576);
    if (!imgs.length) { fail('這個資料夾裡沒有支援的影像檔（jpg/png/tif/bmp/webp/gif）。'); return; }
    // 貪婪打包:每批 < BATCH_MB 且 < BATCH_N 檔;單檔超大就自成一批
    const MAXB = BATCH_MB * 1048576, batches = []; let cur = [], cb = 0;
    for (const f of imgs) {
      if (cur.length && (cb + (f.size || 0) > MAXB || cur.length >= BATCH_N)) { batches.push(cur); cur = []; cb = 0; }
      cur.push(f); cb += (f.size || 0);
    }
    if (cur.length) batches.push(cur);
    // token 輪詢:後端把路徑收下、寫回 #cs-upstatus token 變化 → 完成(依失敗數決定收起或提示)
    const el = statusEl(); base = el ? el.value : '';
    clearInterval(poll);
    poll = setInterval(() => {
      const e2 = statusEl();
      if (e2 && e2.value && e2.value !== base) {
        const nf = window.__cs_up_failed || 0;
        if (nf > 0) { ov.classList.add('err'); ov.classList.remove('det');
          if (titleEl()) titleEl().textContent = '⚠️ 部分批次未上傳';
          if (sub()) sub().innerHTML = '已載入成功上傳的影像，但有 <b>' + nf + '</b> 批失敗（可能太大或中斷）。可稍後重試遺漏的部分。';
          clearInterval(poll); poll = null; busy = false;
        } else hide();
      }
    }, 300);
    let sent = 0, failed = 0; const paths = [];
    for (let i = 0; i < batches.length; i++) {
      const bBytes = batches[i].reduce((s, f) => s + (f.size || 0), 0);
      try {
        const res = await xhrUpload(batches[i], (loaded) => setProgress((sent + loaded) / (total || 1), i + 1, batches.length));
        if (Array.isArray(res)) for (const p of res) paths.push(p);
      } catch (err) { failed++; }
      sent += bBytes; setProgress(sent / (total || 1), i + 1, batches.length);
    }
    if (!paths.length) {              // 全數失敗
      fail('上傳失敗' + (failed ? '（' + failed + ' 批皆未成功）' : '') + '。可能檔案太大或網路中斷，請減少張數後再試。');
      return;
    }
    // 交後端:路徑 JSON → textbox → 點隱藏鈕(其 .click 帶當下 textbox 值送後端)
    window.__cs_up_failed = failed;
    if (sub()) sub().textContent = '伺服器處理中…';
    const tb = document.querySelector('#cs-uppaths textarea, #cs-uppaths input');
    const btn = document.querySelector('#cs-uppaths-btn button') || document.getElementById('cs-uppaths-btn');
    if (tb && btn) { setNative(tb, JSON.stringify(paths)); btn.click(); }
    else fail('內部錯誤：找不到上傳橋接元件。');
  };
  // 自建 directory input(不靠 Gradio 內建上傳,自己控制分批)
  const inp = document.createElement('input');
  inp.type = 'file'; inp.multiple = true;
  try { inp.webkitdirectory = true; } catch (_) {}
  inp.setAttribute('webkitdirectory', ''); inp.style.display = 'none';
  document.body.appendChild(inp);
  inp.addEventListener('change', () => { const fs = Array.from(inp.files || []); inp.value = ''; if (fs.length) startUpload(fs); });
  // 「📁 上傳影像」鈕 → 開瀏覽器資料夾選擇器並分批上傳。
  // 但站主切到「本地模式」時(#cs-mode=local)不攔截 → 交給 Gradio 後端 on_pick_folder_pub 走伺服器端原生選夾。
  document.addEventListener('click', (e) => {
    if (e.target.closest && e.target.closest('#cs-upload')) {
      const m = document.querySelector('#cs-mode textarea, #cs-mode input');
      if (m && m.value === 'local') return;   // 本地模式:放行給後端原生選夾
      e.preventDefault(); e.stopPropagation(); inp.click();
    }
  }, true);
}
""").replace("__BATCH_MB__", str(BATCH_MB)).replace("__BATCH_N__", str(BATCH_N))


# 站內 Google 登入前端:載入 GSI SDK → 匿名時把官方 Google 按鈕 render 進身分列的 #cs-gbtn;
# 拿到 credential(ID token)POST /auth/google 成功即 reload(讓 on_identity 依 cookie 重判身分);
# 「登出」鈕 POST /auth/logout 後 reload。僅 AUTH_ENABLED(有 client id)時掛。
AUTH_JS = ("""
() => {
  if (window.__cs_auth_init) return;
  window.__cs_auth_init = true;
  const CID = "__GOOGLE_CLIENT_ID__";
  const post = (url) => fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' } });
  // 登出(事件委派,身分列可能重繪)
  document.addEventListener('click', (e) => {
    if (e.target && e.target.id === 'cs-logout') {
      e.preventDefault();
      post('/auth/logout').finally(() => location.reload());
    }
  }, true);
  const onCred = (resp) => {
    if (!resp || !resp.credential) return;
    fetch('/auth/google', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ credential: resp.credential })
    }).then(r => r.json().then(j => ({ ok: r.ok, j }))).then(({ ok, j }) => {
      if (ok) location.reload();
      else alert('登入失敗：' + ((j && j.error) || '請確認使用站主 Google 帳號'));
    }).catch(() => alert('登入失敗（網路問題），請再試一次。'));
  };
  let inited = false;
  const initGsi = () => {
    if (inited || !(window.google && google.accounts && google.accounts.id)) return;
    inited = true;
    google.accounts.id.initialize({ client_id: CID, callback: onCred, auto_select: false });
  };
  // render 官方 Google 按鈕進 #cs-gbtn(身分列由 on_identity 非同步填入 → 輪詢等它出現)
  let rendered = null;
  const tryRender = () => {
    initGsi();
    const el = document.getElementById('cs-gbtn');
    if (el && inited && el !== rendered && !el.childElementCount) {
      try { google.accounts.id.renderButton(el, { type: 'standard', theme: 'outline', size: 'medium', text: 'signin', shape: 'pill' }); rendered = el; } catch (_) {}
    }
  };
  // 載入 GSI SDK
  const s = document.createElement('script');
  s.src = 'https://accounts.google.com/gsi/client'; s.async = true; s.defer = true;
  s.onload = () => { initGsi(); tryRender(); };
  document.head.appendChild(s);
  setInterval(tryRender, 700);   // 身分列重繪 / SDK 慢載都能補上按鈕
}
""").replace("__GOOGLE_CLIENT_ID__", GOOGLE_CLIENT_ID)


def build():
    with gr.Blocks(title="ClinScope — 臨床顯微辨識工作台") as demo:
        gr.HTML(BRAND_HTML, elem_classes="brandwrap")
        identity_html = gr.HTML("", elem_id="cs-identity")   # 公開站身分列(由 on_identity 於載入時填)
        if not AVAILABLE:
            gr.Markdown("⚠️ 找不到任何模型權重,請先完成至少一條 track 的訓練。")
            return demo
        _init_discs = visible_disciplines() or DISCIPLINES
        # 公開站預設停在範例模型(OpenFungi),配合預先分類的範例畫面;否則第一個可見科別
        _sample_spec = next((m for m in AVAILABLE
                             if m["key"] == SAMPLE_MODEL_KEY and _visible_model(m["key"])), None)
        if PUBLIC and _sample_spec:
            init_disc = disc_of(_sample_spec["key"])
            init_label = _sample_spec["label"]
        else:
            init_disc = _init_discs[0]
            init_label = (labels_for(init_disc) or
                          [m["label"] for m in AVAILABLE if disc_of(m["key"]) == init_disc])[0]
        S = gr.State(new_state(init_label))

        # --- 上排:模型(左) + 顯示(右) ---
        with gr.Row(equal_height=False, elem_classes="toprow"):
            with gr.Column(scale=5, elem_id="cs-model-panel", elem_classes="panel cs-model"):
                gr.HTML(panel_head("模型", "MODEL", "🟢≥.85 🟡≥.70 🟠≥.50 🔴<.50 成績分級"))
                if PUBLIC:   # 站主:模型篩選 icon(絕對定位到「模型」標題列右側,預設隱藏)
                    btn_filter = gr.Button("⚙︎ 篩選", visible=False, elem_classes="cs-filterbtn", scale=0)
                with gr.Row(elem_classes="modelrow"):
                    disc_dd = gr.Dropdown(_init_discs, value=init_disc, label="分類（科別）", scale=2)
                    model_dd = gr.Dropdown(model_choices(init_disc), value=init_label, label="模型", scale=5)
                card_md = gr.Markdown(card_text(init_label), elem_id="cs-modelcard", elem_classes="card")
                thr = gr.Slider(0.0, 1.0, value=0.5, step=0.01, label="信心門檻（分類）",
                                info="Top-1 機率低於此值 → 歸入 Null 待確認", elem_classes="thr")
            with gr.Column(scale=6, elem_classes="panel"):
                gr.HTML(panel_head("顯示", "DISPLAY", ""))
                preview_img = gr.Image(show_label=False, height=520, elem_id="cs-preview", elem_classes="preview")
                preview_meta = gr.Markdown("", visible=False, elem_classes="pmeta")

        # --- 下排:推論(滿版) ---
        with gr.Row(elem_classes="editrow"):
            with gr.Column(elem_classes="panel cs-infer"):
                gr.HTML(panel_head("推論", "INFERENCE", "批次／單張計算 · 分類分頁 · 匯出"))
                if PUBLIC:   # 站主本地模式切換:絕對定位到「推論」標題列右側(預設隱藏,on_identity 對站主開放)
                    mode_chk = gr.Checkbox(label="🖥️ 本地模式", value=False, visible=False,
                                           container=False, elem_classes="cs-modechk")
                # 上排 ICON 列:按鈕 + 兩個短下拉(檢視 / 切換分類,各自前綴 👁 / ⇄ 自我標示)
                with gr.Row(elem_classes="toolbar"):
                    if PUBLIC:   # 公開站:純 Button + 自建 directory input,前端分批上傳(繞過 CF 100MB)
                        btn_folder = gr.Button("📁 上傳影像", elem_id="cs-upload", elem_classes="tbtn", scale=0)
                    else:        # 本機桌面/dev:client 即 server,原生選夾直接可用
                        btn_folder = gr.Button("📁 選擇資料夾", elem_classes="tbtn", scale=0)
                    view_dd = gr.Dropdown(["👁 詳細資料", "👁 圖示"], value="👁 詳細資料",
                                          show_label=False, container=False, elem_classes="shortdd")
                    btn_batch = gr.Button("▶ 批次計算", elem_classes="tbtn primary", scale=0)
                    btn_single = gr.Button("◉ 單張計算", elem_classes="tbtn navy2", scale=0)
                    switch_dd = gr.Dropdown([SW_PLACEHOLDER], value=SW_PLACEHOLDER, show_label=False,
                                            container=False, interactive=False, elem_classes="shortdd")
                    btn_apply = gr.Button("⚖️ 套用門檻", elem_classes="tbtn", scale=0)
                    btn_csv = gr.Button("⤓ 匯出 CSV", elem_classes="tbtn", scale=0)
                    btn_save = gr.Button("💾 打包下載 ZIP" if PUBLIC else "💾 另存資料夾",
                                         elem_classes="tbtn", scale=0)
                count_md = gr.Markdown("", visible=False, elem_classes="countbar")
                page_radio = gr.Radio([TOTAL], value=TOTAL, show_label=False,
                                      container=False, elem_classes="tabstrip")
                gallery = gr.Gallery(show_label=False, columns=8, object_fit="cover",
                                     allow_preview=False, elem_id="cs-gallery", elem_classes="grid")
                # 詳細資料 = 自訂 HTML 表(分類欄內嵌 <select> 下拉);互動經 bridge 回傳
                # 注意:建立時 visible=True 才會掛載,初始可見性交給首次 render 決定(visible=False 建立者不掛載)
                detail_html = gr.HTML("", elem_id="cs-results", elem_classes="detail")
                # bridge:前端 JS 寫入指令的隱藏 textbox + 觸發鈕。Gradio 不認「合成 input 事件」,
                # 故改用「JS 設 textbox 值 → 點隱藏按鈕」觸發後端(按鈕的合成 click 才會送事件)。兩者都用 CSS 藏。
                bridge = gr.Textbox("", elem_id="cs-bridge", show_label=False)
                bridge_btn = gr.Button("bridge", elem_id="cs-trigger")
                # 上傳完成信號(公開站):handler 每次寫入單調 token,前端進度層偵測到變化即收起。CSS 藏。
                up_status = gr.Textbox("", elem_id="cs-upstatus", show_label=False)
                # 分批上傳橋接(公開站):前端把已上傳的伺服器路徑 JSON 寫入 textbox → 點隱藏鈕觸發後端。CSS 藏。
                up_paths = gr.Textbox("", elem_id="cs-uppaths", show_label=False)
                up_paths_btn = gr.Button("uppaths", elem_id="cs-uppaths-btn")
                # 模式旗標(公開/本地):on_mode_change 寫入,UPLOAD_JS 讀它決定是否攔截資料夾鈕。CSS 藏。
                mode_flag = gr.Textbox("public", elem_id="cs-mode", show_label=False)
                csv_out = gr.File(label="下載 CSV", visible=False)
                zip_out = gr.File(label="下載 ZIP", visible=False)   # 公開站打包下載用

        # 另存/打包:檔名格式對話框(覆蓋全頁、置中);按鈕先跳此框,確認後桌面版開資料夾選擇框、公開站打包 zip
        with gr.Column(visible=False, elem_classes="cs-modal-backdrop") as save_modal:
            with gr.Column(elem_classes="cs-modal-box"):
                _init_modal_title = (_save_modal_title(False) if PUBLIC else
                    '<div class="cs-modal-title">📁 另存資料夾 — 檔名格式</div>'
                    '<div class="cs-modal-sub">選擇複製後的檔名命名方式（分類取自目前分頁／各檔指派）</div>')
                save_modal_title = gr.HTML(_init_modal_title)
                save_fmt = gr.Radio([("分類_檔名", "c_n"), ("檔名_分類", "n_c"), ("檔名 (分類)", "n_pc")],
                                    value="c_n", show_label=False, elem_classes="cs-fmt")
                with gr.Row(elem_classes="cs-modal-actions"):
                    save_cancel = gr.Button("取消", elem_classes="tbtn", scale=0)
                    save_confirm = gr.Button("確認另存", elem_classes="tbtn accent", scale=0)

        if PUBLIC:   # 站主模型篩選對話框(分數門檻 slider + 手動隱藏,變更套用到所有訪客)
            with gr.Column(visible=False, elem_classes="cs-modal-backdrop") as filter_modal:
                with gr.Column(elem_classes="cs-modal-box"):
                    gr.HTML('<div class="cs-modal-title">🔎 模型篩選（站主）</div>'
                            '<div class="cs-modal-sub">設定顯示門檻與手動隱藏;變更會即時套用到所有訪客</div>')
                    filter_score = gr.Slider(0, 100, value=int((MODEL_FILTER.get("min_score") or 0) * 100),
                                             step=1, label="分數門檻（%）",
                                             info="只顯示成績 ≥ 此門檻的模型;0 = 全部顯示（預設 85）")
                    filter_hide = gr.Dropdown([(m["label"], m["key"]) for m in AVAILABLE],
                                              value=MODEL_FILTER.get("hidden", []), multiselect=True,
                                              label="手動隱藏的模型（可多選）", filterable=True)
                    with gr.Row(elem_classes="cs-modal-actions"):
                        filter_cancel = gr.Button("取消", elem_classes="tbtn", scale=0)
                        filter_save = gr.Button("儲存篩選", elem_classes="tbtn accent", scale=0)

        gr.HTML(FOOTER_HTML)

        MODEL_OUT = [S, card_md, thr, page_radio, gallery, detail_html, count_md,
                     switch_dd, preview_img, preview_meta]
        FOLDER_OUT = [S, page_radio, gallery, detail_html, count_md, preview_img, preview_meta]
        COMPUTE_OUT = [S, page_radio, gallery, detail_html, count_md, preview_img, preview_meta]
        PANEL_OUT = [S, page_radio, gallery, detail_html, count_md]
        SELECT_OUT = [S, switch_dd, preview_img, preview_meta]
        BRIDGE_OUT = [S, detail_html, gallery, page_radio, count_md, switch_dd, preview_img, preview_meta]

        disc_dd.change(on_disc, [disc_dd], [model_dd]).then(on_model, [model_dd, S], MODEL_OUT)
        model_dd.change(on_model, [model_dd, S], MODEL_OUT)
        if PUBLIC:
            # 上傳完成後由隱藏鈕把伺服器路徑送進後端;資料夾鈕本身只在「站主+本地模式」時走原生選夾
            up_paths_btn.click(on_uploaded_paths, [up_paths, S], FOLDER_OUT + [up_status])
            btn_folder.click(on_pick_folder_pub, [S], FOLDER_OUT)
            mode_chk.change(on_mode_change, [mode_chk, S],
                            [S, btn_folder, btn_save, mode_flag, save_modal_title])
        else:
            btn_folder.click(on_pick_folder, [S], FOLDER_OUT)
        view_dd.change(on_view, [view_dd, S], PANEL_OUT)
        switch_dd.input(on_switch, [switch_dd, S], PANEL_OUT)
        page_radio.change(on_page, [page_radio, S], COMPUTE_OUT)
        btn_batch.click(on_batch, [model_dd, thr, S], COMPUTE_OUT)
        btn_single.click(on_single, [model_dd, thr, S], COMPUTE_OUT)
        btn_apply.click(on_apply_thr, [thr, S], PANEL_OUT)
        btn_csv.click(on_csv, [model_dd, S], [csv_out])
        btn_save.click(on_saveas, [S], [save_modal])                       # 跳出檔名格式對話框
        save_cancel.click(lambda: gr.update(visible=False), None, [save_modal])
        if PUBLIC:   # 公開站:本地模式(站主)→ 原生另存;否則 → 打包下載 ZIP
            save_confirm.click(on_saveas_confirm_dispatch, [save_fmt, S], [save_modal, zip_out])
        else:        # 桌面版:確認 → 開資料夾選擇框並複製到本機
            save_confirm.click(on_saveas_confirm, [save_fmt, S], [save_modal])
        if PUBLIC:   # 站主模型篩選:開/存(存後重建科別+模型下拉,並刷新模型卡)
            btn_filter.click(on_open_filter, None, [filter_modal])
            filter_cancel.click(lambda: gr.update(visible=False), None, [filter_modal])
            filter_save.click(on_save_filter, [filter_score, filter_hide, S],
                              [filter_modal, disc_dd, model_dd]).then(on_model, [model_dd, S], MODEL_OUT)
        gallery.select(on_select, [S, model_dd, thr], SELECT_OUT)
        bridge_btn.click(on_bridge, [bridge, S, model_dd, thr], BRIDGE_OUT)  # 詳細資料 HTML 表的選列/改分類
        _ev = demo.load(on_model, [model_dd, S], MODEL_OUT)
        if PUBLIC:   # 預設畫面載入預先分類好的範例;.then 確保在 on_model 重設「之後」才跑(避免 load 事件競態把結果洗掉)
            _ev.then(on_load_sample, [S], FOLDER_OUT)
        demo.load(None, None, None, js=ARROW_JS)  # 方向鍵切換檔案(圖示 / 詳細資料 皆可)
        if PUBLIC:
            demo.load(None, None, None, js=UPLOAD_JS)  # 公開站:上傳進度覆蓋層
            demo.load(None, None, None, js=WEBTOUR_JS)  # 首次進站導覽(訪客 spotlight + 右下 ? 重播鈕)
            # 身分判定:站主 → 按鈕可用;訪客 → 虛化停用(唯讀)。後端另有 _guard_owner 縱深防禦。
            demo.load(on_identity, None,
                      [identity_html, btn_folder, btn_batch, btn_single, btn_apply, btn_csv, btn_save,
                       mode_chk, btn_filter])
            if AUTH_ENABLED:
                demo.load(None, None, None, js=AUTH_JS)  # 站內 Google 登入(登入/登出鈕 + GSI)
    return demo


def _unsafe_to_serve(p) -> bool:
    """公開站不得對外提供的路徑:repo 根(含 models/、data/、reports/、packaging/autostart/env.local.bat…)
    與使用者家目錄(含 .cloudflared 憑證、桌面/下載裡的影像)—— 本身、其內任何位置、或涵蓋它們的上層(如 D:\\、C:\\)。"""
    try:
        p = Path(p).resolve()
    except Exception:
        return True
    for s in (ROOT, Path.home()):
        s = s.resolve()
        if p == s or p.is_relative_to(s) or s.is_relative_to(p):
            return True
    return False


def _allowed_paths():
    """Gradio allowed_paths —— 清單內任何檔案都能被**任何人**經 /gradio_api/file=<絕對路徑> 直接下載。
    桌面/dev(非 PUBLIC,只綁 127.0.0.1):沿用預設開放各磁碟根,或 CLINSCOPE_ALLOWED_PATHS(分號分隔)。
    公開站 fail-closed:預設只開桌面安裝檔;CLINSCOPE_ALLOWED_PATHS 中涵蓋/位於 repo 根或家目錄的項目一律剔除。
    (工作台影像不靠 allowed_paths:圖示檢視由 _servable 複製進 Gradio 快取;匯出見 _export_dir。)"""
    env_paths = os.environ.get("CLINSCOPE_ALLOWED_PATHS", "").strip()
    if env_paths:
        allowed = [p.strip() for p in env_paths.split(";")
                   if p.strip() and Path(p.strip()).exists()]
    elif PUBLIC:
        allowed = []
    else:
        allowed = [f"{d}:\\" for d in "CDEFGHIJKLMNOPQRSTUVWXYZ" if Path(f"{d}:\\").exists()]
    if PUBLIC:
        dropped = [p for p in allowed if _unsafe_to_serve(p)]
        if dropped:
            print(f"[ClinScope] ⚠ 公開站忽略會曝露 repo/家目錄的 CLINSCOPE_ALLOWED_PATHS 項目: {dropped}")
        allowed = [p for p in allowed if not _unsafe_to_serve(p)]
        if _SETUP_EXE.exists():          # 下載桌面版按鈕(BRAND_HTML)只開這一個檔
            allowed.append(str(_SETUP_EXE))
    return allowed


def launch_app(prevent_thread_lock=False, inbrowser=False, server_port=None, **overrides):
    """建構並啟動工作台。供 `python app.py`(__main__)與 packaging/run_desktop.py
    (pywebview 桌面視窗,需 prevent_thread_lock=True)共用,避免啟動參數兩處重複。"""
    allowed = _allowed_paths()
    # auth:預設無(本機單人直用);公開站設 CLINSCOPE_AUTH_USER + CLINSCOPE_AUTH_PASS
    # 即啟用 Gradio 內建「帳號+密碼」登入(密碼只走環境變數,不寫進原始碼/版控)。
    _u, _p = os.environ.get("CLINSCOPE_AUTH_USER"), os.environ.get("CLINSCOPE_AUTH_PASS")
    auth = (_u, _p) if (_u and _p) else None
    port = int(server_port or os.environ.get("GRADIO_SERVER_PORT", 7860) or 7860)
    print(f"[ClinScope] device={DEVICE.type} | {len(AVAILABLE)} models | "
          f"disciplines={DISCIPLINES} | allowed_paths={allowed} | auth={'on' if auth else 'off'} | "
          f"mode={'PUBLIC(upload)' if PUBLIC else 'local(native-pick)'} | "
          f"google-login={'ON' if AUTH_ENABLED else 'off'}")
    if PUBLIC and os.environ.get("CLINSCOPE_FORCE_OWNER", "").strip():
        print("[ClinScope] ⚠ CLINSCOPE_FORCE_OWNER 已移除,公開模式一律忽略(站主只能經站內 Google 登入)。")
    if PUBLIC:
        _precompute_sample()      # 預先分類範例夾(供預設畫面),失敗不影響啟動
    demo = build()
    if AUTH_ENABLED:
        # 站內 Google 登入:掛自訂 /auth/* 路由到 Gradio 底層 FastAPI,再用 uvicorn 起(127.0.0.1,經 tunnel)。
        import uvicorn
        from fastapi import FastAPI
        fastapi_app = FastAPI()
        _add_auth_routes(fastapi_app)
        mounted = gr.mount_gradio_app(fastapi_app, demo, path="/", allowed_paths=allowed,
                                      theme=THEME, css=CSS, show_error=True)
        server = uvicorn.Server(uvicorn.Config(mounted, host="127.0.0.1", port=port, log_level="warning"))
        if prevent_thread_lock:
            import threading
            threading.Thread(target=server.run, daemon=True).start()
            return mounted
        server.run()
        return mounted
    kw = dict(theme=THEME, css=CSS, allowed_paths=allowed,
              prevent_thread_lock=prevent_thread_lock, inbrowser=inbrowser)
    if auth:
        kw["auth"] = auth
    if server_port:
        kw["server_port"] = server_port
    kw.update(overrides)
    return demo.launch(**kw)


if __name__ == "__main__":
    launch_app()
