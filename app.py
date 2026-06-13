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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image, ImageFile
import gradio as gr

ImageFile.LOAD_TRUNCATED_IMAGES = True
ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
]

AVAILABLE = [m for m in REGISTRY if m["path"].exists()]
BY_LABEL = {m["label"]: m for m in AVAILABLE}

_cls_cache: dict = {}
_yolo_cache: dict = {}


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


@torch.no_grad()
def run(image, model_label, topk, conf):
    if image is None:
        return {}, None, "請先上傳一張顯微影像。"
    spec = BY_LABEL.get(model_label)
    if spec is None:
        return {}, None, "找不到該模型(權重檔可能還在訓練中)。"

    if spec["kind"] == "cls":
        model, classes, tf = load_cls(spec["path"])
        x = tf(image.convert("RGB")).unsqueeze(0).to(DEVICE)
        probs = F.softmax(model(x), dim=1)[0]
        k = min(int(topk), len(classes))
        cvals, idx = probs.topk(k)
        result = {prettify(classes[i]): float(c) for c, i in zip(cvals, idx)}
        note = f"**{spec['blurb']}**\n\n類別數:{len(classes)} · 裝置:{DEVICE.type.upper()}"
        return result, None, note

    # detection
    model = load_yolo(spec["path"])
    res = model.predict(source=image.convert("RGB"), conf=float(conf), verbose=False)[0]
    annotated = res.plot()[:, :, ::-1]  # BGR -> RGB
    counts: dict = {}
    for c in res.boxes.cls.tolist():
        name = res.names[int(c)]
        counts[name] = counts.get(name, 0) + 1
    if counts:
        body = " · ".join(f"**{k}**: {v}" for k, v in sorted(counts.items()))
    else:
        body = "(未偵測到物件,可調低信心門檻)"
    note = f"**{spec['blurb']}**\n\n計數 → {body}"
    return {}, annotated, note


def build():
    with gr.Blocks(title="ClinScope — 臨床顯微辨識") as demo:
        gr.Markdown(
            "# 🔬 ClinScope — 臨床顯微影像辨識\n"
            "上傳顯微影像,選擇對應模型。**分類**模型回傳機率;**偵測**模型回傳標註圖與計數。")
        if not AVAILABLE:
            gr.Markdown("⚠️ 找不到任何模型權重,請先完成至少一條 track 的訓練。")
            return demo
        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Image(type="pil", label="顯微影像", height=300)
                sel = gr.Dropdown(choices=list(BY_LABEL.keys()),
                                  value=list(BY_LABEL.keys())[0], label="辨識模型")
                topk = gr.Slider(1, 6, value=3, step=1, label="Top-K(分類用)")
                conf = gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="信心門檻(偵測用)")
                btn = gr.Button("辨識", variant="primary")
            with gr.Column(scale=1):
                out_label = gr.Label(num_top_classes=6, label="分類結果(機率)")
                out_img = gr.Image(label="偵測標註圖", height=300)
                out_note = gr.Markdown()
        btn.click(run, inputs=[inp, sel, topk, conf],
                  outputs=[out_label, out_img, out_note])
        gr.Markdown(
            "> ⚠️ 研究/作品集用途,**非臨床診斷**。每個模型只在其資料集分布內準確;"
            "跨到你自己的顯微鏡照片(不同染色法/相機/倍率)準確率預期下降。")
    return demo


if __name__ == "__main__":
    print(f"[ClinScope] device={DEVICE.type} | available models: "
          f"{[m['key'] for m in AVAILABLE]}")
    build().launch()
