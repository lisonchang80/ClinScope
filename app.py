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
    {"key": "puma", "kind": "det",
     "label": "🔬 黑色素瘤組織核偵測 PUMA (H&E, 10 核類, YOLO)",
     "path": ROOT / "reports" / "yolo" / "puma" / "weights" / "best.pt",
     "blurb": "組織病理(本平台首條皮膚黑色素瘤 track):原發+轉移黑色素瘤 H&E ROI,偵測 10 種細胞核 — 腫瘤核/淋巴球/基質/內皮/組織球/嗜黑素細胞 melanophage/漿細胞/嗜中性球/凋亡核/上皮。YOLOv8s val mAP50 0.193 / mAP50-95 0.132 —— 誠實偏低:10 類極不平衡(腫瘤核 57k vs 嗜中性 366、嗜黑素 695),稀有核型近乎學不起來、密集小核偵測難;主力腫瘤核/淋巴球為主要貢獻。含黑色素瘤特有的 melanophage 等核型,為既有 PanNuke/NuInsSeg 所無。ROI 級切分(原發/轉移),CC0 授權。"},
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
