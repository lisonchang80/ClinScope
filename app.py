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
     "blurb": "組織病理 H&E 乳腺腫瘤二元分類:良性 benign vs 惡性 malignant(40–400× 放大倍率混訓,magnification-independent)。val 0.978。⚠️ 隨機 split 會把同病人不同倍率分到 train/val → 偏樂觀,需 patient-level split 才完全可信。"},
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
     "blurb": "細胞學:甲狀腺細針穿刺 FNA 6 類(原始標註 1–6,疑對應 Bethesda I–VI 報告分級,資料卡未明確定義)。val 0.947、6 類 F1 0.93–0.99。對照子宮頸 Pap,屬不同部位的細胞學。⚠️ 同片多 crop,隨機 split 有病人洩漏。"},
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
     "blurb": "寄生蟲種級:瘧原蟲 4 蟲種分類 — 惡性瘧 Falciparum / 三日瘧 Malariae / 卵形瘧 Ovale / 間日瘧 Vivax(薄血膜 Giemsa,全視野 img + 單寄生蟲 crops)。比 NIH『感染/未感染』更細的種級鑑別。⚠️ val 0.997 嚴重偏樂觀,非真實泛化 —— 資料極度不平衡(falciparum 獨大)且 img/crops 同視野跨 train/val 洩漏(同 LC25000 增強洩漏型),稀有種 Ovale/Malariae val 僅 6–7 張、無統計意義。"},
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
