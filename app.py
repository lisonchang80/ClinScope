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
    {"key": "wbc", "kind": "cls",
     "label": "🩸 白血球分類 WBC differential — PBC (Wright/Giemsa)",
     "path": ROOT / "models" / "pbc_resnet50.pt",
     "blurb": "周邊血液白血球 8 類:嗜中性/嗜酸/嗜鹼性球、淋巴球、單核球、未成熟顆粒球(IG)、紅血球母細胞、血小板。"},
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
    {"key": "bloodcells", "kind": "det",
     "label": "🩸 血球偵測 Blood cells — BCCD (YOLO)",
     "path": ROOT / "reports" / "yolo" / "bccd" / "weights" / "best.pt",
     "blurb": "整片血抹片偵測 + 計數:RBC / WBC / Platelets。"},
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
