"""
ClinScope — classify an image (or a folder of images) with a trained checkpoint.

Example:
    python src/predict.py --model models/resnet50_fungi.pt --input some_photo.jpg
    python src/predict.py --model models/resnet50_fungi.pt --input my_microscope_pics/ --topk 3
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_model(model_path: Path, device):
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    classes = ckpt["classes"]
    img_size = ckpt.get("img_size", 224)
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, len(classes)))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return model, classes, tf


@torch.no_grad()
def predict_one(model, tf, classes, path: Path, device, topk: int):
    img = Image.open(path).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)
    probs = F.softmax(model(x), dim=1)[0]
    k = min(topk, len(classes))
    conf, idx = probs.topk(k)
    return [(classes[i], float(c)) for c, i in zip(conf, idx)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=Path("models/resnet50_fungi.pt"))
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes, tf = load_model(args.model, device)
    print(f"[model] {len(classes)} classes: {classes}\n")

    if args.input.is_dir():
        files = sorted(f for f in args.input.rglob("*") if f.suffix.lower() in IMG_EXT)
    else:
        files = [args.input]
    if not files:
        raise SystemExit(f"No images found at {args.input}")

    for f in files:
        preds = predict_one(model, tf, classes, f, device, args.topk)
        head = "  ".join(f"{name}={p:.1%}" for name, p in preds)
        print(f"{f.name:<40} {head}")


if __name__ == "__main__":
    main()
