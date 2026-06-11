"""
ClinScope — evaluate a trained classifier on an EXTERNAL (out-of-distribution) image set.

The whole point of this project is honesty about the in-distribution vs OOD gap.
This script quantifies it: take images the model never trained on (your own
microscope photos, or a different public dataset) and measure how much it drops.

Two modes (auto-detected, override with --mode):
  labeled   : --data-dir is an ImageFolder  <class>/*.jpg  whose folder names map to
              the model's training classes. Reports overall + per-class accuracy, a
              classification report, a confusion matrix PNG, and the drop vs the
              model's stored in-distribution val_acc.
  unlabeled : --data-dir is a flat folder of images (no class subdirs). No ground
              truth -> no accuracy. Instead reports the predicted-class distribution
              and the mean top-1 confidence, which is how you spot a model that is
              confidently WRONG on OOD input.

If the external folder names differ from the model's class names, pass a JSON map:
  --class-map '{"infected": "Parasitized", "healthy": "Uninfected"}'

Examples:
  python src/eval_external.py --model models/pbc_resnet50.pt --data-dir my_smears/
  python src/eval_external.py --model models/malaria_resnet50.pt --data-dir ext/ \
         --class-map '{"infected": "Parasitized", "healthy": "Uninfected"}'
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
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
    return model, classes, tf, ckpt.get("val_acc")


@torch.no_grad()
def predict_top1(model, tf, path: Path, device):
    img = Image.open(path).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)
    probs = F.softmax(model(x), dim=1)[0]
    conf, idx = probs.max(0)
    return int(idx), float(conf)


def list_images(folder: Path):
    return sorted(f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXT)


def resolve_class(folder_name: str, classes, class_map):
    """Map an external folder name to a model class index, or None if no match."""
    target = class_map.get(folder_name, folder_name)
    by_lower = {c.lower(): c for c in classes}
    if target in classes:
        return classes.index(target)
    if target.lower() in by_lower:
        return classes.index(by_lower[target.lower()])
    return None


def detect_mode(data_dir: Path):
    subdirs = [d for d in data_dir.iterdir() if d.is_dir()]
    if any(list_images(d) for d in subdirs):
        return "labeled"
    return "unlabeled"


def run_labeled(model, tf, classes, val_acc, data_dir, report_dir, device, class_map):
    try:
        from sklearn.metrics import classification_report, confusion_matrix
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        have_metrics = True
    except Exception:
        have_metrics = False

    y_true, y_pred, skipped, unreadable = [], [], [], 0
    for sub in sorted(d for d in data_dir.iterdir() if d.is_dir()):
        ci = resolve_class(sub.name, classes, class_map)
        if ci is None:
            skipped.append(sub.name)
            continue
        for f in list_images(sub):
            try:
                idx, _ = predict_top1(model, tf, f, device)
            except Exception:
                unreadable += 1
                continue
            y_true.append(ci)
            y_pred.append(idx)

    if skipped:
        print(f"[warn] folders not matched to any model class (skipped): {skipped}")
        print(f"       model classes = {classes}\n       (use --class-map to map them)")
    if unreadable:
        print(f"[warn] skipped {unreadable} unreadable image(s)")
    if not y_true:
        raise SystemExit("No labeled images matched the model's classes. "
                         "Check folder names or pass --class-map.")

    n = len(y_true)
    acc = sum(int(t == p) for t, p in zip(y_true, y_pred)) / n
    print(f"\n[external] {n} images | top-1 accuracy = {acc:.4f}")
    if val_acc is not None:
        drop = val_acc - acc
        print(f"[in-dist]  stored val_acc = {val_acc:.4f}  ->  drop = {drop:+.4f} "
              f"({drop / val_acc:+.1%} relative)")

    report_dir.mkdir(parents=True, exist_ok=True)
    if have_metrics:
        present = sorted(set(y_true) | set(y_pred))
        names = [classes[i] for i in present]
        rep = classification_report(y_true, y_pred, labels=present,
                                    target_names=names, digits=4, zero_division=0)
        (report_dir / "external_report.txt").write_text(rep, encoding="utf-8")
        print("\n" + rep)
        cm = confusion_matrix(y_true, y_pred, labels=present)
        fig, ax = plt.subplots(figsize=(max(6, len(names)), max(5, len(names) * 0.9)))
        im = ax.imshow(cm, cmap="Oranges")
        ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
        ax.set_yticks(range(len(names)), names)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"External / OOD — acc={acc:.3f}"
                     + (f" (in-dist {val_acc:.3f})" if val_acc is not None else ""))
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        fig.colorbar(im); fig.tight_layout()
        fig.savefig(report_dir / "external_confusion_matrix.png", dpi=130)
        print(f"[report] wrote {report_dir}/external_report.txt + external_confusion_matrix.png")

    with open(report_dir / "external_summary.json", "w", encoding="utf-8") as f:
        json.dump({"mode": "labeled", "n_images": n, "external_acc": acc,
                   "in_dist_val_acc": val_acc, "skipped_folders": skipped}, f, indent=2)


def run_unlabeled(model, tf, classes, data_dir, report_dir, device):
    files = sorted(f for f in data_dir.rglob("*") if f.suffix.lower() in IMG_EXT)
    if not files:
        raise SystemExit(f"No images found under {data_dir}")
    dist, confs, unreadable = Counter(), [], 0
    for f in files:
        try:
            idx, conf = predict_top1(model, tf, f, device)
        except Exception:
            unreadable += 1
            continue
        dist[classes[idx]] += 1
        confs.append(conf)

    n = len(confs)
    mean_conf = sum(confs) / n if n else 0.0
    low = sum(1 for c in confs if c < 0.5)
    print(f"\n[external/unlabeled] {n} images | mean top-1 confidence = {mean_conf:.3f}")
    print(f"[caution] {low}/{n} ({low / n:.0%}) below 0.5 confidence -> possible OOD"
          if n else "")
    print("\npredicted-class distribution:")
    for name, c in dist.most_common():
        print(f"  {name:<28} {c:>5}  ({c / n:.1%})")
    if unreadable:
        print(f"\n[warn] skipped {unreadable} unreadable image(s)")

    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_dir / "external_summary.json", "w", encoding="utf-8") as f:
        json.dump({"mode": "unlabeled", "n_images": n, "mean_top1_conf": mean_conf,
                   "low_conf_count": low, "distribution": dict(dist)}, f, indent=2)
    print(f"\n[report] wrote {report_dir}/external_summary.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--mode", choices=["auto", "labeled", "unlabeled"], default="auto")
    ap.add_argument("--class-map", default="{}",
                    help='JSON mapping external folder names -> model class names')
    ap.add_argument("--report-dir", type=Path, default=None)
    args = ap.parse_args()

    if not args.data_dir.is_dir():
        raise SystemExit(f"--data-dir must be a folder: {args.data_dir}")
    report_dir = args.report_dir or Path("reports") / f"{args.model.stem}_xdomain"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes, tf, val_acc = load_model(args.model, device)
    print(f"[model] {args.model.name} | {len(classes)} classes: {classes}")
    print(f"[device] {device}")

    mode = args.mode if args.mode != "auto" else detect_mode(args.data_dir)
    print(f"[mode] {mode}")
    if mode == "labeled":
        run_labeled(model, tf, classes, val_acc, args.data_dir, report_dir, device,
                    json.loads(args.class_map))
    else:
        run_unlabeled(model, tf, classes, args.data_dir, report_dir, device)


if __name__ == "__main__":
    main()
