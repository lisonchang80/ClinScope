"""
ClinScope — ResNet-50 transfer learning for fungal microscopy classification.

Trains on an ImageFolder-style dataset:
    <data-dir>/train/<class>/*.jpg
    <data-dir>/val/<class>/*.jpg

Two-phase schedule by default: freeze the backbone and train only the new head
for `--freeze-epochs`, then unfreeze everything and fine-tune at a lower LR.
Designed to fit an 8 GB GPU (RTX 4060 Ti) via AMP + batch 32 @ 224px.

Example:
    python src/train_resnet.py --data-dir data/micro_split --epochs 30
"""
import argparse
import json
import time
from pathlib import Path

# IMPORT ORDER MATTERS: load the matplotlib / sklearn (numpy+MKL/OpenMP) stack
# BEFORE torch. After `datasets` pulled in numpy 2.x, importing torch *before*
# sklearn/matplotlib segfaults at startup on this machine (native OpenMP/MKL
# load-order conflict; reproduced deterministically). sklearn-first is safe.
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import classification_report, confusion_matrix
    HAVE_METRICS = True
except Exception:
    HAVE_METRICS = False

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from PIL import ImageFile

from keepawake import keep_awake  # local: block system sleep during the run

# Clinical microscopy archives (e.g. DIBaS) ship some truncated TIFs; tolerate them
# instead of crashing the whole DataLoader. Genuinely-corrupt files are pruned upstream.
ImageFile.LOAD_TRUNCATED_IMAGES = True


def build_loaders(data_dir: Path, img_size: int, batch_size: int, workers: int):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_tf = transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_ds = datasets.ImageFolder(data_dir / "train", train_tf)
    val_ds = datasets.ImageFolder(data_dir / "val", eval_tf)
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=workers, pin_memory=True, drop_last=False)
    val_ld = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=True)
    return train_ds, val_ds, train_ld, val_ld


def set_backbone_trainable(model: nn.Module, trainable: bool):
    for name, p in model.named_parameters():
        if not name.startswith("fc."):
            p.requires_grad = trainable


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    all_pred, all_true = [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=(device.type == "cuda")):
            out = model(x)
            loss = criterion(out, y)
        loss_sum += loss.item() * x.size(0)
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += x.size(0)
        all_pred.extend(pred.cpu().tolist())
        all_true.extend(y.cpu().tolist())
    return loss_sum / total, correct / total, all_true, all_pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--freeze-epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-finetune", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=Path("models/resnet50_fungi.pt"))
    ap.add_argument("--report-dir", type=Path, default=Path("reports"))
    ap.add_argument("--title", default="ClinScope", help="confusion-matrix title prefix")
    ap.add_argument("--class-weight", choices=["inverse", "sqrt", "none"], default="inverse",
                    help="loss class-weighting vs imbalance: inverse-freq (default); "
                         "gentler sqrt(inverse-freq) for extreme imbalance (e.g. 21-class "
                         "bone marrow with a 6-image class); or none")
    args = ap.parse_args()

    keep_awake()  # don't let the machine sleep mid-training (monitor may still sleep)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device} "
          f"({torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'})")

    train_ds, val_ds, train_ld, val_ld = build_loaders(
        args.data_dir, args.img_size, args.batch_size, args.workers)
    classes = train_ds.classes
    print(f"[data] {len(train_ds)} train / {len(val_ds)} val | "
          f"{len(classes)} classes: {classes}")

    # Class-weighted loss to counter imbalance. Raw inverse-freq can be too
    # aggressive under extreme imbalance (a 6-image class would get ~1000x weight
    # and dominate) -> sqrt softens it, none disables.
    counts = torch.zeros(len(classes))
    for _, y in train_ds.samples:
        counts[y] += 1
    inv = counts.sum() / (len(classes) * counts.clamp(min=1))
    if args.class_weight == "none":
        weights = torch.ones(len(classes))
    elif args.class_weight == "sqrt":
        weights = inv.sqrt()
    else:
        weights = inv
    weights = weights.to(device)
    print(f"[loss] class-weight={args.class_weight} "
          f"(min={weights.min():.3f} max={weights.max():.3f})")
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, len(classes)))
    model.to(device)

    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
    best_acc, best_true, best_pred = 0.0, None, None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        frozen = epoch <= args.freeze_epochs
        set_backbone_trainable(model, trainable=not frozen)
        lr = args.lr_head if frozen else args.lr_finetune
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)

        model.train()
        t0 = time.time()
        run_loss, seen = 0.0, 0
        for x, y in train_ld:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=(device.type == "cuda")):
                out = model(x)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            run_loss += loss.item() * x.size(0)
            seen += x.size(0)

        val_loss, val_acc, y_true, y_pred = evaluate(model, val_ld, device, criterion)
        tag = "frozen" if frozen else "finetune"
        print(f"[ep {epoch:02d}/{args.epochs}] ({tag}) "
              f"train_loss={run_loss/seen:.3f} val_loss={val_loss:.3f} "
              f"val_acc={val_acc:.4f} ({time.time()-t0:.0f}s)")

        if val_acc > best_acc:
            best_acc, best_true, best_pred = val_acc, y_true, y_pred
            torch.save({"state_dict": model.state_dict(), "classes": classes,
                        "img_size": args.img_size, "val_acc": val_acc}, args.out)
            print(f"    -> saved best ({val_acc:.4f}) to {args.out}")

    print(f"\n[done] best val_acc = {best_acc:.4f}")

    if HAVE_METRICS and best_true is not None:
        rep = classification_report(best_true, best_pred, target_names=classes, digits=4)
        (args.report_dir / "classification_report.txt").write_text(rep, encoding="utf-8")
        print(rep)
        cm = confusion_matrix(best_true, best_pred)
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
        ax.set_yticks(range(len(classes)), classes)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"{args.title} — ResNet-50 (best val_acc={best_acc:.3f})")
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        fig.colorbar(im); fig.tight_layout()
        fig.savefig(args.report_dir / "confusion_matrix.png", dpi=130)
        print(f"[report] wrote {args.report_dir}/classification_report.txt + confusion_matrix.png")

    with open(args.report_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"best_val_acc": best_acc, "classes": classes,
                   "n_train": len(train_ds), "n_val": len(val_ds)}, f, indent=2)


if __name__ == "__main__":
    main()
