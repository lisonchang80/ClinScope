"""
ClinScope — train a YOLO detector (ultralytics) on a YOLO-format dataset.

The detection counterpart to train_resnet.py: use this when one image holds many
objects that must be localised + counted (blood cells, parasite eggs, AFB), rather
than a single centred object to classify.

Example:
    python src/train_yolo.py --data data/bccd_yolo/data.yaml --epochs 80 --name bccd
"""
import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="path to data.yaml")
    ap.add_argument("--weights", default="yolov8s.pt",
                    help="pretrained checkpoint (auto-downloaded): yolov8n/s/m...")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16, help="fits 8 GB at 640px for n/s")
    ap.add_argument("--patience", type=int, default=20, help="early-stop patience")
    ap.add_argument("--project", type=Path, default=Path("reports/yolo"))
    ap.add_argument("--name", default="run")
    ap.add_argument("--device", default="0", help="'0' for first GPU, 'cpu' to force CPU")
    args = ap.parse_args()

    model = YOLO(args.weights)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        project=str(args.project),
        name=args.name,
        device=args.device,
        verbose=True,
    )
    metrics = model.val(data=str(args.data), project=str(args.project),
                        name=f"{args.name}_val")
    best = args.project / args.name / "weights" / "best.pt"
    print(f"\n[done] best weights -> {best}")
    print(f"mAP50-95 = {metrics.box.map:.4f} | mAP50 = {metrics.box.map50:.4f}")


if __name__ == "__main__":
    main()
