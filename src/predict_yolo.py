"""
ClinScope — run a trained YOLO detector on an image or folder; save annotated output
and print per-class counts (useful for "how many parasites / cells in this field").

Example:
    python src/predict_yolo.py --model reports/yolo/bccd/weights/best.pt --input some.jpg
"""
import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--input", type=Path, required=True, help="image file or folder")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", type=Path, default=Path("reports/yolo/predict"))
    args = ap.parse_args()

    model = YOLO(str(args.model))
    results = model.predict(source=str(args.input), conf=args.conf, save=True,
                            project=str(args.out.parent), name=args.out.name,
                            exist_ok=True)
    for r in results:
        counts = {}
        for c in r.boxes.cls.tolist():
            name = r.names[int(c)]
            counts[name] = counts.get(name, 0) + 1
        print(f"{Path(r.path).name:<32} {counts}")
    print(f"\n[done] annotated images saved under {args.out}")


if __name__ == "__main__":
    main()
