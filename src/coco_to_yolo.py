"""ClinScope — convert a COCO-format detection dataset (e.g. Roboflow/RF100 export with
per-split <split>/_annotations.coco.json + images) into the YOLO layout ultralytics wants:

    <out>/images/{train,val}/*.jpg
    <out>/labels/{train,val}/*.txt        # one line per box: "<cls> xc yc w h" (normalised)
    <out>/data.yaml

prepare_yolo.py only handles Pascal VOC; this is the COCO path the add-dataset-track recipe
points at ("COCO 來源要先轉成這格式").

    python src/coco_to_yolo.py --coco-root data/foo_coco --out data/foo_yolo \
      --splits "train:train,valid:val"
"""
import argparse
import json
import shutil
from pathlib import Path


def conv_split(coco_json: Path, img_src: Path, out_img: Path, out_lbl: Path, cat_map):
    data = json.load(open(coco_json, encoding="utf-8"))
    images = {im["id"]: im for im in data["images"]}
    anns = {}
    for a in data["annotations"]:
        anns.setdefault(a["image_id"], []).append(a)
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)
    n = 0
    for iid, im in images.items():
        fn = im["file_name"]
        w, h = im["width"], im["height"]
        src = img_src / fn
        if not src.exists():
            continue
        shutil.copy(src, out_img / fn)
        lines = []
        for a in anns.get(iid, []):
            if a["category_id"] not in cat_map:
                continue
            x, y, bw, bh = a["bbox"]  # COCO bbox = top-left x,y + w,h
            lines.append(f"{cat_map[a['category_id']]} "
                         f"{(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}")
        (out_lbl / (Path(fn).stem + ".txt")).write_text("\n".join(lines))
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco-root", required=True, help="dir containing <split>/_annotations.coco.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default="train:train,valid:val",
                    help="comma list of src:dst split pairs")
    args = ap.parse_args()
    root, out = Path(args.coco_root), Path(args.out)

    # category map from the train json: real object classes -> 0..N-1.
    # Roboflow/RF100 put a placeholder supercategory at id 0 (supercategory 'none'); drop it.
    cats = json.load(open(root / "train" / "_annotations.coco.json", encoding="utf-8"))["categories"]
    real = [c for c in cats if c.get("supercategory", "none") != "none"] or cats
    real = sorted(real, key=lambda c: c["id"])
    cat_map = {c["id"]: i for i, c in enumerate(real)}
    names = [c["name"] for c in real]

    for pair in args.splits.split(","):
        s, dst = pair.split(":")
        cj = root / s / "_annotations.coco.json"
        if not cj.exists():
            print(f"  (skip {s}: no json)")
            continue
        n = conv_split(cj, root / s, out / "images" / dst, out / "labels" / dst, cat_map)
        print(f"  {s} -> {dst}: {n} images")

    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\n"
        f"nc: {len(names)}\nnames: {names}\n")
    print("classes:", names, "| data.yaml ->", out / "data.yaml")


if __name__ == "__main__":
    main()
