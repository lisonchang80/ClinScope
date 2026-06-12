"""
ClinScope — convert RationAI/PanNuke parquet (per-instance nucleus masks) into a
YOLO detection dataset (images/{train,val} + labels/{train,val} + data.yaml).

PanNuke parquet row:
  image       : struct{bytes,path}            -- 256x256 H&E tile
  instances   : list<struct{bytes,path}>      -- one binary mask image per nucleus
  categories  : list<int64>                   -- nucleus type per instance
  tissue      : int64                          -- tissue type (unused here)

For each instance mask we take the bbox of the nonzero region -> normalized
YOLO box `cls xc yc w h`. Folds 1+2 -> train, fold 3 -> val (PanNuke convention).
"""
import argparse
import io
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# PanNuke nucleus types in category-index order (0..4). RationAI/PanNuke uses
# 0 neoplastic, 1 inflammatory, 2 connective/soft-tissue, 3 dead, 4 epithelial.
CLASS_NAMES = ["neoplastic", "inflammatory", "connective", "dead", "epithelial"]


def bbox_from_mask(mask_bytes):
    m = np.array(Image.open(io.BytesIO(mask_bytes)).convert("L"))
    ys, xs = np.nonzero(m)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()), m.shape[1], m.shape[0]


def process(files, out, split, cat_offset):
    imgdir = out / "images" / split
    lbldir = out / "labels" / split
    imgdir.mkdir(parents=True, exist_ok=True)
    lbldir.mkdir(parents=True, exist_ok=True)
    n_img = n_box = n_empty = 0
    cat_seen = set()
    for fp in files:
        pf = pq.ParquetFile(fp)
        for gi in range(pf.num_row_groups):
            t = pf.read_row_group(gi, columns=["image", "instances", "categories"])
            imgs = t.column("image").to_pylist()
            insts = t.column("instances").to_pylist()
            cats = t.column("categories").to_pylist()
            for i, (img, inst, cat) in enumerate(zip(imgs, insts, cats)):
                try:
                    im = Image.open(io.BytesIO(img["bytes"])).convert("RGB")
                    W, H = im.size
                    lines = []
                    for mobj, c in zip(inst, cat):
                        cat_seen.add(c)
                        mb = mobj["bytes"] if isinstance(mobj, dict) else mobj
                        bb = bbox_from_mask(mb)
                        if bb is None:
                            continue
                        x0, y0, x1, y1, mw, mh = bb
                        cls = c - cat_offset
                        if not (0 <= cls < len(CLASS_NAMES)):
                            continue
                        xc = ((x0 + x1) / 2) / mw
                        yc = ((y0 + y1) / 2) / mh
                        w = (x1 - x0 + 1) / mw
                        h = (y1 - y0 + 1) / mh
                        lines.append(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                    name = f"{fp.stem}_{gi:03d}_{i:04d}"
                    im.save(imgdir / f"{name}.png")
                    (lbldir / f"{name}.txt").write_text("\n".join(lines))
                    n_img += 1
                    n_box += len(lines)
                    if not lines:
                        n_empty += 1
                except Exception:
                    pass
        print(f"  [{split}] {fp.name}: imgs={n_img} boxes={n_box} empty={n_empty}", flush=True)
    print(f"  [{split}] categories seen: {sorted(cat_seen)}", flush=True)
    return n_img, n_box


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="dir with fold*.parquet")
    ap.add_argument("--out", type=Path, required=True, help="YOLO dataset root")
    ap.add_argument("--train-folds", default="fold1,fold2")
    ap.add_argument("--val-folds", default="fold3")
    ap.add_argument("--cat-offset", type=int, default=0,
                    help="subtract from category int (set 1 if categories are 1-indexed)")
    args = ap.parse_args()

    allp = sorted(args.src.glob("**/*.parquet"))
    def pick(folds):
        keys = [f.strip() for f in folds.split(",")]
        return [p for p in allp if any(k in p.name for k in keys)]
    train_files, val_files = pick(args.train_folds), pick(args.val_folds)
    print(f"train parquet: {[p.name for p in train_files]}")
    print(f"val   parquet: {[p.name for p in val_files]}")

    ntr, btr = process(train_files, args.out, "train", args.cat_offset)
    nva, bva = process(val_files, args.out, "val", args.cat_offset)

    yaml = args.out / "data.yaml"
    yaml.write_text(
        f"path: {args.out.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: {CLASS_NAMES}\n"
    )
    print(f"\n[done] train {ntr} imgs/{btr} boxes | val {nva} imgs/{bva} boxes")
    print(f"[done] wrote {yaml}")


if __name__ == "__main__":
    main()
