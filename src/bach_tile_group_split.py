"""
ClinScope — decode large H&E microscopy images from a HF parquet export, tile
each into non-overlapping patches, and write a train/val ImageFolder split that
is DISJOINT at the source-image level (all tiles of one image go to one side).

Built for BACH (ICIAR 2018): ~270 train images, 2048x1536, 4 severity classes
(Benign / InSitu / Invasive / Normal). Whole-image classification with tiling
gives more samples without leaking tiles of the same image across train/val.

Writes <out>/{train,val}/<class>/<rowidx>_<tile>.jpg
"""
import argparse
import io
import json
import random
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    from keepawake import keep_awake
except Exception:
    def keep_awake():
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="dir with *.parquet")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--image-col", default="image")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--class-names-json", required=True,
                    help='JSON list of ClassLabel names in index order')
    ap.add_argument("--skip-classes-json", default='[]',
                    help='JSON list of class names to skip (e.g. ["Unknown"])')
    ap.add_argument("--file-glob", default="*", help="only parquet whose name contains this")
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quality", type=int, default=90)
    args = ap.parse_args()

    keep_awake()
    names = json.loads(args.class_names_json)
    skip = set(json.loads(args.skip_classes_json))
    files = [p for p in sorted(args.src.glob("**/*.parquet")) if args.file_glob in p.name]
    if not files:
        raise SystemExit(f"no parquet matching '{args.file_glob}' under {args.src}")
    print(f"parquet: {[p.name for p in files]}")

    # pass 1: gather (file, rg, row) -> label, then stratified image-level val pick
    rows = []  # (fi, gi, ri, label_name)
    for fi, fp in enumerate(files):
        pf = pq.ParquetFile(fp)
        for gi in range(pf.num_row_groups):
            labs = pf.read_row_group(gi, columns=[args.label_col]).column(args.label_col).to_pylist()
            for ri, lab in enumerate(labs):
                cname = names[lab] if isinstance(lab, int) else str(lab)
                if cname in skip:
                    continue
                rows.append((fi, gi, ri, cname))
    by_cls = defaultdict(list)
    for r in rows:
        by_cls[r[3]].append(r)
    rng = random.Random(args.seed)
    val_set = set()
    for c, rs in by_cls.items():
        rng.shuffle(rs)
        n_val = max(1, int(round(len(rs) * args.val_frac)))
        for r in rs[:n_val]:
            val_set.add((r[0], r[1], r[2]))
    print(f"images: {len(rows)} | per-class: { {c: len(v) for c, v in by_cls.items()} } | val images: {len(val_set)}")

    # pass 2: decode + tile + route
    n_img = n_tile = 0
    per = {"train": defaultdict(int), "val": defaultdict(int)}
    for fi, fp in enumerate(files):
        pf = pq.ParquetFile(fp)
        for gi in range(pf.num_row_groups):
            tbl = pf.read_row_group(gi, columns=[args.image_col, args.label_col])
            imgs = tbl.column(args.image_col).to_pylist()
            labs = tbl.column(args.label_col).to_pylist()
            for ri, (img, lab) in enumerate(zip(imgs, labs)):
                cname = names[lab] if isinstance(lab, int) else str(lab)
                if cname in skip:
                    continue
                split = "val" if (fi, gi, ri) in val_set else "train"
                try:
                    im = Image.open(io.BytesIO(img["bytes"] if isinstance(img, dict) else img)).convert("RGB")
                    W, H = im.size
                    T = args.tile
                    cdir = args.out / split / cname
                    cdir.mkdir(parents=True, exist_ok=True)
                    ti = 0
                    for y in range(0, H - T + 1, T):
                        for x in range(0, W - T + 1, T):
                            im.crop((x, y, x + T, y + T)).save(
                                cdir / f"{fi}_{gi}_{ri}_{ti}.jpg", quality=args.quality)
                            ti += 1
                            n_tile += 1
                            per[split][cname] += 1
                    n_img += 1
                except Exception:
                    pass
        print(f"  {fp.name}: imgs={n_img} tiles={n_tile}", flush=True)

    print("\n[done] per split/class (tiles):")
    for sp in ("train", "val"):
        print(f" [{sp}]")
        for c in sorted(per[sp]):
            print(f"   {c:12s} {per[sp][c]}")
    print(f"total: {n_img} imgs -> {n_tile} tiles")


if __name__ == "__main__":
    main()
