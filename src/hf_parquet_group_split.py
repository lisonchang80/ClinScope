"""
ClinScope — decode a Hugging Face image-classification parquet export into a
GROUP-DISJOINT train/val ImageFolder split.

Unlike a random split, every row sharing the same --group-col value (e.g.
subject_id / patient / slide id) goes ENTIRELY to train OR val, never both.
That yields an honest, leakage-free validation set whenever the dataset carries
patient/slide grouping (the bias that inflated several random-split tracks).

Writes:
    <out>/train/<label>/<shard>_<rg>_<row>.jpg
    <out>/val/<label>/<shard>_<rg>_<row>.jpg

Image extraction mirrors hf_parquet_to_imagefolder.py (struct {bytes,path} or
raw bytes). The label is used as-is when it is already a string; an int
ClassLabel is mapped via --class-names-json when provided, else stringified.

Example (C-NMC leukemia, patient-disjoint val):
    python src/hf_parquet_group_split.py --src data/cnmc_raw/data \
      --out data/cnmc_split --image-col image --label-col label \
      --group-col subject_id --val-frac 0.15 --max-side 256
"""
import argparse
import io
import json
import random
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
    ap.add_argument("--src", type=Path, required=True, help="dir holding *.parquet shards")
    ap.add_argument("--out", type=Path, required=True, help="split root to write (train/ val/)")
    ap.add_argument("--image-col", default="image")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--group-col", required=True,
                    help="patient/slide id column; the train/val split is DISJOINT on this")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-side", type=int, default=256,
                    help="downscale so the long side <= this (0 = keep original)")
    ap.add_argument("--quality", type=int, default=90)
    ap.add_argument("--class-names-json", default=None,
                    help='JSON list of ClassLabel names in index order (only if label is int)')
    args = ap.parse_args()

    keep_awake()
    class_names = json.loads(args.class_names_json) if args.class_names_json else None
    files = sorted(args.src.glob("*.parquet"))
    if not files:
        raise SystemExit(f"no *.parquet under {args.src}")

    # pass 1: enumerate unique groups (read only the group column)
    groups_all = set()
    for fp in files:
        pf = pq.ParquetFile(fp)
        for gi in range(pf.num_row_groups):
            t = pf.read_row_group(gi, columns=[args.group_col])
            groups_all.update(str(x) for x in t.column(args.group_col).to_pylist())
    groups_sorted = sorted(groups_all)
    rng = random.Random(args.seed)
    rng.shuffle(groups_sorted)
    n_val = max(1, int(round(len(groups_sorted) * args.val_frac)))
    val_groups = set(groups_sorted[:n_val])
    print(f"[split] {len(groups_sorted)} groups on '{args.group_col}' -> "
          f"{len(val_groups)} val / {len(groups_sorted) - len(val_groups)} train (DISJOINT)",
          flush=True)

    # pass 2: decode + route by group
    (args.out / "train").mkdir(parents=True, exist_ok=True)
    (args.out / "val").mkdir(parents=True, exist_ok=True)
    n_ok = n_bad = 0
    per = {"train": {}, "val": {}}
    for fi, fp in enumerate(files):
        pf = pq.ParquetFile(fp)
        for gi in range(pf.num_row_groups):
            t = pf.read_row_group(gi, columns=[args.image_col, args.label_col, args.group_col])
            imgs = t.column(args.image_col).to_pylist()
            labs = t.column(args.label_col).to_pylist()
            grps = t.column(args.group_col).to_pylist()
            for ri, (img, lab, grp) in enumerate(zip(imgs, labs, grps)):
                try:
                    raw = img["bytes"] if isinstance(img, dict) else img
                    im = Image.open(io.BytesIO(raw)).convert("RGB")
                    if args.max_side and max(im.size) > args.max_side:
                        im.thumbnail((args.max_side, args.max_side))
                    if class_names and isinstance(lab, int):
                        cname = class_names[lab]
                    else:
                        cname = str(lab)
                    cname = cname.replace("/", "-").strip()
                    split = "val" if str(grp) in val_groups else "train"
                    cdir = args.out / split / cname
                    cdir.mkdir(parents=True, exist_ok=True)
                    im.save(cdir / f"{fi:02d}_{gi:03d}_{ri:04d}.jpg", quality=args.quality)
                    n_ok += 1
                    per[split][cname] = per[split].get(cname, 0) + 1
                except Exception:
                    n_bad += 1
        print(f"[{fi + 1}/{len(files)}] {fp.name}: ok={n_ok} bad={n_bad}", flush=True)

    print("\n[done] per split/class:")
    for sp in ("train", "val"):
        print(f" [{sp}]")
        for k in sorted(per[sp]):
            print(f"   {k:24s} {per[sp][k]}")
    print(f"total: ok={n_ok} bad={n_bad}")


if __name__ == "__main__":
    main()
