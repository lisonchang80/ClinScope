"""
ClinScope — decode a Hugging Face image-classification parquet export into an
ImageFolder tree that prepare_data.py / train_resnet.py can consume.

Many HF datasets ship as parquet shards with each image embedded as a struct
``{bytes, path}`` plus an integer ClassLabel. That is not the ImageFolder layout
our ResNet pipeline expects, so this reads the shards with pyarrow, decodes each
image with PIL, optionally downscales the long side (clinical microscopy frames
are large; ResNet trains at 224 anyway), and writes:

    <out>/<class_name>/<shard>_<group>_<row>.jpg

Class names come from --class-names-json (the HF ClassLabel order) so folders are
human-readable instead of 0..N. Inspect them once with:

    load_dataset(repo, split="train", streaming=True).features["label"].names

Example:
    python src/hf_parquet_to_imagefolder.py \
      --src data/parasite_egg_raw --out data/parasite_egg_cells --max-side 512 \
      --class-names-json "[\"Ascaris lumbricoides\", ...]"
"""
import argparse
import io
import json
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image, ImageFile

# Some embedded JPEGs are truncated; tolerate rather than abort the whole shard.
ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    from keepawake import keep_awake  # local: block system sleep during the run
except Exception:  # keepawake is optional
    def keep_awake():
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="dir holding *.parquet shards")
    ap.add_argument("--out", type=Path, required=True, help="ImageFolder root to write")
    ap.add_argument("--image-col", default="image")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--max-side", type=int, default=512,
                    help="downscale so the long side <= this (0 = keep original)")
    ap.add_argument("--quality", type=int, default=90)
    ap.add_argument("--class-names-json", default=None,
                    help='JSON list of ClassLabel names in index order, e.g. \'["a","b"]\'')
    ap.add_argument("--group-col", default=None,
                    help="optional 2nd column (e.g. organ/site) to prefix the folder "
                         "name as <group>_<class>, recovering composite classes")
    ap.add_argument("--group-names-json", default=None,
                    help='JSON list of group ClassLabel names in index order')
    args = ap.parse_args()

    keep_awake()

    class_names = json.loads(args.class_names_json) if args.class_names_json else None
    group_names = json.loads(args.group_names_json) if args.group_names_json else None
    files = sorted(args.src.glob("*.parquet"))
    if not files:
        raise SystemExit(f"no *.parquet under {args.src}")
    args.out.mkdir(parents=True, exist_ok=True)

    n_ok, n_bad, per_class = 0, 0, {}
    for fi, fp in enumerate(files):
        pfile = pq.ParquetFile(fp)
        for gi in range(pfile.num_row_groups):
            cols = [args.image_col, args.label_col]
            if args.group_col:
                cols.append(args.group_col)
            tbl = pfile.read_row_group(gi, columns=cols)
            images = tbl.column(args.image_col).to_pylist()
            labels = tbl.column(args.label_col).to_pylist()
            groups = tbl.column(args.group_col).to_pylist() if args.group_col else None
            for ri, (img, lab) in enumerate(zip(images, labels)):
                try:
                    raw = img["bytes"] if isinstance(img, dict) else img
                    im = Image.open(io.BytesIO(raw)).convert("RGB")
                    if args.max_side and max(im.size) > args.max_side:
                        im.thumbnail((args.max_side, args.max_side))
                    cname = class_names[lab] if class_names else str(lab)
                    if groups is not None:
                        gv = groups[ri]
                        gname = group_names[gv] if group_names else str(gv)
                        cname = f"{gname}_{cname}"
                    cname = cname.replace("/", "-").strip()
                    cdir = args.out / cname
                    cdir.mkdir(exist_ok=True)
                    im.save(cdir / f"{fi:02d}_{gi:03d}_{ri:04d}.jpg", quality=args.quality)
                    n_ok += 1
                    per_class[cname] = per_class.get(cname, 0) + 1
                except Exception:
                    n_bad += 1
        print(f"[{fi + 1}/{len(files)}] {fp.name}: ok={n_ok} bad={n_bad}", flush=True)

    print("\n[done] per-class counts:")
    for k in sorted(per_class):
        print(f"  {k:34s} {per_class[k]}")
    print(f"total: ok={n_ok} bad={n_bad} | classes={len(per_class)}")


if __name__ == "__main__":
    main()
