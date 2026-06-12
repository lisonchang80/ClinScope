"""ClinScope — decode a Hugging Face dataset split into an ImageFolder tree using the
`datasets` library (not raw pyarrow). Use this when hf_parquet_to_imagefolder.py fails on
the schema — some mirrors flatten the HF Image feature into top-level bytes/path columns
or otherwise hide the label, which raw pyarrow can't address but `datasets` decodes cleanly.

    python src/hf_dataset_to_imagefolder.py --repo owner/name --split train \
      --out data/foo_if --label-col label --max-side 0

Writes <out>/<class_name>/<row>.jpg (class names from the ClassLabel feature when present,
else the raw label value).
"""
import argparse
from pathlib import Path

try:
    from keepawake import keep_awake
except Exception:
    def keep_awake():
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="HF dataset repo id")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--image-col", default="image")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--max-side", type=int, default=0,
                    help="downscale long side <= this (0 = keep original)")
    ap.add_argument("--quality", type=int, default=92)
    args = ap.parse_args()
    keep_awake()

    from datasets import load_dataset
    ds = load_dataset(args.repo, split=args.split)
    feat = ds.features.get(args.label_col)
    names = getattr(feat, "names", None)
    args.out.mkdir(parents=True, exist_ok=True)

    per, n_bad = {}, 0
    for i, ex in enumerate(ds):
        try:
            im = ex[args.image_col]
            im = im.convert("RGB")
            if args.max_side and max(im.size) > args.max_side:
                im.thumbnail((args.max_side, args.max_side))
            lab = ex[args.label_col]
            cname = (names[lab] if names else str(lab))
            cname = cname.replace("/", "-").replace(" ", "_").strip()
            cdir = args.out / cname
            cdir.mkdir(exist_ok=True)
            im.save(cdir / f"{i:06d}.jpg", quality=args.quality)
            per[cname] = per.get(cname, 0) + 1
        except Exception:
            n_bad += 1
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1} decoded ...", flush=True)

    print("\n[done] per-class counts:")
    for k in sorted(per):
        print(f"  {k:28s} {per[k]}")
    print(f"total: ok={sum(per.values())} bad={n_bad} | classes={len(per)}")


if __name__ == "__main__":
    main()
