"""
ClinScope — IPD-Brain glioma: tile large H&E PNGs (in labelled.zip) into patches,
label each by its case's Subtype (from IPD_Brain.csv), and split PATIENT-DISJOINT
on Case Number (no case crosses train/val). Streams the zip; caps tiles/image.

Filenames look like `IN Brain-0004(a).png`; the CSV `Case Number` may carry the
`(a)`/`(b)`. We match the label by full stem first, then by the case base
(suffix stripped), and we GROUP for the split by the case base so both (a)/(b)
images of one patient stay on the same side.

Writes <out>/{train,val}/<SUBTYPE>/<stem>_<tile>.jpg
"""
import argparse
import csv
import io
import random
import re
import zipfile
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageFile

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


def case_base(stem):
    return re.sub(r"\([A-Za-z0-9]+\)\s*$", "", stem).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--max-tiles", type=int, default=30)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cmap = {}
    with open(args.csv, encoding="utf-8-sig", errors="ignore") as fh:
        for row in csv.DictReader(fh):
            cn = (row.get("Case Number") or "").strip()
            st = (row.get("Subtype") or "").strip()
            if cn and st:
                cmap[cn] = st
    base_map = {}
    for cn, st in cmap.items():
        base_map.setdefault(case_base(cn), st)

    with zipfile.ZipFile(args.zip) as z:
        names = [n for n in z.namelist()
                 if n.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))]
    items = []
    for n in names:
        stem = Path(n).stem
        st = cmap.get(stem) or base_map.get(case_base(stem))
        if st:
            items.append((n, case_base(stem), st))
    print(f"images: {len(names)} | matched to subtype: {len(items)}", flush=True)

    by_sub = defaultdict(set)
    for n, cb, st in items:
        by_sub[st].add(cb)
    rng = random.Random(args.seed)
    val_cases = set()
    for st, cbs in by_sub.items():
        cbs = sorted(cbs)
        rng.shuffle(cbs)
        nv = max(1, int(round(len(cbs) * args.val_frac)))
        val_cases.update(cbs[:nv])
    print(f"subtype cases: { {st: len(c) for st, c in by_sub.items()} } | val cases: {len(val_cases)}", flush=True)

    n_ok = 0
    saved = {"train": defaultdict(int), "val": defaultdict(int)}
    with zipfile.ZipFile(args.zip) as z:
        for n, cb, st in items:
            sp = "val" if cb in val_cases else "train"
            d = args.out / sp / st
            d.mkdir(parents=True, exist_ok=True)
            try:
                im = Image.open(io.BytesIO(z.read(n))).convert("RGB")
                W, H = im.size
                T = args.tile
                coords = [(x, y) for y in range(0, H - T + 1, T) for x in range(0, W - T + 1, T)]
                rng.shuffle(coords)
                for i, (x, y) in enumerate(coords[:args.max_tiles]):
                    im.crop((x, y, x + T, y + T)).save(d / f"{Path(n).stem}_{i}.jpg", quality=90)
                    n_ok += 1
                    saved[sp][st] += 1
            except Exception:
                pass
    print(f"\n[done] tiles: {n_ok}", flush=True)
    for sp in ("train", "val"):
        print(f" [{sp}] {dict(saved[sp])}")


if __name__ == "__main__":
    main()
