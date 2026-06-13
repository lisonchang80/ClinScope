"""
ClinScope — build a WSI-disjoint, class-balanced ImageFolder split from
PatchGastricADC22 (flat patches `<WSI_id>_<hash>.jpg` + captions.csv).

Each WSI has one subtype caption; every patch of a WSI inherits its label.
We (a) collapse the verbose subtypes into 4 robust differentiation grades,
(b) split by WSI id so patches of one slide never cross train/val (no leakage),
(c) cap patches-per-WSI to keep the set tractable. Streams the zip (no full
extract). Ultra-rare special types (Signet ring 17 WSI, Mucinous 3) are dropped
for insufficient WSIs to split — reported, not silent.

Writes <out>/{train,val}/<grade>/<wsi>_<hash>.jpg
"""
import argparse
import csv
import io
import random
import zipfile
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# verbose subtype -> 4 differentiation grades (None = drop)
GRADE = {
    "Well differentiated tubular adenocarcinoma": "Well",
    "Well to moderately differentiated tubular adenocarcinoma": "Well",
    "Moderately differentiated tubular adenocarcinoma": "Moderate",
    "Moderately to poorly differentiated adenocarcinoma": "Poor",
    "Poorly differentiated adenocarcinoma, non-solid type": "Poor",
    "Poorly differentiated adenocarcinoma, solid type": "Poor",
    "Papillary adenocarcinoma": "Papillary",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--captions", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-per-wsi", type=int, default=40)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-side", type=int, default=256)
    args = ap.parse_args()

    # WSI id -> grade
    wsi_grade = {}
    dropped = defaultdict(int)
    with open(args.captions, encoding="utf-8", errors="ignore") as fh:
        r = csv.DictReader(fh)
        for row in r:
            g = GRADE.get((row.get("subtype") or "").strip())
            if g:
                wsi_grade[row["id"]] = g
            else:
                dropped[(row.get("subtype") or "")[:40]] += 1
    print(f"WSIs kept: {len(wsi_grade)} | dropped subtypes: {dict(dropped)}", flush=True)

    # WSI-level stratified split
    by_grade = defaultdict(list)
    for w, g in wsi_grade.items():
        by_grade[g].append(w)
    rng = random.Random(args.seed)
    val_wsis = set()
    for g, ws in by_grade.items():
        rng.shuffle(ws)
        nv = max(1, int(round(len(ws) * args.val_frac)))
        val_wsis.update(ws[:nv])
    print(f"grade WSI counts: { {g: len(ws) for g, ws in by_grade.items()} } | val WSIs: {len(val_wsis)}", flush=True)

    # stream zip, route + cap
    per_wsi = defaultdict(int)
    n_ok = n_skip = 0
    saved = {"train": defaultdict(int), "val": defaultdict(int)}
    for sp in ("train", "val"):
        for g in by_grade:
            (args.out / sp / g).mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip) as z:
        for name in z.namelist():
            if not name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            base = name.rsplit("/", 1)[-1]
            wsi = base.split("_", 1)[0]
            g = wsi_grade.get(wsi)
            if g is None:
                n_skip += 1
                continue
            if per_wsi[wsi] >= args.max_per_wsi:
                continue
            per_wsi[wsi] += 1
            sp = "val" if wsi in val_wsis else "train"
            try:
                im = Image.open(io.BytesIO(z.read(name))).convert("RGB")
                if args.max_side and max(im.size) > args.max_side:
                    im.thumbnail((args.max_side, args.max_side))
                im.save(args.out / sp / g / base, quality=90)
                n_ok += 1
                saved[sp][g] += 1
            except Exception:
                pass
    print(f"\n[done] saved {n_ok} patches (skipped {n_skip} dropped-subtype)", flush=True)
    for sp in ("train", "val"):
        print(f" [{sp}] {dict(saved[sp])}")


if __name__ == "__main__":
    main()
