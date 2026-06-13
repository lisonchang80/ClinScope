"""
ClinScope — build a GROUP-DISJOINT train/val ImageFolder split from a raw image
tree (files keep their ORIGINAL names), grouping by a token parsed out of each
filename / path.

Unlike a random split, every image whose group id (patient / slide / source
field-of-view) matches goes ENTIRELY to train OR val, never both. That removes
the leakage that inflated several random-split tracks (e.g. BreakHis patients,
MP-IDB same-field crops, slide-level cytology).

Label per image:
  --label-regex  : regex searched against the POSIX *relative path*; group(1)
                   is the class name (e.g. "(benign|malignant)").
  (omitted)      : the image's immediate parent directory name is the class.

Group per image:
  --group-regex  : regex searched against the filename *stem*; group(1) is the
                   group id the split is made disjoint on. REQUIRED.

Writes:
    <out>/train/<class>/<group>__<n>.<ext>
    <out>/val/<class>/<group>__<n>.<ext>

Example (BreakHis, patient-disjoint binary):
    python src/imagefolder_group_split.py \
      --src data/breakhis_raw/BreaKHis_v1 --out data/breakhis_split \
      --label-regex "/(benign|malignant)/" \
      --group-regex "-(\\d+-\\d+)-" --val-frac 0.25 --max-side 700
"""
import argparse
import re
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    from keepawake import keep_awake
except Exception:
    def keep_awake():
        return False

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="raw image tree root")
    ap.add_argument("--out", type=Path, required=True, help="split root to write (train/ val/)")
    ap.add_argument("--label-regex", default=None,
                    help="regex on POSIX relative path; group(1)=class. omit => parent dir name")
    ap.add_argument("--group-regex", required=True,
                    help="regex on filename stem; group(1)=group id (split is DISJOINT on this)")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-side", type=int, default=0,
                    help="downscale so long side <= this (0 = keep original)")
    ap.add_argument("--quality", type=int, default=92)
    ap.add_argument("--ext", default=None,
                    help="comma list of extensions to keep (default: common image types)")
    ap.add_argument("--dry-run", action="store_true", help="only print the plan, write nothing")
    args = ap.parse_args()

    keep_awake()
    exts = ({"." + e.lstrip(".").lower() for e in args.ext.split(",")} if args.ext else EXTS)
    lab_re = re.compile(args.label_regex) if args.label_regex else None
    grp_re = re.compile(args.group_regex)

    # pass 1: enumerate (path, class, group); collect groups per class
    items = []  # (path, cname, gid)
    n_no_label = n_no_group = 0
    for p in sorted(args.src.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in exts:
            continue
        rel = p.relative_to(args.src).as_posix()
        if lab_re:
            m = lab_re.search(rel)
            if not m:
                n_no_label += 1
                continue
            cname = m.group(1)
        else:
            cname = p.parent.name
        gm = grp_re.search(p.stem)
        if not gm:
            n_no_group += 1
            continue
        items.append((p, cname.replace("/", "-").strip(), gm.group(1)))

    if not items:
        raise SystemExit(f"no labelled+grouped images under {args.src} "
                         f"(skipped: no-label={n_no_label} no-group={n_no_group})")

    # group -> majority class (a group should be single-class; keep its class for stratifying)
    grp_cls = {}
    grp_n = defaultdict(int)
    for _, c, g in items:
        grp_n[(g, c)] += 1
    for (g, c), n in grp_n.items():
        if g not in grp_cls or n > grp_n[(g, grp_cls[g])]:
            grp_cls[g] = c

    by_cls_groups = defaultdict(set)
    for g, c in grp_cls.items():
        by_cls_groups[c].add(g)

    # stratified group-disjoint val pick: val_frac of GROUPS per class
    rng = random.Random(args.seed)
    val_groups = set()
    for c, gs in by_cls_groups.items():
        gl = sorted(gs)
        rng.shuffle(gl)
        n_val = max(1, int(round(len(gl) * args.val_frac)))
        val_groups.update(gl[:n_val])

    n_groups = len(grp_cls)
    print(f"[scan] images={len(items)} groups={n_groups} classes={sorted(by_cls_groups)}")
    print(f"[scan] skipped: no-label={n_no_label} no-group={n_no_group}")
    print(f"[split] val groups={len(val_groups)} / train groups={n_groups - len(val_groups)} "
          f"(DISJOINT on group id)")
    for c in sorted(by_cls_groups):
        gv = len(by_cls_groups[c] & val_groups)
        print(f"   {c:16s} groups: {len(by_cls_groups[c])} ({gv} val / {len(by_cls_groups[c]) - gv} train)")
    if args.dry_run:
        print("[dry-run] no files written")
        return

    # pass 2: copy/route + optional downscale
    (args.out / "train").mkdir(parents=True, exist_ok=True)
    (args.out / "val").mkdir(parents=True, exist_ok=True)
    n_ok = n_bad = 0
    per = {"train": defaultdict(int), "val": defaultdict(int)}
    seq = defaultdict(int)
    for p, cname, gid in items:
        split = "val" if gid in val_groups else "train"
        cdir = args.out / split / cname
        cdir.mkdir(parents=True, exist_ok=True)
        key = (split, cname, gid)
        seq[key] += 1
        safe_g = re.sub(r"[^A-Za-z0-9._-]", "_", gid)
        ext = p.suffix.lower() if p.suffix.lower() != ".jpeg" else ".jpg"
        try:
            if args.max_side:
                im = Image.open(p).convert("RGB")
                if max(im.size) > args.max_side:
                    im.thumbnail((args.max_side, args.max_side))
                ext = ".jpg"
                im.save(cdir / f"{safe_g}__{seq[key]:04d}{ext}", quality=args.quality)
            else:
                import shutil
                shutil.copy2(p, cdir / f"{safe_g}__{seq[key]:04d}{ext}")
            n_ok += 1
            per[split][cname] += 1
        except Exception:
            n_bad += 1

    print("\n[done] per split/class (images):")
    for sp in ("train", "val"):
        print(f" [{sp}]")
        for c in sorted(per[sp]):
            print(f"   {c:16s} {per[sp][c]}")
    print(f"total: ok={n_ok} bad={n_bad}")


if __name__ == "__main__":
    main()
