"""
FungiScope — prepare the OpenFungi archive into an ImageFolder train/val split.

Steps:
  1. Extract openfungi.zip (skipped if already extracted).
  2. Locate the requested subset directory (default: the folder whose name
     contains "micro"); treat its immediate subdirectories as classes.
  3. Stratified train/val split (seeded) -> data/<subset>_split/{train,val}/<class>/

Example:
    python src/prepare_data.py --zip data/openfungi.zip --subset micro --val-frac 0.2
"""
import argparse
import random
import shutil
import zipfile
from pathlib import Path

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def extract(zip_path: Path, out_root: Path) -> Path:
    if out_root.exists() and any(out_root.iterdir()):
        print(f"[extract] {out_root} already populated, skipping unzip")
        return out_root
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"[extract] unzipping {zip_path} -> {out_root} (this can take a while for 8 GB)")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out_root)
    return out_root


def find_subset_root(extracted: Path, keyword: str) -> Path:
    """Find a directory whose name contains `keyword` and that holds class subdirs."""
    candidates = [d for d in extracted.rglob("*")
                  if d.is_dir() and keyword.lower() in d.name.lower()]
    # Prefer the candidate with the most image-bearing immediate subdirectories.
    best, best_score = None, -1
    for d in candidates:
        score = sum(1 for sub in d.iterdir()
                    if sub.is_dir() and any(f.suffix.lower() in IMG_EXT
                                            for f in sub.rglob("*")))
        if score > best_score:
            best, best_score = d, score
    if best is None or best_score == 0:
        raise SystemExit(
            f"Could not auto-locate a '{keyword}' subset with class subfolders under "
            f"{extracted}. Inspect the tree and pass --src explicitly.")
    print(f"[locate] subset root = {best}  ({best_score} class folders)")
    return best


def is_readable(path: Path) -> bool:
    """True if PIL can fully decode the file (truncated-but-valid passes)."""
    try:
        with Image.open(path) as im:
            im.convert("RGB").load()
        return True
    except Exception:
        return False


def collect(class_root: Path, verify: bool = True):
    classes = {}
    dropped = 0
    for sub in sorted(p for p in class_root.iterdir() if p.is_dir()):
        imgs = [f for f in sub.rglob("*") if f.suffix.lower() in IMG_EXT]
        if verify:
            good = [f for f in imgs if is_readable(f)]
            dropped += len(imgs) - len(good)
            imgs = good
        if imgs:
            classes[sub.name] = imgs
    if verify and dropped:
        print(f"[verify] dropped {dropped} unreadable/corrupt image(s)")
    return classes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path, default=Path("data/openfungi.zip"))
    ap.add_argument("--extracted", type=Path, default=Path("data/openfungi_extracted"))
    ap.add_argument("--src", type=Path, default=None,
                    help="Explicit class-folder root; skips auto-detection.")
    ap.add_argument("--subset", default="micro", help="keyword: micro | macro")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verify", action=argparse.BooleanOptionalAction, default=True,
                    help="open each image with PIL; drop unreadable/corrupt ones (default on)")
    args = ap.parse_args()

    out = args.out or Path(f"data/{args.subset}_split")
    src = args.src
    if src is None:
        root = extract(args.zip, args.extracted)
        src = find_subset_root(root, args.subset)

    classes = collect(src, verify=args.verify)
    if not classes:
        raise SystemExit(f"No class folders with images found under {src}")

    rng = random.Random(args.seed)
    if out.exists():
        shutil.rmtree(out)
    total_tr = total_va = 0
    print(f"[split] val_frac={args.val_frac} seed={args.seed}")
    for cls, imgs in classes.items():
        imgs = sorted(imgs)
        rng.shuffle(imgs)
        n_val = max(1, int(round(len(imgs) * args.val_frac)))
        val, train = imgs[:n_val], imgs[n_val:]
        for split, files in (("train", train), ("val", val)):
            dst = out / split / cls
            dst.mkdir(parents=True, exist_ok=True)
            for i, f in enumerate(files):
                shutil.copy2(f, dst / f"{cls}_{i:04d}{f.suffix.lower()}")
        total_tr += len(train); total_va += len(val)
        print(f"    {cls:<28} train={len(train):4d}  val={len(val):4d}")
    print(f"[done] {total_tr} train + {total_va} val -> {out}")


if __name__ == "__main__":
    main()
