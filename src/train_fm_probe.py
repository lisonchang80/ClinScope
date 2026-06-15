# -*- coding: utf-8 -*-
"""ClinScope FM 引擎(分類): Phikon-v2(病理 ViT-L, frozen)特徵 + LogisticRegression head。
對「WSI 亞型/小資料」這類 ResNet 從頭訓會過擬合的 cls track,frozen FM 特徵 + 線性 head 通常顯著拉分
(見 skill pick-model-arch 實測)。骨幹凍結, 只訓 head, 8GB 充裕。
用法: python src/train_fm_probe.py --data-dir data/<k>_split --out models/<k>_fmprobe.joblib \
        --report-dir reports/<k>_fm --grid 3 --title "..."
產物: joblib(clf+scaler+classes+grid+model) 供 app.py 載入推論。"""
import os, sys, argparse, numpy as np

# sklearn 先載(避免 torch import 順序 segfault, 見 reference_torch_import_segfault)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fm_features as fm

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_split(d, cap=None, seed=42):
    classes = sorted([c for c in os.listdir(d) if os.path.isdir(os.path.join(d, c))])
    cidx = {c: i for i, c in enumerate(classes)}
    paths, ys = [], []
    rng = np.random.RandomState(seed)
    for c in classes:
        fs = [os.path.join(d, c, f) for f in sorted(os.listdir(os.path.join(d, c)))
              if f.lower().endswith(IMG_EXT)]
        if cap and len(fs) > cap:
            fs = list(rng.choice(fs, cap, replace=False))
        paths += fs; ys += [cidx[c]] * len(fs)
    return paths, np.array(ys), classes


def feats_for(paths, grid, bs=24):
    out = []
    for s in range(0, len(paths), bs):
        imgs = [Image.open(p) for p in paths[s:s + bs]]
        out.append(fm.extract_batch(imgs, DEV, grid=grid))
        if (s // bs) % 20 == 0:
            print(f"  feat {s+len(imgs)}/{len(paths)}", flush=True)
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report-dir", required=True)
    ap.add_argument("--grid", type=int, default=3)
    ap.add_argument("--cap-train", type=int, default=None, help="每類 train 上限(大資料加速)")
    ap.add_argument("--title", default="")
    args = ap.parse_args()
    print(f"[fm-probe] {args.title or args.data_dir} | grid={args.grid} | dev={DEV}", flush=True)

    trp, ytr, classes = load_split(os.path.join(args.data_dir, "train"), cap=args.cap_train)
    vap, yva, _ = load_split(os.path.join(args.data_dir, "val"))
    print(f"[data] train {len(trp)} / val {len(vap)} | {len(classes)} classes: {classes}", flush=True)

    print("[extract] train ...", flush=True); Xtr = feats_for(trp, args.grid)
    print("[extract] val ...", flush=True);  Xva = feats_for(vap, args.grid)

    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0)
    clf.fit(sc.transform(Xtr), ytr)
    pred = clf.predict(sc.transform(Xva))
    acc = accuracy_score(yva, pred); mf1 = f1_score(yva, pred, average="macro")
    wf1 = f1_score(yva, pred, average="weighted")
    rep = classification_report(yva, pred, target_names=classes, digits=4, zero_division=0)

    os.makedirs(args.report_dir, exist_ok=True)
    with open(os.path.join(args.report_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"# Phikon-v2 frozen + LogReg | {args.title}\n")
        f.write(f"val_acc={acc:.4f} macro-F1={mf1:.4f} weighted-F1={wf1:.4f}\n\n{rep}\n")
    joblib.dump({"clf": clf, "scaler": sc, "classes": classes, "grid": args.grid,
                 "model": fm.MODEL_NAME, "num_classes": len(classes)}, args.out)
    import json
    json.dump({"best_val_acc": round(float(acc), 4), "macro_f1": round(float(mf1), 4),
               "n_train": len(trp), "n_val": len(vap), "classes": classes,
               "engine": "phikon-v2-frozen+logreg"},
              open(os.path.join(args.report_dir, "summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    print(f"\n[done] val_acc={acc:.4f} | macro-F1={mf1:.4f} | weighted-F1={wf1:.4f}", flush=True)
    print(rep, flush=True)
    print(f"[saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
