# -*- coding: utf-8 -*-
"""為 ClinScope 每個 REGISTRY 模型產出結構化「識別卡」JSON -> model_cards/<key>.json + index.json。
來源: app.py REGISTRY(label/blurb/kind) + reports/(成績) + 別名對應報表夾。可重跑(成績更新自動帶入)。

用法: python src/gen_model_cards.py
"""
import os, re, json, csv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "model_cards")

# REGISTRY key -> reports 夾名(不一致者)
CLS_DIR = {"wbc": "pbc", "bacteria": "dibas", "ana": "hep2", "fungi_koh": "defungi",
           "sperm": "smids", "her2_ihc": "her2ihc", "pap": "sipakmed", "fungi": "openfungi"}
DET_DIR = {"bloodcells": "bccd", "bbbc041_stage": "bbbc041", "mrose_gram": "mrose",
           "dataset19_colony": "dataset19", "giardia_crypto": "giardia"}
BATCH6 = {"kupbc13", "mdsbm27", "sicap", "sbsd", "evisan", "bcg", "puma",
          "tb_det", "tb_cls", "nuinsseg", "effusion",
          "msi", "orchid", "kidney", "mihic", "tvmi3k", "tcga",
          "tanzania", "riva", "placenta", "koh", "csf", "ivcm"}
BATCH7 = {"cryonuseg", "blastocyst", "thalassemia", "osteosarcoma", "breast_idc", "sperm_dfi",
          "ebhi_seg"}

# 來源(blurb 未必含, 這裡 curated; 找不到再退回 blurb 解析)
SOURCE = {
    "kupbc13": "Zenodo 17333317", "mdsbm27": "Figshare 28737170", "sicap": "Zenodo 14178894 (SICAPv2)",
    "sbsd": "Zenodo 15676241", "evisan": "Zenodo 4303768", "bcg": "Figshare 26004610",
    "puma": "Zenodo 13859989", "tb_det": "Kaggle saife245/tuberculosis-image-datasets",
    "tb_cls": "Mendeley yhybmfrmgs", "nuinsseg": "Zenodo 10518968",
    "effusion": "Kaggle cmacus/body-cavity-fluid-cytology-images",
    "msi": "Zenodo 2530835 (Kather CRC-DX MSI)", "orchid": "Zenodo 12646943 (ORCHID oral)",
    "mihic": "Zenodo 10065510 (MIHIC)", "tvmi3k": "Zenodo 6534086 (TVMI3K)",
    "tcga": "Zenodo 6604094 (TCGA-TILs)", "kidney": "Zenodo 4299694 (glomeruli)",
    "tanzania": "Harvard Dataverse (Tanzania malaria thick/thin)", "riva": "Zenodo 17288879 (RIVA)",
    "placenta": "Zenodo 10535022 (HAPPY)", "koh": "Figshare 14678514 (hyphae KOH)",
    "csf": "Zenodo 6543147 (CSF cytology)", "ivcm": "Figshare 19838083 (IVCM-Keratitis)",
    "cryonuseg": "Kaggle ipateam/segmentation-of-nuclei-in-cryosectioned-he-images (CryoNuSeg)",
    "blastocyst": "Figshare 20123153 (human blastocyst IVF benchmark)",
    "thalassemia": "Mendeley rfdz6wfzn4 (RBCdataset 9-class)",
    "osteosarcoma": "Kaggle tatsuyaryu/osteosarcoma-histopathology-classification (TCIA UT Southwestern)",
    "breast_idc": "Mendeley w7jjcx7gj6 (BreastCancer IDC Grade)",
    "sperm_dfi": "Figshare 30120811 (sperm SCD DNA-fragmentation)",
    "ebhi_seg": "Figshare 21540159 (EBHI-SEG colorectal)",
}

DISCIPLINE_KW = ["微生物", "黴菌", "寄生蟲", "血液/骨髓", "血液腫瘤", "血液", "組織病理", "細胞學",
                 "免疫組化", "免疫病理", "免疫", "男科", "眼科", "細胞遺傳學", "尿液分析", "尿液",
                 "生殖醫學", "生殖", "胚胎學"]


def parse_registry(src):
    """擷取每個 entry 的 key/kind/label/blurb/weights"""
    out = []
    pat = re.compile(
        r'\{"key":\s*"([^"]+)",\s*"kind":\s*"([^"]+)",\s*'
        r'"label":\s*"([^"]+)",\s*'
        r'"path":\s*(.+?),\s*'
        r'"blurb":\s*"(.+?)"\}', re.DOTALL)
    for m in pat.finditer(src):
        key, kind, label, path_expr, blurb = m.groups()
        quoted = re.findall(r'"([^"]+)"', path_expr)  # ROOT / "models" / "x.pt"
        weights = "/".join(quoted)
        out.append(dict(key=key, kind=kind, label=label, blurb=blurb, weights=weights))
    return out


def split_name(label):
    disp = re.sub(r'^[^\wA-Za-z一-鿿]+', '', label).strip()  # 去開頭 emoji
    before = disp.split("(")[0].split("（")[0]
    before = re.split(r'\s*[—–]\s*', before)[0]  # 破折號前(資料集名常另列在破折號後)
    zh = "".join(re.findall(r'[一-鿿＋+0-9／/]+', before)).strip("/").strip()
    en = " ".join(re.findall(r'[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9\-\./]*', before)).strip()
    return {"display": disp, "zh": zh, "en": en}


def discipline_of(blurb, label):
    head = blurb.split("(")[0].split("（")[0].split(":")[0].split("：")[0][:12]
    for kw in DISCIPLINE_KW:
        if kw in head or kw in blurb[:20]:
            return kw
    return None


def caveats_of(blurb):
    parts = re.split(r'⚠️', blurb)
    return [("⚠️ " + p.strip().rstrip("。") + "。") for p in parts[1:] if p.strip()]


def summary_of(blurb):
    s = re.split(r'⚠️', blurb)[0].strip()
    return s


def find_ref(blurb):
    for pat, fmt in [(r'Zenodo[`\s]*(\d+)', "Zenodo {}"), (r'Figshare[`\s]*(\d+)', "Figshare {}"),
                     (r'Mendeley[`\s]*([0-9a-z]+)', "Mendeley {}"), (r'Broad\s*(BBBC\d+)', "Broad {}"),
                     (r'Kaggle[`\s]*([\w\-]+/[\w\-]+)', "Kaggle {}")]:
        m = re.search(pat, blurb)
        if m:
            return fmt.format(m.group(1))
    return None


def find_license(blurb):
    for pat in ["CC[\\s\\-]?BY[\\s\\-]?SA[\\s\\-]?4\\.0", "CC[\\s\\-]?BY[\\s\\-]?NC[\\s\\-]?SA",
                "CC[\\s\\-]?BY[\\s\\-]?4\\.0", "CC[\\s\\-]?BY", "CC0", "AGPL[\\s\\-]?3\\.0", "© 原作者"]:
        m = re.search(pat, blurb)
        if m:
            return m.group(0)
    return None


def split_method(blurb):
    for kw, val in [("patient-level", "patient-level (病人級)"), ("slide-disjoint", "slide-disjoint (片級)"),
                    ("WSI 級", "WSI-level"), ("片級", "slide-level (片級)"), ("病人級", "patient-level (病人級)"),
                    ("source-disjoint", "source-disjoint"), ("video-disjoint", "video-block-disjoint"),
                    ("器官分層", "organ-stratified"), ("官方", "official split"),
                    ("影像級隨機", "image-level random (⚠️可能洩漏)"), ("隨機 split", "random (⚠️可能洩漏)"),
                    ("隨機切", "random (⚠️可能洩漏)")]:
        if kw in blurb:
            return val
    return None


def cls_metrics(rdir):
    out = {}
    sj = os.path.join(ROOT, "reports", rdir, "summary.json")
    if os.path.exists(sj):
        d = json.load(open(sj, encoding="utf-8"))
        out["val_acc"] = round(d.get("best_val_acc"), 4) if d.get("best_val_acc") is not None else None
        out["n_train"] = d.get("n_train"); out["n_val"] = d.get("n_val")
        if d.get("classes"):
            out["num_classes"] = len(d["classes"]); out["classes"] = d["classes"]
    cr = os.path.join(ROOT, "reports", rdir, "classification_report.txt")
    if os.path.exists(cr):
        for line in open(cr, encoding="utf-8"):
            if line.strip().startswith("macro avg"):
                out["macro_f1"] = float(line.split()[-2])
            if line.strip().startswith("weighted avg"):
                out["weighted_f1"] = float(line.split()[-2])
    return out or None


def det_metrics(rdir):
    p = os.path.join(ROOT, "reports", "yolo", rdir, "results.csv")
    if not os.path.exists(p):
        return None
    rows = list(csv.DictReader(open(p)))
    if not rows:
        return None
    last = rows[-1]
    g = lambda s: next((k for k in last if s in k), None)
    def fv(s):
        k = g(s); return round(float(last[k]), 4) if k else None
    return {"mAP50": fv("mAP50(B)"), "mAP50_95": fv("mAP50-95(B)"),
            "precision": fv("precision(B)"), "recall": fv("recall(B)"), "epochs": len(rows)}


def main():
    os.makedirs(OUT, exist_ok=True)
    src = open(os.path.join(ROOT, "app.py"), encoding="utf-8").read()
    entries = parse_registry(src)
    index = []
    for e in entries:
        key, kind = e["key"], e["kind"]
        is_det = kind == "det"
        rdir = (DET_DIR if is_det else CLS_DIR).get(key, key)
        metrics = det_metrics(rdir) if is_det else cls_metrics(rdir)
        weights_abs = os.path.join(ROOT, e["weights"])
        card = {
            "key": key,
            "abbr": key,
            "name": split_name(e["label"]),
            "discipline": discipline_of(e["blurb"], e["label"]),
            "task": "detection" if is_det else "classification",
            "model_type": "YOLOv8s" if is_det else "ResNet-50",
            "metrics": metrics,
            "experiment_summary": summary_of(e["blurb"]),
            "caveats": caveats_of(e["blurb"]),
            "source": SOURCE.get(key) or find_ref(e["blurb"]),
            "license": find_license(e["blurb"]),
            "split_method": split_method(e["blurb"]),
            "weights": e["weights"],
            "status": "registered" if os.path.exists(weights_abs) else "pending",
            "batch": ("batch-7 (2026-06-14)" if key in BATCH7
                      else "batch-6 (2026-06-14)" if key in BATCH6 else "batch 1-5"),
        }
        if metrics and "num_classes" in (metrics or {}):
            card["num_classes"] = metrics.pop("num_classes")
            card["classes"] = metrics.pop("classes", None)
        json.dump(card, open(os.path.join(OUT, key + ".json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        index.append({"key": key, "name_zh": card["name"]["zh"], "name_en": card["name"]["en"],
                      "task": card["task"], "discipline": card["discipline"],
                      "metrics": metrics, "status": card["status"]})
    json.dump({"platform": "ClinScope", "n_models": len(index), "cards": index},
              open(os.path.join(OUT, "index.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[model_cards] wrote {len(entries)} cards -> {OUT}")
    miss = [e["key"] for e in entries if not (det_metrics(DET_DIR.get(e['key'],e['key'])) if e['kind']=='det' else cls_metrics(CLS_DIR.get(e['key'],e['key'])))]
    print(f"[model_cards] 無成績(reports 缺)的 {len(miss)}: {miss}")


if __name__ == "__main__":
    main()
