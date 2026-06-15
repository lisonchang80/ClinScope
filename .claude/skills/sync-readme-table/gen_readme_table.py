# -*- coding: utf-8 -*-
"""從 model_cards/*.json 重生 README 的模型表 / 資料來源表 / 科目統計,並對帳卡片數 vs app.py REGISTRY。

用法(在 repo 根目錄跑,只用標準庫):
  python .claude/skills/sync-readme-table/gen_readme_table.py              # 全表 + 來源表 + 統計 + 對帳
  python .claude/skills/sync-readme-table/gen_readme_table.py --batch "batch-9 (2026-06-15)"  # 只出該批新列
  python .claude/skills/sync-readme-table/gen_readme_table.py --summary    # 只出科目統計
  python .claude/skills/sync-readme-table/gen_readme_table.py --check      # 只對帳數量

設計:成績欄用卡片 metrics + caveats 組「規範化字串」(不還原手寫敘事,圖的是一致 + 不漏)。
引擎欄 FM / SAHI 兩個特例用 ENGINE_OVERRIDE 補(卡片 model_type 不分 FM)。表到 stdout、統計/對帳到 stderr。
"""
import os, re, json, sys, glob
from collections import Counter

# 強制 utf-8 輸出:成績欄含 ⚠️(U+26A0) 等字元,cp950 console 直接 print 會 UnicodeEncodeError。
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CARDS = os.path.join(ROOT, "model_cards")

ENGINE_OVERRIDE = {"ubc_ocean": "Phikon-v2 FM", "puma": "YOLOv8s + SAHI"}
DISC_ORDER = ["血液", "血液/骨髓", "血液腫瘤", "微生物", "黴菌", "寄生蟲", "組織病理",
              "免疫組化", "免疫病理", "免疫", "細胞學", "男科", "生殖醫學", "生殖", "胚胎學",
              "尿液分析", "尿液", "細胞遺傳學", "眼科", "獸醫"]


def load_cards():
    out = []
    for p in sorted(glob.glob(os.path.join(CARDS, "*.json"))):
        if os.path.basename(p) == "index.json":
            continue
        out.append(json.load(open(p, encoding="utf-8")))
    return out


def engine(c):
    if c["key"] in ENGINE_OVERRIDE:
        return ENGINE_OVERRIDE[c["key"]]
    return c.get("model_type") or ("YOLOv8s" if c["task"] == "detection" else "ResNet-50")


def score(c):
    m = c.get("metrics") or {}
    if c["task"] == "detection":
        s = f"mAP50 {m.get('mAP50')}" if m.get("mAP50") is not None else "—"
        if m.get("mAP50_95") is not None:
            s += f" / mAP50-95 {m['mAP50_95']}"
    else:
        s = f"val {m.get('val_acc')}" if m.get("val_acc") is not None else "—"
        if m.get("macro_f1") is not None:
            s += f" / F1 {m['macro_f1']}"
    if c.get("caveats"):
        s += " ⚠️"
    return s


def disc_sort(c):
    d = c.get("discipline") or "其他"
    return (DISC_ORDER.index(d) if d in DISC_ORDER else len(DISC_ORDER), d, c["key"])


def name_of(c):
    n = c.get("name") or {}
    return n.get("zh") or n.get("en") or c["key"]


def model_table(cards):
    rows = ["| 科別 | Track | 引擎 | 成績 | Batch |", "|---|---|---|---|---|"]
    for c in sorted(cards, key=disc_sort):
        rows.append(f"| {c.get('discipline') or '—'} | {name_of(c)} | {engine(c)} | {score(c)} | {c.get('batch','—')} |")
    return "\n".join(rows)


def source_table(cards):
    rows = ["| Track | 來源 |", "|---|---|"]
    for c in sorted(cards, key=disc_sort):
        rows.append(f"| {name_of(c)} | {c.get('source') or '—'} |")
    return "\n".join(rows)


def summary(cards):
    cnt = Counter((c.get("discipline") or "其他") for c in cards)
    lines = ["科目統計:"]
    for d in sorted(cnt, key=lambda x: (DISC_ORDER.index(x) if x in DISC_ORDER else 99)):
        lines.append(f"  {d}: {cnt[d]}")
    lines.append(f"  合計: {sum(cnt.values())}")
    return "\n".join(lines)


def check(cards):
    src = open(os.path.join(ROOT, "app.py"), encoding="utf-8").read()
    n_reg = len(re.findall(r'\{"key":\s*"[^"]+",\s*"kind":', src))
    n_card = len(cards)
    ok = "OK" if n_reg == n_card else "不一致!!"
    return f"對帳: app.py REGISTRY {n_reg} 條 vs model_cards {n_card} 張 -> {ok}"


def main():
    cards = load_cards()
    args = sys.argv[1:]
    if "--check" in args:
        print(check(cards)); return
    if "--summary" in args:
        print(summary(cards)); return
    if "--batch" in args:
        tag = args[args.index("--batch") + 1]
        sub = [c for c in cards if c.get("batch") == tag]
        print(f"# {tag} 新增 {len(sub)} 列\n")
        print(model_table(sub)); return
    # 預設:全表 + 來源表(stdout) + 統計/對帳(stderr)
    print("<!-- MODELS:START (由 gen_readme_table.py 重生) -->")
    print(model_table(cards))
    print("<!-- MODELS:END -->\n")
    print("<!-- SOURCES:START -->")
    print(source_table(cards))
    print("<!-- SOURCES:END -->")
    sys.stderr.write("\n" + summary(cards) + "\n" + check(cards) + "\n")


if __name__ == "__main__":
    main()
