"""從本機現有權重產生 packaging/weights_manifest.json。

manifest 鍵 = repo 相對路徑（POSIX），值 = {hf_path, sha256, size_mb, bytes}。
HF 上傳與 app 端 lazy download 都靠這張表。

用法（repo 根目錄、venv python）：
    .venv/Scripts/python.exe packaging/gen_weights_manifest.py
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from app import REGISTRY          # noqa: E402  (匯入即取得 registry，launch 在 __main__ 不會跑)
import weights as W               # noqa: E402


def sha256(p: Path, buf: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(buf), b""):
            h.update(c)
    return h.hexdigest()


def main() -> None:
    man: dict = {}
    missing: list = []
    for spec in REGISTRY:
        for rel in W.spec_rels(spec):
            if rel in man:
                continue
            fp = ROOT / rel
            if not fp.exists():
                missing.append(rel)
                continue
            sz = fp.stat().st_size
            man[rel] = {
                "hf_path": rel,
                "sha256": sha256(fp),
                "size_mb": round(sz / 1e6, 1),
                "bytes": sz,
            }

    out = ROOT / "packaging" / "weights_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(man, indent=2, ensure_ascii=False), encoding="utf-8")

    tot = sum(v["bytes"] for v in man.values())
    print(f"[manifest] {len(man)} files, {tot / 1e9:.2f} GB -> {out}")
    if missing:
        print(f"[manifest] ⚠️ {len(missing)} 檔本機不存在(略過): {missing[:10]}"
              + (" …" if len(missing) > 10 else ""))


if __name__ == "__main__":
    main()
