"""上傳所有 manifest 權重 + manifest 本身到 **公開** HF Hub repo。

安全：預設 dry-run，只列清單與總量；加 --yes 才真的建 repo + 上傳。
需本機已設 HF token（huggingface-cli login 或 HF_TOKEN）。

用法：
    .venv/Scripts/python.exe packaging/upload_weights_hf.py            # dry-run
    .venv/Scripts/python.exe packaging/upload_weights_hf.py --yes      # 真上傳
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO_ID = "tinghaochang/clinscope-weights"
MAN_PATH = ROOT / "packaging" / "weights_manifest.json"


def main() -> None:
    if not MAN_PATH.exists():
        sys.exit("找不到 weights_manifest.json，請先跑 gen_weights_manifest.py")
    man = json.loads(MAN_PATH.read_text(encoding="utf-8"))
    tot = sum(v["bytes"] for v in man.values())
    print(f"repo={REPO_ID}（public） | {len(man)} files | {tot / 1e9:.2f} GB")

    if "--yes" not in sys.argv:
        for rel, m in list(man.items())[:8]:
            print(f"  ↑ {rel}  ({m['size_mb']} MB)")
        if len(man) > 8:
            print(f"  … 其餘 {len(man) - 8} 個")
        print("\n[dry-run] 確認無誤後加 --yes 真正上傳到公開 repo。")
        return

    from huggingface_hub import HfApi, create_repo
    api = HfApi()
    create_repo(REPO_ID, repo_type="model", private=False, exist_ok=True)
    print(f"repo ready: https://huggingface.co/{REPO_ID}")

    # 已在 Hub 的檔（每檔 upload_file 是原子 commit，出現在清單＝完整）→ 可續傳、不重傳
    try:
        existing = set(api.list_repo_files(REPO_ID, repo_type="model"))
    except Exception:
        existing = set()

    # manifest 先傳，讓 repo 自帶一份對照
    api.upload_file(path_or_fileobj=str(MAN_PATH), path_in_repo="weights_manifest.json",
                    repo_id=REPO_ID, repo_type="model")

    done = skipped = 0
    n = len(man)
    for i, (rel, m) in enumerate(man.items(), 1):
        fp = ROOT / rel
        if not fp.exists():
            print(f"  ⚠️ 略過(本機缺): {rel}")
            continue
        if m["hf_path"] in existing:
            skipped += 1
            print(f"  ✓ [{i}/{n}] 已在 Hub，跳過: {rel}", flush=True)
            continue
        print(f"  ↑ [{i}/{n}] {rel}  ({m['size_mb']} MB)", flush=True)
        api.upload_file(path_or_fileobj=str(fp), path_in_repo=m["hf_path"],
                        repo_id=REPO_ID, repo_type="model")
        done += 1
    print(f"done: uploaded {done} new, skipped {skipped} existing "
          f"-> https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
