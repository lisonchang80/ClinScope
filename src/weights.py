"""weights.py — ClinScope 權重 lazy-download resolver（見 docs/DEPLOY_PLAN.md B2）。

開發環境（權重就在 repo ROOT 下）→ 每個函式都是 pass-through，app 行為不變。
打包安裝環境（權重缺席）→ 某模型首次被用到時，從 **公開** HF Hub repo 抓對應權重檔，
存到 %LOCALAPPDATA%\\ClinScope\\models（鏡像 repo 相對路徑）、sha256 校驗、之後重用。

manifest：packaging/weights_manifest.json   {rel-path: {hf_path, sha256, size_mb, bytes}}
由 packaging/gen_weights_manifest.py 從本機權重產生。

⚠️ 本機（及多數無 admin 的 Windows）不能建 symlink → hf_hub_download 必用 local_dir 強制複製。
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

if getattr(sys, "frozen", False):     # PyInstaller 凍結:資料/模組都在 _MEIPASS(_internal)
    ROOT = Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
else:
    ROOT = Path(__file__).resolve().parent.parent      # repo / 開發根目錄
REPO_ID = os.environ.get("CLINSCOPE_WEIGHTS_REPO", "tinghaochang/clinscope-weights")

# 快取放在（可能唯讀的）安裝目錄之外
_base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_CACHE_HOME") or str(Path.home())
CACHE_DIR = Path(_base) / "ClinScope" / "models"

_MANIFEST_PATH = ROOT / "packaging" / "weights_manifest.json"
try:
    MANIFEST: dict = (
        json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
        if _MANIFEST_PATH.exists() else {}
    )
except Exception:
    MANIFEST = {}


def _norm_rel(rel) -> str:
    """把任意權重路徑（絕對或相對）正規化成 repo 相對的 POSIX 字串（manifest 鍵）。"""
    p = Path(rel)
    if not p.is_absolute():
        return p.as_posix()
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except Exception:
        return p.as_posix()


def spec_rels(spec: dict) -> list:
    """一個 REGISTRY spec 需要的所有權重檔（主 .pt/best.pt + 選用的 fm joblib）。"""
    rels = []
    if spec.get("path") is not None:
        rels.append(_norm_rel(spec["path"]))
    if spec.get("fm"):
        rels.append(_norm_rel(spec["fm"]))
    return rels


def runtime_rels(spec: dict) -> list:
    """推論時『實際會載入』的檔：FM 模型只用 joblib，其餘用主 .pt。
    （spec_rels 列全部以決定 availability；下載/提示用這個較精準的子集，避免下無用檔。）"""
    if spec.get("fm"):
        return [_norm_rel(spec["fm"])]
    if spec.get("path") is not None:
        return [_norm_rel(spec["path"])]
    return []


def is_downloadable(spec: dict) -> bool:
    """spec 需要的每個檔都『本機已有』或『manifest 有登錄』時為 True（決定是否列入 AVAILABLE）。"""
    rels = spec_rels(spec)
    if not rels:
        return False
    for rel in rels:
        if (ROOT / rel).exists():
            continue
        if rel in MANIFEST:
            continue
        return False
    return True


def pending_rels(spec: dict) -> list:
    """spec 推論所需、但本機與快取都還沒有、且 manifest 可下載的檔（rel 列）。空＝已就緒。"""
    out = []
    for rel in runtime_rels(spec):
        if (ROOT / rel).exists() or (CACHE_DIR / rel).exists():
            continue
        if rel in MANIFEST:
            out.append(rel)
    return out


def pending_size_mb(spec: dict) -> float:
    """spec 尚未下載檔的總大小（MB，四捨五入 1 位）。"""
    return round(sum(float(MANIFEST.get(r, {}).get("size_mb", 0)) for r in pending_rels(spec)), 1)


def _sha256(path: Path, buf: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch(rel: str, progress=None) -> Path:
    """確保單一 repo-相對檔存在本機；缺則從 HF 下載、校驗。回傳可用的絕對路徑。"""
    local = ROOT / rel
    if local.exists():                     # 開發環境：直接用 repo 內權重
        return local
    cached = CACHE_DIR / rel
    if cached.exists():                    # 已下載過
        return cached

    man = MANIFEST.get(rel)
    if not man:
        raise FileNotFoundError(
            f"權重不存在且不在 manifest:{rel}"
            f"（請先跑 packaging/gen_weights_manifest.py 並上傳 HF）")

    from huggingface_hub import hf_hub_download
    cached.parent.mkdir(parents=True, exist_ok=True)
    if progress is not None:
        try:
            progress(0.0, desc=f"下載 {Path(rel).name} … 約 {man.get('size_mb', '?')} MB")
        except Exception:
            pass

    # local_dir 強制複製（不走 symlink 快取，避開無 admin 的 Windows symlink 限制）
    fpath = Path(hf_hub_download(
        repo_id=REPO_ID, filename=man["hf_path"],
        local_dir=str(CACHE_DIR)))

    want = man.get("sha256")
    if want:
        got = _sha256(fpath)
        if got != want:
            try:
                fpath.unlink()
            except Exception:
                pass
            raise IOError(f"下載校驗失敗 {rel}:sha256 不符（期望 {want[:12]}…，得到 {got[:12]}…）")
    return fpath


def ensure_file(rel, progress=None) -> Path:
    """確保單一權重檔（rel 字串或路徑）就緒，回傳可用 Path。供 load_fm 等直接帶相對路徑者用。"""
    return _fetch(_norm_rel(rel), progress=progress)


def ensure_local(spec: dict, progress=None) -> Path:
    """確保 spec 的主權重就緒，回傳其路徑。供 load_cls/load_yolo/load_sahi 用。"""
    return ensure_file(spec["path"], progress=progress)
