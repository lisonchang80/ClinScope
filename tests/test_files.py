# -*- coding: utf-8 -*-
"""檔案曝險:Gradio 的 /gradio_api/file=<絕對路徑> 會把 allowed_paths 內任何檔案交給**任何人**。
公開站不可涵蓋 repo 根(models/、data/、env.local.bat…)、使用者家目錄(.cloudflared 憑證等)。"""
from pathlib import Path

import httpx

from conftest import A, OWNER, ROOT, TMP, cls_label, fake_request, make_png, start_site


def _get(site, p):
    return httpx.get(f"{site.base}/gradio_api/file={Path(p).as_posix()}", timeout=10)


def test_default_public_site_does_not_serve_repo_files(site):
    # 舊版預設開放「各磁碟根」(Windows C:\、D:\…)→ 匿名可抓 repo 內任何檔(Linux 無磁碟根故此條在雲端本就過)
    for p in (ROOT / "app.py", ROOT / "README.md", ROOT / "model_cards" / "fungi.json"):
        assert _get(site, p).status_code == 403, p


def test_allowed_paths_env_covering_repo_is_ignored_on_public_site(monkeypatch):
    # public-tunnel skill 建議的 CLINSCOPE_ALLOWED_PATHS=D:\ 會涵蓋 repo 根 → 舊版匿名可直接下載 app.py / env.local.bat
    monkeypatch.setenv("CLINSCOPE_ALLOWED_PATHS", str(ROOT.parent))
    s = start_site()
    assert _get(s, ROOT / "app.py").status_code == 403


def test_unsafe_to_serve_covers_repo_home_and_their_parents():
    for p in (ROOT, ROOT.parent, ROOT / "models", ROOT / "data", ROOT / "packaging" / "autostart",
              Path.home(), Path(Path.home().anchor)):
        assert A._unsafe_to_serve(p), p


def _safe_share(monkeypatch):
    """repo 與家目錄以外的資料夾。Windows 的 %TEMP% 位在家目錄內 → 測試改用假家目錄。"""
    fake_home = TMP / "fake_home"
    fake_home.mkdir(exist_ok=True)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    ok = TMP / "public_share"
    ok.mkdir(exist_ok=True)
    return ok


def test_dir_outside_repo_and_home_is_servable(monkeypatch):
    assert not A._unsafe_to_serve(_safe_share(monkeypatch))


def test_public_allowed_paths_only_keep_safe_entries(monkeypatch):
    ok = _safe_share(monkeypatch)
    monkeypatch.setenv("CLINSCOPE_ALLOWED_PATHS",
                       ";".join(map(str, [ROOT.parent, ROOT, Path.home(), Path.home() / "Desktop", ok])))
    extra = {str(A._SETUP_EXE)} if A._SETUP_EXE.exists() else set()
    assert set(A._allowed_paths()) == {str(ok)} | extra
    monkeypatch.delenv("CLINSCOPE_ALLOWED_PATHS")
    assert set(A._allowed_paths()) == extra          # 預設:只開桌面安裝檔(若已打包)


def test_public_gallery_uses_gradio_cache_copy_not_original():
    # 圖示檢視改交 Gradio 快取副本(內容雜湊命名),原資料夾不必、也不會進 allowed_paths
    src = make_png(TMP / "local_folder_img.png")
    S = A.new_state(cls_label())
    S["files"] = [str(src)]
    (p, _cap), = A.gallery_items(S, S["files"])
    assert Path(p).resolve().is_relative_to(A._GRADIO_TMP)
    assert Path(p).read_bytes() == src.read_bytes()


def _owner_state_with_result():
    img = make_png(TMP / "export_src.png")
    S = A.new_state(cls_label())
    S["files"] = S["page_files"] = [str(img)]
    S["results"] = {str(img): {"page": "x", "prob": 0.9, "ranked": [("x", 0.9)], "manual": False}}
    return S


def test_public_exports_not_written_into_repo_or_served_dir():
    # 舊版寫進 ROOT/exports(檔名可預測 + 被磁碟根 allowed_paths 涵蓋 → 匿名可猜檔名下載站主匯出)
    S = _owner_state_with_result()
    csv_path = Path(A.on_csv(cls_label(), S, fake_request(OWNER))["value"]).resolve()
    _modal, z = A.on_saveas_confirm_zip("c_n", S, fake_request(OWNER))
    zip_path = Path(z["value"]).resolve()
    for p in (csv_path, zip_path):
        assert p.is_file()
        assert not p.is_relative_to(ROOT), p
        assert not p.is_relative_to(A._GRADIO_TMP), p
