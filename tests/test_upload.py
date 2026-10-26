# -*- coding: utf-8 -*-
"""分批上傳:公開站只有站主會上傳 → Gradio 的上傳端點對非站主一律 403;單檔大小、單次總量/檔數上限;
暫存檔定期清理;on_uploaded_paths 只收 Gradio 上傳暫存根內的影像(擋路徑注入)。"""
import json
import os

import httpx
import pytest

from conftest import A, OWNER, TMP, VISITORS, cookie_for, fake_request, make_png, start_site, upload


@pytest.mark.parametrize("who", VISITORS)
def test_visitors_cannot_upload(site, who):
    r = upload(site, [make_png(TMP / "visitor_up.png")], cookie_for(who))
    assert r.status_code == 403


@pytest.mark.parametrize("who", VISITORS)
def test_visitors_cannot_post_recordings(site, who):
    ck = cookie_for(who)
    r = httpx.post(f"{site.base}/gradio_api/process_recording",
                   files=[("video", ("x.mp4", b"\0" * 1024, "video/mp4"))],
                   headers={"Cookie": ck} if ck else {}, timeout=30)
    assert r.status_code == 403


def test_owner_can_upload(site):
    r = upload(site, [make_png(TMP / "owner_up.png")], cookie_for("owner"))
    assert r.status_code == 200 and len(r.json()) == 1


def test_single_file_size_limit(monkeypatch):
    monkeypatch.setattr(A, "UPLOAD_MAX_FILE_MB", 1)
    s = start_site()
    big = TMP / "big.png"
    big.write_bytes(os.urandom(2 * 1048576))
    assert upload(s, [big], cookie_for("owner")).status_code == 413


def test_uploaded_temp_files_are_cleaned_periodically(site):
    freq, age = site.demo.delete_cache
    assert 0 < freq <= 3600 and 0 < age <= 2 * 86400


def _in_tmp(name, color=(1, 2, 3)):
    d = A._GRADIO_TMP / "unit"
    d.mkdir(parents=True, exist_ok=True)
    return str(make_png(d / name, color))


def test_uploaded_paths_rejects_injection():
    good = _in_tmp("ok.png")
    outside = str(make_png(TMP / "outside.png"))
    txt = A._GRADIO_TMP / "unit" / "notes.txt"
    txt.write_text("x")
    raw = [good, outside, str(A._GRADIO_TMP / ".." / "outside.png"), str(txt), 123, None,
           str(A.ROOT / "app.py")]
    S = A.new_state(A.AVAILABLE[0]["label"])
    out = A.on_uploaded_paths(json.dumps(raw), S, fake_request(OWNER))
    assert out[0]["files"] == [good]


def test_uploaded_paths_count_and_total_caps(monkeypatch):
    files = [_in_tmp(f"cap_{i}.png", (i, i, i)) for i in range(3)]
    S = A.new_state(A.AVAILABLE[0]["label"])
    monkeypatch.setattr(A, "UPLOAD_MAX_FILES", 2)
    out = A.on_uploaded_paths(json.dumps(files), S, fake_request(OWNER))
    assert len(out[0]["files"]) == 2 and "略過 1" in out[6]
    monkeypatch.setattr(A, "UPLOAD_MAX_FILES", 100)
    one = os.path.getsize(files[0])
    monkeypatch.setattr(A, "UPLOAD_MAX_TOTAL_MB", 1.5 * one / 1048576)
    out = A.on_uploaded_paths(json.dumps(files), S, fake_request(OWNER))
    assert len(out[0]["files"]) == 1 and "略過 2" in out[6]


def test_public_site_without_site_login_also_blocks_uploads(monkeypatch):
    # 未設 CLINSCOPE_GOOGLE_CLIENT_ID 的公開站走 demo.launch() 分支,中介層經 app_kwargs 掛上
    monkeypatch.setattr(A, "AUTH_ENABLED", False)
    s = start_site()
    assert upload(s, [make_png(TMP / "nologin_up.png")]).status_code == 403
