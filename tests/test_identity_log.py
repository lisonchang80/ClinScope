# -*- coding: utf-8 -*-
"""identity_debug.log:匿名載入頁面不寫檔(否則任何人狂打 on_identity 就能無限長檔);登入者照記,檔案有上限。"""
from conftest import A, cookie_for


def _size():
    return A._ID_LOG.stat().st_size if A._ID_LOG.exists() else 0


def test_anonymous_page_loads_do_not_write_log(site):
    before = _size()
    for _ in range(5):
        assert site.call("on_identity")["success"]
    assert _size() == before


def test_logged_in_identity_is_logged_with_size_cap(site, monkeypatch):
    before = _size()
    site.call("on_identity", cookie=cookie_for("owner"))
    assert _size() > before and "owner@example.com" in A._ID_LOG.read_text(encoding="utf-8")
    monkeypatch.setattr(A, "_ID_LOG_MAX_BYTES", _size())
    capped = _size()
    site.call("on_identity", cookie=cookie_for("guest"))
    assert _size() == capped
