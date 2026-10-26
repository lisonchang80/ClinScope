# -*- coding: utf-8 -*-
"""身分:站主判定、Cloudflare 標頭、cookie 簽章、放行捷徑、ID token 驗證、登出。"""
from conftest import A, OWNER, fake_request, owner_cookie_header

CF_HEADER = "Cf-Access-Authenticated-User-Email"


def _bar(msg):
    assert msg["success"], msg
    return msg["output"]["data"][0]


# ── Cloudflare Access 標頭(Access 已改 Bypass → 任何人都能自帶)──
def test_cf_access_header_alone_does_not_grant_owner():
    assert not A._is_owner(fake_request(headers={CF_HEADER.lower(): OWNER}))
    assert not A._is_owner(fake_request(headers={CF_HEADER: OWNER}))


def test_cf_access_header_spoof_rejected_end_to_end(site):
    assert "訪客" in _bar(site.call("on_identity", headers={CF_HEADER: OWNER}))
    assert "已登入站主" in _bar(site.call("on_identity", cookie=owner_cookie_header()))


def test_cf_access_header_only_with_explicit_opt_in_and_no_site_login(monkeypatch):
    monkeypatch.setattr(A, "TRUST_CF_ACCESS_HEADER", True)
    assert not A._is_owner(fake_request(headers={CF_HEADER: OWNER}))   # 站內登入啟用時仍不信任
    monkeypatch.setattr(A, "AUTH_ENABLED", False)
    assert A._is_owner(fake_request(headers={CF_HEADER: OWNER}))       # 舊「Access 強制登入」部署才用
