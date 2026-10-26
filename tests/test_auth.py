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


# ── cookie 簽章金鑰:不可用公開 repo 裡的預設值 ──
def _forge(secret):
    from itsdangerous import URLSafeTimedSerializer
    payload = A._serializer().loads(A.make_session(OWNER))      # 與 app 目前的 cookie 格式一致
    return URLSafeTimedSerializer(secret, salt="clinscope-session").dumps(payload)


def test_cookie_forged_with_public_default_secret_is_rejected():
    # conftest 刻意不設 CLINSCOPE_SESSION_SECRET(= 正式站忘了設 env.local.bat 的情況)
    assert not A._is_owner(fake_request(cookie_token=_forge("dev-insecure-change-me")))


def test_session_secret_resolution():
    strong = "s" * 40
    assert A._session_secret(strong) == strong
    for weak in (None, "", "dev-insecure-change-me", "short-secret"):
        got = A._session_secret(weak)
        assert got != weak and len(got) >= 32
    assert A._session_secret(None) != A._session_secret(None)   # 每次啟動隨機


# ── dev/桌面放行捷徑在公開模式一定關閉 ──
def test_force_owner_env_is_ignored_on_public_site(monkeypatch):
    monkeypatch.setenv("CLINSCOPE_FORCE_OWNER", "1")
    assert not A._is_owner(fake_request())
    assert not A._is_owner(None)


def test_desktop_mode_is_single_user_owner(monkeypatch):
    monkeypatch.setattr(A, "PUBLIC", False)
    assert A._is_owner(None)


def test_owner_email_compare_is_case_and_space_insensitive():
    assert A._is_owner(fake_request(email="  Owner@Example.COM "))
    assert not A._is_owner(fake_request(email="owner@example.com.evil.test"))
    assert not A._is_owner(fake_request(email=""))


def test_site_login_cookie_ignored_when_site_login_disabled(monkeypatch):
    # 拿掉 CLINSCOPE_GOOGLE_CLIENT_ID = 公開站沒有站主(全員唯讀);之前發出的 cookie 不能繼續當站主
    monkeypatch.setattr(A, "AUTH_ENABLED", False)
    assert not A._is_owner(fake_request(email=OWNER))
