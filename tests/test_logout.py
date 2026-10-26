# -*- coding: utf-8 -*-
"""登出必須在伺服器端讓該 session 失效(不只是請瀏覽器刪 cookie):被複製走的 cookie 值也不能再用。"""
from itsdangerous import URLSafeTimedSerializer

from conftest import A, OWNER, auth_client, fake_request


def _logout(tok):
    return auth_client().post("/auth/logout", json={},
                              headers={"Cookie": f"{A.COOKIE_NAME}={tok}"})


def test_logout_revokes_session_server_side():
    tok = A.make_session(OWNER)
    assert A._is_owner(fake_request(cookie_token=tok))
    r = _logout(tok)
    assert r.status_code == 200
    sc = r.headers["set-cookie"].lower()
    assert A.COOKIE_NAME in sc and "max-age=0" in sc and "secure" in sc and "httponly" in sc
    assert not A._is_owner(fake_request(cookie_token=tok))


def test_logout_only_revokes_that_session():
    a, b = A.make_session(OWNER), A.make_session(OWNER)
    _logout(a)
    assert not A._is_owner(fake_request(cookie_token=a))
    assert A._is_owner(fake_request(cookie_token=b))


def test_revocation_survives_server_restart():
    tok = A.make_session(OWNER)
    _logout(tok)
    A._REVOKED.clear()
    A._REVOKED.update(A._load_revoked())          # 模擬重啟:只靠持久化檔還原
    assert not A._is_owner(fake_request(cookie_token=tok))


def test_legacy_cookie_without_session_id_rejected():
    legacy = URLSafeTimedSerializer(A.SESSION_SECRET, salt="clinscope-session").dumps({"email": OWNER})
    assert not A._is_owner(fake_request(cookie_token=legacy))
