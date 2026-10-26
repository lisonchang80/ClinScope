# -*- coding: utf-8 -*-
"""Google ID token 驗證:用測試自產的 RSA 金鑰真的簽 JWT,只把「抓 Google 公鑰」那一步換成假 transport,
其餘(簽章 / aud / iss / exp)走 google-auth 真實流程。不連網、不含任何真 token。"""
import json
import time
import types

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from conftest import A, CLIENT_ID, OWNER, auth_client

KID = "test-kid"


def _keypair():
    k = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    priv = k.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                           serialization.NoEncryption()).decode()
    pub = k.public_key().public_bytes(serialization.Encoding.PEM,
                                      serialization.PublicFormat.SubjectPublicKeyInfo).decode()
    return priv, pub


PRIV, PUB = _keypair()
OTHER_PRIV, _ = _keypair()


@pytest.fixture(autouse=True)
def fake_google_certs(monkeypatch):
    """google.oauth2.id_token 以 request(certs_url) 取公鑰 → 回我們的測試公鑰。"""
    from google.auth.transport import requests as g_requests

    class _Req:
        def __call__(self, url, method="GET", **kw):
            return types.SimpleNamespace(status=200, data=json.dumps({KID: PUB}).encode())

    monkeypatch.setattr(g_requests, "Request", _Req)


def _token(priv=PRIV, **over):
    from google.auth import crypt, jwt
    now = int(time.time())
    claims = {"iss": "https://accounts.google.com", "aud": CLIENT_ID, "sub": "1234567890",
              "email": OWNER, "email_verified": True, "iat": now, "exp": now + 3600}
    claims.update(over)
    claims = {k: v for k, v in claims.items() if v is not None}
    return jwt.encode(crypt.RSASigner.from_string(priv, key_id=KID), claims).decode()


def test_valid_token_accepted():
    assert A.verify_google_credential(_token())["email"] == OWNER


@pytest.mark.parametrize("why, tok", [
    ("aud 不是本站 client id", lambda: _token(aud="someone-else.apps.googleusercontent.com")),
    ("iss 不是 Google", lambda: _token(iss="https://evil.example.com")),
    ("已過期", lambda: _token(iat=int(time.time()) - 7200, exp=int(time.time()) - 3600)),
    ("別把金鑰簽的", lambda: _token(priv=OTHER_PRIV)),
    ("email 未驗證", lambda: _token(email_verified=False)),
    ("缺 email_verified", lambda: _token(email_verified=None)),
    ("缺 email", lambda: _token(email=None)),
])
def test_bad_tokens_rejected(why, tok):
    with pytest.raises(Exception):
        A.verify_google_credential(tok())


def test_auth_route_sets_hardened_cookie_and_normalizes_email():
    c = auth_client()
    r = c.post("/auth/google", json={"credential": _token(email="  Owner@Example.COM")})
    assert r.status_code == 200 and r.json() == {"email": OWNER, "owner": True}
    sc = r.headers["set-cookie"].lower()
    for flag in ("httponly", "secure", "samesite=lax", "path=/", "max-age="):
        assert flag in sc, flag
    tok = r.cookies.get(A.COOKIE_NAME)
    assert A._is_owner(types.SimpleNamespace(headers={}, cookies={A.COOKIE_NAME: tok}))


def test_auth_route_rejects_unverified_email_without_cookie():
    r = auth_client().post("/auth/google", json={"credential": _token(email_verified=False)})
    assert r.status_code == 401
    assert A.COOKIE_NAME not in r.headers.get("set-cookie", "")
