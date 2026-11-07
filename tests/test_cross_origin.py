# -*- coding: utf-8 -*-
"""跨子網域防線(審查 2026-11-07 GA-3 / GA-8 / GA-10)。

- SameSite=Lax 擋不住同站(*.tinghaochang.com 兄弟子網域)送來的 cookie;Gradio 內建 CORS 在 Host 不是
  localhost 時對任何 Origin 都回 ACAO + credentials → 「只收 application/json 就會被預檢擋」的前提不成立。
  所以:非 GET/HEAD/OPTIONS 一律要同源(Sec-Fetch-Site same-origin/none;舊瀏覽器退回 Origin 對 Host),
  非同源的回應拿掉 ACAO / ACAC(連 GET 也讀不到),Secure 時 cookie 改名 __Host-cs_session。
- 所有回應帶防框架標頭(同站 iframe 會帶站主 cookie → 點擊劫持)。
- /auth/google 失敗時不把例外內文回給匿名呼叫者。"""
import httpx
import pytest

from conftest import A, OWNER, auth_client, cookie_for, fake_request

PUBLIC_HOST = "clinscope.tinghaochang.com"
EVIL = "https://evil.tinghaochang.com"


def _join(site, cookie=None, **headers):
    fid = site.fn_index("on_disc")
    if cookie:
        headers["Cookie"] = cookie
    return httpx.post(f"{site.base}/gradio_api/queue/join",
                      json={"data": [None] * site.n_inputs(fid), "fn_index": fid,
                            "session_hash": "xorigin", "event_data": None, "trigger_id": None},
                      headers=headers, timeout=10)


# ── GA-3 同源寫入防線 ──
@pytest.mark.parametrize("headers", [
    {"Sec-Fetch-Site": "same-site"},                       # 兄弟子網域(現代瀏覽器)
    {"Sec-Fetch-Site": "cross-site"},
    {"Origin": EVIL},                                      # 沒有 Fetch Metadata 的舊瀏覽器 → 比 Origin 與 Host
    {"Origin": "null"},
])
def test_cross_site_queue_join_with_owner_cookie_is_403(site, headers):
    r = _join(site, cookie=cookie_for("owner"), **headers)
    assert r.status_code == 403, r.text[:200]


def test_cross_site_upload_is_403_even_for_owner(site):
    ck = cookie_for("owner")
    r = httpx.post(f"{site.base}/gradio_api/upload", files=[("files", ("a.png", b"x", "image/png"))],
                   headers={"Cookie": ck, "Sec-Fetch-Site": "same-site"}, timeout=10)
    assert r.status_code == 403


def test_cross_site_logout_is_403_and_owner_stays_logged_in(site):
    tok = A.make_session(OWNER)
    r = httpx.post(f"{site.base}/auth/logout", json={},
                   headers={"Cookie": f"{A.COOKIE_NAME}={tok}", "Sec-Fetch-Site": "same-site"}, timeout=10)
    assert r.status_code == 403
    assert A._is_owner(fake_request(cookie_token=tok))


def test_cross_site_google_login_is_403_without_cookie(site):
    r = httpx.post(f"{site.base}/auth/google", json={"credential": "x"},
                   headers={"Sec-Fetch-Site": "same-site"}, timeout=10)
    assert r.status_code == 403
    assert "set-cookie" not in r.headers


@pytest.mark.parametrize("headers", [
    {"Sec-Fetch-Site": "same-origin"},
    {},                                                    # 非瀏覽器(curl/腳本)不會自動帶受害者 cookie → 放行
])
def test_same_origin_writes_still_work(site, headers):
    disc = A.disc_of(A.AVAILABLE[0]["key"])
    msg = site.call("on_disc", {0: disc}, headers=headers)
    assert msg["success"], msg


def test_same_origin_fallback_compares_origin_with_host(site):
    host = site.base.split("://", 1)[1]
    r = _join(site, Origin=site.base)
    assert r.status_code == 200
    r = _join(site, Origin=site.base.replace(host, "127.0.0.1.evil.example"))
    assert r.status_code == 403


# ── GA-3 非同源拿掉 ACAO / ACAC ──
@pytest.mark.parametrize("method,path", [
    ("OPTIONS", "/gradio_api/queue/join"),
    ("OPTIONS", "/gradio_api/upload"),
    ("OPTIONS", "/auth/google"),
    ("GET", "/config"),
    ("GET", "/gradio_api/info"),
])
def test_cross_origin_responses_carry_no_cors_grant(site, method, path):
    h = {"Host": PUBLIC_HOST, "Origin": EVIL}
    if method == "OPTIONS":
        h["Access-Control-Request-Method"] = "POST"
        h["Access-Control-Request-Headers"] = "content-type"
    r = httpx.request(method, site.base + path, headers=h, timeout=10)
    assert "access-control-allow-origin" not in r.headers, (path, dict(r.headers))
    assert "access-control-allow-credentials" not in r.headers, path


def test_same_origin_cors_headers_are_kept(site):
    r = httpx.get(site.base + "/config",
                  headers={"Host": PUBLIC_HOST, "Origin": f"https://{PUBLIC_HOST}"}, timeout=10)
    assert r.headers.get("access-control-allow-origin") == f"https://{PUBLIC_HOST}"


# ── GA-3 cookie 名稱 ──
def test_secure_cookie_uses_host_prefix_and_http_dev_keeps_old_name():
    assert A._cookie_name(True) == "__Host-cs_session"
    assert A._cookie_name(False) == "cs_session"
    assert A.COOKIE_SECURE and A.COOKIE_NAME == "__Host-cs_session"   # conftest 用預設(Secure)


def test_old_cookie_name_no_longer_grants_owner():
    req = fake_request()
    req.cookies["cs_session"] = A.make_session(OWNER)     # 兄弟子網域可用 Domain= 丟進來的那種
    assert not A._is_owner(req)


def test_logout_also_clears_legacy_cookie():
    tok = A.make_session(OWNER)
    r = auth_client().post("/auth/logout", json={},
                           headers={"Cookie": f"{A.COOKIE_NAME}={tok}; cs_session=old"})
    sc = [v.lower() for k, v in r.headers.multi_items() if k.lower() == "set-cookie"]
    assert any(v.startswith("__host-cs_session=") and "max-age=0" in v for v in sc), sc
    assert any(v.startswith("cs_session=") and "max-age=0" in v for v in sc), sc
    assert not any("domain=" in v for v in sc), "__Host- cookie 不可帶 Domain"


# ── GA-8 防框架標頭 ──
@pytest.mark.parametrize("path", ["/", "/config"])
def test_public_site_sends_anti_framing_headers(site, path):
    r = httpx.get(site.base + path, timeout=10)
    assert r.headers.get("x-frame-options") == "SAMEORIGIN"
    assert "frame-ancestors 'self'" in r.headers.get("content-security-policy", "")


# ── GA-10 登入失敗不回例外內文 ──
@pytest.mark.parametrize("cred", ["garbage", "a.b.c", ""])
def test_google_login_failure_returns_generic_error(cred):
    r = auth_client().post("/auth/google", json={"credential": cred})
    assert r.status_code == 401
    assert r.json() == {"error": "invalid credential"}
