# -*- coding: utf-8 -*-
"""非公開模式(桌面/dev,沒設 CLINSCOPE_PUBLIC)只服務本機(審查 2026-11-07 CS-1 / GA-1)。

共用 cloudflared tunnel 常駐、ingress 把 clinscope.tinghaochang.com 轉到 localhost:7860,前面沒有
Cloudflare Access。非公開模式 allowed_paths=各磁碟根、_is_owner 一律 True → 只要非公開實例被 tunnel
轉到,網路上任何人都是站主、可 GET /gradio_api/file=C:/... 讀整顆磁碟。
防線:非公開模式掛 _LocalOnly(帶 Cloudflare 標頭或 Host 非本機 → 403 / websocket 1008);
另把非公開預設埠移出 tunnel 入口(7862)。公開模式不掛(公開站本來就要接 tunnel 流量)。"""
import asyncio
from pathlib import Path

import httpx
import pytest

from conftest import A, _free_port

TUNNEL_HOST = "clinscope.tinghaochang.com"


# ── 直接對中介層送 ASGI scope ──
def _run(scope):
    calls, sent = [], []

    async def inner(scope, receive, send):
        calls.append(scope["type"])
        if scope["type"] == "http":
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(msg):
        sent.append(msg)

    asyncio.run(A._LocalOnly(inner)(scope, receive, send))
    return calls, sent


def _scope(kind="http", host="127.0.0.1:7862", headers=()):
    hs = [(b"host", host.encode("latin-1"))] if host is not None else []
    hs += [(k.encode("latin-1"), v.encode("latin-1")) for k, v in headers]
    s = {"type": kind, "path": "/gradio_api/file=C:/Windows/win.ini", "headers": hs,
         "query_string": b"", "client": ("127.0.0.1", 50000), "server": ("127.0.0.1", 7862)}
    if kind == "http":
        s["method"] = "GET"
    return s


def _status(sent):
    return next(m["status"] for m in sent if m["type"] == "http.response.start")


def _body(sent):
    return b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")


@pytest.mark.parametrize("host,headers", [
    (TUNNEL_HOST, ()),
    (TUNNEL_HOST + ":443", ()),
    ("scopelab.tinghaochang.com", ()),
    ("127.0.0.1:7862", [("cf-connecting-ip", "203.0.113.9")]),
    ("127.0.0.1:7862", [("cf-ray", "8f0000000000abcd-TPE")]),
    ("127.0.0.1:7862", [("cf-visitor", '{"scheme":"https"}')]),
    ("127.0.0.1:7862", [("cdn-loop", "cloudflare")]),
    ("localhost", [("CDN-Loop", "CloudFlare; loops=1")]),
    ("127.0.0.1.nip.io:7862", ()),          # DNS rebinding 類
    ("localhost.evil.example", ()),
    ("[::1].evil.example", ()),
    (None, ()),                              # 沒有 Host → fail closed
])
def test_http_from_tunnel_or_foreign_host_is_403(host, headers):
    calls, sent = _run(_scope("http", host, headers))
    assert calls == [], "被擋的請求不得往下送"
    assert _status(sent) == 403
    assert "CLINSCOPE_PUBLIC=1" in _body(sent).decode("utf-8")


@pytest.mark.parametrize("host", ["127.0.0.1:7862", "127.0.0.1", "localhost", "LOCALHOST:7862",
                                  "[::1]:7862", "[::1]", "::1"])
def test_local_host_passes(host):
    calls, sent = _run(_scope("http", host))
    assert calls == ["http"]
    assert _status(sent) == 200


def test_websocket_from_tunnel_is_closed_1008():
    calls, sent = _run(_scope("websocket", TUNNEL_HOST))
    assert calls == []
    assert sent == [{"type": "websocket.close", "code": 1008}]
    calls, sent = _run(_scope("websocket", "127.0.0.1:7862", [("cf-ray", "x")]))
    assert calls == [] and sent == [{"type": "websocket.close", "code": 1008}]


def test_websocket_local_passes():
    calls, _ = _run(_scope("websocket", "[::1]:7862"))
    assert calls == ["websocket"]


def test_lifespan_passes_through():
    calls, _ = _run({"type": "lifespan"})
    assert calls == ["lifespan"]


# ── launch_app 各啟動路徑都掛上 / 預設埠 ──
class _FakeDemo:
    def __init__(self):
        self.kw = None

    def launch(self, **kw):
        self.kw = kw
        return self


def _mw_classes(kw):
    return [m.cls for m in (kw.get("app_kwargs") or {}).get("middleware") or []]


def _fake_launch(monkeypatch, public, env_port=None, **launch_kw):
    monkeypatch.setattr(A, "PUBLIC", public)
    monkeypatch.setattr(A, "AUTH_ENABLED", False)
    monkeypatch.setattr(A, "_precompute_sample", lambda: None)
    if env_port is None:
        monkeypatch.delenv("GRADIO_SERVER_PORT", raising=False)
    else:
        monkeypatch.setenv("GRADIO_SERVER_PORT", str(env_port))
    demo = _FakeDemo()
    monkeypatch.setattr(A, "build", lambda: demo)
    A.launch_app(prevent_thread_lock=True, **launch_kw)
    return demo.kw


def test_desktop_launch_mounts_local_only_and_defaults_off_tunnel_port(monkeypatch):
    kw = _fake_launch(monkeypatch, public=False)
    # 本機限定在最外層;防框架 + 同源寫入防線(GA-8 / GA-3)接在後面
    assert _mw_classes(kw) == [A._LocalOnly, A._FrameGuard, A._SameOriginWriteGuard]
    assert kw["server_port"] == 7862, "非公開沒指定埠時不可落在 tunnel 入口 7860"


def test_desktop_launch_port_overrides(monkeypatch):
    assert _fake_launch(monkeypatch, public=False, env_port=7899)["server_port"] == 7899
    assert _fake_launch(monkeypatch, public=False, server_port=7898)["server_port"] == 7898


def test_desktop_launch_keeps_caller_middleware(monkeypatch):
    from starlette.middleware import Middleware

    class _Other:
        def __init__(self, app):
            self.app = app

    kw = _fake_launch(monkeypatch, public=False, app_kwargs={"middleware": [Middleware(_Other)],
                                                              "docs_url": None})
    assert _mw_classes(kw) == [A._LocalOnly, A._FrameGuard, A._SameOriginWriteGuard, _Other], \
        "本機限定要在最外層,且不可蓋掉呼叫端的中介層"
    assert kw["app_kwargs"]["docs_url"] is None


def test_public_launch_unchanged(monkeypatch):
    kw = _fake_launch(monkeypatch, public=True)
    assert _mw_classes(kw) == [A._FrameGuard, A._SameOriginWriteGuard, A._BodySizeLimit, A._OwnerOnlyUploads], \
        "公開站要接 tunnel 流量,不可掛本機限定"
    assert "server_port" not in kw, "公開站沿用啟動器的 GRADIO_SERVER_PORT=7860,不可搬家"


def test_uvicorn_path_mounts_local_only_outermost_when_not_public(monkeypatch):
    # 目前 AUTH_ENABLED 必為 PUBLIC,這條是縱深:將來若拆開,非公開的 uvicorn 路徑也要擋
    import uvicorn
    monkeypatch.setattr(A, "PUBLIC", False)
    monkeypatch.setattr(A, "AUTH_ENABLED", True)
    monkeypatch.setattr(A, "build", lambda: object())
    monkeypatch.setattr(A.gr, "mount_gradio_app", lambda app, demo, **k: app)

    class _FakeServer:
        def __init__(self, config):
            self.config = config

        def run(self):
            pass

    monkeypatch.setattr(uvicorn, "Server", _FakeServer)
    mounted = A.launch_app(prevent_thread_lock=False, server_port=_free_port())
    assert mounted.user_middleware[0].cls is A._LocalOnly
    assert A._OwnerOnlyUploads in [m.cls for m in mounted.user_middleware]


def test_uvicorn_public_path_stack_order(monkeypatch):
    # 外層 FastAPI 包住掛在 "/" 的 Gradio:拿掉 ACAO 的那層一定要在 Gradio 自己的 CORS 外面(GA-3)
    import uvicorn
    monkeypatch.setattr(A, "PUBLIC", True)
    monkeypatch.setattr(A, "AUTH_ENABLED", True)
    monkeypatch.setattr(A, "_precompute_sample", lambda: None)
    monkeypatch.setattr(A, "build", lambda: object())
    monkeypatch.setattr(A.gr, "mount_gradio_app", lambda app, demo, **k: app)

    class _FakeServer:
        def __init__(self, config):
            self.config = config

        def run(self):
            pass

    monkeypatch.setattr(uvicorn, "Server", _FakeServer)
    mounted = A.launch_app(prevent_thread_lock=False, server_port=_free_port())
    assert [m.cls for m in mounted.user_middleware] == [
        A._FrameGuard, A._StripCrossOriginCors, A._SameOriginWriteGuard, A._BodySizeLimit,   # 本文上限(審查 GA-12)
        A._OwnerOnlyUploads]


# ── 真的起一個非公開實例(隨機埠),用 tunnel 會送的 Host 打 ──
@pytest.fixture
def desktop_site(monkeypatch):
    monkeypatch.setattr(A, "PUBLIC", False)
    monkeypatch.setattr(A, "AUTH_ENABLED", False)
    monkeypatch.delenv("CLINSCOPE_ALLOWED_PATHS", raising=False)
    holder = {}
    orig_build = A.build

    def _capture():
        holder["demo"] = orig_build()
        return holder["demo"]

    monkeypatch.setattr(A, "build", _capture)
    port = _free_port()
    assert port not in (7860, 7861)
    A.launch_app(prevent_thread_lock=True, server_port=port, quiet=True)
    base = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            if httpx.get(base + "/", timeout=2).status_code < 500:
                break
        except Exception:
            import time
            time.sleep(0.1)
    try:
        yield base
    finally:
        holder["demo"].close()


WIN_INI = Path("C:/Windows/win.ini")


def test_live_desktop_mode_blocks_tunnel_requests(desktop_site):
    url = f"{desktop_site}/gradio_api/file={WIN_INI.as_posix()}"
    r = httpx.get(url, headers={"Host": TUNNEL_HOST}, timeout=10)
    assert r.status_code == 403, r.text[:200]
    r = httpx.get(url, headers={"cf-connecting-ip": "203.0.113.9", "cf-ray": "abc-TPE"}, timeout=10)
    assert r.status_code == 403
    r = httpx.get(f"{desktop_site}/", headers={"Host": TUNNEL_HOST}, timeout=10)
    assert r.status_code == 403
    # 本機照常可用
    assert httpx.get(f"{desktop_site}/", timeout=10).status_code == 200
    if WIN_INI.exists():   # 證明上面的 403 來自本機限定(本機 Host 下桌面模式本來就開放磁碟根)
        assert httpx.get(url, timeout=10).status_code == 200


def test_live_desktop_mode_frame_and_cross_site_guards(desktop_site):
    # 桌面版 Host 是本機 → _LocalOnly 放行;但使用者瀏覽器裡的惡意網頁仍可對 127.0.0.1 發 POST(GA-3 / GA-8)
    r = httpx.get(f"{desktop_site}/", timeout=10)
    assert r.headers.get("x-frame-options") == "SAMEORIGIN"
    assert "frame-ancestors 'self'" in r.headers.get("content-security-policy", "")
    r = httpx.post(f"{desktop_site}/gradio_api/queue/join", json={"data": [], "fn_index": 0, "session_hash": "x"},
                   headers={"Sec-Fetch-Site": "cross-site", "Origin": "https://evil.example"}, timeout=10)
    assert r.status_code == 403
