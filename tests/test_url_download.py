# -*- coding: utf-8 -*-
"""公開站不替任何人下載外部 URL(審查 2026-11-07 CS-2)。

Gradio preprocess 對每個非 State 輸入都跑 async_move_files_to_cache:值長得像 FileData、path 是 http URL
就直接 async_ssrf_protected_download(整個回應串流寫進快取、沒有大小上限、query 不同就不去重),
而且發生在事件函式(_guard_owner)之前。本站沒有任何事件需要伺服器端抓 URL → 公開模式整條關掉。
這裡把真正連網的那一步(async_ssrf_protected_get)換成計次假函式,不連外。"""
import types
import uuid

import httpx
import pytest

from conftest import A, VISITORS, cookie_for


@pytest.fixture
def fetches(monkeypatch):
    from gradio import processing_utils as pu
    calls = []

    async def _fake_get(url):
        calls.append(url)

        async def _chunks():
            yield b"x" * 1024

        return types.SimpleNamespace(status_code=200, has_redirect_location=False,
                                     aiter_bytes=_chunks, url=url, headers={})

    monkeypatch.setattr(pu, "async_ssrf_protected_get", _fake_get)
    return calls


def _file_url(i):
    # query 每次不同(攻擊情境)→ Gradio 以 hash_url 命名的快取夾不會命中,一定會去抓
    return {"path": f"https://example.invalid/huge.iso?n={i}-{uuid.uuid4().hex}", "url": None,
            "meta": {"_type": "gradio.FileData"}}


@pytest.mark.parametrize("who", VISITORS + ["owner"])
def test_filedata_url_inputs_never_trigger_server_downloads(site, fetches, who):
    ck = cookie_for(who)
    for i, name in enumerate(["on_disc", "on_view", "on_page"]):
        site.call(name, {0: _file_url(i)}, cookie=ck)
    assert fetches == [], "公開站不得替呼叫者下載外部 URL"


def test_download_functions_are_really_replaced(site):
    # Gradio 升級改名時要在啟動就炸(_disable_url_downloads 會 raise),不能靜默失效
    from gradio import blocks, processing_utils as pu
    assert blocks.processing_utils is pu
    for name in ("async_ssrf_protected_download", "ssrf_protected_download", "save_url_to_cache"):
        assert getattr(pu, name) in (A._refuse_url_download, A._refuse_url_download_async), name


def test_disable_url_downloads_fails_loudly_if_gradio_renames(monkeypatch):
    from gradio import processing_utils as pu
    monkeypatch.delattr(pu, "async_ssrf_protected_download")
    with pytest.raises(RuntimeError):
        A._disable_url_downloads()


@pytest.mark.parametrize("who", VISITORS)
def test_visitors_cannot_post_component_server(site, who):
    ck = cookie_for(who)
    r = httpx.post(f"{site.base}/gradio_api/component_server",
                   files=[("data", ("x.bin", b"\0" * 4096, "application/octet-stream"))],
                   headers={"Cookie": ck} if ck else {}, timeout=30)
    assert r.status_code == 403
