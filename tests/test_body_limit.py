# -*- coding: utf-8 -*-
"""請求本文上限(審查 2026-11-07 GA-12):/auth/* 64KB、queue/join 等其他端點 8MB、上傳端點交給 max_file_size。

以前外層 FastAPI 只有 _OwnerOnlyUploads 等閘門:匿名對 /auth/google 送 8MB JSON,伺服器整包讀入並解析才回 401;
/gradio_api/queue/join 對匿名也是無上限的 JSON 解析。沒有 Content-Length 的 chunked 請求也要擋。
"""
import hashlib
import json
import os
import tempfile
import time

import httpx
import pytest

from conftest import A, TMP, cookie_for, make_png, start_site, upload

MB = 1024 * 1024


@pytest.fixture(scope="module")
def site():
    """本檔自己起一個站(同 launch_app 路徑):大本文、chunked、提早回 413 的連線不和其他測試共用同一個伺服器。"""
    return start_site()


def _paths_payload(names):
    """與 on_uploaded_paths 實際收到的同一種本文:前端 JSON.stringify(上傳路徑陣列) 塞進 Textbox,再包一層 queue/join。"""
    tmp = os.path.join(tempfile.gettempdir(), "gradio")
    paths = [os.path.join(tmp, hashlib.sha256(str(i).encode()).hexdigest(), n) for i, n in enumerate(names)]
    inner = json.dumps(paths, ensure_ascii=False)
    return json.dumps({"data": [inner, None], "event_data": None, "fn_index": 123, "trigger_id": 456,
                       "session_hash": "abcdefghij1"}, ensure_ascii=False).encode("utf-8")


def test_queue_limit_at_least_twice_measured_5000_paths():
    n = A.UPLOAD_MAX_FILES
    assert n == 5000
    sizes = {
        "typical": len(_paths_payload([f"IMG_20240315_143022_{i:04d}.jpg" for i in range(n)])),
        "long_cjk": len(_paths_payload([f"台大醫院檢驗科_血液抹片_顯微鏡400倍_病患編號A12345678_第{i:04d}張_重新拍攝版本_最終.tif"
                                        for i in range(n)])),
        "124cjk": len(_paths_payload([("顯" * 120) + f"{i:04d}.png" for i in range(n)])),
    }
    worst = max(sizes.values())
    assert worst < 2.6 * MB, sizes
    assert A.QUEUE_BODY_LIMIT == 8 * MB and A.QUEUE_BODY_LIMIT >= 2 * worst, sizes


def test_limits_by_path():
    assert A._body_limit("/auth/google") == 64 * 1024
    assert A._body_limit("/auth/logout") == 64 * 1024
    assert A._body_limit("/gradio_api/queue/join") == 8 * MB
    assert A._body_limit("/gradio_api/upload") is None
    assert A._body_limit("/gradio_api/process_recording") is None


def _chunks(total, size=64 * 1024):
    sent = 0
    while sent < total:
        n = min(size, total - sent)
        sent += n
        yield b" " * n


def test_auth_google_large_body_413_fast(site):
    t = time.monotonic()
    r = httpx.post(f"{site.base}/auth/google", content=b'{"credential":"' + b"a" * (8 * MB) + b'"}',
                   headers={"Content-Type": "application/json"}, timeout=30)
    assert r.status_code == 413, r.text[:200]
    assert time.monotonic() - t < 4


def test_auth_google_chunked_over_limit_413(site):
    r = httpx.post(f"{site.base}/auth/google", content=_chunks(200 * 1024),
                   headers={"Content-Type": "application/json"}, timeout=30)
    assert r.request.headers.get("transfer-encoding") == "chunked"
    assert r.status_code == 413, r.text[:200]


def test_auth_google_small_chunked_still_works(site):
    body = json.dumps({"credential": "x.y"}).encode()
    r = httpx.post(f"{site.base}/auth/google", content=iter([body[:5], body[5:]]),
                   headers={"Content-Type": "application/json"}, timeout=30)
    assert r.request.headers.get("transfer-encoding") == "chunked"
    assert r.status_code == 401, r.text[:200]


def _join(site, pad, cookie=None):
    fid = site.fn_index("on_disc")
    body = json.dumps({"data": [None] * site.n_inputs(fid), "fn_index": fid, "session_hash": "bodylimit",
                       "event_data": None, "trigger_id": None, "pad": "x" * pad}).encode()
    headers = {"Content-Type": "application/json"}
    if cookie:
        headers["Cookie"] = cookie
    return httpx.post(f"{site.base}/gradio_api/queue/join", content=body, headers=headers, timeout=30)


def test_queue_join_over_8mb_413_even_for_owner(site):
    assert _join(site, 9 * MB).status_code == 413
    assert _join(site, 9 * MB, cookie_for("owner")).status_code == 413


def test_queue_join_chunked_over_8mb_413(site):
    r = httpx.post(f"{site.base}/gradio_api/queue/join", content=_chunks(9 * MB, 256 * 1024),
                   headers={"Content-Type": "application/json"}, timeout=30)
    assert r.status_code == 413


def test_owner_5000_paths_payload_accepted(site):
    """站主分批上傳後送回 5,000 筆長中文檔名路徑(約 2.4MB)不被擋。"""
    names = [("顯" * 120) + f"{i:04d}.png" for i in range(5000)]
    tmp = os.path.join(tempfile.gettempdir(), "gradio")
    paths = json.dumps([os.path.join(tmp, hashlib.sha256(str(i).encode()).hexdigest(), n)
                        for i, n in enumerate(names)], ensure_ascii=False)
    fid = site.fn_index("on_uploaded_paths")
    vals = [None] * site.n_inputs(fid)
    vals[0] = paths
    body = json.dumps({"data": vals, "fn_index": fid, "session_hash": "bodylimit-owner",
                       "event_data": None, "trigger_id": None}, ensure_ascii=False).encode("utf-8")
    assert len(body) > 2 * MB
    r = httpx.post(f"{site.base}/gradio_api/queue/join", content=body,
                   headers={"Content-Type": "application/json", "Cookie": cookie_for("owner")}, timeout=30)
    assert r.status_code == 200, r.text[:200]


def test_queue_join_chunked_under_limit_ok(site):
    fid = site.fn_index("on_disc")
    body = json.dumps({"data": [None] * site.n_inputs(fid), "fn_index": fid, "session_hash": "bodylimit2",
                       "event_data": None, "trigger_id": None, "pad": "x" * (512 * 1024)}).encode()
    r = httpx.post(f"{site.base}/gradio_api/queue/join",
                   content=iter([body[i:i + 65536] for i in range(0, len(body), 65536)]),
                   headers={"Content-Type": "application/json"}, timeout=30)
    assert r.request.headers.get("transfer-encoding") == "chunked"
    assert r.status_code == 200, r.text[:200]


def test_upload_exempt_from_8mb(site):
    """上傳端點不受 8MB 限制(交給 max_file_size,UPLOAD_MAX_FILE_MB);非站主照樣被 _OwnerOnlyUploads 403。"""
    big = TMP / "body_limit_big.png"
    big.write_bytes(os.urandom(9 * MB))
    r = upload(site, [big], cookie_for("owner"))
    assert r.status_code == 200, r.text[:200]
    assert upload(site, [make_png(TMP / "body_limit_small.png")], None).status_code == 403
