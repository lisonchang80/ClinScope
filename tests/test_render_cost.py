# -*- coding: utf-8 -*-
"""渲染與打包的成本(審查 2026-11-07 CS-7 / CS-8)。

- CS-7:公開站圖示檢視每次 render 都對目前分頁每個檔做 save_file_to_cache(整檔 SHA-256 + 複製)。
  改成以 (路徑, 大小, mtime) 記住快取副本,命中且副本還在就直接用。
- CS-8:打包 ZIP 對 JPEG/PNG 這類已壓縮格式用 DEFLATE 只是白花 CPU → 改 STORED;BMP/TIFF 照壓。"""
import os
import time
import zipfile
from pathlib import Path

import pytest

from conftest import A, OWNER, TMP, cls_label, fake_request, make_png


@pytest.fixture
def cache_calls(monkeypatch):
    from gradio import processing_utils as pu
    real, calls = pu.save_file_to_cache, []

    def _spy(path, cache_dir):
        calls.append(str(path))
        return real(path, cache_dir)

    monkeypatch.setattr(pu, "save_file_to_cache", _spy)
    A._SERVABLE_CACHE.clear()
    return calls


def test_servable_reuses_cached_copy(cache_calls):
    src = make_png(TMP / "servable_a.png", (10, 20, 30))
    first = A._servable(str(src))
    for _ in range(3):
        assert A._servable(str(src)) == first
    assert len(cache_calls) == 1, "同一個未變動的檔不可每次 render 都重 hash"
    assert Path(first).resolve().is_relative_to(A._GRADIO_TMP)


def test_servable_recomputes_when_cached_copy_was_deleted(cache_calls):
    src = make_png(TMP / "servable_b.png", (40, 50, 60))
    first = A._servable(str(src))
    os.remove(first)                                    # delete_cache 清掉了
    again = A._servable(str(src))
    assert len(cache_calls) == 2 and Path(again).is_file()


def test_servable_recomputes_when_source_changes(cache_calls):
    src = TMP / "servable_c.png"
    make_png(src, (1, 1, 1))
    first = A._servable(str(src))
    time.sleep(0.02)
    make_png(src, (200, 1, 1))
    st = src.stat()
    os.utime(src, ns=(st.st_atime_ns, st.st_mtime_ns + 10_000_000))
    second = A._servable(str(src))
    assert len(cache_calls) == 2 and second != first
    assert Path(second).read_bytes() == src.read_bytes()


def test_servable_cache_is_bounded(cache_calls, monkeypatch):
    monkeypatch.setattr(A, "_SERVABLE_MAX", 3)
    for i in range(6):
        A._servable(str(make_png(TMP / f"servable_lru_{i}.png", (i, 2, 3))))
    assert len(A._SERVABLE_CACHE) == 3


def test_gallery_render_twice_hashes_once(cache_calls):
    S = A.new_state(cls_label())
    S["files"] = [str(make_png(TMP / f"servable_g_{i}.png", (i, 7, 7))) for i in range(4)]
    A.gallery_items(S, S["files"])
    A.gallery_items(S, S["files"])
    assert len(cache_calls) == 4


def test_zip_stores_already_compressed_images_and_deflates_bitmaps():
    from PIL import Image
    files = [make_png(TMP / "zip_a.png"), TMP / "zip_b.jpg", TMP / "zip_c.bmp"]
    Image.new("RGB", (16, 16), (1, 2, 3)).save(files[1])
    Image.new("RGB", (16, 16), (4, 5, 6)).save(files[2])
    S = A.new_state(cls_label())
    S["files"] = S["page_files"] = [str(p) for p in files]
    S["results"] = {str(p): {"page": "x", "prob": 0.9, "ranked": [("x", 0.9)], "manual": False}
                    for p in files}
    _modal, z = A.on_saveas_confirm_zip("c_n", S, fake_request(OWNER))
    with zipfile.ZipFile(z["value"]) as zf:
        kinds = {Path(i.filename).suffix: i.compress_type for i in zf.infolist()}
        assert zf.testzip() is None
    assert kinds == {".png": zipfile.ZIP_STORED, ".jpg": zipfile.ZIP_STORED, ".bmp": zipfile.ZIP_DEFLATED}
