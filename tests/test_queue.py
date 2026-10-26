# -*- coding: utf-8 -*-
"""濫用上限:佇列長度有上限(匿名洗大量事件不會無限堆積);批次/單張推論共用單一 GPU 槽;
其餘事件每個 listener 同時只跑 1 個(明寫在程式,不受 GRADIO_DEFAULT_CONCURRENCY_LIMIT 環境變數影響)。"""
from conftest import A


def _fns(site, *names):
    return [f for f in site.demo.fns.values() if getattr(f.fn, "__name__", None) in names]


def test_queue_is_bounded(site):
    q = site.demo._queue
    assert q.max_size is not None and 0 < q.max_size <= A.QUEUE_MAX_SIZE
    assert q.default_concurrency_limit == 1


def test_batch_and_single_inference_share_one_gpu_slot(site):
    fns = _fns(site, "on_batch", "on_single")
    assert len(fns) == 2
    assert {f.concurrency_id for f in fns} == {"gpu"}
    assert all(f.concurrency_limit == 1 for f in fns)


def test_queue_full_is_rejected_not_queued(site, monkeypatch):
    import httpx
    monkeypatch.setattr(site.demo._queue, "max_size", 0)
    fid = next(i for i, f in site.demo.fns.items() if getattr(f.fn, "__name__", None) == "on_view")
    r = httpx.post(f"{site.base}/gradio_api/queue/join",
                   json={"data": [None, None], "fn_index": fid, "session_hash": "qfull"}, timeout=10)
    assert r.status_code == 503
