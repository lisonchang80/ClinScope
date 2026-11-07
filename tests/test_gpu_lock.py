# -*- coding: utf-8 -*-
"""GPU 推論同時只跑一份(審查 2026-11-07 CS-6)。

on_batch / on_single 共用 concurrency_id="gpu",但點圖偵測(on_select / on_bridge → _select_core → det_infer)
不在那個槽裡 → 批次跑到一半點縮圖就會再起一份推論。修法是推論呼叫本身包一把模組層級鎖:
只有真正的推論序列化,訪客點選、改分類這些不推論的事件不受影響。"""
import threading
import time

from conftest import A, OWNER, TMP, cls_label, det_label, fake_request, make_png


class _Probe:
    def __init__(self):
        self.active = 0
        self.peak = 0
        self.calls = 0
        self._m = threading.Lock()

    def enter(self):
        with self._m:
            self.active += 1
            self.calls += 1
            self.peak = max(self.peak, self.active)
        time.sleep(0.12)
        with self._m:
            self.active -= 1


def _state(label, n=3):
    S = A.new_state(label)
    S["files"] = S["page_files"] = [str(make_png(TMP / f"gpu_{label[:2]}_{i}.png", (i * 40, 9, 9)))
                                    for i in range(n)]
    return S


def _run_together(*fns):
    ts = [threading.Thread(target=f) for f in fns]
    for t in ts:
        t.start()
        time.sleep(0.03)
    for t in ts:
        t.join(20)


def test_click_detection_waits_for_running_batch(monkeypatch):
    probe = _Probe()

    def det(image, spec, conf, progress=None):
        import numpy as np
        probe.enter()
        return {"obj": 1}, np.zeros((8, 8, 3), dtype=np.uint8), 1

    monkeypatch.setattr(A, "det_infer", det)
    lab, owner = det_label(), fake_request(OWNER)
    S1, S2 = _state(lab), _state(lab)
    noprog = lambda *a, **k: None   # noqa: E731

    _run_together(lambda: A.on_batch(lab, 0.25, S1, progress=noprog, request=owner),
                  lambda: [A._select_core(S2, lab, 0.25, i, progress=noprog, request=owner) for i in range(3)],
                  lambda: A.on_single(lab, 0.25, {**_state(lab, 1), "selected": S1["files"][0]},
                                      progress=noprog, request=owner))
    assert probe.calls == 3 + 3 + 1
    assert probe.peak == 1, "同時進入 GPU 推論的數量必須是 1"


def test_classification_batch_and_single_share_the_lock(monkeypatch):
    probe = _Probe()

    def cls(image, spec, progress=None):
        probe.enter()
        return [("a", 0.9), ("b", 0.1)]

    monkeypatch.setattr(A, "cls_rank", cls)
    lab, owner = cls_label(), fake_request(OWNER)
    S1, S2 = _state(lab), _state(lab, 1)
    S2["selected"] = S2["files"][0]
    noprog = lambda *a, **k: None   # noqa: E731
    _run_together(lambda: A.on_batch(lab, 0.5, S1, progress=noprog, request=owner),
                  lambda: A.on_single(lab, 0.5, S2, progress=noprog, request=owner))
    assert probe.calls == 4 and probe.peak == 1


def test_lock_is_not_held_by_non_inference_clicks():
    # 分類型點圖不推論 → 不需要也不該搶鎖(鎖被別人拿著時照樣立即回應)
    lab = cls_label()
    S = _state(lab, 1)
    with A._GPU_LOCK:
        t0 = time.time()
        A._select_core(S, lab, 0.5, 0, request=fake_request())
        assert time.time() - t0 < 1.0
