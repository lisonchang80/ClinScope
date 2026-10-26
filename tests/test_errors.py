# -*- coding: utf-8 -*-
"""非預期例外的內文(常含伺服器路徑)不回給瀏覽器;刻意的 gr.Error 訊息(如「僅站主…」)照常顯示。"""
import json

from conftest import A, cls_label


def test_internal_exception_text_not_sent_to_client(site, monkeypatch):
    def boom(S):
        raise RuntimeError("內部細節 D:/Claude code/ClinScope/packaging/autostart/env.local.bat")
    monkeypatch.setattr(A, "render", boom)
    msg = site.call("on_view", {0: "👁 圖示"})
    assert not msg["success"]
    assert "env.local.bat" not in json.dumps(msg, ensure_ascii=False)


def test_intended_gr_error_still_shown(site):
    msg = site.call("on_batch", {0: cls_label(), 1: 0.5})
    assert "僅站主" in msg["output"]["error"]
