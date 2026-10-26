# -*- coding: utf-8 -*-
"""點圖偵測(_select_core 的 det 分支)會跑 GPU 推論 → 僅站主;訪客點圖只看原圖。
gallery.select(on_select)與詳細資料表 bridge(on_bridge {"a":"sel"})兩條入口都要擋。"""
import json

import pytest

from conftest import VISITORS, cookie_for, det_label, select_model, visitor_session


def _click(site, via, ck, sh):
    if via == "bridge":
        return site.call("on_bridge", {0: json.dumps({"a": "sel", "i": 0}), 2: det_label(), 3: 0.25},
                         cookie=ck, session=sh)
    return site.call("on_select", {1: det_label(), 2: 0.25}, cookie=ck, session=sh,
                     event_data={"index": 0, "value": None, "selected": True})


@pytest.mark.parametrize("via", ["bridge", "gallery"])
@pytest.mark.parametrize("who", VISITORS)
def test_visitor_click_on_det_model_does_not_run_inference(site, sample, infer_calls, who, via):
    ck = cookie_for(who)
    sh = visitor_session(site, who, sample)
    select_model(site, det_label(), cookie=ck, session=sh)
    msg = _click(site, via, ck, sh)
    assert msg["success"], msg                # 仍可看原圖,只是不推論
    assert infer_calls["det"] == 0


@pytest.mark.parametrize("via", ["bridge", "gallery"])
def test_owner_click_on_det_model_runs_inference(site, sample, infer_calls, via):
    ck = cookie_for("owner")
    sh = visitor_session(site, "owner", sample)
    select_model(site, det_label(), cookie=ck, session=sh)
    assert _click(site, via, ck, sh)["success"]
    assert infer_calls["det"] == 1
