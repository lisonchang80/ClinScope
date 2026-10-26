# -*- coding: utf-8 -*-
"""站主功能在**伺服器端**擋住:每個 Gradio 事件都能被直接打(/gradio_api/queue/join + fn_index,
與 api_visibility 無關),所以這裡完全不經 UI、直接打佇列,逐一驗證會寫檔 / 開本機選夾 / 吃 GPU 的事件。

公開站事件清單(build() 內 PUBLIC 分支):
  寫檔    on_csv(CSV)、on_saveas_confirm_dispatch(ZIP / 本地另存)、on_save_filter(model_filter.json)
  本機路徑 on_pick_folder_pub(伺服器端原生選夾)、on_mode_change(切本地模式)、on_uploaded_paths(收上傳路徑)
  GPU     on_batch、on_single、on_select / on_bridge 的點圖偵測(見 test_gpu_guard.py)
  唯讀    on_disc、on_model、on_view、on_page、on_switch、on_apply_thr、on_load_sample、on_saveas(只開對話框)
"""
import json
import uuid

import pytest

from conftest import (A, TMP, VISITORS, cls_label, cookie_for, det_label, make_png, new_session,
                      select_model, upload, visitor_session)

@pytest.mark.parametrize("who", VISITORS)
@pytest.mark.parametrize("fn, data", [
    ("on_batch", lambda: {0: cls_label(), 1: 0.5}),
    ("on_single", lambda: {0: cls_label(), 1: 0.5}),
    ("on_csv", lambda: {0: cls_label()}),
    ("on_uploaded_paths", lambda: {0: "[]"}),
])
def test_owner_only_events_raise_for_visitors(site, sample, infer_calls, who, fn, data):
    sh = visitor_session(site, who, sample)
    msg = site.call(fn, data(), cookie=cookie_for(who), session=sh)
    assert not msg["success"]
    assert "僅站主" in (msg["output"].get("error") or "")
    assert infer_calls == {"cls": 0, "det": 0}


@pytest.mark.parametrize("who", VISITORS)
def test_model_filter_write_refused_for_visitors(site, monkeypatch, who):
    monkeypatch.setattr(A, "MODEL_FILTER_FILE", TMP / f"mf_{uuid.uuid4().hex}.json")
    monkeypatch.setattr(A, "MODEL_FILTER", dict(A.MODEL_FILTER))
    before = dict(A.MODEL_FILTER)
    site.call("on_save_filter", {0: 0, 1: ["fungi"]}, cookie=cookie_for(who))
    assert A.MODEL_FILTER == before
    assert not A.MODEL_FILTER_FILE.exists()


@pytest.mark.parametrize("who", VISITORS)
def test_zip_export_refused_for_visitors(site, sample, who):
    sh = visitor_session(site, who, sample)
    msg = site.call("on_saveas_confirm_dispatch", {0: "c_n"}, cookie=cookie_for(who), session=sh)
    assert msg["success"]
    assert not (msg["output"]["data"][1] or {}).get("value")


@pytest.mark.parametrize("who", VISITORS)
def test_local_mode_and_native_picker_refused_for_visitors(site, monkeypatch, who):
    picked = []
    monkeypatch.setattr(A, "_native_dir", lambda: picked.append(1) or "")
    sh = new_session()
    site.call("on_mode_change", {0: True}, cookie=cookie_for(who), session=sh)
    site.call("on_pick_folder_pub", cookie=cookie_for(who), session=sh)
    site.call("on_saveas_confirm_dispatch", {0: "c_n"}, cookie=cookie_for(who), session=sh)
    assert picked == []


def test_owner_local_mode_reaches_native_picker(site, monkeypatch):
    picked = []
    monkeypatch.setattr(A, "_native_dir", lambda: picked.append(1) or "")
    ck, sh = cookie_for("owner"), new_session()
    assert site.call("on_mode_change", {0: True}, cookie=ck, session=sh)["success"]
    assert site.call("on_pick_folder_pub", cookie=ck, session=sh)["success"]
    assert picked == [1]


def test_owner_full_flow_upload_batch_export(site, infer_calls):
    ck, sh = cookie_for("owner"), new_session()
    imgs = [make_png(TMP / f"own_{i}.png", (i * 70, 10, 200)) for i in range(3)]
    r = upload(site, imgs, ck)
    assert r.status_code == 200, r.text
    m = site.call("on_uploaded_paths", {0: json.dumps(r.json())}, cookie=ck, session=sh)
    assert m["success"] and m["output"]["data"][-1].endswith(":3"), m
    assert site.call("on_batch", {0: cls_label(), 1: 0.5}, cookie=ck, session=sh)["success"]
    assert infer_calls["cls"] == 3
    m = site.call("on_csv", {0: cls_label()}, cookie=ck, session=sh)
    assert m["success"] and m["output"]["data"][0]["value"]["path"].endswith(".csv")
    sel = json.dumps({"a": "sel", "i": 0})
    assert site.call("on_bridge", {0: sel, 2: cls_label(), 3: 0.5}, cookie=ck, session=sh)["success"]
    assert site.call("on_single", {0: cls_label(), 1: 0.5}, cookie=ck, session=sh)["success"]
    assert infer_calls["cls"] == 4
    m = site.call("on_saveas_confirm_dispatch", {0: "c_n"}, cookie=ck, session=sh)
    assert m["success"] and m["output"]["data"][1]["value"]["path"].endswith(".zip")
    assert site.call("on_view", {0: "👁 圖示"}, cookie=ck, session=sh)["success"]   # 圖示檢視走 Gradio 快取


def test_owner_det_click_runs_detection(site, infer_calls):
    ck, sh = cookie_for("owner"), new_session()
    imgs = [make_png(TMP / "own_det.png")]
    r = upload(site, imgs, ck)
    site.call("on_uploaded_paths", {0: json.dumps(r.json())}, cookie=ck, session=sh)
    select_model(site, det_label(), cookie=ck, session=sh)
    m = site.call("on_bridge", {0: json.dumps({"a": "sel", "i": 0}), 2: det_label(), 3: 0.25},
                  cookie=ck, session=sh)
    assert m["success"] and infer_calls["det"] == 1


def test_visitor_can_still_browse_sample(site, sample):
    sh = visitor_session(site, "anonymous", sample)
    for fn, data in [("on_view", {0: "👁 圖示"}), ("on_page", {0: A.TOTAL}),
                     ("on_bridge", {0: json.dumps({"a": "sel", "i": 1}), 2: cls_label(), 3: 0.5})]:
        assert site.call(fn, data, session=sh)["success"], fn
