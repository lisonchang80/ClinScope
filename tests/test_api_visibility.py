# -*- coding: utf-8 -*-
"""事件 API 曝光:本站沒有要對外提供的 API → 所有事件 api_visibility="private"(gradio 6 取代 show_api=False)。
注意:private 只是不列在 API 文件、gradio_client 不能用名字呼叫;直接打 /gradio_api/queue/join + fn_index
仍然打得到 —— 真正的防線是函式內的站主檢查(見 test_guards.py,它們全程用 fn_index 直打)。"""
import pytest
from gradio_client import Client

from conftest import cls_label


def test_every_event_is_private(site):
    leaked = [f.api_name for f in site.demo.fns.values() if f.fn and f.api_visibility != "private"]
    assert leaked == []


def test_gradio_client_sees_no_endpoints(site):
    info = Client(site.base, verbose=False).view_api(return_format="dict", print_info=False)
    assert info["named_endpoints"] == {}


def test_gradio_client_cannot_call_owner_event_by_name(site):
    c = Client(site.base, verbose=False)
    with pytest.raises(Exception):
        c.predict(cls_label(), 0.5, api_name="/on_batch")


def test_desktop_build_still_works_and_is_private(monkeypatch):
    from conftest import A
    monkeypatch.setattr(A, "PUBLIC", False)
    demo = A.build()
    names = {getattr(f.fn, "__name__", None) for f in demo.fns.values()}
    assert {"on_pick_folder", "on_saveas_confirm", "on_batch"} <= names
    assert all(f.api_visibility == "private" for f in demo.fns.values() if f.fn)
