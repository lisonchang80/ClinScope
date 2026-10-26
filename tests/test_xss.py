# -*- coding: utf-8 -*-
"""XSS 回歸:走 HTML 的地方(詳細資料表 gr.HTML、身分列 gr.HTML)必須跳脫檔名 / 類別 / 自訂分頁 / email;
模型卡與影像說明走 gr.Markdown(sanitize_html 預設開啟)。審查時未發現可利用點,這些測試防止日後退化。"""
import json

from conftest import A, OWNER, TMP, cls_label, det_label, fake_request, make_png

EVIL = '"><img src=x onerror=alert(1)>'


def _evil_file():
    return str(make_png(TMP / f"{EVIL}.png"))


def _assert_inert(html):
    assert "<img src=x" not in html and "<script" not in html.lower()
    assert "&lt;img src=x" in html or "&lt;script" in html.lower()


def test_detail_table_escapes_filename_class_and_custom_page():
    p = _evil_file()
    S = A.new_state(cls_label())
    S["files"] = S["page_files"] = [p]
    S["custom_pages"] = ["<script>alert(1)</script>"]
    S["results"] = {p: {"page": "<script>alert(1)</script>", "prob": 0.9, "ranked": [], "manual": True}}
    html = A.detail_html_str(S)
    _assert_inert(html)
    assert "&lt;script&gt;" in html


def test_detail_table_escapes_det_class_names():
    p = _evil_file()
    S = A.new_state(det_label())
    S["files"] = S["page_files"] = [p]
    S["results"] = {p: {"counts": {EVIL: 3}, "total": 3}}
    _assert_inert(A.detail_html_str(S))


def test_custom_page_added_via_bridge_is_escaped():
    p = _evil_file()
    S = A.new_state(cls_label())
    S["files"] = [p]
    S["results"] = {p: {"page": "x", "prob": 0.9, "ranked": [], "manual": False}}
    out = A.on_bridge(json.dumps({"a": "addpage", "name": "<img src=x onerror=alert(1)>"}), S,
                      cls_label(), 0.5)
    html = A.detail_html_str(out[0])
    assert "<img src=x" not in html


def test_identity_bar_escapes_email():
    bar = A.on_identity(fake_request(email='<img src=x onerror=alert(1)>@example.com'))[0]
    assert "<img src=x" not in bar and "&lt;img" in bar
    assert OWNER in A.on_identity(fake_request(email=OWNER))[0]


def test_model_cards_render_through_sanitized_markdown(site):
    md = [b for b in site.demo.blocks.values() if type(b).__name__ == "Markdown"]
    assert md and all(b.sanitize_html for b in md)


def test_model_cards_contain_no_active_html():
    import re
    bad = re.compile(r"<\s*(script|iframe|object|embed|svg|img)\b|\bon\w+\s*=|javascript:", re.I)
    for key in A.CARDS:
        text = A.card_text(next(m["label"] for m in A.AVAILABLE if m["key"] == key)) \
            if any(m["key"] == key for m in A.AVAILABLE) else json.dumps(A.CARDS[key], ensure_ascii=False)
        assert not bad.search(text), key
