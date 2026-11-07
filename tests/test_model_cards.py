# -*- coding: utf-8 -*-
"""模型卡缺 classes 時不可在事件裡載權重(審查 2026-11-07 CS-5)。

on_model → render → page_choices → all_classes → model_classes 整條沒有身分檢查:卡片沒有 classes 就退回
load_cls(W.ensure_local(spec)) → 匿名切到該模型就把 ResNet 載進 GPU 常駐(桌面版則無進度靜默下載 ~90MB)。
conftest 已把 load_cls / weights.ensure_local 換成「一呼叫就失敗」。"""
import json

from conftest import A, ROOT


def _cls_specs():
    return [m for m in A.REGISTRY if m["kind"] == "cls"]


def test_public_model_classes_without_card_classes_does_not_load_weights(monkeypatch):
    spec = next(m for m in A.AVAILABLE if m["kind"] == "cls" and not m.get("fm"))
    card = dict(A.CARDS.get(spec["key"]) or {})
    card["classes"] = None
    monkeypatch.setitem(A.CARDS, spec["key"], card)
    assert A.model_classes(spec) == []


def test_public_model_classes_without_card_classes_fm_does_not_load_either(monkeypatch):
    spec = next((m for m in A.AVAILABLE if m.get("fm")), None)
    if spec is None:
        return
    monkeypatch.setitem(A.CARDS, spec["key"], {**(A.CARDS.get(spec["key"]) or {}), "classes": None})
    assert A.model_classes(spec) == []


def test_every_classification_card_lists_its_classes():
    missing = []
    for m in _cls_specs():
        p = ROOT / "model_cards" / f"{m['key']}.json"
        card = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        if not card.get("classes"):
            missing.append(m["key"])
    assert missing == [], f"這些分類模型卡沒有 classes(公開站列類別時會缺):{missing}"


def test_pap_card_has_score_so_the_threshold_filter_applies():
    card = json.loads((ROOT / "model_cards" / "pap.json").read_text(encoding="utf-8"))
    assert card["num_classes"] == len(card["classes"]) == 5
    assert (card.get("metrics") or {}).get("val_acc") is not None
