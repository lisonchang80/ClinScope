# -*- coding: utf-8 -*-
"""骨架自檢:app 可在無模型下 import、公開站可起、訪客可正常瀏覽(切模型)。"""
from conftest import A, cls_label


def test_app_imports_in_public_auth_mode():
    assert A.PUBLIC and A.AUTH_ENABLED
    assert A.AVAILABLE, "至少要有一個模型(權重存在或可下載)才能建 UI"


def test_visitor_can_browse_models(site):
    msg = site.call("on_model", {0: cls_label()})
    assert msg["success"], msg
