# -*- coding: utf-8 -*-
"""ClinScope 網頁/API 層測試的共用設定(不載入任何真模型、不下載權重)。

- 以「公開站 + 站內 Google 登入」的設定 import app(與正式站同一條分支)。
- 推論層(cls_rank / det_infer)換成假函式並計次;模型載入、權重下載一律 raise,
  任何測試誤觸都會立刻失敗,不會吃 GPU 或連 HF。
- 沒裝 torch 的環境(例如雲端 CI)塞最小 torch/torchvision 替身;有真 torch 就用真的。
- identity_debug.log / model_filter.json 等執行期檔改寫到暫存夾,不污染 repo。
- 測試資料一律假帳號(example.com),不含真實 email / token / 病患影像。
"""
import json
import os
import socket
import sys
import tempfile
import time
import types
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TMP = Path(tempfile.mkdtemp(prefix="clinscope_test_"))

OWNER = "owner@example.com"
GUEST = "guest@example.com"
CLIENT_ID = "test-client-id.apps.googleusercontent.com"

# ── import 前設好環境(app 在 import 時讀環境變數)──
for _k in ("CLINSCOPE_SESSION_SECRET", "CLINSCOPE_FORCE_OWNER", "CLINSCOPE_ALLOWED_PATHS",
           "CLINSCOPE_AUTH_USER", "CLINSCOPE_AUTH_PASS", "CLINSCOPE_TRUST_CF_ACCESS_HEADER"):
    os.environ.pop(_k, None)
os.environ.update({
    "CLINSCOPE_PUBLIC": "1",
    "CLINSCOPE_GOOGLE_CLIENT_ID": CLIENT_ID,
    "CLINSCOPE_OWNER_EMAIL": OWNER,
    "CLINSCOPE_SAMPLE_DIR": str(TMP / "no_sample_dir"),
    "GRADIO_TEMP_DIR": str(TMP / "gradio"),
    "GRADIO_ANALYTICS_ENABLED": "False",
})


def _stub_torch_if_missing():
    """只在 import 不到 torch 時才塞替身;替身只夠 app.py import 期間用,真的推論會 raise。"""
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
        return
    except Exception:
        pass

    def _no(*a, **k):
        raise AssertionError("測試不得載入 torch 權重")

    torch = types.ModuleType("torch")
    torch.device = lambda t: types.SimpleNamespace(type=str(t).split(":")[0])
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.no_grad = lambda: (lambda f: f)
    torch.load = _no
    torch.Tensor = type("Tensor", (), {})
    nn = types.ModuleType("torch.nn")
    fn = types.ModuleType("torch.nn.functional")
    nn.functional = fn
    torch.nn = nn
    tv = types.ModuleType("torchvision")
    tv.models = types.ModuleType("torchvision.models")
    tv.transforms = types.ModuleType("torchvision.transforms")
    sys.modules.update({"torch": torch, "torch.nn": nn, "torch.nn.functional": fn,
                        "torchvision": tv, "torchvision.models": tv.models,
                        "torchvision.transforms": tv.transforms})


_stub_torch_if_missing()
sys.path.insert(0, str(ROOT))
import app as A  # noqa: E402
import weights as W  # noqa: E402

# ── 推論層替身 ──
CALLS = {"cls": 0, "det": 0}


def _forbid(name):
    def _f(*a, **k):
        raise AssertionError(f"測試不得呼叫 {name}(會載入真模型或下載權重)")
    return _f


for _n in ("load_cls", "load_yolo", "load_sahi", "load_fm"):
    setattr(A, _n, _forbid(_n))
W.ensure_file = _forbid("weights.ensure_file")
W.ensure_local = _forbid("weights.ensure_local")
W.pending_rels = lambda spec: []
W.pending_size_mb = lambda spec: 0


def fake_cls_rank(image, spec, progress=None):
    CALLS["cls"] += 1
    raw = (A.CARDS.get(spec["key"]) or {}).get("classes") or ["a", "b"]
    names = [A.prettify(x) for x in raw]
    probs = [0.9] + [0.1 / max(1, len(names) - 1)] * (len(names) - 1)
    return list(zip(names, probs))


def fake_det_infer(image, spec, conf, progress=None):
    import numpy as np
    CALLS["det"] += 1
    return {"obj": 2}, np.zeros((8, 8, 3), dtype=np.uint8), 2


A.cls_rank = fake_cls_rank
A.det_infer = fake_det_infer

# 執行期檔寫到暫存夾
A._ID_LOG = TMP / "identity_debug.log"
A.MODEL_FILTER_FILE = TMP / "model_filter.json"


@pytest.fixture
def infer_calls():
    CALLS["cls"] = CALLS["det"] = 0
    return CALLS


# ── 假 request(單元測試用)──
def fake_request(email=None, headers=None, cookie_token=None):
    """email → 用 app 自己的 make_session 簽出 cookie;headers 原樣帶入。"""
    cookies = {}
    if email is not None:
        cookies[A.COOKIE_NAME] = A.make_session(email)
    if cookie_token is not None:
        cookies[A.COOKIE_NAME] = cookie_token
    return types.SimpleNamespace(headers=dict(headers or {}), cookies=cookies)


def owner_cookie_header():
    return f"{A.COOKIE_NAME}={A.make_session(OWNER)}"


def det_label():
    return next(m["label"] for m in A.AVAILABLE if m["kind"] == "det")


def cls_label():
    return next(m["label"] for m in A.AVAILABLE
                if m["kind"] == "cls" and (A.CARDS.get(m["key"]) or {}).get("classes"))


def make_png(path, color=(200, 30, 30)):
    from PIL import Image
    Image.new("RGB", (16, 16), color).save(path)
    return path


# ── 真的起一個公開站(同 launch_app 的站內 Google 登入分支),供 HTTP/佇列層測試 ──
def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class LiveSite:
    def __init__(self, base, demo):
        self.base = base
        self.demo = demo

    def fn_index(self, name, nth=0):
        hits = [fid for fid, f in sorted(self.demo.fns.items())
                if getattr(f.fn, "__name__", None) == name]
        if len(hits) <= nth:
            raise KeyError(f"找不到事件 {name}(nth={nth})")
        return hits[nth]

    def n_inputs(self, fid):
        return len(self.demo.fns[fid].inputs)

    def call(self, name, data=None, cookie=None, session=None, nth=0, event_data=None, timeout=30):
        """直接打 /gradio_api/queue/join(與前端/攻擊者同一條路,不經 UI),回傳 process_completed 訊息。
        data 為 dict {輸入序號: 值} 或 list;未給的輸入(含 State)送 None。"""
        import httpx
        fid = self.fn_index(name, nth)
        n = self.n_inputs(fid)
        if isinstance(data, dict) or data is None:
            vals = [None] * n
            for i, v in (data or {}).items():
                vals[i] = v
        else:
            vals = list(data)
        sh = session or uuid.uuid4().hex[:12]
        headers = {"Cookie": cookie} if cookie else {}
        r = httpx.post(f"{self.base}/gradio_api/queue/join",
                       json={"data": vals, "fn_index": fid, "session_hash": sh,
                             "event_data": event_data, "trigger_id": None},
                       headers=headers, timeout=timeout)
        r.raise_for_status()
        with httpx.stream("GET", f"{self.base}/gradio_api/queue/data",
                          params={"session_hash": sh}, headers=headers, timeout=timeout) as s:
            for line in s.iter_lines():
                if not line.startswith("data:"):
                    continue
                msg = json.loads(line[5:])
                if msg.get("msg") == "process_completed":
                    return msg
        raise AssertionError("佇列未回 process_completed")


def start_site():
    """以目前 app 模組設定呼叫真正的 launch_app(prevent_thread_lock)並等埠就緒。"""
    import httpx
    holder = {}
    orig_build = A.build

    def _capture():
        holder["demo"] = orig_build()
        return holder["demo"]

    A.build = _capture
    try:
        port = _free_port()
        A.launch_app(prevent_thread_lock=True, server_port=port)
    finally:
        A.build = orig_build
    base = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            if httpx.get(base + "/", timeout=2).status_code < 500:
                break
        except Exception:
            time.sleep(0.1)
    return LiveSite(base, holder["demo"])


@pytest.fixture(scope="session")
def site():
    return start_site()
