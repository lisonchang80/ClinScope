# -*- coding: utf-8 -*-
"""登出必須在伺服器端讓該 session 失效(不只是請瀏覽器刪 cookie):被複製走的 cookie 值也不能再用。"""
import pytest
from itsdangerous import URLSafeTimedSerializer

from conftest import A, OWNER, auth_client, fake_request


def _logout(tok):
    return auth_client().post("/auth/logout", json={},
                              headers={"Cookie": f"{A.COOKIE_NAME}={tok}"})


def test_logout_revokes_session_server_side():
    tok = A.make_session(OWNER)
    assert A._is_owner(fake_request(cookie_token=tok))
    r = _logout(tok)
    assert r.status_code == 200
    sc = r.headers["set-cookie"].lower()
    assert A.COOKIE_NAME.lower() in sc and "max-age=0" in sc and "secure" in sc and "httponly" in sc
    assert not A._is_owner(fake_request(cookie_token=tok))


def test_logout_only_revokes_that_session():
    a, b = A.make_session(OWNER), A.make_session(OWNER)
    _logout(a)
    assert not A._is_owner(fake_request(cookie_token=a))
    assert A._is_owner(fake_request(cookie_token=b))


def test_revocation_survives_server_restart():
    tok = A.make_session(OWNER)
    _logout(tok)
    A._REVOKED.clear()
    A._REVOKED.update(A._load_revoked())          # 模擬重啟:只靠持久化檔還原
    assert not A._is_owner(fake_request(cookie_token=tok))


def test_legacy_cookie_without_session_id_rejected():
    legacy = URLSafeTimedSerializer(A.SESSION_SECRET, salt="clinscope-session").dumps({"email": OWNER})
    assert not A._is_owner(fake_request(cookie_token=legacy))


def test_auth_routes_refuse_cross_site_form_posts():
    # 跨站 <form> 只能送 text/plain 等「簡單」型別;要求 application/json 迫使跨站請求走 CORS 預檢 → 被擋
    tok = A.make_session(OWNER)
    c = auth_client()
    r = c.post("/auth/logout", content="{}",
               headers={"Content-Type": "text/plain", "Cookie": f"{A.COOKIE_NAME}={tok}"})
    assert r.status_code == 415
    assert A._is_owner(fake_request(cookie_token=tok))          # 站主沒被別站登出
    r = c.post("/auth/google", content='{"credential": "x"}', headers={"Content-Type": "text/plain"})
    assert r.status_code == 415
    assert "set-cookie" not in r.headers


# ── 作廢清單:只記站主、有上限、原子寫入、壞檔 fail-closed(審查 2026-11-07 CS-9)──
import json as _json  # noqa: E402
import os as _os  # noqa: E402
import time as _time  # noqa: E402

from conftest import GUEST  # noqa: E402


def test_guest_logout_is_not_recorded():
    before = dict(A._REVOKED)
    _logout(A.make_session(GUEST))
    assert A._REVOKED == before, "非站主的 cookie 沒有權限,不必(也不該讓任何帳號能)灌大作廢清單"


def test_revoked_list_is_capped(monkeypatch):
    monkeypatch.setattr(A, "_REVOKED_MAX", 3, raising=False)
    toks = [A.make_session(OWNER) for _ in range(5)]
    for t in toks:
        _logout(t)
    assert len(A._REVOKED) <= 3
    assert not A._is_owner(fake_request(cookie_token=toks[-1]))
    assert len(_json.loads(A._REVOKED_FILE.read_text(encoding="utf-8"))) <= 3


def test_revoked_file_is_written_atomically(monkeypatch):
    seen = []
    real = A.os.replace

    def _spy(src, dst):
        seen.append((str(src), str(dst)))
        return real(src, dst)

    monkeypatch.setattr(A.os, "replace", _spy)
    _logout(A.make_session(OWNER))
    assert seen and seen[-1][1] == str(A._REVOKED_FILE)
    assert _os.path.dirname(seen[-1][0]) == _os.path.dirname(str(A._REVOKED_FILE)), "tmp 要在同目錄(os.replace 才是原子)"


def test_truncated_revoked_file_fails_closed(isolated_revoked):
    tok = A.make_session(OWNER)
    _logout(tok)
    raw = A._REVOKED_FILE.read_bytes()
    A._REVOKED_FILE.write_bytes(raw[: len(raw) // 2])        # 寫到一半被殺(重開機/taskkill)
    A._REVOKED.clear()
    A._REVOKED.update(A._load_revoked())                      # 模擬重啟
    assert not A._is_owner(fake_request(cookie_token=tok)), "壞檔不可讓已登出(可能外流)的 cookie 復活"
    assert list(A._REVOKED_FILE.parent.glob(A._REVOKED_FILE.name + ".corrupt*")), "壞檔要改名保留"
    _time.sleep(1.1)                                          # itsdangerous 時戳以秒計
    assert A._is_owner(fake_request(cookie_token=A.make_session(OWNER))), "重新登入的 cookie 要能用"


def test_missing_revoked_file_is_a_normal_first_start(monkeypatch):
    monkeypatch.setattr(A, "_NOT_BEFORE", 0.0, raising=False)
    monkeypatch.setattr(A, "_REVOKED_FILE", A._REVOKED_FILE.with_name("never_written.json"))
    assert A._load_revoked() == {}
    assert A._NOT_BEFORE == 0.0
    assert A._is_owner(fake_request(cookie_token=A.make_session(OWNER)))


# ── 壞檔 fail-closed 要撐過「之後的每一次」重啟,直到舊 cookie 自然過期(CS-9 驗收補修)──
import uuid as _uuid  # noqa: E402

from conftest import TMP  # noqa: E402

_NB_KEY = "__not_before__"


@pytest.fixture
def isolated_revoked(monkeypatch):
    """每條測試自己的作廢檔與記憶體狀態;結束後還原(不讓 not_before 漏到其他測試)。"""
    monkeypatch.setattr(A, "_REVOKED_FILE", TMP / f"revoked_{_uuid.uuid4().hex[:8]}.json")
    monkeypatch.setattr(A, "_REVOKED", {})
    monkeypatch.setattr(A, "_NOT_BEFORE", 0.0, raising=False)
    return monkeypatch


def _restart(monkeypatch):
    """新行程:模組全域重新初始化(_NOT_BEFORE 回到 0、_REVOKED 只靠檔案還原)。"""
    monkeypatch.setattr(A, "_NOT_BEFORE", 0.0, raising=False)
    A._REVOKED.clear()
    A._REVOKED.update(A._load_revoked())


def _file_json():
    return _json.loads(A._REVOKED_FILE.read_text(encoding="utf-8"))


def test_corrupt_file_fail_closed_survives_second_restart(isolated_revoked):
    tok = A.make_session(OWNER)
    _logout(tok)
    raw = A._REVOKED_FILE.read_bytes()
    A._REVOKED_FILE.write_bytes(raw[: len(raw) // 2])
    _restart(isolated_revoked)
    assert not A._is_owner(fake_request(cookie_token=tok))
    assert _NB_KEY in _file_json(), "not_before 要寫進作廢檔本身,不能只放記憶體"
    _restart(isolated_revoked)
    assert not A._is_owner(fake_request(cookie_token=tok)), "第二次重啟後已作廢的站主 cookie 不可復活"
    _restart(isolated_revoked)
    assert not A._is_owner(fake_request(cookie_token=tok))


def test_not_before_key_survives_logouts_and_is_not_a_sid_or_counted(isolated_revoked):
    nb = _time.time() - 5
    A._REVOKED_FILE.write_text(_json.dumps({_NB_KEY: nb}), encoding="utf-8")
    _restart(isolated_revoked)
    assert A._NOT_BEFORE == nb and _NB_KEY not in A._REVOKED
    isolated_revoked.setattr(A, "_REVOKED_MAX", 2, raising=False)
    toks = [A.make_session(OWNER) for _ in range(4)]
    for t in toks:
        _logout(t)
    d = _file_json()
    assert d.get(_NB_KEY) == nb, "登出寫檔、剔除舊項目都要保留 not_before"
    assert len([k for k in d if k != _NB_KEY]) == 2 and len(A._REVOKED) == 2, "not_before 不算進上限"
    _restart(isolated_revoked)
    assert A._NOT_BEFORE == nb and _NB_KEY not in A._REVOKED


def test_not_before_key_dropped_after_max_age(isolated_revoked):
    old = _time.time() - A.SESSION_MAX_AGE - 60
    live_sid_exp = _time.time() + 3600
    A._REVOKED_FILE.write_text(_json.dumps({_NB_KEY: old, "somesid": live_sid_exp}), encoding="utf-8")
    _restart(isolated_revoked)
    assert A._NOT_BEFORE == 0.0, "過了 SESSION_MAX_AGE,之前簽發的 cookie 早已自然過期 → 不必再擋"
    assert A._REVOKED == {"somesid": live_sid_exp}
    assert A._is_owner(fake_request(cookie_token=A.make_session(OWNER)))
    _logout(A.make_session(OWNER))
    assert _NB_KEY not in _file_json()


def test_corrupt_file_put_back_when_not_before_cannot_be_written(isolated_revoked):
    tok = A.make_session(OWNER)
    _logout(tok)
    raw = A._REVOKED_FILE.read_bytes()
    A._REVOKED_FILE.write_bytes(raw[: len(raw) // 2])
    real_write = A._write_revoked
    isolated_revoked.setattr(A, "_write_revoked", lambda *a, **k: False)   # 磁碟滿:新檔寫不進去
    _restart(isolated_revoked)
    assert not A._is_owner(fake_request(cookie_token=tok))
    assert A._REVOKED_FILE.exists(), "寫不進新檔時要把壞檔搬回原處"
    isolated_revoked.setattr(A, "_write_revoked", real_write)
    _restart(isolated_revoked)
    assert not A._is_owner(fake_request(cookie_token=tok)), "下次重啟仍要 fail-closed,舊 cookie 不可復活"
