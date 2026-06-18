"""run_desktop.py — ClinScope 桌面啟動器(B3,見 docs/DEPLOY_PLAN.md)。

在背景執行緒起 Gradio 伺服器(127.0.0.1:PORT),等埠就緒後用 pywebview 開一個
原生桌面視窗指向它;若沒裝 pywebview / 開窗失敗,退回用系統預設瀏覽器開。

打包後(PyInstaller)以本檔為進入點 → 使用者雙擊 exe 即見桌面視窗,不需懂瀏覽器/網址。

環境變數:
  CLINSCOPE_PORT          伺服器埠(預設 7860)
  CLINSCOPE_DESKTOP_SMOKE =1 時只驗「伺服器起得來、HTTP 回 200」就結束,不開視窗(供 CI/無頭測試)
"""
from __future__ import annotations

import os
import socket
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# 讓 import app / weights 可運作(dev 與 PyInstaller 解包後皆適用)
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# windowed 凍結 exe 可能沒有 stdout/stderr(None)→ print 會炸,補上吞掉用的替身
for _name in ("stdout", "stderr"):
    if getattr(sys, _name, None) is None:
        try:
            setattr(sys, _name, open(os.devnull, "w", encoding="utf-8"))
        except Exception:
            pass

PORT = int(os.environ.get("CLINSCOPE_PORT", "7860"))
URL = f"http://127.0.0.1:{PORT}"
SMOKE = os.environ.get("CLINSCOPE_DESKTOP_SMOKE") == "1"


def _start_server() -> None:
    """在本執行緒啟動 Gradio(非阻塞);伺服器自身另起 daemon thread 服務請求。"""
    import app
    app.launch_app(prevent_thread_lock=True, server_port=PORT, inbrowser=False, quiet=True)


def _wait_until_up(timeout: float = 120.0) -> bool:
    """輪詢直到埠可連線且 HTTP 回應;逾時回 False。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", PORT), timeout=1):
                pass
            with urllib.request.urlopen(URL, timeout=3) as r:
                if r.status < 500:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def _open_window() -> bool:
    """嘗試用 pywebview 開原生視窗(阻塞至關窗)。成功(視窗已關)回 True;不可用回 False。"""
    try:
        import webview
    except Exception as e:
        print(f"[desktop] 找不到 pywebview({e});改用瀏覽器。")
        return False
    try:
        webview.create_window("ClinScope — 臨床顯微辨識工作台", URL,
                              width=1480, height=940, min_size=(1100, 720))
        webview.start()   # 阻塞直到使用者關閉視窗
        return True
    except Exception as e:
        print(f"[desktop] pywebview 開窗失敗({e});改用瀏覽器。")
        return False


def main() -> None:
    # 伺服器在背景執行緒起(launch_app 內部已 prevent_thread_lock,故這裡再包一層 thread
    # 只是為了讓主執行緒能立刻往下做「等就緒 → 開窗」)
    t = threading.Thread(target=_start_server, daemon=True)
    t.start()

    if not _wait_until_up():
        print("[desktop] 伺服器在時限內未就緒,結束。")
        sys.exit(1)
    print(f"[desktop] 伺服器就緒:{URL}")

    if SMOKE:
        # 寫出 marker 檔以驗證凍結後真的載入模型 + manifest(windowed exe 無 stdout 時也可讀)
        try:
            import json as _json
            import app
            import weights as _W
            marker = {
                "registry": len(app.REGISTRY),
                "available": len(app.AVAILABLE),
                "disciplines": app.DISCIPLINES,
                "manifest_files": len(_W.MANIFEST),
                "root": str(app.ROOT),
                "sample_downloadable": bool(app.AVAILABLE) and _W.is_downloadable(app.AVAILABLE[0]),
            }
        except Exception as e:   # noqa: BLE001
            marker = {"error": repr(e)}
        out = os.environ.get("CLINSCOPE_SMOKE_MARKER") or str(Path.cwd() / "_smoke_marker.json")
        try:
            Path(out).write_text(__import__("json").dumps(marker, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass
        print("SMOKE OK", marker)
        os._exit(0)   # 跳過 Gradio daemon thread 的收尾,直接乾淨結束

    if not _open_window():
        # 退路:開系統瀏覽器,主執行緒掛住讓伺服器持續服務
        import webbrowser
        webbrowser.open(URL)
        print(f"[desktop] 已用瀏覽器開啟 {URL};關閉此視窗即停止伺服器。")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
