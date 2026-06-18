# clinscope.spec — PyInstaller 規格(B4,見 docs/DEPLOY_PLAN.md)。
#
# ⚠️ 必須從「乾淨 CPU-only venv」建置(requirements-runtime.txt),不要用 dev 的 cu124 venv,
#    否則會把 ~2.5GB CUDA runtime 打進安裝包(且文書機用不到)。建置走 packaging/build_exe.ps1。
#
#    pyinstaller --noconfirm --clean packaging/clinscope.spec \
#        --distpath packaging/dist --workpath packaging/build
#
# 產物:packaging/dist/ClinScope/ClinScope.exe(onedir;權重不在內,首次選模型才下載)。
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules

ROOT = Path(SPECPATH).resolve().parent          # SPECPATH = packaging/ → ROOT = repo 根
SRC = str(ROOT / "src")
sys.path.insert(0, SRC)

datas, binaries, hiddenimports = [], [], []


def _add(pkg):
    """collect_all 一個套件(資料+動態庫+子模組);失敗只警告不中斷。"""
    try:
        d, b, h = collect_all(pkg)
        datas.extend(d); binaries.extend(b); hiddenimports.extend(h)
        print(f"[spec] collected {pkg}: +{len(d)} datas +{len(b)} bins +{len(h)} hidden")
    except Exception as e:                       # noqa: BLE001
        print(f"[spec] collect_all({pkg}) 跳過:{e}")


# 有執行期模板/設定/資料、PyInstaller 靜態分析常漏掉的套件 → 全收
for _pkg in ("gradio", "gradio_client", "safehttpx", "groovy",
             "ultralytics", "sahi", "timm", "transformers",
             "huggingface_hub", "tokenizers", "webview",
             "torch", "torchvision"):
    _add(_pkg)

# sklearn / scipy 子模組(joblib 載入 LogReg head 時用)
hiddenimports += collect_submodules("sklearn")
hiddenimports += collect_submodules("scipy")

# 專案自有資料 + 模組(app.py 執行期 sys.path.insert(ROOT/'src') 後 import weights/fm_features)
datas += [
    (str(ROOT / "model_cards"), "model_cards"),
    (str(ROOT / "packaging" / "weights_manifest.json"), "packaging"),
    (str(ROOT / "app.py"), "."),
]
for _py in (ROOT / "src").glob("*.py"):
    datas.append((str(_py), "src"))

# 額外隱藏匯入:lazy import(函式內)/ pywebview Windows 後端 / tkinter 原生資料夾對話框
hiddenimports += [
    "app", "weights", "fm_features",
    "tkinter", "tkinter.filedialog",
    "webview.platforms.winforms", "webview.platforms.edgechromium",
    "clr", "PIL._tkinter_finder",
]

# gradio 必須保留原始 .py(它執行期會讀自身原始碼/模板,純 .pyc 會壞)
module_collection_mode = {"gradio": "py", "gradio_client": "py"}

a = Analysis(
    [str(ROOT / "packaging" / "run_desktop.py")],
    pathex=[str(ROOT), SRC],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["kaggle", "gdown", "datasets", "pyarrow",          # 僅 data/_pipeline prep 用
              "tensorflow", "flax", "jax"],                       # transformers 可選後端,不用
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="ClinScope",
    console=False,            # 桌面 app:不開黑窗(出錯訊息走 run_desktop 的瀏覽器退路)
    icon=None,
)

coll = COLLECT(
    exe, a.binaries, a.datas,
    strip=False, upx=False,
    name="ClinScope",
)
