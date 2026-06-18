# build_exe.ps1 — 從乾淨 CPU-only venv 用 PyInstaller 打包 ClinScope 桌面 exe(B4)。
# 見 docs/DEPLOY_PLAN.md。產物:packaging\dist\ClinScope\ClinScope.exe(onedir,不含權重)。
#
# 用法(於 repo 根):  powershell -ExecutionPolicy Bypass -File packaging\build_exe.ps1
$ErrorActionPreference = "Stop"
$ROOT = Split-Path -Parent $PSScriptRoot      # 本檔在 packaging\ → repo 根
Set-Location $ROOT

$buildPy = ".\.venv-build\Scripts\python.exe"
if (-not (Test-Path $buildPy)) {
    Write-Host "[build] 由 dev venv 衍生乾淨 venv .venv-build(不含 dev 套件)..."
    & ".\.venv\Scripts\python.exe" -m venv .venv-build
}

Write-Host "[build] 安裝 runtime 依賴(CPU torch wheel index)+ pyinstaller ..."
& $buildPy -m pip install --upgrade pip
& $buildPy -m pip install -r packaging\requirements-runtime.txt
& $buildPy -m pip install pyinstaller

Write-Host "[build] PyInstaller 打包(--clean)..."
& $buildPy -m PyInstaller --noconfirm --clean packaging\clinscope.spec `
    --distpath packaging\dist --workpath packaging\build

Write-Host "[build] 完成 → packaging\dist\ClinScope\ClinScope.exe"
