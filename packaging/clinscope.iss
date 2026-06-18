; clinscope.iss — Inno Setup 6 安裝腳本(B5,見 docs/DEPLOY_PLAN.md)。
;
; 把 PyInstaller 產物 packaging\dist\ClinScope\ 整包成 ClinScope-Setup.exe。
; 每使用者安裝(免系統管理員 / 不跳 UAC),最符合「越容易安裝越好」。
;
; 編譯(需先裝 Inno Setup 6,提供 ISCC.exe):
;   "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" packaging\clinscope.iss
; 產物:packaging\dist\ClinScope-Setup.exe
;
; 前置:先成功跑 packaging\build_exe.ps1 產生 dist\ClinScope\ClinScope.exe。

#define AppName "ClinScope"
#define AppVer "0.1.0"
#define AppPublisher "ClinScope"
#define AppExe "ClinScope.exe"

[Setup]
AppId={{8F3A1C2E-7B4D-4E9A-9C1F-2A6B5D3E8F10}
AppName={#AppName}
AppVersion={#AppVer}
AppPublisher={#AppPublisher}
; 每使用者安裝到 %LOCALAPPDATA%\Programs\ClinScope → 免 admin
DefaultDirName={localappdata}\Programs\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
; 安裝腳本與來源都在 packaging\ 下;輸出到 packaging\dist
SourceDir=.
OutputDir=dist
OutputBaseFilename=ClinScope-Setup
UninstallDisplayIcon={app}\{#AppExe}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Tasks]
Name: "desktopicon"; Description: "建立桌面捷徑"; GroupDescription: "附加捷徑:"

[Files]
; PyInstaller onedir 整包(權重不在內,首次選模型才從 HF 下載到 %LOCALAPPDATA%\ClinScope\models)
Source: "dist\ClinScope\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{group}\解除安裝 {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "立即啟動 {#AppName}"; Flags: nowait postinstall skipifsilent

; 註:解除安裝「不」刪除模型快取 %LOCALAPPDATA%\ClinScope\models(使用者已下載的權重視為使用者資料,
;     避免重裝得重抓數 GB)。如需清除請手動刪該資料夾。
