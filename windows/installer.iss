; Copyright 2024-2026 Hartmut Häntze
; Licensed under the Apache License, Version 2.0
; http://www.apache.org/licenses/LICENSE-2.0
;
; Inno Setup script for a real MRSegmentator installer: installs the
; already-built folder distribution to a stable per-user location and adds
; Desktop / Start Menu shortcuts, so the end user only ever unpacks the
; multi-GB weights once (at install time), not on every launch -- the
; problem onefile mode has on the PyInstaller backend, and the one this
; installer is meant to replace for anyone who wants a single file to hand
; out without paying that runtime cost. See the "Installer" section of
; windows/README.md.
;
; build_windows_exe.py --installer compiles this with Inno Setup's ISCC.exe,
; passing MyAppVersion and SourceDir on the command line (/D...); the
; #ifndef defaults below only matter if you run ISCC.exe on this file
; directly, e.g. while editing it.
;
; Requires Inno Setup 6 (https://jrsoftware.org/isdl.php) -- a build-time
; tool only, not something end users need.

#define MyAppName "MRSegmentator"
#define MyAppPublisher "AIAH Lab"
#define MyAppURL "https://github.com/hhaentze/MRSegmentator"
#define MyAppExeName "mrsegmentator.exe"

#ifndef MyAppVersion
  #define MyAppVersion "0.0.0"
#endif
#ifndef SourceDir
  #define SourceDir "..\build\windows\mrseg_entry.dist"
#endif

[Setup]
; Fixed once, never regenerate: this is what lets a newer installer detect
; and cleanly upgrade an existing install instead of creating a duplicate
; Add/Remove Programs entry.
AppId={{1FF2E7D9-D65E-4301-8C7E-90298B77BF52}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
; Per-user install, no admin/UAC prompt needed -- matches the rest of this
; build tooling, which never requires elevation either.
DefaultDirName={localappdata}\Programs\{#MyAppName}
PrivilegesRequired=lowest
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
UninstallDisplayIcon={app}\{#MyAppExeName}
OutputBaseFilename={#MyAppName}-Setup-{#MyAppVersion}
; The weights dominate the payload and are already dense float data, so
; spending more time for a better compression ratio buys little; "fast"
; keeps installer *build* time reasonable for a multi-GB source folder.
Compression=lzma2/fast
SolidCompression=yes
WizardStyle=modern

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional shortcuts:"

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent
