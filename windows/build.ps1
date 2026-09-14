# Copyright 2024-2026 Hartmut Häntze
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0

<#
.SYNOPSIS
    One-shot build of the MRSegmentator Windows executable.

.DESCRIPTION
    Creates a throwaway virtual environment, installs MRSegmentator plus the
    chosen compiler into it, and runs windows\build_windows_exe.py.
    The repository itself is never modified.

.EXAMPLE
    .\windows\build.ps1
    .\windows\build.ps1 -Backend pyinstaller
    .\windows\build.ps1 -Cuda -Zip
    .\windows\build.ps1 -NoWeights
    .\windows\build.ps1 -OneFile
    .\windows\build.ps1 -OneFile -Icon my_logo.ico
    .\windows\build.ps1 -Installer
#>

[CmdletBinding()]
param(
    [ValidateSet('nuitka', 'pyinstaller')]
    [string]$Backend = 'nuitka',

    # Ship CUDA-enabled torch. Adds several GB; the CPU build runs anywhere.
    [switch]$Cuda,

    # Do not bundle weights (they are then downloaded on first run).
    [switch]$NoWeights,

    # Single self-contained .exe (weights embedded) instead of a folder.
    # See windows/README.md for the Nuitka-vs-PyInstaller tradeoff this makes.
    [switch]$OneFile,

    # Path to an icon image (.ico, .png, ...). Re-derived into a proper
    # square, multi-resolution .ico before it reaches the compiler.
    [string]$Icon = '',

    # Also produce a distributable .zip.
    [switch]$Zip,

    # Also build a real installer with Inno Setup: installs to a stable
    # per-user location with a Desktop shortcut, so weights are unpacked once
    # at install time rather than on every launch. Requires Inno Setup
    # (https://jrsoftware.org/isdl.php) and is incompatible with -OneFile.
    [switch]$Installer,

    # Path to Inno Setup's ISCC.exe, if not in the default install location
    # or on PATH.
    [string]$Iscc = '',

    # Reuse an existing virtual environment instead of creating one -- only
    # actually reused if it was built from the same -Python; otherwise it is
    # recreated automatically (see below).
    [string]$VenvPath = '',

    [string]$Python = 'py -3.11'
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot

Write-Host "MRSegmentator Windows build" -ForegroundColor Cyan
Write-Host "  repository : $repo"
Write-Host "  backend    : $Backend"
Write-Host "  torch      : $(if ($Cuda) { 'CUDA' } else { 'CPU only' })"
Write-Host "  layout     : $(if ($OneFile) { 'single .exe (weights embedded)' } else { 'folder (weights\ next to the .exe)' })"
Write-Host "  icon       : $(if ($Icon) { $Icon } else { 'none' })"
Write-Host "  installer  : $(if ($Installer) { 'yes (Inno Setup)' } else { 'no' })"

if (-not $VenvPath) { $VenvPath = Join-Path $repo 'build\windows\venv' }

# -Python may be a launcher command ("py -3.11"), not a direct path, so
# resolve it to the actual interpreter it runs -- that's what gets compared
# below to decide whether an existing venv still matches what was asked for.
$resolvedPython = (Invoke-Expression "$Python -c `"import sys; print(sys.executable)`"").Trim()
$pythonMarker = Join-Path $VenvPath '.mrseg-build-python'

if (Test-Path $VenvPath) {
    $previousPython = if (Test-Path $pythonMarker) { (Get-Content $pythonMarker -Raw).Trim() } else { $null }
    if ($previousPython -ne $resolvedPython) {
        # Without this check, an existing venv is reused unconditionally --
        # silently keeping whatever interpreter it was originally created
        # from even after -Python is changed to point somewhere else. That
        # interpreter is baked into the venv's own DLLs (python3.dll etc.),
        # so a stale conda/miniforge-based venv is exactly what makes "I
        # switched to a non-conda Python" appear to have no effect, and
        # both Nuitka and PyInstaller inherit whatever DLL mismatch that
        # interpreter carries into the compiled .exe.
        Write-Host "`n=== Existing venv at $VenvPath was built from a different Python" -ForegroundColor Yellow
        Write-Host "    previous : $previousPython"
        Write-Host "    requested: $resolvedPython"
        Write-Host "    Recreating it (this venv is throwaway; nothing outside build\ is touched)."
        Remove-Item -Recurse -Force $VenvPath
    }
}

if (-not (Test-Path $VenvPath)) {
    Write-Host "`n=== Creating virtual environment at $VenvPath" -ForegroundColor Cyan
    Invoke-Expression "$Python -m venv `"$VenvPath`""
    Set-Content -Path $pythonMarker -Value $resolvedPython -NoNewline
}

$venvPython = Join-Path $VenvPath 'Scripts\python.exe'
if (-not (Test-Path $venvPython)) { throw "No python.exe in $VenvPath" }

Write-Host "`n=== Installing dependencies" -ForegroundColor Cyan
& $venvPython -m pip install --upgrade pip wheel

if (-not $Cuda) {
    # Pin the CPU wheels first so the CUDA ones are never pulled in as a
    # dependency of MRSegmentator; they would add several GB to the build.
    & $venvPython -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    if ($LASTEXITCODE -ne 0) { throw 'Installing CPU torch failed' }
}

& $venvPython -m pip install -e "$repo"
if ($LASTEXITCODE -ne 0) { throw 'Installing MRSegmentator failed' }

if ($Backend -eq 'nuitka') {
    & $venvPython -m pip install "nuitka>=2.4" ordered-set zstandard
} else {
    & $venvPython -m pip install pyinstaller pyinstaller-hooks-contrib
}
if ($LASTEXITCODE -ne 0) { throw 'Installing the build backend failed' }

$buildArgs = @((Join-Path $repo 'windows\build_windows_exe.py'), '--backend', $Backend)
if ($NoWeights)  { $buildArgs += '--no-weights' }
if ($OneFile)    { $buildArgs += '--onefile' }
if ($Icon)       { $buildArgs += @('--icon', $Icon) }
if ($Zip)        { $buildArgs += '--zip' }
if ($Installer)  { $buildArgs += '--installer' }
if ($Iscc)       { $buildArgs += @('--iscc', $Iscc) }

Write-Host "`n=== Building (this takes a while: 15-90 minutes)" -ForegroundColor Cyan
& $venvPython @buildArgs
if ($LASTEXITCODE -ne 0) { throw "Build failed with exit code $LASTEXITCODE" }

Write-Host "`nBuild complete." -ForegroundColor Green
