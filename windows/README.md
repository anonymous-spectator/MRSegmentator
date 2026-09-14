# Building a Windows executable

This folder builds a self-contained Windows `.exe` of MRSegmentator that
behaves like the pip installation -- same CLI, same flags -- without the user
ever installing Python or managing model weights.

Nothing in `src/` is modified; this directory is entirely additive (remove it
and the project is unchanged), and it is excluded from `make lint` / `make type`.

```
windows/
  build_windows_exe.py   build driver (weights + manifest + compiler + packaging)
  build.ps1               convenience wrapper: venv, deps, build
  fetch_weights.py        downloads, verifies and stages weights at build time
  mrseg_entry.py          the entry point that gets compiled
  frozen_support.py       runtime fixes that only apply to the frozen build
  mrseg_gui.py            the GUI shown on a no-argument (double-click) launch
  installer.iss           Inno Setup script for --installer
```

Only `mrsegmentator.exe` is built -- there is no separate `dcm_helper.exe`.
DICOM input still works: `mrsegmentator.main` converts a DICOM directory
passed as `--input` itself, so `dicom_helper` just ships as a normal bundled
dependency.

## Quick start

On a Windows machine with Python 3.10-3.13 and
[Visual Studio Build Tools 2022](https://visualstudio.microsoft.com/downloads/)
("Desktop development with C++"), plus
[Inno Setup](https://jrsoftware.org/isdl.php) if you want a real installer:

```powershell
git clone https://github.com/hhaentze/MRSegmentator
cd MRSegmentator
.\windows\build.ps1 -Installer -Icon path\to\your_logo.ico
```

That produces a per-user installer (Desktop shortcut, uninstaller) which
downloads the model weights once, right after installation -- see
[Installer & weights](#installer--weights) below. `-Icon` takes any image
Pillow can read (`.ico`, `.png`, ...) -- see [Icon](#icon) below.

Without `-Installer` you get a plain folder build in
`build\windows\mrseg_entry.dist\`, weights included by default:

```powershell
.\windows\build.ps1
```

```
mrsegmentator.exe        same CLI as the pip `mrsegmentator` console script
weights\base\            shipped; nothing downloaded at first run
weights\body_comp\
README.txt
<runtime DLLs and data>
```

Double-clicking `mrsegmentator.exe` opens a GUI; from a terminal it behaves
exactly like the pip installation: `mrsegmentator.exe --input scan.nii.gz
--outdir segmentations`.

For a single self-contained `.exe` instead, add `-OneFile` (weights get
embedded in the binary -- see [Onefile](#onefile) for the cost this has on
PyInstaller before reaching for it).

## Graphical interface

Starting `mrsegmentator.exe` with **no arguments at all** -- what a
double-click in Explorer always does -- opens a simplified GUI
(`mrseg_gui.py`) instead of the CLI's usual "print `--help` and exit". Any
real argument still takes the normal CLI path untouched.

The GUI never calls into inference code directly: each run is the same CLI,
invoked as a subprocess with a fixed `--fast --split_level 1`. It adds one
input picker (files or a folder, DICOM included), an output directory, a
Base/Body-composition model choice, and a log pane with a progress bar parsed
from the CLI's own tqdm output. It's built entirely on `tkinter` (ships with
Python), so nothing under `src/` is touched and no new dependency is added.

A CLI tool and a windowed GUI normally need different Windows "subsystems".
`--windows-console-mode=attach` (Nuitka) and the `--windowed` build +
`frozen_support.attach_console_if_present()` (PyInstaller) make one exe act
like a console app from a terminal and a windowed app from Explorer.

## Installer & weights

`-Installer` compiles `windows/installer.iss` with
[Inno Setup](https://jrsoftware.org/isdl.php)'s `ISCC.exe` (a build-time tool
only; found automatically, or point `-Iscc` at it) into a single installer
`.exe`. Double-clicking it once extracts everything to
`%LOCALAPPDATA%\Programs\MRSegmentator` (no admin/UAC prompt) and adds a
Desktop shortcut, an uninstaller and an Add/Remove Programs entry; from then
on `mrsegmentator.exe` is a normal file on disk with no per-launch unpack cost
on either backend. `AppId` is a fixed GUID, so installing a newer build
upgrades in place instead of duplicating the entry.

**Weights are downloaded after install, not baked into the build.** A
`--installer` build always skips weight staging (no internet needed to build),
and `installer.iss` runs `mrsegmentator.exe --mrseg-install-weights` as its
last step, which:

* downloads each model's weights straight into `<install dir>\weights` (a few
  GB, needs internet) with a small Tk progress window of its own -- this step
  has no console at all to print tqdm's usual progress bar to, and a
  multi-GB download with zero visible feedback looks exactly like a frozen
  installer, so `install_weights()` reports into that window instead;
* skips any model that's already there and current -- re-running the
  installer (e.g. while iterating on it locally) does not re-download;
* moves weights already sitting in `~/.mrsegmentator` (e.g. from local
  `pip install`-based development) into place instead of re-downloading them.

This reuses `mrsegmentator.config.ensure_model()` verbatim (see
`frozen_support.install_weights()`), so the version/checksum logic lives in
exactly one place.

`--installer` is incompatible with `--onefile`: an installed folder already
has no per-launch unpack cost, which is the only problem onefile mode solves.

## Onefile

Without `-OneFile`, weights ship as a `weights\` folder next to the `.exe`
(the default). With it, weights are embedded in the single compiled binary.
The backends differ in what that costs at *run* time:

* **Nuitka** caches its one-time unpack in a stable, version-keyed directory,
  so only the first launch after installing (or updating) is slow.
* **PyInstaller** has no such cache: it re-extracts the whole multi-GB payload
  on *every* launch. `build_windows_exe.py` warns loudly if you combine
  `--onefile`, `--backend pyinstaller` and weights.

If you want one file to hand out without either cost, prefer `--zip` (a
zipped folder build) or `--installer` (an installed, un-zipped copy) instead.

## Nuitka vs. PyInstaller

`--backend nuitka` (default) compiles Python to machine code: faster startup,
real source protection, but a 30-90 min build needing a C compiler (MSVC).
`--backend pyinstaller` just bundles CPython + `.pyc` files: a 5-15 min build
needing nothing extra, slightly slower startup, `.pyc` is trivially
recoverable. Neither changes segmentation speed -- that's PyTorch, compiled
either way. The two share the entry point, weight staging and packaging, so
switching is a one-flag change; Nuitka is preferred, PyInstaller is the
fallback if a Nuitka build goes sideways.

## Options

```
python windows\build_windows_exe.py [options]

  --backend {nuitka,pyinstaller}   compiler (default: nuitka)
  --onefile                        single self-contained .exe (weights embedded)
  --no-weights                     do not ship weights (ignored with --installer,
                                    which never bakes weights in either way)
  --models base body_comp          which models to ship (default: both)
  --icon PATH                      .ico file, passed straight to the compiler
  --zip                            also produce a distributable archive
  --installer                      also build an installer with Inno Setup
                                    (incompatible with --onefile)
  --iscc PATH                      path to ISCC.exe, if not found automatically
  --skip-compile                   re-package an existing build
  --jobs N                         parallel compile jobs (nuitka)
```

Downloaded weight archives are cached in `%TEMP%\mrseg_weight_cache` between
non-installer builds, so re-running does not re-download several GB.

## What makes this work

Freezing this application needed four fixes, all in `frozen_support.py`:

1. **Weights** -- `MRSEG_WEIGHTS_PATH` (not `_DIR`) is pointed at the bundled
   `weights\` folder before anything else runs, with a `version.json` staged
   alongside each model so `ensure_model()` never re-downloads offline. A
   user-supplied `MRSEG_WEIGHTS_PATH` still wins.
2. **nnU-Net's dynamic class lookup** -- nnU-Net resolves its trainer,
   resampling functions and reader/writer by *scanning the file system* under
   `nnunetv2.__path__[0]` (`recursive_find_python_class()`), which finds
   nothing in a frozen build (no `.py` files to walk) and fails with *"Could
   not find trainer class"*. `build_windows_exe.py` records every module nnU-Net
   could look up into `nnunet_module_manifest.json` at build time; a
   `sys.meta_path` post-import hook in `frozen_support.py` replaces the lookup
   function with one that falls back to that manifest, installed before any
   call site can bind the original by value.
3. **multiprocessing** -- Windows has no `fork`, so nnU-Net's workers
   re-launch the executable; `mrseg_entry.py` calls
   `multiprocessing.freeze_support()` first, *after* installing the hook from
   (2), since spawned workers re-enter the entry point too.
4. **Console vs. GUI dispatch** -- see [Graphical interface](#graphical-interface).

Run with `MRSEG_FROZEN_DEBUG=1` set to see which weights directory, manifest
and dynamic lookups this machinery actually used.

## Icon

`--icon PATH` (`-Icon PATH` in build.ps1) accepts any image Pillow can read
-- a `.png`, `.jpg`, or `.ico` -- and always re-derives it into a proper
square, multi-resolution `.ico` (16/24/32/48/64/128/256px) before handing it
to the compiler. A non-square source is padded onto a transparent square
rather than stretched, so a single hand-drawn square logo just works;
quick "convert my logo to .ico" web tools are exactly what produces the
distorted, single-resolution `.ico` this step exists to fix. Needs Pillow
(already a transitive dependency via matplotlib); without it, a `.ico`
input is copied through as-is (a warning is printed) and anything else
fails outright.

**Windows caches file icons.** If you rebuild with a different icon and
Explorer still shows the old one on the `.exe`, that's the icon cache, not a
build issue -- moving/renaming the file, or logging off and back on, forces
a refresh.

## GPU or CPU

The default build ships CPU-only torch (~2 GB + weights, runs anywhere).
`.\windows\build.ps1 -Cuda` ships CUDA torch instead (several GB larger,
only helps machines with a matching NVIDIA driver) -- ship it as a separate
build if your users need the speed.

## Verifying a build

```powershell
cd build\windows\mrseg_entry.dist
.\mrsegmentator.exe --help
.\mrsegmentator.exe --input <a real scan> --outdir out --fast --cpu_only
```

The build already runs `--help` as a smoke test and fails if the executable
doesn't start -- that only covers the CLI path, so also double-click
`mrsegmentator.exe` at least once to confirm the GUI opens and a run
completes. For an `--installer` build, additionally run the installer once:
confirm no UAC prompt, that weights download (or are skipped if already
present), that the Desktop shortcut works, and that uninstalling removes
everything.

## Known Windows issues

* **Antivirus** flags freshly compiled binaries routinely; signing the
  executable is the only real fix at distribution scale. Installer `.exe`s
  (Inno Setup's included) get flagged too, sometimes more readily.
* **`MAX_PATH`** -- torch and nnU-Net produce deep paths. Enable long paths
  (`HKLM\SYSTEM\CurrentControlSet\Control\FileSystem\LongPathsEnabled = 1`)
  or build near the drive root.
* **Worker memory** -- every worker loads its own copy of torch under spawn;
  `--nproc 1 --nproc_export 2` is a good default on modest machines.
* **Defender real-time scanning** roughly doubles Nuitka build times;
  excluding the build directory helps.
* **Building with conda/miniforge/anaconda Python** is fragile: both
  backends bundle whatever DLLs they find next to the interpreter used to
  build, and conda's own C runtime/MKL/OpenMP DLLs routinely conflict in
  version or exports with what pip-installed torch/numpy/scipy ship. The
  symptom is the built `.exe` refusing to start with an *"ordinal ... not
  found in DLL ..."* error that never shows up during the build itself.
  `build_windows_exe.py` detects this and warns loudly; the fix is to build
  with an official python.org CPython 3.10-3.13 instead (`py -3.11`, which
  is `build.ps1`'s default -- don't pass `-Python` at a conda interpreter).
  `build.ps1` also remembers which interpreter its venv (`build\windows\venv`
  by default) was created from and recreates it automatically when `-Python`
  changes, so switching away from a conda interpreter can't silently keep
  using a venv built from the old one -- rebuild once after upgrading to get
  that check.
