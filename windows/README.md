# Building a Windows executable

This folder builds a self-contained Windows `.exe` of MRSegmentator that
behaves like the pip installation — same CLI, same flags — but ships its model
weights, so the end user never deals with weight management or downloads.

Nothing in `src/` is modified. This directory is entirely additive: remove it
and the project is unchanged. It is also excluded from `make lint` / `make type`,
which only cover `src` and `tests`.

```
windows/
  build_windows_exe.py   build driver (weights + manifest + compiler + packaging)
  build.ps1              convenience wrapper: venv, deps, build
  fetch_weights.py       downloads, verifies and stages the weights
  mrseg_entry.py         the entry point that gets compiled
  frozen_support.py      runtime fixes that only apply to the frozen build
  mrseg_gui.py           the GUI shown on a no-argument (double-click) launch
  icon.ico               default .exe / window icon (override with --icon)
```

## Quick start

On a Windows machine with Python 3.10–3.13 and
[Visual Studio Build Tools 2022](https://visualstudio.microsoft.com/downloads/)
("Desktop development with C++") installed:

```powershell
git clone https://github.com/hhaentze/MRSegmentator
cd MRSegmentator
.\windows\build.ps1
```

The result lands in `build\windows\mrseg_entry.dist\`:

```
mrsegmentator.exe        same CLI as the `mrsegmentator` console script
dcm_helper.exe           same CLI as the `dcm_helper` console script
weights\base\            shipped; nothing is downloaded at first run
weights\body_comp\
README.txt               end-user instructions
<runtime DLLs and data>
```

The end user unzips that folder. Double-clicking `mrsegmentator.exe` opens a
GUI (see below); from a terminal it behaves exactly like the pip installation:

```
mrsegmentator.exe --input scan.nii.gz --outdir segmentations
```

For a single self-contained `.exe` instead (weights embedded, your own icon),
add `-OneFile` and/or `-Icon` — see [`--onefile`](#--onefile-a-single-exe-with-the-weights-baked-in)
and [Icon](#icon) below for what each actually costs/needs before reaching for them:

```powershell
.\windows\build.ps1 -OneFile -Icon path\to\your_logo.png
```

`-Icon` accepts a `.ico` directly, or any common raster image
(`.png`/`.jpg`/`.bmp`/...) — it gets converted to a proper multi-resolution
`.ico` automatically, so a hand-designed logo doesn't need pre-converting.

## Graphical interface

Starting `mrsegmentator.exe` with **no arguments at all** — what a
double-click in Explorer always does — opens a simplified GUI
(`mrseg_gui.py`) instead of the CLI's usual "print `--help` and exit" for a
bare invocation. Any real argument, from a terminal or a script, still takes
the normal CLI path untouched; `dcm_helper.exe` is unaffected and stays
CLI-only.

The GUI itself never calls into inference code directly — each run is just
the same CLI, invoked as a subprocess per selected input with a fixed
`--fast --split_level 1`:

* **Add files...** / **Add folder...** — one or more images, or a folder
  (batched as a single run, same as `--input <dir>` on the CLI; a folder of
  DICOM files works too, exactly as it does on the CLI).
* **Output directory** — shared by all runs in the batch.
* **Model** — Base (default) or Body composition, mirroring `--body_comp`.
* **Run** processes the queued inputs one at a time and shows a log pane
  with the CLI's own output, including tqdm progress lines; a percentage
  parsed out of the current line drives the progress bar. **Cancel** stops
  after the current file.

The header sits on its own tinted panel: the project name and subtitle, a
short note that this is a CPU-friendly light mode (single fold, fast
settings), and three colored, rounded badge links to the codebase and both
papers — styled like a shields.io/GitHub README badge, each opened with the
system's default browser via `webbrowser.open()`. The badges are drawn on a
plain `tk.Canvas` (rounded-rect polygon + centered text) since neither
`tk` nor `ttk` has a built-in rounded button.

Nothing under `src/` is touched by the GUI either, and it adds no new
dependency: it's built entirely on `tkinter`, which ships with Python.

### Console vs. GUI at the OS level

A CLI tool and a windowed GUI normally need two different Windows
"subsystems" — a console app always gets a console window (even when
double-clicked), a windowed app never does (even run from a terminal, its
output goes nowhere visible). One executable needs to behave like a console
app from a terminal and like a windowed app from Explorer, which is what
`--windows-console-mode=attach` (Nuitka) and the `--windowed` build +
`frozen_support.attach_console_if_present()` (PyInstaller) below provide.

## Nuitka vs. PyInstaller

Both are supported; `--backend nuitka` is the default.

|                   | Nuitka                                | PyInstaller                          |
| ----------------- | ------------------------------------- | ------------------------------------ |
| How it works      | Compiles Python to C, then to machine code | Bundles CPython + your `.pyc` files |
| Build time (this stack) | 30–90 min                       | 5–15 min                             |
| Startup           | Noticeably faster                     | Slower (unpacking, bytecode import)  |
| Runtime speed     | Marginal gain here — the work is inside PyTorch kernels, which are precompiled C++ either way | baseline |
| Distribution size | Comparable; dominated by torch either way | comparable                       |
| Source protection | Real (compiled)                       | Minimal (`.pyc` is trivially recovered) |
| Requires          | A C compiler (MSVC)                   | Nothing extra                        |
| PyTorch support   | Dedicated `torch` plugin, but the long tail of dynamic imports is more fragile | Mature, well-trodden hooks |

**Recommendation:** Nuitka, as preferred — the startup win is real for a CLI
tool, and compiled code is a nicer artifact to hand out. Keep PyInstaller as
the fallback: torch plus nnU-Net is the stress case for Nuitka, and if a build
goes sideways, `--backend pyinstaller` will usually get you an artifact within
the hour. The two share the entry point, the weight staging and the packaging,
so switching is a one-flag change.

Note that neither tool makes the segmentation itself faster. Runtime is
dominated by PyTorch, which is already compiled C++/CUDA.

## What actually needed solving

Freezing this application is not just "point a compiler at `main.py`". Four
things break, and `frozen_support.py` fixes each one at startup.

### 1. Weights

`config._resolve_root()` checks `MRSEG_WEIGHTS_PATH` first and otherwise
downloads into `~/.mrsegmentator`. The build stages the weights into
`weights\` next to the executable, using the multi-model layout
(`weights\base\`, `weights\body_comp\`), and the entry point points
`MRSEG_WEIGHTS_PATH` at it before anything else runs.

`version.json` is staged alongside each model, which is what makes this work
offline: `ensure_model()` compares it against `MODEL_REGISTRY` and, finding the
weights current, returns without downloading. (Without that file the version
reads as 0.0 and every run would try to re-download into a directory the user
may not even be able to write to.) `fetch_weights.py` writes the file if the
release archive does not already contain it.

Because the layout is the regular one rather than a raw nnU-Net directory,
legacy mode stays off and `--body_comp` keeps working.

A user-supplied `MRSEG_WEIGHTS_PATH` still wins, so the exe stays as
configurable as the pip installation. If no bundled weights are present
(`--no-weights` builds) the variable is left alone and the normal
download-to-`~/.mrsegmentator` behaviour applies.

> Note: the variable is `MRSEG_WEIGHTS_PATH`, not `MRSEG_WEIGHTS_DIR`.

### 2. nnU-Net's dynamic class lookup — the one that would really bite

nnU-Net does not import its trainer, its resampling functions or its image
reader/writer with `import` statements. It reads their *names* from
`plans.json` / `dataset.json` and resolves them at runtime with
`recursive_find_python_class()`, which walks the **file system** underneath
`nnunetv2.__path__[0]` using `pkgutil.iter_modules`.

A frozen build has no `.py` files to walk. The scan returns nothing, the lookup
returns `None`, and inference dies with *"Could not find trainer class"* — and
no amount of `--hidden-import` fixes it, because the failure is a directory
listing coming back empty, not a missing module.

The fix has two halves:

* **Build time** — `build_windows_exe.py` records every module under the
  packages nnU-Net searches (`nnunetv2.training.nnUNetTrainer`,
  `nnunetv2.preprocessing.resampling`, `nnunetv2.imageio`,
  `nnunetv2.utilities.label_handling`) into `nnunet_module_manifest.json`, and
  force-includes all of them in the binary. The same manifest covers
  `dynamic_network_architectures.architectures`, which nnU-Net resolves through
  `pydoc.locate()` from the `network_class_name` in `plans.json`.
* **Run time** — `frozen_support.py` replaces `recursive_find_python_class`
  with one that tries the original first and otherwise resolves the name
  against the manifest.

The replacement is installed through a `sys.meta_path` hook that fires when
`nnunetv2.utilities.find_class_by_name` is first imported. That matters for two
reasons: call sites do `from ... import recursive_find_python_class`, binding
the function *by value* at their own import time, so patching has to happen
before they load; and the hook itself imports nothing, which preserves the
deferred-import trick in `main.py` that keeps `--help` fast.

`recursive_find_python_class()`'s own signature has grown across nnU-Net
versions — the package range in `setup.cfg` (`nnunetv2>=2.2.1,<=2.8.0`) spans
both a plain `(folder, class_name, current_module)` and, since a later
release, a 4th positional `base_folder` plus keyword-only `verbose` and
`cleanup_imports_from_base_folder` used by the external-trainer-path fallback.
The replacement therefore takes `*args, **kwargs` and only ever reads out
`class_name` and `current_module` (positional or keyword, whichever the
call site used) — it stays a drop-in regardless of which signature the
installed nnU-Net actually has, instead of hard-coding one arity and breaking
on the other with `TypeError: ... takes N positional arguments but M were
given`.

### 3. multiprocessing

nnU-Net runs preprocessing and export in worker processes. Windows has no
`fork`, so each worker re-launches the executable, which without
`multiprocessing.freeze_support()` means every worker re-runs the whole program
— a fork bomb. `mrseg_entry.py` calls it before anything else.

Subtle consequence: the workers re-enter the entry point, so the hook from (2)
has to be installed *before* `freeze_support()`. nnU-Net's export workers
resolve resampling functions in the child process, not the parent.

### 4. Console vs. GUI dispatch

`mrseg_entry.py` opens the GUI when started with no arguments and runs the
CLI otherwise (see [Graphical interface](#graphical-interface) above). On the
PyInstaller backend the exe is built `--windowed`, i.e. with no console of
its own, so `frozen_support.attach_console_if_present()` reattaches
stdout/stderr to the launching terminal's console when one exists — before
anything prints, including argparse's own `--help` — so a terminal
invocation still behaves like a normal CLI tool. It has to run before
`bootstrap()`. On Nuitka this is a no-op: `--windows-console-mode=attach`
already did the equivalent at the C level, before Python even started.

## Options

```
python windows\build_windows_exe.py [options]

  --backend {nuitka,pyinstaller}   compiler (default: nuitka)
  --onefile                        single, self-contained .exe (weights embedded)
                                    instead of a folder
  --no-weights                     do not ship weights
  --no-dcm-helper                  only build mrsegmentator.exe, skip dcm_helper.exe
  --models base body_comp          which models to ship (default: both)
  --icon PATH                      custom icon, .ico or any raster image (default:
                                    windows/icon.ico; pass "" for no custom icon)
  --zip                            also produce a distributable archive
  --skip-compile                   re-package an existing build
  --jobs N                         parallel compile jobs (nuitka)
```

Weight archives are cached in `%TEMP%\mrseg_weight_cache` between builds, and
staged weights are reused, so re-running the build does not re-download 2 GB.

### `--onefile`: a single .exe with the weights baked in

Without `--onefile` the weights ship as a `weights\` folder next to the .exe
(the default, and what the rest of this doc assumes). With it, the weights
are embedded directly in the compiled binary via `--include-data-dir`
(Nuitka) / `--add-data` (PyInstaller) — `frozen_support.bundled_weights_dir()`
already searches the onefile unpack location first, so no runtime code needed
to change for this to work. The result really is one file: sharing it is
copying that one `.exe` (a few GB, dominated by the weights), nothing else.

The two backends differ in what "unpack" costs at run time, because only one
of them can cache it:

* **Nuitka** sets `--onefile-tempdir-spec` to a stable, version-keyed cache
  directory (no `{PID}`/`{TIME}` in it) instead of Nuitka's own default of a
  fresh temp dir per run. Nuitka reuses that cache and skips re-extracting
  when it's already there and matches this build, so only the *first* launch
  after installing (or after updating to a new version) pays the unpack cost;
  every launch after that starts close to instantly.
* **PyInstaller** has no equivalent option: its onefile bootstrap always
  extracts to a fresh temp directory and deletes it on exit, so **every
  single launch** re-unpacks the whole multi-GB payload before the window
  even appears -- noticeably slower to open, and noticeably more CPU/disk
  activity (decompressing multiple GB), every time, not just once. Bearable
  for a small onefile app; a real, repeated cost once weights are embedded.
  `build_windows_exe.py` prints a loud warning at build time whenever this
  exact combination (`--onefile` + `--backend pyinstaller` + weights) is
  chosen, so it's a decision rather than a surprise. If that cost matters
  more to you than Nuitka's longer compile time, prefer the Nuitka backend
  for a onefile build.

`add_second_entry_point()` hard-links `dcm_helper.exe` to `mrsegmentator.exe`
rather than copying it, so the two file names don't double the on-disk size
of a onefile build (they're the same bytes; hard links only cost extra space
if you later zip the folder, since a zip has no concept of a hard link).

If you'd rather not accept either onefile trade-off but still want a single
file to hand someone, `--zip` (below) already gives you that for the default
folder build, without any unpack cost at every launch.

## Icon

`windows/icon.ico` (a simple blue-to-teal rounded badge with a brain glyph,
in the same colors as the GUI's own header/badges) is used automatically --
`--icon PATH` (`-Icon PATH` in build.ps1) overrides it, or an empty string
(`--icon ""` / `-Icon ''`, the PowerShell default) builds without a custom
icon. It sets the .exe's own file icon in both backends, and is separately
bundled as a plain data file under the fixed name `icon.ico` so
`mrseg_gui.py` can also set it as the actual window/taskbar icon at runtime
(`frozen_support.find_data_file("icon.ico")`) -- Tk does not inherit the
hosting .exe's icon on its own.

Swap in your own design by pointing `--icon` at it: `stage_icon()` accepts a
`.ico` or any other common raster format (`.png`, `.jpg`, `.bmp`, ...) --
no pre-conversion needed for a hand-designed logo. Either way it is always
re-derived through Pillow (already installed as a `matplotlib` dependency,
so nothing extra to install in the usual case) and re-exported at a fixed
set of sizes, `.ico` inputs included: a non-square *source* image gets
padded onto a transparent square first rather than stretched to fit, and
that matters for a `.ico` too -- a single non-square frame is a common
result of quick "convert my logo to .ico" tools that resize instead of pad,
and copying such a file as-is would carry the distortion straight through
to a visibly stretched icon. (What re-deriving *can't* fix: a source image
whose pixels are already distorted by an earlier lossy resize somewhere
upstream -- start from an undistorted, ideally-square source for a clean
result.) Not supported: vector formats like `.svg` (export a PNG from your
design tool first).

**Windows caches file icons.** If you rebuild with a different icon and
Explorer still shows the old one on the `.exe`, that's the icon cache, not a
build issue -- moving/renaming the file, or logging off and back on, forces
a refresh. The window/taskbar icon the GUI itself sets at runtime is read
fresh from the file on every launch and is not affected by this.

## GPU or CPU

The default build ships CPU-only torch. That runs on any machine and keeps the
distribution around 2 GB plus weights.

`.\windows\build.ps1 -Cuda` ships CUDA torch instead: several GB larger, and it
only helps users who have a matching NVIDIA driver. For a general-purpose
download, CPU-only plus `--fast` is usually the better trade; ship a separate
CUDA build if your users need the speed.

## Verifying a build

```powershell
cd build\windows\mrseg_entry.dist
.\mrsegmentator.exe --help
.\mrsegmentator.exe --input <a real scan> --outdir out --fast --cpu_only
```

The build script already runs the `--help` check and fails the build if the
executable does not start. That only exercises the CLI path, though — also
double-click `mrsegmentator.exe` in Explorer (or run it with no arguments
from a terminal) at least once to confirm the GUI opens and a run completes.

If something misbehaves at runtime, the frozen-build machinery explains itself:

```powershell
set MRSEG_FROZEN_DEBUG=1
.\mrsegmentator.exe --input scan.nii.gz --outdir out
```

This prints which weights directory was chosen, whether the manifest was found,
and every dynamic class lookup it resolves.

## Known Windows issues

* **Antivirus.** Freshly compiled binaries are routinely flagged. Signing the
  executable is the only real fix for distribution at scale.
* **`MAX_PATH`.** torch and nnU-Net produce deep paths. Enable long paths
  (`Computer\HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem\
  LongPathsEnabled = 1`) or build somewhere near the drive root.
* **Worker memory.** Under spawn, every worker loads its own copy of torch.
  `--nproc 1 --nproc_export 2` is a good default on modest machines.
* **Defender real-time scanning** roughly doubles Nuitka build times. Excluding
  the build directory helps.
* **"Matplotlib is building the font cache"** on the very first run ever
  (nnU-Net pulls in matplotlib transitively). A few hundred ms to a couple of
  seconds, and matplotlib caches the result in `%LOCALAPPDATA%\matplotlib`
  itself after that -- every run after the first is unaffected, no code here
  needed to change. Pre-building the cache at compile time and shipping it
  would need its own writable, version-matched cache directory (the same
  class of problem the weights and the onefile temp dir already solve), for
  a one-time saving of at most a couple of seconds -- not worth the added
  moving parts.
