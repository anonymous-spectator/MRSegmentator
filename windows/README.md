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

The end user unzips that folder and runs:

```
mrsegmentator.exe --input scan.nii.gz --outdir segmentations
```

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

Freezing this application is not just "point a compiler at `main.py`". Three
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

### 3. multiprocessing

nnU-Net runs preprocessing and export in worker processes. Windows has no
`fork`, so each worker re-launches the executable, which without
`multiprocessing.freeze_support()` means every worker re-runs the whole program
— a fork bomb. `mrseg_entry.py` calls it before anything else.

Subtle consequence: the workers re-enter the entry point, so the hook from (2)
has to be installed *before* `freeze_support()`. nnU-Net's export workers
resolve resampling functions in the child process, not the parent.

## Options

```
python windows\build_windows_exe.py [options]

  --backend {nuitka,pyinstaller}   compiler (default: nuitka)
  --onefile                        single .exe instead of a folder
  --no-weights                     do not ship weights
  --models base body_comp          which models to ship (default: both)
  --zip                            also produce a distributable archive
  --skip-compile                   re-package an existing build
  --jobs N                         parallel compile jobs (nuitka)
```

Weight archives are cached in `%TEMP%\mrseg_weight_cache` between builds, and
staged weights are reused, so re-running the build does not re-download 2 GB.

`--onefile` is supported but not recommended: the runtime unpacks several GB to
a temporary directory on *every* launch. The folder build starts faster and is
just as easy to distribute as a zip.

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
executable does not start.

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
