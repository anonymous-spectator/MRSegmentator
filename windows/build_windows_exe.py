# Copyright 2024-2026 Hartmut Häntze
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0

"""Build a self-contained Windows executable of MRSegmentator.

The result behaves like the pip installation -- same CLI, same flags -- except
that the model weights travel with it, so the user never has to think about
weight management or network access::

    MRSegmentator\\
        mrsegmentator.exe        <- same CLI as `mrsegmentator`; a plain
                                     double-click opens a simplified GUI
        weights\\base\\ ...        <- shipped, no download at first run
        <runtime files>

Nothing under src/ is touched.  The only extra code that ends up in the binary
is windows/mrseg_entry.py, windows/frozen_support.py and windows/mrseg_gui.py.

Usage (on Windows, inside the environment that has MRSegmentator installed):

    python windows\\build_windows_exe.py                      # nuitka, weights
    python windows\\build_windows_exe.py --backend pyinstaller
    python windows\\build_windows_exe.py --no-weights         # lean build
    python windows\\build_windows_exe.py --models base        # skip body_comp
    python windows\\build_windows_exe.py --zip                # + release archive

Prerequisites:
    pip install -e .
    pip install nuitka        (or: pip install pyinstaller)
    Nuitka additionally needs a C compiler; MSVC (Visual Studio Build Tools
    2022, "Desktop development with C++") is the reliable choice for torch.
"""

import argparse
import json
import os
import pkgutil
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fetch_weights import read_model_registry, stage_weights  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
WINDOWS_DIR = ROOT / "windows"
ENTRY_SCRIPT = WINDOWS_DIR / "mrseg_entry.py"
BUILD_DIR = ROOT / "build" / "windows"

APP_NAME = "mrsegmentator"
PRODUCT_NAME = "MRSegmentator"
COMPANY_NAME = "AIAH Lab"

# ---------------------------------------------------------------------------
# What has to go into the bundle
# ---------------------------------------------------------------------------
# Must be importable for the build to make sense at all.
REQUIRED_PACKAGES = [
    "mrsegmentator",
    "dicom_helper",
    "nnunetv2",
    "dynamic_network_architectures",
    "batchgenerators",
    "acvl_utils",
    "SimpleITK",
    "torch",
]

# Packages that must be pulled in *whole*, because parts of them are reached
# through strings (plans.json / dataset.json / plugin registries) rather than
# import statements, so neither backend can find them by following imports.
#
# torch, scipy, skimage, sklearn, pandas and matplotlib are deliberately NOT
# here.  Both backends ship dedicated support for them, and forcing every
# submodule of torch in particular turns a 30-minute Nuitka build into an
# all-day one (or an out-of-memory crash) for no benefit.
FORCE_INCLUDE_PACKAGES = [
    "mrsegmentator",
    "dicom_helper",
    "nnunetv2",
    "dynamic_network_architectures",
    "batchgenerators",
    "acvl_utils",
    "SimpleITK",
]

OPTIONAL_FORCE_INCLUDE = [
    "batchgeneratorsv2",
    "pydicom",
    "highdicom",
    "einops",
    "cc3d",
    "fft_conv_pytorch",
    "yacs",
    "tqdm",
]

# Packages whose non-Python files (json, pyd, dll, headers) must be copied too.
PACKAGE_DATA = [
    "nnunetv2",
    "dynamic_network_architectures",
    "batchgenerators",
    "SimpleITK",
    "pydicom",
    "highdicom",
    "torch",
]

# Sub-packages nnU-Net searches at runtime with recursive_find_python_class().
# Every module below is listed in the manifest that frozen_support.py reads,
# and is force-included in the build so the import can actually succeed.
DYNAMIC_LOOKUP_PACKAGES = [
    "nnunetv2.training.nnUNetTrainer",
    "nnunetv2.preprocessing.resampling",
    "nnunetv2.imageio",
    "nnunetv2.utilities.label_handling",
]

# Resolved through pydoc.locate() from plans.json "network_class_name".
NETWORK_ARCHITECTURE_PACKAGES = ["dynamic_network_architectures.architectures"]

MANIFEST_NAME = "nnunet_module_manifest.json"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def info(message: str) -> None:
    print(f"\n=== {message}", flush=True)


def _looks_like_conda(base_prefix: Path) -> bool:
    """Heuristic: is the underlying interpreter a conda/miniforge/anaconda one?

    ``sys.base_prefix`` (not ``sys.prefix``) is what points at a conda
    install even when running inside a plain ``venv`` created *from* that
    conda Python -- which is exactly the setup ``build.ps1 -Python <conda
    python.exe>`` produces. ``conda-meta`` is conda's own package-database
    directory and exists in every conda environment (base or not); the name
    check is a fallback for layouts where it is missing.
    """
    name = str(base_prefix).lower()
    return (base_prefix / "conda-meta").is_dir() or any(
        marker in name for marker in ("conda", "miniforge", "anaconda")
    )


def package_available(name: str) -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def available_packages(names: List[str]) -> List[str]:
    return [name for name in names if package_available(name)]


def force_include_packages() -> List[str]:
    return available_packages(FORCE_INCLUDE_PACKAGES + OPTIONAL_FORCE_INCLUDE)


def extra_dynamic_modules(manifest: Path) -> List[str]:
    """Manifest modules not already covered by a force-included package.

    Whole-package inclusion already pulls these in, and Windows caps a command
    line at 32767 characters, so listing a few hundred redundant module names
    is a real way to break the build.
    """
    covered = set(force_include_packages())
    modules = json.loads(manifest.read_text(encoding="utf-8"))["modules"]
    return [name for name in modules if name.split(".")[0] not in covered]


def check_environment(backend: str) -> None:
    info("Checking build environment")

    if sys.version_info < (3, 10) or sys.version_info >= (3, 14):
        raise SystemExit(
            f"MRSegmentator supports Python 3.10-3.13, this is "
            f"{sys.version_info.major}.{sys.version_info.minor}"
        )

    if os.name != "nt":
        print(
            "WARNING: not running on Windows. The build will produce a binary "
            "for THIS platform; cross-compiling to Windows is not supported by "
            "either backend. Run this on a Windows machine.",
            file=sys.stderr,
        )

    if _looks_like_conda(Path(sys.base_prefix)):
        print(
            "\nWARNING: building with a conda/miniforge/anaconda Python "
            f"({sys.executable}). Both backends bundle whatever DLLs they find "
            "next to that interpreter, and conda's own C runtime, MKL and "
            "OpenMP DLLs routinely conflict in version/exports with the ones "
            "pip-installed torch/numpy/scipy ship -- the usual symptom is the "
            "built .exe failing to start with an 'ordinal ... not found in "
            "DLL' error that never shows up in the build itself. Building "
            "with an official python.org CPython 3.10-3.13 (e.g. `py -3.11`, "
            "as build.ps1 defaults to) avoids this class of bug entirely.\n",
            file=sys.stderr,
        )

    missing = [name for name in REQUIRED_PACKAGES if not package_available(name)]
    if missing:
        raise SystemExit(
            f"Missing required packages: {missing}\n"
            f"Install MRSegmentator and its dependencies first: pip install -e ."
        )

    if not package_available("nuitka" if backend == "nuitka" else "PyInstaller"):
        tool = "nuitka" if backend == "nuitka" else "pyinstaller"
        raise SystemExit(f"Backend '{backend}' not installed. Run: pip install {tool}")

    print(f"python      : {sys.version.split()[0]} ({sys.executable})")
    print(f"backend     : {backend}")
    optional = available_packages(OPTIONAL_FORCE_INCLUDE)
    print(f"bundling    : {', '.join(force_include_packages())}")
    print(f"optional    : {', '.join(optional) or 'none'}")


# ---------------------------------------------------------------------------
# Manifest of dynamically-imported nnU-Net modules
# ---------------------------------------------------------------------------
def collect_dynamic_modules() -> List[str]:
    """List every module nnU-Net may look up by name at runtime.

    nnU-Net resolves the trainer, the resampling functions and the image
    reader/writer by scanning directories with pkgutil.  A frozen build has no
    directories to scan, so the names are recorded here at build time instead.
    """
    import importlib

    modules: List[str] = []

    for package_name in DYNAMIC_LOOKUP_PACKAGES + NETWORK_ARCHITECTURE_PACKAGES:
        try:
            package = importlib.import_module(package_name)
        except Exception as error:
            print(f"  WARNING: cannot import {package_name}: {error}", file=sys.stderr)
            continue

        modules.append(package_name)
        paths = getattr(package, "__path__", None)
        if not paths:
            continue

        for module_info in pkgutil.walk_packages(paths, prefix=package_name + "."):
            modules.append(module_info.name)

    modules = sorted(set(modules))

    if not modules:
        raise SystemExit(
            "No dynamically imported modules found. Without this manifest the "
            "executable compiles fine but dies at inference time with "
            "'Could not find trainer class'.\n"
            "Check that nnU-Net is installed and that DYNAMIC_LOOKUP_PACKAGES "
            "still matches this nnunetv2 version's layout."
        )

    print(f"  {len(modules)} modules recorded")
    return modules


def write_manifest(modules: List[str], destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "generated_by": "windows/build_windows_exe.py",
        "python": sys.version.split()[0],
        "modules": modules,
    }
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return destination


# ---------------------------------------------------------------------------
# Icon
# ---------------------------------------------------------------------------
ICO_SIZES = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]


def normalize_icon(icon: Path, destination: Path) -> Path:
    """Re-derive ``icon`` into a proper, square, multi-resolution ``.ico``.

    Handed straight to Pillow regardless of source format, ``.ico`` included:
    a plain copy would trust the source file's own frames, and a single
    non-square frame -- a common result of quick "convert my logo to .ico"
    tools that resize instead of pad -- would carry that distortion straight
    through and show up visibly stretched once Windows displays it in a
    square icon slot. Padding onto a transparent square first avoids that
    either way, whatever the source actually is.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        from PIL import Image
    except ImportError as error:
        if icon.suffix.lower() == ".ico":
            print(
                f"  WARNING: Pillow not installed, copying {icon} as-is -- a "
                f"non-square or single-resolution .ico will look stretched"
            )
            shutil.copy2(icon, destination)
            return destination
        raise SystemExit(
            f"--icon {icon} is not a .ico file, and Pillow is not installed to "
            f"convert it (it normally comes in already, as a matplotlib "
            f"dependency). Either supply a real .ico, or run: pip install pillow"
        ) from error

    # Pillow's ICO reader opens the largest frame in the file by default, so
    # re-deriving from an already-multi-resolution .ico loses nothing.
    image = Image.open(icon).convert("RGBA")
    if image.width != image.height:
        size = max(image.width, image.height)
        square = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        square.paste(image, ((size - image.width) // 2, (size - image.height) // 2), image)
        image = square

    image.save(destination, format="ICO", sizes=ICO_SIZES)
    return destination


# ---------------------------------------------------------------------------
# Backend command lines
# ---------------------------------------------------------------------------
def nuitka_command(
    manifest: Path,
    output_dir: Path,
    onefile: bool,
    jobs: Optional[int],
    version: str,
    embedded_weights: Optional[Path] = None,
    icon: Optional[Path] = None,
) -> List[str]:
    command = [
        sys.executable,
        "-m",
        "nuitka",
        "--onefile" if onefile else "--standalone",
        "--assume-yes-for-downloads",
        # Attach to the launching terminal's console when there is one (so a
        # CLI invocation prints in place, as usual); stay windowed with no
        # console of its own otherwise, so a double-click opens only the GUI.
        "--windows-console-mode=attach",
        # nnU-Net spawns worker processes; the plugin makes the frozen binary
        # re-enter itself correctly on Windows.
        "--enable-plugin=multiprocessing",
        "--enable-plugin=torch",
        # The GUI shown on a no-argument (double-click) launch is Tkinter.
        "--enable-plugin=tk-inter",
        # Compiling all of torch with LTO costs hours for no runtime benefit.
        "--lto=no",
        "--noinclude-pytest-mode=nofollow",
        "--noinclude-unittest-mode=nofollow",
        "--noinclude-setuptools-mode=nofollow",
        f"--output-dir={output_dir}",
        f"--output-filename={APP_NAME}.exe",
        f"--company-name={COMPANY_NAME}",
        f"--product-name={PRODUCT_NAME}",
        f"--file-version={version}",
        f"--product-version={version}",
        f"--file-description={PRODUCT_NAME} - Multi-Modality Segmentation of 40+10 Classes",
    ]

    if icon is not None:
        command.append(f"--windows-icon-from-ico={icon}")

    if jobs:
        command.append(f"--jobs={jobs}")

    for name in force_include_packages():
        command.append(f"--include-package={name}")

    for name in available_packages(PACKAGE_DATA):
        command.append(f"--include-package-data={name}")

    # Anything that only exists as a string in plans.json / dataset.json and is
    # not already covered by a whole-package inclusion above.
    for name in extra_dynamic_modules(manifest):
        command.append(f"--include-module={name}")

    command.append(f"--include-data-files={manifest}={MANIFEST_NAME}")

    if embedded_weights is not None:
        # Bakes the weights into the single .exe payload instead of shipping
        # them as a sibling folder. frozen_support.bundled_weights_dir()
        # already searches the onefile unpack location (sys._MEIPASS is
        # checked first by _data_dir_candidates()), so no runtime code needs
        # to change for this to be picked up.
        command.append(f"--include-data-dir={embedded_weights}=weights")
        # Without this, Nuitka's onefile bootstrap extracts the whole
        # multi-GB payload to a fresh temp dir on *every* launch and deletes
        # it on exit. A stable spec (no {PID}/{TIME}) makes Nuitka reuse the
        # same cache directory and skip re-extraction once it is already
        # there and matches this build -- so only the first run after
        # installing (or updating to a new version) pays the unpack cost.
        # {CACHE_DIR}/{COMPANY}/{PRODUCT}/{VERSION} are Nuitka's own runtime
        # placeholders (filled in from --company-name/--product-name/
        # --product-version, all already passed above), not Python f-string
        # substitution.
        command.append("--onefile-tempdir-spec={CACHE_DIR}/{COMPANY}/{PRODUCT}/onefile_{VERSION}")

    command.append(str(ENTRY_SCRIPT))
    return command


def pyinstaller_command(
    manifest: Path,
    output_dir: Path,
    onefile: bool,
    version: str,
    embedded_weights: Optional[Path] = None,
    icon: Optional[Path] = None,
) -> List[str]:
    separator = ";" if os.name == "nt" else ":"

    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        # Windowed subsystem: no console of its own, so a double-click opens
        # only the GUI. frozen_support.attach_console_if_present() reattaches
        # stdout/stderr to the launching terminal when the exe is run from one,
        # so a CLI invocation still behaves like a normal console tool.
        "--windowed",
        "--onefile" if onefile else "--onedir",
        f"--name={APP_NAME}",
        f"--distpath={output_dir / 'dist'}",
        f"--workpath={output_dir / 'work'}",
        f"--specpath={output_dir}",
        f"--paths={WINDOWS_DIR}",
        f"--add-data={manifest}{separator}.",
    ]

    if icon is not None:
        command.append(f"--icon={icon}")

    if embedded_weights is not None:
        # See the matching comment in nuitka_command(): this bakes the
        # weights into the single .exe; frozen_support.py finds them via
        # sys._MEIPASS without any runtime code change. Unlike Nuitka,
        # PyInstaller's onefile bootstrap has no persistent-cache option: it
        # re-extracts the whole payload to a fresh temp dir on *every*
        # launch and deletes it on exit, so every run pays the unpack cost
        # for however large the weights are. Prefer the Nuitka backend for a
        # single .exe if that matters more than compile time.
        command.append(f"--add-data={embedded_weights}{separator}weights")

    for name in force_include_packages():
        command.append(f"--collect-all={name}")

    for name in extra_dynamic_modules(manifest):
        command.append(f"--hidden-import={name}")

    command.append(str(ENTRY_SCRIPT))
    return command


def run(command: List[str]) -> None:
    printable = " ".join(command[:6])
    print(f"  running: {printable} ... ({len(command)} args total)", flush=True)

    started = time.time()
    result = subprocess.run(command, cwd=ROOT)
    elapsed = time.time() - started

    if result.returncode != 0:
        raise SystemExit(f"Build failed after {elapsed / 60:.1f} min (exit {result.returncode})")
    print(f"  finished in {elapsed / 60:.1f} min")


# ---------------------------------------------------------------------------
# Distribution assembly
# ---------------------------------------------------------------------------
def locate_dist(backend: str, output_dir: Path, onefile: bool) -> Path:
    """Return the folder holding the built executable."""
    if backend == "nuitka":
        if onefile:
            return output_dir
        candidates = [
            output_dir / f"{ENTRY_SCRIPT.stem}.dist",
            output_dir / f"{APP_NAME}.dist",
        ]
    else:
        if onefile:
            return output_dir / "dist"
        candidates = [output_dir / "dist" / APP_NAME]

    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise SystemExit(f"Could not find build output. Looked in: {[str(c) for c in candidates]}")


def executable_path(dist: Path) -> Path:
    for name in (f"{APP_NAME}.exe", APP_NAME, f"{ENTRY_SCRIPT.stem}.exe", ENTRY_SCRIPT.stem):
        candidate = dist / name
        if candidate.is_file():
            return candidate
    raise SystemExit(f"No executable found in {dist}")


def copy_weights(dist: Path, staged_weights: Path) -> None:
    target = dist / "weights"
    if target.exists():
        shutil.rmtree(target)

    print(f"  copying weights into {target}")
    shutil.copytree(staged_weights, target)

    size = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
    print(f"  weights: {size / 2**30:.2f} GiB")


def write_readme(
    dist: Path,
    with_weights: bool,
    backend: str,
    version: str,
    onefile: bool,
    installer: bool = False,
) -> None:
    if installer:
        weights_note = (
            "Model weights are downloaded once, right after installation, into\n"
            "the weights\\ folder next to mrsegmentator.exe (internet required "
            "then;\nnothing is downloaded on later runs)."
        )
    elif not with_weights:
        weights_note = (
            "Model weights are NOT included. On first use they are downloaded\n"
            "to %USERPROFILE%\\.mrsegmentator (about 2 GB, internet required)."
        )
    elif onefile:
        weights_note = (
            "Model weights are embedded directly in mrsegmentator.exe -- there is\n"
            "no separate weights folder, nothing is downloaded, and no internet\n"
            "connection is needed."
        )
        if backend == "nuitka":
            weights_note += (
                "\nThe first launch (or the first after updating to a new version)\n"
                "unpacks the embedded weights to a per-user cache and takes a bit\n"
                "longer; every launch after that reuses the cache and starts normally."
            )
        else:
            weights_note += (
                "\nBecause this is a PyInstaller build, every single launch re-unpacks\n"
                "the embedded weights to a temporary folder and deletes it again on\n"
                "exit -- expect a real delay (disk-speed dependent) before the window\n"
                "appears on every run, not just the first. A Nuitka build avoids this\n"
                "by reusing its unpacked cache between runs."
            )
    else:
        weights_note = (
            "Model weights are included in the weights\\ folder next to the "
            "executable.\nNothing is downloaded, and no internet connection is "
            "needed."
        )

    folder_note = (
        "* mrsegmentator.exe is fully self-contained -- copy or share just that one file."
        if onefile
        else "* Keep the whole folder together; the .exe needs the files next to it."
    )

    (dist / "README.txt").write_text(
        f"""{PRODUCT_NAME} {version} - Windows build ({backend})

Usage
-----
Double-click mrsegmentator.exe for a simplified GUI: add one or more input
files or folders, pick an output directory and a model (base or body
composition), and click Run. GUI runs always use --fast --split_level 1.

For the full command-line interface, open a terminal (cmd or PowerShell) in
this folder and run it with arguments instead, e.g.:

    mrsegmentator.exe --input <file or directory> --outdir <output directory>

All options of the pip installation are available:

    mrsegmentator.exe --help

Weights
-------
{weights_note}

To use a different weights directory, set the environment variable before
running, exactly as with the pip installation:

    set MRSEG_WEIGHTS_PATH=D:\\my_weights
    mrsegmentator.exe -i input -o output

Notes
-----
{folder_note}
* Without a CUDA GPU the segmentation runs on CPU. It works, but is slow.
  Use --fast, or --fold 0, to trade some accuracy for speed.
* Each worker process loads its own copy of PyTorch on Windows. If memory is
  tight, lower --nproc and --nproc_export (e.g. --nproc 1 --nproc_export 2).
* Large images can exhaust memory; use --split_level 1 or 2.
* For verbose diagnostics of the frozen-build machinery itself:
      set MRSEG_FROZEN_DEBUG=1
""",
        encoding="utf-8",
    )
    print("  wrote README.txt")


def make_archive(dist: Path, output_dir: Path, version: str) -> Path:
    name = f"{PRODUCT_NAME}-{version}-windows-x64"
    print(f"  creating {name}.zip (this takes a while for a multi-GB build)")
    archive = shutil.make_archive(
        str(output_dir / name), "zip", root_dir=dist.parent, base_dir=dist.name
    )
    return Path(archive)


def find_iscc(override: Optional[Path]) -> Optional[Path]:
    """Locate Inno Setup's command-line compiler, ISCC.exe."""
    if override is not None:
        return override if override.is_file() else None

    candidates = [
        Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
        / "Inno Setup 6"
        / "ISCC.exe",
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Inno Setup 6" / "ISCC.exe",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    found = shutil.which("ISCC.exe") or shutil.which("iscc")
    return Path(found) if found else None


def build_installer(dist: Path, output_dir: Path, version: str, iscc: Path) -> Path:
    """Compile windows/installer.iss into a single installer .exe.

    Installs the already-built folder distribution to a stable per-user
    location with Desktop/Start Menu shortcuts, so the weights are only ever
    unpacked once (at install time) rather than on every launch -- see the
    "Installer" section of windows/README.md for why this exists alongside
    --onefile rather than instead of it everywhere.
    """
    installer_dir = output_dir / "installer"
    installer_dir.mkdir(parents=True, exist_ok=True)

    command = [
        str(iscc),
        f"/DMyAppVersion={version}",
        f"/DSourceDir={dist}",
        f"/O{installer_dir}",
        str(WINDOWS_DIR / "installer.iss"),
    ]
    run(command)

    candidates = list(installer_dir.glob(f"{PRODUCT_NAME}-Setup-*.exe"))
    if not candidates:
        raise SystemExit(f"Inno Setup did not produce an installer in {installer_dir}")
    return candidates[0]


def smoke_test(executable: Path) -> bool:
    """Run `--help` to confirm the binary starts and finds its dependencies."""
    info("Smoke test")
    try:
        result = subprocess.run(
            [str(executable), "--help"],
            capture_output=True,
            text=True,
            timeout=600,
        )
    except Exception as error:
        print(f"  FAILED to launch: {error}", file=sys.stderr)
        return False

    # The parser exits with 0 for --help.
    output = (result.stdout or "") + (result.stderr or "")
    if result.returncode == 0 and "--input" in output:
        print("  ok: executable starts and prints its help")
        return True

    print(f"  FAILED (exit {result.returncode})", file=sys.stderr)
    print(output[-4000:], file=sys.stderr)
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def read_version() -> str:
    """Read the version from setup.cfg, normalised to Windows' x.y.z.w form."""
    import configparser

    parser = configparser.ConfigParser()
    parser.read(ROOT / "setup.cfg", encoding="utf-8")
    version = parser.get("metadata", "version", fallback="0.0.0")

    parts = [p for p in version.split(".") if p.isdigit()]
    while len(parts) < 4:
        parts.append("0")
    return ".".join(parts[:4])


def parse_args() -> argparse.Namespace:
    registry = read_model_registry()

    parser = argparse.ArgumentParser(
        description="Build a Windows executable of MRSegmentator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--backend",
        choices=("nuitka", "pyinstaller"),
        default="nuitka",
        help="compiler to use",
    )
    parser.add_argument(
        "--onefile",
        action="store_true",
        help="produce a single self-contained .exe (weights embedded, nothing "
        "else to ship) instead of a folder; Nuitka caches its one-time unpack "
        "between runs, PyInstaller re-unpacks on every launch -- see windows/README.md",
    )
    parser.add_argument(
        "--no-weights",
        dest="with_weights",
        action="store_false",
        help="do not ship weights; fall back to downloading at first run",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=sorted(registry),
        choices=sorted(registry),
        help="which models to ship",
    )
    parser.add_argument("--output-dir", type=Path, default=BUILD_DIR, help="build directory")
    parser.add_argument(
        "--weights-cache",
        type=Path,
        default=Path(tempfile.gettempdir()) / "mrseg_weight_cache",
        help="where downloaded weight archives are cached between builds",
    )
    parser.add_argument("--jobs", type=int, default=None, help="parallel compile jobs (nuitka)")
    parser.add_argument(
        "--icon",
        type=Path,
        default=None,
        help="path to an image (.ico, .png, ...) used as the executable's "
        "icon; re-derived through Pillow into a proper square, "
        "multi-resolution .ico before it reaches the compiler",
    )
    parser.add_argument("--zip", action="store_true", help="also produce a distributable .zip")
    parser.add_argument(
        "--installer",
        action="store_true",
        help="also build a real installer (Inno Setup): installs to a stable "
        "per-user location with a Desktop shortcut, and downloads the model "
        "weights once, right after install (the build itself ships without "
        "them and needs no internet access). Requires Inno Setup "
        "(https://jrsoftware.org/isdl.php) and a folder build -- "
        "incompatible with --onefile, whose whole point this replaces",
    )
    parser.add_argument(
        "--iscc",
        type=Path,
        default=None,
        help="path to Inno Setup's ISCC.exe, if not in the default install "
        "location or on PATH",
    )
    parser.add_argument(
        "--skip-compile",
        action="store_true",
        help="reuse an existing build; only redo weights and packaging",
    )
    parser.add_argument("--skip-smoke-test", action="store_true", help="do not run --help")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    version = read_version()

    if args.installer and args.onefile:
        raise SystemExit(
            "--installer builds an installer for the folder distribution and is "
            "incompatible with --onefile: the installer already solves the "
            "single-file-to-share problem without onefile's per-launch unpack "
            "cost. Drop one or the other."
        )

    iscc: Optional[Path] = None
    if args.installer:
        iscc = find_iscc(args.iscc)
        if iscc is None:
            raise SystemExit(
                "--installer requires Inno Setup's ISCC.exe, which was not found "
                "in the default install location or on PATH. Install Inno Setup "
                "(https://jrsoftware.org/isdl.php), or pass --iscc <path to ISCC.exe>."
            )
        # The installer downloads weights itself, once, right after install
        # (mrsegmentator.exe --mrseg-install-weights, see installer.iss) --
        # so the build itself never needs weights or internet access.
        args.with_weights = False

    check_environment(args.backend)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    icon: Optional[Path] = None
    if args.icon:
        if not args.icon.is_file():
            raise SystemExit(f"--icon {args.icon} not found")
        info(f"Normalizing icon: {args.icon}")
        icon = normalize_icon(args.icon, output_dir / "generated" / "icon.ico")

    info("Recording nnU-Net's dynamically imported modules")
    manifest = write_manifest(collect_dynamic_modules(), output_dir / "generated" / MANIFEST_NAME)

    staged_weights: Optional[Path] = None
    if args.with_weights:
        info(f"Staging weights: {', '.join(args.models)}")
        staged_weights = stage_weights(
            output_dir / "weights_staging", list(args.models), args.weights_cache
        )
    elif args.installer:
        info("Skipping weight staging: the installer downloads them after install")

    # Only a one-file build embeds the weights in the executable itself; a
    # folder build keeps shipping them as a sibling weights/ directory
    # (copy_weights(), below), same as before.
    embedded_weights = staged_weights if args.onefile else None

    if embedded_weights is not None and args.backend == "pyinstaller":
        size_gib = sum(f.stat().st_size for f in embedded_weights.rglob("*") if f.is_file())
        size_gib /= 2**30
        print(
            f"\nWARNING: --onefile with --backend pyinstaller re-extracts the "
            f"embedded weights (~{size_gib:.1f} GiB) to a temp folder on "
            f"*every* launch, not just the first -- expect a real delay and "
            f"noticeable CPU/disk use before the window appears, every single "
            f"time. --backend nuitka caches its one-time unpack instead (much "
            f"longer compile, near-instant launches after the first). If "
            f"neither tradeoff is acceptable, drop --onefile and use --zip "
            f"instead: a single archive to share, with no per-launch cost.\n",
            file=sys.stderr,
        )

    if args.skip_compile:
        info("Skipping compilation (--skip-compile)")
    else:
        info(f"Compiling with {args.backend} (expect 15-90 minutes)")
        if args.backend == "nuitka":
            command = nuitka_command(
                manifest, output_dir, args.onefile, args.jobs, version, embedded_weights, icon
            )
        else:
            command = pyinstaller_command(
                manifest, output_dir, args.onefile, version, embedded_weights, icon
            )
        run(command)

    info("Assembling distribution")
    dist = locate_dist(args.backend, output_dir, args.onefile)
    executable = executable_path(dist)
    print(f"  executable: {executable}")

    if staged_weights is not None and embedded_weights is None:
        copy_weights(dist, staged_weights)

    write_readme(
        dist,
        staged_weights is not None,
        args.backend,
        version,
        args.onefile,
        installer=args.installer,
    )

    ok = True
    if not args.skip_smoke_test:
        ok = smoke_test(executable)

    if args.zip:
        info("Packaging")
        archive = make_archive(dist, output_dir, version)
        print(f"  {archive}")

    if args.installer:
        assert iscc is not None  # checked at the top of main()
        info("Building installer (Inno Setup)")
        installer = build_installer(dist, output_dir, version, iscc)
        print(f"  {installer}")

    info("Done")
    print(f"Distribution: {dist.resolve()}")
    print(f"Run it with : {executable.name} --input <image> --outdir <dir>")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
