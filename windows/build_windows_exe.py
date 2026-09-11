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
        dcm_helper.exe           <- same CLI as `dcm_helper`
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
SECOND_EXE = "dcm_helper"
DEFAULT_ICON = WINDOWS_DIR / "icon.ico"

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


ICO_SIZES = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]


def stage_icon(icon: Path, destination: Path) -> Path:
    """Produce a proper multi-resolution ``.ico`` at a fixed filename, so
    both backends embed it the same way regardless of the source file's own
    name or format, and mrseg_gui.py can look it up at runtime with a name
    it knows ahead of time (see frozen_support.find_data_file("icon.ico")).

    ``icon`` may already be a ``.ico`` (copied as-is) or any image Pillow can
    read (``.png``, ``.jpg``, ``.bmp``, ...), converted here -- a hand-drawn
    logo does not need to be pre-converted to design your own icon.  A
    non-square source is padded onto a transparent square first so it isn't
    stretched.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)

    if icon.suffix.lower() == ".ico":
        shutil.copy2(icon, destination)
        return destination

    try:
        from PIL import Image
    except ImportError as error:
        raise SystemExit(
            f"--icon {icon} is not a .ico file, and Pillow is not installed to "
            f"convert it (it normally comes in already, as a matplotlib "
            f"dependency). Either supply a real .ico, or run: pip install pillow"
        ) from error

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
        # Also embed it as an ordinary data file (not just the .exe's PE
        # resource) so mrseg_gui.py can set it as the actual window/taskbar
        # icon at runtime via frozen_support.find_data_file("icon.ico") --
        # Tk does not inherit the hosting exe's own icon automatically.
        command.append(f"--include-data-files={icon}=icon.ico")

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
        # --icon only sets the .exe's own PE resource icon; also bundle it
        # as an ordinary data file so mrseg_gui.py can set the actual
        # window/taskbar icon at runtime (see the matching comment in
        # nuitka_command()). icon is already staged under the fixed name
        # "icon.ico" by stage_icon(), so this lands at that same name.
        command.append(f"--add-data={icon}{separator}.")

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


def add_second_entry_point(dist: Path, executable: Path) -> Optional[Path]:
    """Provide dcm_helper.exe as a hard link to the built binary.

    mrseg_entry.py dispatches on the executable's file name, so one build
    provides both console scripts of the pip installation -- a one-file
    build is self-contained, so this works there as well.  A hard link
    (same file, two directory entries) rather than a copy matters once
    weights are embedded in a one-file build: a real copy would double the
    multi-GB payload on disk for no reason.  Falls back to a copy if hard
    links are not available (e.g. across filesystems, or unsupported).
    """
    target = dist / (SECOND_EXE + executable.suffix)
    if target.exists():
        target.unlink()
    try:
        os.link(executable, target)
    except OSError as error:
        print(f"  hard link failed ({error}), copying instead")
        shutil.copy2(executable, target)
    print(f"  added {target.name}")
    return target


def copy_weights(dist: Path, staged_weights: Path) -> None:
    target = dist / "weights"
    if target.exists():
        shutil.rmtree(target)

    print(f"  copying weights into {target}")
    shutil.copytree(staged_weights, target)

    size = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
    print(f"  weights: {size / 2**30:.2f} GiB")


def write_readme(
    dist: Path, with_weights: bool, backend: str, version: str, onefile: bool
) -> None:
    if not with_weights:
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

DICOM conversion helper (same CLI as the `dcm_helper` console script):

    dcm_helper.exe --help

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
        default=DEFAULT_ICON if DEFAULT_ICON.is_file() else None,
        help="custom icon for the executable: a .ico, or any raster image "
        "(.png/.jpg/.bmp/...) auto-converted to one (default: windows/icon.ico); "
        "pass an empty string to build without a custom icon",
    )
    parser.add_argument("--zip", action="store_true", help="also produce a distributable .zip")
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

    check_environment(args.backend)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    icon: Optional[Path] = None
    if args.icon and str(args.icon):
        if Path(args.icon).is_file():
            icon = stage_icon(Path(args.icon), output_dir / "generated" / "icon.ico")
        else:
            print(f"WARNING: --icon {args.icon} not found, building without a custom icon")

    info("Recording nnU-Net's dynamically imported modules")
    manifest = write_manifest(collect_dynamic_modules(), output_dir / "generated" / MANIFEST_NAME)

    staged_weights: Optional[Path] = None
    if args.with_weights:
        info(f"Staging weights: {', '.join(args.models)}")
        staged_weights = stage_weights(
            output_dir / "weights_staging", list(args.models), args.weights_cache
        )

    # Only a one-file build embeds the weights in the executable itself; a
    # folder build keeps shipping them as a sibling weights/ directory
    # (copy_weights(), below), same as before.
    embedded_weights = staged_weights if args.onefile else None

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

    add_second_entry_point(dist, executable)

    if staged_weights is not None and embedded_weights is None:
        copy_weights(dist, staged_weights)

    write_readme(dist, staged_weights is not None, args.backend, version, args.onefile)

    ok = True
    if not args.skip_smoke_test:
        ok = smoke_test(executable)

    if args.zip:
        info("Packaging")
        archive = make_archive(dist, output_dir, version)
        print(f"  {archive}")

    info("Done")
    print(f"Distribution: {dist.resolve()}")
    print(f"Run it with : {executable.name} --input <image> --outdir <dir>")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
