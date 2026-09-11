# Copyright 2024-2026 Hartmut Häntze
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0

"""Entry point compiled into the Windows executable.

It reproduces what the two console scripts of the pip installation do::

    mrsegmentator = mrsegmentator.main:main
    dcm_helper    = dicom_helper.main:main

Which one runs is decided by the *name of the executable*, so a single build
can serve both: the build script copies ``mrsegmentator.exe`` to
``dcm_helper.exe`` inside the same distribution folder.

Everything else (argument parsing, logging, inference) is the unmodified
MRSegmentator code -- ``sys.argv`` is passed through untouched.

For ``mrsegmentator.exe`` there is one addition on top of the pip CLI:
started with no arguments at all -- what always happens when the exe is
double-clicked in Explorer -- it opens ``mrseg_gui.py`` instead of falling
into ``parser.initialize()``'s bare "print help and exit" path. Any real
argument (as any terminal invocation supplies) still takes the normal CLI
path untouched. ``dcm_helper.exe`` is unaffected and stays CLI-only.
"""

import multiprocessing
import sys
from pathlib import Path
from typing import Callable

DCM_HELPER_NAMES = ("dcm_helper", "dicom_helper")


def _executable_stem() -> str:
    """File name of the running executable, without extension.

    ``sys.executable`` is the frozen binary in a Nuitka/PyInstaller build.  When
    this file is run through a plain interpreter it points at python itself, in
    which case the script name is the meaningful one.
    """
    stem = Path(sys.executable).stem.lower()
    if stem.startswith("python"):
        stem = Path(sys.argv[0]).stem.lower()
    return stem


def _resolve_main() -> Callable[[], None]:
    """Pick the CLI to run based on the executable's file name."""
    stem = _executable_stem()

    if any(stem.startswith(name) for name in DCM_HELPER_NAMES):
        from dicom_helper.main import main as dcm_main

        return dcm_main

    # No arguments at all means Explorer double-click rather than a terminal
    # invocation (a real CLI call always passes at least --input). Without
    # this, parser.initialize() would just print --help and exit(2).
    if len(sys.argv) == 1:
        import mrseg_gui

        return mrseg_gui.main

    from mrsegmentator.main import main as mrseg_main

    return mrseg_main


def main() -> None:
    _resolve_main()()


if __name__ == "__main__":
    # Order matters.  The hooks installed by bootstrap() are lazy and cheap,
    # and they must already be in place when freeze_support() takes over in a
    # spawned child process: nnU-Net's export workers resolve resampling
    # functions through nnU-Net's dynamic class lookup, in the child.
    import frozen_support

    # Must run before anything prints (including argparse's own --help):
    # on the PyInstaller backend the exe is built windowed (no console of
    # its own) so double-clicking it shows only the GUI, and this attaches
    # to the launching terminal's console when there is one so a terminal
    # invocation still behaves like a normal CLI tool. No-op elsewhere.
    frozen_support.attach_console_if_present()
    frozen_support.bootstrap()
    multiprocessing.freeze_support()
    main()
