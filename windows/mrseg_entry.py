# Copyright 2024-2026 Hartmut Häntze
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0

"""Entry point compiled into the Windows executable.

It reproduces what the ``mrsegmentator`` console script of the pip
installation does: ``mrsegmentator.main:main``. Everything else (argument
parsing, logging, inference) is the unmodified MRSegmentator code --
``sys.argv`` is passed through untouched.

There are two additions on top of the pip CLI:

* started with no arguments at all -- what always happens when the exe is
  double-clicked in Explorer -- it opens ``mrseg_gui.py`` instead of falling
  into ``parser.initialize()``'s bare "print help and exit" path. Any real
  argument (as any terminal invocation supplies) still takes the normal CLI
  path untouched.
* started with the single hidden argument ``--mrseg-install-weights``, it
  downloads/moves the model weights into place next to the executable and
  exits, instead of doing anything CLI- or GUI-related. This is what the
  Inno Setup installer's post-install step calls -- see
  ``frozen_support.install_weights()``.

This build only ever produces ``mrsegmentator.exe`` -- no second entry point
for the ``dicom_helper`` (``dcm_helper``) console script. DICOM *input*
still works from here: ``mrsegmentator.main`` itself converts a DICOM
directory given as ``--input`` (see ``src/mrsegmentator/main.py``), so
``dicom_helper`` stays a normal bundled dependency of this exe either way.
"""

import multiprocessing
import sys
from typing import Callable

INSTALL_WEIGHTS_FLAG = "--mrseg-install-weights"


def _resolve_main() -> Callable[[], None]:
    """Pick the GUI or the CLI, based on whether any arguments were given."""
    # No arguments at all means Explorer double-click rather than a terminal
    # invocation (a real CLI call always passes at least --input). Without
    # this, parser.initialize() would just print --help and exit(2).
    if len(sys.argv) == 1:
        import mrseg_gui

        return mrseg_gui.main

    from mrsegmentator.main import main as mrseg_main

    return mrseg_main


def main() -> None:
    if sys.argv[1:] == [INSTALL_WEIGHTS_FLAG]:
        import frozen_support

        sys.exit(frozen_support.install_weights())

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
