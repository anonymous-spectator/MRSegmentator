# Copyright 2024-2026 Hartmut Häntze
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0

"""Runtime support for the frozen (Windows .exe) build of MRSegmentator.

This module is imported *only* by ``mrseg_entry.py``, the entry point that
Nuitka / PyInstaller compile.  A regular ``pip install mrsegmentator`` never
sees it, and nothing under ``src/`` is modified or monkeypatched away from its
documented behaviour.

Three things break when this application is frozen; this module fixes all of
them before ``mrsegmentator.main.main()`` is called.

1. **Weights.**  ``config._resolve_root()`` looks at ``MRSEG_WEIGHTS_PATH``
   first and falls back to ``~/.mrsegmentator`` (downloading on demand).  The
   frozen build ships a ``weights/`` directory next to the executable, so we
   simply point ``MRSEG_WEIGHTS_PATH`` at it.  The layout is the regular
   multi-model one (``weights/base/``, ``weights/body_comp/``), so
   ``ensure_model()`` finds an up-to-date ``version.json`` and never downloads.
   If the directory is absent the variable is left alone and the normal
   download-to-``~/.mrsegmentator`` behaviour applies.

2. **nnU-Net's dynamic class lookup.**  ``recursive_find_python_class()`` walks
   the *file system* under ``nnunetv2.__path__[0]`` with ``pkgutil.iter_modules``
   to find the trainer, the resampling functions and the reader/writer.  A
   frozen build has no ``.py`` files to walk, so the lookup silently returns
   ``None`` and inference dies with "Could not find trainer class".  We install
   a post-import hook that replaces the function with one that resolves names
   against a manifest of module names recorded at build time.  Its signature
   has also grown a 4th positional ``base_folder`` plus keyword-only
   ``verbose``/``cleanup_imports_from_base_folder`` in some nnU-Net versions,
   so the replacement takes ``*args, **kwargs`` rather than hard-coding one
   arity (see ``_find_class_call_args``).

3. **multiprocessing.**  Handled in ``mrseg_entry.py`` (``freeze_support()``),
   but note that spawned children re-enter that entry point, so the hook from
   (2) must be installed *before* ``freeze_support()`` runs -- nnU-Net's export
   workers resolve resampling functions in the child process.

4. **Console vs. GUI.**  ``mrseg_entry.py`` opens ``mrseg_gui.py`` when the
   executable is started with no arguments (an Explorer double-click), and
   runs the normal CLI otherwise.  On the PyInstaller backend the exe is
   built windowed (no console of its own), so double-clicking it shows only
   the GUI window and nothing else; ``attach_console_if_present()`` below
   reattaches stdout/stderr to the launching terminal's console when one
   exists, so a terminal invocation still prints in place like a normal CLI
   tool.  The Nuitka backend gets the same behaviour for free from
   ``--windows-console-mode=attach``, before Python even starts, which makes
   this a safe no-op there.
"""

import importlib
import importlib.abc
import importlib.util
import json
import os
import sys
import types
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

MANIFEST_NAME = "nnunet_module_manifest.json"
WEIGHTS_DIR_NAME = "weights"
WEIGHTS_ENV_VAR = "MRSEG_WEIGHTS_PATH"
DEBUG_ENV_VAR = "MRSEG_FROZEN_DEBUG"

# Module whose ``recursive_find_python_class`` is the single definition all
# nnU-Net call sites import from.
_FIND_CLASS_MODULE = "nnunetv2.utilities.find_class_by_name"
_FIND_CLASS_ATTR = "recursive_find_python_class"

_manifest_cache: Optional[List[str]] = None


# ---------------------------------------------------------------------------
# Debug output (stderr, opt-in via MRSEG_FROZEN_DEBUG=1)
# ---------------------------------------------------------------------------
def _debug(message: str) -> None:
    if os.environ.get(DEBUG_ENV_VAR) and sys.stderr is not None:
        print(f"[mrseg-frozen] {message}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Locating files inside the frozen bundle
# ---------------------------------------------------------------------------
def is_frozen() -> bool:
    """True when running from a Nuitka or PyInstaller build."""
    return bool(getattr(sys, "frozen", False)) or "__compiled__" in globals()


def attach_console_if_present() -> None:
    """Reattach stdio to the launching terminal's console, if there is one.

    A windowed-subsystem executable (the PyInstaller backend's build) starts
    with no console of its own -- exactly what a double-click should produce,
    since only the GUI is meant to appear. When Windows could not give the
    process a usable stdin/stdout/stderr at all (no console, nothing
    redirected -- the double-click case, or certain shells that launch a
    windowed subsystem exe without passing handles), CPython leaves the
    corresponding ``sys.std*`` as ``None``. Only in that situation do we look
    for a console-owning parent process, attach to it, and repoint the
    missing streams there -- before anything else runs, since this has to
    happen before argparse can print so much as ``--help``. If there is no
    parent console either (e.g. launched by Inno Setup's ``[Run]`` step),
    any stream still ``None`` afterwards is pointed at ``os.devnull`` instead
    -- leaving it ``None`` would crash the first thing that writes to it
    (tqdm's progress bar included), rather than just producing no output.

    Deliberately does **not** touch a stream that is already usable: both a
    real terminal invocation (Nuitka's ``--windows-console-mode=attach``
    already resolved this at the C level before Python started, and most
    shells hand a windowed exe working, inherited console handles anyway)
    and a piped subprocess call (the build's own ``--help`` smoke test
    captures stdout this way) must keep whatever valid stream they already
    have -- reassigning it here would silently break output capture.
    A no-op on non-Windows and on an unfrozen run, where this is never
    exercised.
    """
    if os.name != "nt" or not is_frozen():
        return
    if sys.stdout is not None and sys.stderr is not None and sys.stdin is not None:
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        attach_parent_process = -1
        if kernel32.AttachConsole(attach_parent_process):
            if sys.stdout is None:
                sys.stdout = open("CONOUT$", "w", encoding="utf-8", errors="replace")
            if sys.stderr is None:
                sys.stderr = open("CONOUT$", "w", encoding="utf-8", errors="replace")
            if sys.stdin is None:
                sys.stdin = open("CONIN$", "r", encoding="utf-8", errors="replace")
            _debug("attached to parent console")
    except Exception as error:  # pragma: no cover - defensive
        _debug(f"console attach skipped: {error}")

    # No console to attach to either (e.g. launched by Inno Setup's [Run]
    # step, which has none of its own) -- sys.std* are still None at this
    # point. Leaving them that way is a landmine: anything that writes to
    # them (tqdm's progress bar in config._download_model(), not just a
    # stray print()) crashes with "'NoneType' object has no attribute
    # 'write'" instead of the download just proceeding silently. Devnull
    # streams keep every such call a harmless no-op.
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w")
    if sys.stdin is None:
        sys.stdin = open(os.devnull, "r")


def app_dir() -> Path:
    """Directory containing the executable (or this file when not frozen)."""
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def _data_dir_candidates() -> List[Path]:
    """Places a bundled data file may live, most specific first.

    PyInstaller one-dir puts data under ``_internal`` (``sys._MEIPASS``),
    Nuitka standalone puts it next to the executable.
    """
    candidates: List[Path] = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(Path(meipass))
    candidates.append(app_dir())
    candidates.append(Path(__file__).resolve().parent)
    candidates.append(app_dir() / "_internal")

    unique: List[Path] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique


def find_data_file(name: str) -> Optional[Path]:
    """Return the path of a bundled data file, or None if it is not there."""
    for directory in _data_dir_candidates():
        path = directory / name
        if path.is_file():
            return path
    return None


# ---------------------------------------------------------------------------
# (1) Weights
# ---------------------------------------------------------------------------
def bundled_weights_dir() -> Optional[Path]:
    """Return the shipped weights directory if it exists and is non-empty."""
    for directory in _data_dir_candidates():
        weights = directory / WEIGHTS_DIR_NAME
        if weights.is_dir() and any(weights.iterdir()):
            return weights
    return None


def configure_weights_dir() -> Optional[Path]:
    """Point MRSEG_WEIGHTS_PATH at the bundled weights, if any.

    An existing MRSEG_WEIGHTS_PATH set by the user always wins, so the frozen
    build stays as configurable as the pip installation.
    """
    existing = os.environ.get(WEIGHTS_ENV_VAR)
    if existing:
        _debug(f"{WEIGHTS_ENV_VAR} already set to {existing}, leaving it alone")
        return Path(existing)

    weights = bundled_weights_dir()
    if weights is None:
        _debug("no bundled weights found, falling back to ~/.mrsegmentator")
        return None

    os.environ[WEIGHTS_ENV_VAR] = str(weights)
    _debug(f"{WEIGHTS_ENV_VAR} = {weights}")
    return weights


def silence_nnunet_path_warnings() -> None:
    """Same as ``config.disable_nnunet_path_warnings()``, but without importing
    ``mrsegmentator`` (and therefore ``tqdm``/``torch``) this early."""
    for var in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
        if os.environ.get(var) is None:
            os.environ[var] = "empty"


# ---------------------------------------------------------------------------
# (2) nnU-Net dynamic class lookup
# ---------------------------------------------------------------------------
def load_manifest() -> List[str]:
    """Module names recorded at build time, or [] if the manifest is missing."""
    global _manifest_cache
    if _manifest_cache is not None:
        return _manifest_cache

    path = find_data_file(MANIFEST_NAME)
    if path is None:
        _debug(f"{MANIFEST_NAME} not bundled, dynamic lookup fallback disabled")
        _manifest_cache = []
        return _manifest_cache

    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        _manifest_cache = list(payload.get("modules", []))
        _debug(f"loaded {len(_manifest_cache)} module names from {path}")
    except Exception as error:  # pragma: no cover - defensive
        _debug(f"could not read {path}: {error}")
        _manifest_cache = []
    return _manifest_cache


def _modules_under(package: str) -> Iterable[str]:
    prefix = package + "."
    for name in load_manifest():
        if name == package or name.startswith(prefix):
            yield name


def _find_class_call_args(args: tuple, kwargs: dict) -> tuple:
    """Pull ``class_name`` and ``current_module`` out of a call meant for
    nnU-Net's ``recursive_find_python_class``.

    Its signature has grown across nnU-Net versions -- 2.2.1 takes exactly
    ``(folder, class_name, current_module)``; 2.8.0 adds a 4th positional
    ``base_folder`` plus keyword-only ``verbose`` and
    ``cleanup_imports_from_base_folder``, and some call sites pass
    ``current_module`` by keyword instead of positionally. Only ``class_name``
    and ``current_module`` matter for the manifest-based fallback below, so
    accepting ``*args, **kwargs`` and picking those two out (positional if
    present, keyword otherwise) keeps this a drop-in replacement regardless of
    which signature the installed nnU-Net actually uses.
    """
    class_name = args[1] if len(args) > 1 else kwargs.get("class_name")
    current_module = args[2] if len(args) > 2 else kwargs.get("current_module")
    return class_name, current_module


def _make_patched_finder(original: Optional[Callable[..., Any]]) -> Callable[..., Any]:
    """Build a drop-in replacement for ``recursive_find_python_class``.

    The original is tried first so that a non-frozen run (or a build that
    happens to keep source files around) behaves exactly as before.  Only if it
    comes up empty do we resolve the name against the build-time manifest.
    """

    def recursive_find_python_class(*args: Any, **kwargs: Any) -> Any:
        if original is not None:
            try:
                found = original(*args, **kwargs)
                if found is not None:
                    return found
            except Exception as error:  # frozen builds: folder does not exist
                _debug(f"original lookup failed: {error}")

        class_name, current_module = _find_class_call_args(args, kwargs)
        if class_name is None or current_module is None:
            _debug(
                f"cannot resolve without class_name/current_module (args={args!r}, "
                f"kwargs={kwargs!r})"
            )
            return None

        for module_name in _modules_under(current_module):
            try:
                module = importlib.import_module(module_name)
            except Exception as error:
                _debug(f"skipping {module_name}: {error}")
                continue

            candidate = getattr(module, class_name, None)
            if candidate is None:
                continue

            # A package exposes its already-imported submodules as attributes,
            # so nnunetv2.training.nnUNetTrainer has an attribute called
            # "nnUNetTrainer" that is the *module*, not the trainer class.
            # nnU-Net's own implementation only ever inspects non-package
            # modules, so skip module objects here too.
            if isinstance(candidate, types.ModuleType):
                _debug(f"ignoring module {module_name}.{class_name}")
                continue

            _debug(f"resolved {class_name} from {module_name}")
            return candidate

        _debug(f"could not resolve {class_name} under {current_module}")
        return None

    return recursive_find_python_class


def _patch_find_class_module(module: Any) -> None:
    """Replace the lookup in its defining module and in anything that already
    imported it by value (``from ... import recursive_find_python_class``)."""
    original = getattr(module, _FIND_CLASS_ATTR, None)
    if getattr(original, "_mrseg_frozen_patch", False):
        return

    patched = _make_patched_finder(original)
    patched._mrseg_frozen_patch = True  # type: ignore[attr-defined]
    setattr(module, _FIND_CLASS_ATTR, patched)

    for loaded in list(sys.modules.values()):
        if loaded is None or loaded is module:
            continue
        existing = getattr(loaded, _FIND_CLASS_ATTR, None)
        if callable(existing) and not getattr(existing, "_mrseg_frozen_patch", False):
            try:
                setattr(loaded, _FIND_CLASS_ATTR, patched)
            except Exception:  # pragma: no cover - read-only module
                pass
    _debug("patched recursive_find_python_class")


class _PostImportPatcher(importlib.abc.MetaPathFinder):
    """Runs a callback right after a specific module finishes importing.

    Implemented as a meta path finder so that nothing heavy is imported up
    front -- the MRSegmentator CLI defers importing torch/nnU-Net until after
    argument parsing, and the frozen build keeps that property.
    """

    def __init__(self, target: str, callback: Callable[[Any], None]) -> None:
        self.target = target
        self.callback = callback
        self._busy = False

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        if fullname != self.target or self._busy:
            return None

        self._busy = True
        try:
            spec = importlib.util.find_spec(fullname)
        except Exception as error:  # pragma: no cover - defensive
            _debug(f"find_spec({fullname}) failed: {error}")
            return None
        finally:
            self._busy = False

        if spec is None or spec.loader is None:
            return None
        spec.loader = _CallbackLoader(spec.loader, self.callback)
        return spec


class _CallbackLoader(importlib.abc.Loader):
    """Delegating loader that calls ``callback(module)`` after execution."""

    def __init__(self, inner: Any, callback: Callable[[Any], None]) -> None:
        self.inner = inner
        self.callback = callback

    def create_module(self, spec: Any) -> Any:
        return self.inner.create_module(spec)

    def exec_module(self, module: Any) -> None:
        self.inner.exec_module(module)
        try:
            self.callback(module)
        except Exception as error:  # pragma: no cover - defensive
            _debug(f"post-import callback failed: {error}")

    def __getattr__(self, item: str) -> Any:
        return getattr(self.inner, item)


def install_nnunet_import_patch() -> None:
    """Arrange for the dynamic-lookup patch to be applied on first import."""
    already = sys.modules.get(_FIND_CLASS_MODULE)
    if already is not None:
        _patch_find_class_module(already)
        return

    for finder in sys.meta_path:
        if isinstance(finder, _PostImportPatcher) and finder.target == _FIND_CLASS_MODULE:
            return

    sys.meta_path.insert(0, _PostImportPatcher(_FIND_CLASS_MODULE, _patch_find_class_module))
    _debug(f"post-import hook installed for {_FIND_CLASS_MODULE}")


# ---------------------------------------------------------------------------
# One-time weight installation (called by the Inno Setup installer)
# ---------------------------------------------------------------------------
def _run_with_progress_window(events: Any, worker: Callable[[], None]) -> None:
    """Run ``worker`` in a background thread while a small Tk window shows
    ``events`` from it.

    The installer's ``[Run]`` step (see ``attach_console_if_present()``) has
    no console at all, so tqdm's usual progress bar goes to ``os.devnull``
    -- invisible, and a multi-GB download can otherwise look exactly like a
    frozen installer for minutes at a time. A Tk window works with no
    console and needs no new dependency: ``mrseg_gui.py`` already requires
    Tk to be present in this build.

    Falls back to just calling ``worker()`` with no UI if Tk is unavailable
    for any reason (should not normally happen).
    """
    import queue
    import threading

    try:
        import tkinter as tk
        from tkinter import ttk
    except Exception as error:  # pragma: no cover - defensive
        _debug(f"no Tk for progress window ({error}), installing silently")
        worker()
        return

    root = tk.Tk()
    root.title("MRSegmentator Setup")
    root.resizable(False, False)
    root.attributes("-topmost", True)

    status_var = tk.StringVar(value="Preparing to download model weights...")
    tk.Label(root, textvariable=status_var, anchor="w", padx=12).pack(fill="x", pady=(14, 4))

    bar = ttk.Progressbar(root, mode="indeterminate", length=380)
    bar.pack(padx=12, pady=4)
    bar.start(15)

    detail_var = tk.StringVar(value="")
    tk.Label(root, textvariable=detail_var, anchor="w", padx=12, fg="gray").pack(
        fill="x", pady=(0, 14)
    )

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    def poll() -> None:
        try:
            while True:
                event = events.get_nowait()
                kind = event[0]
                if kind == "status":
                    status_var.set(event[1])
                    detail_var.set("")
                elif kind == "start":
                    _, desc, total = event
                    status_var.set(f"Downloading {desc}...")
                    if total:
                        bar.stop()
                        bar.config(mode="determinate", maximum=total)
                        bar["value"] = 0
                    else:
                        bar.config(mode="indeterminate")
                        bar.start(15)
                elif kind == "progress":
                    _, _desc, n, total = event
                    if total:
                        bar["value"] = n
                        detail_var.set(f"{n / 2**20:.0f} / {total / 2**20:.0f} MiB")
                elif kind == "finished":
                    root.after(150, root.destroy)
                    return
        except queue.Empty:
            pass
        root.after(100, poll)

    root.after(100, poll)
    root.mainloop()
    thread.join()


def install_weights() -> int:
    """Get every registered model's weights into ``<app_dir>/weights``.

    This is the ``--mrseg-install-weights`` handler, run once by the Inno
    Setup installer's post-install step so that a build made with
    ``--no-weights`` still ends up with weights on disk before the user ever
    runs an analysis -- rather than the first real run paying that cost.

    Reuses ``mrsegmentator.config.ensure_model()`` for the actual download,
    so it is naturally idempotent: re-running the installer (e.g. while
    iterating on it) skips any model whose weights are already at the
    expected version. Weights already sitting in the default
    ``~/.mrsegmentator`` location (e.g. from local development) are moved
    into place instead of being re-downloaded.

    Progress is shown in a small Tk window (see
    ``_run_with_progress_window()``) rather than tqdm's usual console
    output, which would be invisible here -- this step has no console at
    all (Inno Setup's ``[Run]`` step launches it silently).
    """
    import queue
    import shutil

    from mrsegmentator import config

    target = app_dir() / WEIGHTS_DIR_NAME
    target.mkdir(parents=True, exist_ok=True)
    os.environ[WEIGHTS_ENV_VAR] = str(target)

    default_root = Path.home() / ".mrsegmentator"
    events: "queue.Queue[Any]" = queue.Queue()

    class _ProgressReporter:
        """Stand-in for tqdm that reports into ``events`` instead of
        printing. Only the members ``config._download_model()`` actually
        touches (``total``, ``n``, ``update()``, context-manager protocol)
        need to exist."""

        def __init__(self, *args: Any, total: Optional[int] = None, desc: str = "", **kwargs: Any):
            self.total = total
            self.n = 0
            self.desc = desc
            events.put(("start", desc, total))

        def update(self, n: int = 1) -> None:
            self.n += n
            events.put(("progress", self.desc, self.n, self.total))

        def close(self) -> None:
            pass

        def __enter__(self) -> "_ProgressReporter":
            return self

        def __exit__(self, *exc: Any) -> None:
            self.close()

    # Frozen-build-only monkeypatch, same spirit as the nnU-Net class-lookup
    # patch above: this process exits right after install_weights() returns,
    # so there is nothing to restore it for.
    config.tqdm = _ProgressReporter

    errors: List[BaseException] = []

    def worker() -> None:
        try:
            for name, entry in config.MODEL_REGISTRY.items():
                dest = target / name
                source = default_root / name
                if (not dest.is_dir() or not any(dest.iterdir())) and source.is_dir():
                    if config._read_model_version(source) >= entry["version"]:
                        events.put(("status", f"Using already-downloaded {name} weights..."))
                        shutil.move(str(source), str(dest))

                events.put(("status", f"Checking {name} weights..."))
                config.setup_mrseg(name)
        except BaseException as caught:  # noqa: BLE001 - re-raised on the main thread below
            errors.append(caught)
        finally:
            events.put(("finished", None))

    _run_with_progress_window(events, worker)

    if errors:
        raise errors[0]

    _debug("weights installation complete")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def bootstrap() -> Dict[str, Any]:
    """Prepare the frozen environment.  Cheap: no torch/nnU-Net import here."""
    silence_nnunet_path_warnings()
    weights = configure_weights_dir()
    install_nnunet_import_patch()
    return {"weights_dir": weights, "frozen": is_frozen()}
