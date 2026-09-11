# Copyright 2024-2026 Hartmut Häntze
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0

"""Download and stage model weights for the Windows build.

This is the build-time counterpart of ``mrsegmentator.config._download_model``:
it produces the exact directory layout that ``config._resolve_root()`` expects
from ``MRSEG_WEIGHTS_PATH``::

    weights/
        base/       plans.json, dataset.json, fold_0/ ... version.json
        body_comp/  ...

Because ``version.json`` is staged alongside the weights, ``ensure_model()``
sees an up-to-date model at runtime and never attempts a download -- the
shipped executable works offline and needs no write access to its own folder.

The model URLs and checksums are read straight out of ``src/mrsegmentator/
config.py`` (parsed, not imported, so this runs without the package installed),
so there is exactly one place where the registry lives.

Usage:
    python windows/fetch_weights.py --outdir build/windows/weights
    python windows/fetch_weights.py --outdir ... --models base
"""

import argparse
import ast
import hashlib
import json
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PY = ROOT / "src" / "mrsegmentator" / "config.py"
VERSION_FILE = "version.json"
# One of these must exist for a directory to look like nnU-Net weights.
NNUNET_MARKERS = ("plans.json", "dataset.json")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
def read_model_registry(config_py: Path = CONFIG_PY) -> Dict[str, Dict[str, Any]]:
    """Extract MODEL_REGISTRY from config.py without importing the package."""
    tree = ast.parse(config_py.read_text(encoding="utf-8"), filename=str(config_py))

    for node in tree.body:
        targets: List[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue

        for target in targets:
            if isinstance(target, ast.Name) and target.id == "MODEL_REGISTRY":
                return ast.literal_eval(node.value)  # type: ignore[arg-type]

    raise RuntimeError(f"MODEL_REGISTRY not found in {config_py}")


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------
def _progress(done: int, total: int, label: str) -> None:
    if total > 0:
        pct = min(100.0, 100.0 * done / total)
        bar = f"{done / 2**20:8.1f} / {total / 2**20:.1f} MiB ({pct:5.1f}%)"
    else:
        bar = f"{done / 2**20:8.1f} MiB"
    print(f"\r  {label}: {bar}", end="", flush=True)


def download(url: str, destination: Path) -> None:
    """Stream ``url`` to ``destination`` with a plain-text progress readout."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    label = destination.name

    with urllib.request.urlopen(url) as response:
        total = int(response.headers.get("Content-Length", 0) or 0)
        done = 0
        with open(destination, "wb") as handle:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                handle.write(chunk)
                done += len(chunk)
                _progress(done, total, label)
    print()


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Layout normalisation
# ---------------------------------------------------------------------------
def _looks_like_weights(directory: Path) -> bool:
    return any((directory / marker).is_file() for marker in NNUNET_MARKERS)


def _flatten_single_nested_dir(model_dir: Path) -> None:
    """Move contents up if the archive wrapped everything in one folder.

    Only triggers when the top level does not look like nnU-Net weights but a
    single child directory does, so a correctly shaped archive is left alone.
    """
    if _looks_like_weights(model_dir):
        return

    entries = list(model_dir.iterdir())
    if len(entries) != 1 or not entries[0].is_dir():
        return
    if not _looks_like_weights(entries[0]):
        return

    nested = entries[0]
    print(f"  flattening nested directory {nested.name}/")
    for item in list(nested.iterdir()):
        shutil.move(str(item), str(model_dir / item.name))
    nested.rmdir()


def _ensure_version_file(model_dir: Path, version: float) -> None:
    """Guarantee a version.json so the runtime never re-downloads.

    ``config.ensure_model()`` compares ``version.json`` against the registry and
    downloads whenever the file is missing (version 0.0).  In a shipped build
    that would mean a download on every run, into a directory that may not even
    be writable.
    """
    version_file = model_dir / VERSION_FILE
    if version_file.is_file():
        try:
            current = json.loads(version_file.read_text(encoding="utf-8"))
            if float(current.get("weights_version", 0.0)) >= float(version):
                return
        except Exception:
            pass

    print(f"  writing {VERSION_FILE} (weights_version={version})")
    version_file.write_text(json.dumps({"weights_version": version}, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def stage_model(
    model_name: str,
    entry: Dict[str, Any],
    outdir: Path,
    cache_dir: Path,
    force: bool = False,
) -> Path:
    """Ensure ``outdir/<model_name>`` holds verified, ready-to-ship weights."""
    model_dir = outdir / model_name

    if model_dir.is_dir() and _looks_like_weights(model_dir) and not force:
        print(f"[{model_name}] already staged at {model_dir}, skipping")
        _ensure_version_file(model_dir, entry["version"])
        return model_dir

    if model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    archive = cache_dir / entry["zip_name"]
    expected = entry.get("sha256")

    if archive.is_file() and expected and sha256_of(archive) == expected:
        print(f"[{model_name}] using cached archive {archive}")
    else:
        print(f"[{model_name}] downloading {entry['url']}")
        download(entry["url"], archive)

        if expected:
            actual = sha256_of(archive)
            if actual != expected:
                archive.unlink()
                raise RuntimeError(
                    f"Checksum mismatch for '{model_name}': "
                    f"expected {expected}, got {actual}. The download was removed."
                )
            print(f"  checksum ok ({actual[:16]}...)")

    print(f"[{model_name}] extracting to {model_dir}")
    with zipfile.ZipFile(archive, "r") as zip_ref:
        zip_ref.extractall(model_dir)

    _flatten_single_nested_dir(model_dir)

    if not _looks_like_weights(model_dir):
        raise RuntimeError(
            f"Extracted archive for '{model_name}' does not look like nnU-Net "
            f"weights: none of {NNUNET_MARKERS} found in {model_dir}"
        )

    _ensure_version_file(model_dir, entry["version"])

    size = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
    print(f"[{model_name}] staged, {size / 2**20:.0f} MiB")
    return model_dir


def stage_weights(
    outdir: Path,
    models: List[str],
    cache_dir: Path,
    force: bool = False,
) -> Path:
    registry = read_model_registry()

    unknown = [name for name in models if name not in registry]
    if unknown:
        raise SystemExit(f"Unknown model(s) {unknown}. Available: {sorted(registry)}")

    outdir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    for name in models:
        stage_model(name, registry[name], outdir, cache_dir, force=force)

    return outdir


def main() -> None:
    registry = read_model_registry()

    parser = argparse.ArgumentParser(description="Stage MRSegmentator weights for a frozen build")
    parser.add_argument("--outdir", type=Path, required=True, help="target weights directory")
    parser.add_argument(
        "--models",
        nargs="+",
        default=sorted(registry),
        choices=sorted(registry),
        help="models to stage (default: all)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "mrseg_weight_cache",
        help="where downloaded archives are kept between builds",
    )
    parser.add_argument("--force", action="store_true", help="re-download even if already staged")
    args = parser.parse_args()

    stage_weights(args.outdir, list(args.models), args.cache_dir, force=args.force)
    print(f"\nWeights ready in {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
