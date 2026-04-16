"""Cross-platform build script for the SPY Predictor desktop app.

Usage (from repo root):

    # Install build deps into your current env
    pip install -r requirements.txt pyinstaller

    # Train models so they get bundled (recommended — avoids first-run download)
    python -m scripts.train --no-lstm   # or with --no-lstm omitted if you have torch

    # Build
    python launcher/build.py

The output goes to ``dist/SPYPredictor/`` and contains the .exe (on Windows)
or equivalent binary (on macOS / Linux), plus all dependencies.
"""
from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.check_call(cmd)


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    spec = root / "launcher" / "app.spec"
    if not spec.exists():
        sys.stderr.write(f"ERROR: missing spec file at {spec}\n")
        return 1

    # Clean old build artifacts so we don't accidentally ship stale files.
    for target in ("build", "dist"):
        path = root / target
        if path.exists():
            print(f"Removing old {path}")
            shutil.rmtree(path)

    # Bundle the trained models + data cache if they exist. Users who train
    # first get a ready-to-use app with no network dependency on launch.
    datas_to_stage = []
    for sub in ("models", "data"):
        src = root / sub
        if src.exists() and any(src.iterdir()):
            datas_to_stage.append(src)

    print(f"Building SPYPredictor on {platform.system()} ({platform.machine()})")
    print(f"Python {sys.version.split()[0]}, spec: {spec.relative_to(root)}")

    _run(
        [
            sys.executable,
            "-m",
            "PyInstaller",
            str(spec),
            "--noconfirm",
            "--clean",
            f"--distpath={root / 'dist'}",
            f"--workpath={root / 'build'}",
        ]
    )

    # Post-build: copy optional models/data directories next to the .exe so
    # the frozen launcher can ship with pre-trained weights.
    out_dir = root / "dist" / "SPYPredictor"
    if not out_dir.exists():
        sys.stderr.write(
            "ERROR: expected dist/SPYPredictor/ after build — check PyInstaller output above.\n"
        )
        return 2

    for src in datas_to_stage:
        dest = out_dir / src.name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
        print(f"Staged {src.name}/ -> {dest}")

    print("\nDone. Output:", out_dir)
    if platform.system() == "Windows":
        print(f"    Launch:  {out_dir / 'SPYPredictor.exe'}")
    else:
        print(f"    Launch:  {out_dir / 'SPYPredictor'}")
    print("    Ship the entire SPYPredictor/ folder — the .exe alone won't run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
