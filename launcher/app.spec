# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the SPY prediction dashboard.

Streamlit is notoriously fiddly to freeze because it:
  * uses dynamic imports (plugins, components, backends),
  * ships a ``static/`` front-end bundle that must be copied into the .exe,
  * has a version-config file (``version.py``) that it expects to read at runtime,
  * loads metadata via ``importlib.metadata`` which PyInstaller has to be told
    to preserve.

We handle each of those explicitly below. Build with:

    pyinstaller launcher/app.spec --noconfirm --clean
"""
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_submodules,
    copy_metadata,
)

# ---- paths ------------------------------------------------------------------
# ``SPECPATH`` is injected by PyInstaller and points at the directory
# containing this .spec file. Going up one level gets us the repo root.
ROOT = Path(SPECPATH).resolve().parent


# ---- data files to bundle ---------------------------------------------------
datas: list[tuple[str, str]] = [
    # The actual Streamlit app. PyInstaller places this under ``app/`` inside
    # the bundle, which matches what ``run_app.py`` looks for.
    (str(ROOT / "app" / "dashboard.py"), "app"),
    # The src package — our code that the dashboard imports at runtime.
    (str(ROOT / "src"), "src"),
]

# Streamlit's bundled front-end assets + runtime config. Without these the
# server boots but the browser shows a blank page.
datas += collect_data_files("streamlit")
datas += collect_data_files("altair", include_py_files=False)
datas += collect_data_files("plotly")
datas += collect_data_files("xgboost")
datas += collect_data_files("yfinance")
datas += collect_data_files("sklearn")

# Package metadata — needed because Streamlit (and several deps) call
# ``importlib.metadata.version(...)`` at import time.
for pkg in ("streamlit", "xgboost", "yfinance", "scikit-learn", "plotly", "pandas", "numpy"):
    try:
        datas += copy_metadata(pkg)
    except Exception:
        pass  # If one isn't installed locally we just skip — harmless.


# ---- hidden imports ---------------------------------------------------------
# Modules that PyInstaller's static analyzer won't spot because they're loaded
# dynamically (entry points, plugin systems, version-gated imports, etc).
hiddenimports: list[str] = []
for pkg in (
    "streamlit",
    "streamlit.web",
    "streamlit.web.cli",
    "streamlit.web.bootstrap",
    "streamlit.runtime.scriptrunner.magic_funcs",
    "streamlit.components.v1",
    "altair",
    "plotly",
    "plotly.graph_objects",
    "plotly.express",
    "xgboost",
    "yfinance",
    "sklearn.utils._cython_blas",
    "sklearn.neighbors.typedefs",
    "sklearn.neighbors.quad_tree",
    "sklearn.tree._utils",
):
    try:
        hiddenimports += collect_submodules(pkg)
    except Exception:
        hiddenimports.append(pkg)

# Torch is optional — include it only if it's installed in the build env.
try:
    import torch  # noqa: F401

    hiddenimports += collect_submodules("torch")
    datas += collect_data_files("torch")
except ImportError:
    pass


# ---- analysis / build -------------------------------------------------------
block_cipher = None

a = Analysis(
    [str(ROOT / "launcher" / "run_app.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    # Strip test suites from huge deps to keep the bundle smaller.
    excludes=["tests", "pytest", "IPython", "jupyter"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# One-folder build: faster startup and easier to ship updates than one-file.
# A one-file .exe can be built by passing --onefile to PyInstaller, but it
# pays a 5-10s extraction cost every launch and has issues with some AV
# engines flagging it.
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="SPYPredictor",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,          # UPX compression sometimes trips Windows Defender
    console=True,       # Keep console so errors are visible & Ctrl+C works
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ROOT / "launcher" / "icon.ico") if (ROOT / "launcher" / "icon.ico").exists() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="SPYPredictor",
)
