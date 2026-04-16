"""Desktop launcher for the SPY prediction dashboard.

This is the entry point used when the app is packaged as a Windows .exe
(or a macOS / Linux binary). It:

  1. Resolves the path to ``dashboard.py`` whether we're running from source
     or from a frozen PyInstaller bundle (``sys._MEIPASS``).
  2. Picks a free local port.
  3. Starts the Streamlit server in-process using
     ``streamlit.web.bootstrap.run`` — this avoids the #1 PyInstaller headache
     of trying to spawn the ``streamlit`` CLI as a subprocess.
  4. Opens the user's default browser to the server URL.
  5. Blocks until the user closes the console window (Ctrl+C or closing
     the window), which shuts the server down cleanly.

Keeping this logic in a launcher rather than calling ``streamlit run``
directly is what makes the .exe build reliable.
"""
from __future__ import annotations

import os
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path


def _resource_path(*parts: str) -> Path:
    """Resolve a path relative to either the source tree or a PyInstaller bundle.

    PyInstaller one-file builds extract data to a temp dir and expose it via
    ``sys._MEIPASS``. One-folder builds put data next to the executable.
    Source runs just use the repo root.
    """
    if getattr(sys, "frozen", False):
        base = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    else:
        base = Path(__file__).resolve().parent.parent
    return base.joinpath(*parts)


def _find_free_port(preferred: int = 8501) -> int:
    """Return ``preferred`` if it's free, otherwise ask the OS for any open port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            pass
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _open_browser_when_ready(url: str, delay: float = 2.0) -> None:
    """Open the default browser once the Streamlit server has a second to boot."""
    def _open() -> None:
        time.sleep(delay)
        try:
            webbrowser.open(url, new=2)
        except Exception:
            # If browser launch fails, the console still shows the URL.
            pass

    threading.Thread(target=_open, daemon=True).start()


def main() -> int:
    # Set up env *before* importing streamlit — some of these only take effect
    # if they're present when the server boots.
    port = _find_free_port(8501)
    os.environ.setdefault("STREAMLIT_SERVER_PORT", str(port))
    os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "true")
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    os.environ.setdefault("STREAMLIT_GLOBAL_DEVELOPMENT_MODE", "false")
    # Suppress the "Welcome to Streamlit" / "email" prompt on first launch
    os.environ.setdefault("STREAMLIT_SERVER_ENABLE_CORS", "false")
    os.environ.setdefault("STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION", "false")

    dashboard = _resource_path("app", "dashboard.py")
    if not dashboard.exists():
        sys.stderr.write(
            f"ERROR: dashboard.py not found at {dashboard}\n"
            f"This is a packaging problem — the app was not bundled correctly.\n"
        )
        return 1

    url = f"http://localhost:{port}"
    print("=" * 64)
    print("  SPY Cross-Asset Prediction Dashboard")
    print("=" * 64)
    print(f"  Starting server on {url}")
    print("  A browser window will open automatically.")
    print("  Close this window to shut down the server.")
    print("=" * 64)

    _open_browser_when_ready(url)

    # Import here so PyInstaller's static analysis picks streamlit up as a
    # dependency even in the launcher.
    from streamlit.web import bootstrap  # type: ignore

    # bootstrap.run signature across recent streamlit versions:
    #   run(main_script_path, is_hello, args, flag_options)
    # ``args`` are CLI-style args for the script itself (none here).
    try:
        bootstrap.run(str(dashboard), False, [], flag_options={})
    except TypeError:
        # Older streamlit (<1.12) took a positional `command_line` instead of is_hello.
        bootstrap.run(str(dashboard), "", [], flag_options={})  # type: ignore[arg-type]
    return 0


if __name__ == "__main__":
    sys.exit(main())
