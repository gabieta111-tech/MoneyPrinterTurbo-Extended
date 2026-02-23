#!/usr/bin/env python3
"""
MoneyPrinterTurbo Desktop App
Wraps the Streamlit web UI in a native desktop window using pywebview.
"""

import atexit
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request

# ── Configuration ────────────────────────────────────────────────────────────
APP_TITLE = "MoneyPrinterTurbo"
WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 900
STARTUP_TIMEOUT = 30  # seconds to wait for Streamlit to start


def find_free_port():
    """Find an available port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for_server(url, timeout=STARTUP_TIMEOUT):
    """Wait until the Streamlit server is responding."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except Exception:
            time.sleep(0.3)
    return False


def main():
    # Resolve project root (same directory as this script)
    root_dir = os.path.dirname(os.path.abspath(__file__))
    webui_script = os.path.join(root_dir, "webui", "Main.py")

    if not os.path.isfile(webui_script):
        print(f"ERROR: Could not find {webui_script}")
        sys.exit(1)

    # Pick a free port so we don't collide with anything
    port = find_free_port()
    url = f"http://127.0.0.1:{port}"

    print(f"Starting Streamlit on {url} ...")

    # Build the Streamlit command
    streamlit_cmd = [
        sys.executable, "-m", "streamlit", "run",
        webui_script,
        f"--server.port={port}",
        "--server.address=127.0.0.1",
        "--server.headless=true",           # don't auto-open a browser
        "--browser.gatherUsageStats=false",
        "--server.enableCORS=true",
    ]

    # Start Streamlit as a subprocess
    env = os.environ.copy()
    env["PYTHONPATH"] = root_dir + os.pathsep + env.get("PYTHONPATH", "")

    proc = subprocess.Popen(
        streamlit_cmd,
        cwd=root_dir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid if sys.platform != "win32" else None,
    )

    def cleanup():
        """Kill the Streamlit subprocess tree on exit."""
        if proc.poll() is None:
            print("Shutting down Streamlit server...")
            try:
                if sys.platform == "win32":
                    proc.terminate()
                else:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (ProcessLookupError, OSError):
                pass

    atexit.register(cleanup)

    # Wait for the server to become ready
    print("Waiting for Streamlit server to start...")
    if not wait_for_server(url):
        print("ERROR: Streamlit server did not start within the timeout.")
        print("Streamlit output:")
        if proc.stdout:
            print(proc.stdout.read().decode(errors="replace"))
        cleanup()
        sys.exit(1)

    print(f"Streamlit is ready at {url}")

    # ── Open the native desktop window ───────────────────────────────────
    try:
        import webview
    except ImportError:
        print(
            "\nERROR: pywebview is not installed.\n"
            "Install it with:  pip install pywebview\n\n"
            "On Linux you may also need:\n"
            "  sudo apt install python3-gi gir1.2-webkit2-4.1\n"
            "  (or equivalent for your distribution)\n"
        )
        cleanup()
        sys.exit(1)

    print(f"Opening desktop window: {APP_TITLE}")

    window = webview.create_window(
        title=APP_TITLE,
        url=url,
        width=WINDOW_WIDTH,
        height=WINDOW_HEIGHT,
        resizable=True,
        min_size=(800, 600),
        text_select=True,
    )

    # This blocks until the window is closed
    webview.start(debug=False)

    # Window closed — clean up
    cleanup()
    print("Desktop app closed.")


if __name__ == "__main__":
    main()
