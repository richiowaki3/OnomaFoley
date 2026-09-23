# -*- coding: utf-8 -*-
"""
run_dual_engine.py: Robust launcher for Dual-Engine Demo.
Finds an available free port automatically to prevent "Port is not available" errors.
"""

import sys
import socket
import subprocess
from pathlib import Path


def find_free_port(preferred_ports=[8504, 8505, 8506, 8507, 8508, 8509, 8510]):
    for port in preferred_ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue

    # Fallback: let OS assign dynamic free port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def main():
    root = Path(__file__).resolve().parent
    script = root / "dual_engine_demo.py"

    port = find_free_port()
    print("=" * 65)
    print(f"[OnomaDict] Launching Dual-Engine Synthesizer Demo on Port {port}...")
    print(f"  URL: http://localhost:{port}")
    print("=" * 65)

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(script),
        "--server.port",
        str(port),
        "--browser.gatherUsageStats",
        "false",
    ]
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
