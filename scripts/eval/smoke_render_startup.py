from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT / "backend"


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def main() -> None:
    port = free_port()
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "app:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]

    proc = subprocess.Popen(
        cmd,
        cwd=BACKEND_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        deadline = time.time() + 2.0
        health_ok = False
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError("uvicorn exited before /health became available")
            try:
                resp = requests.get(f"http://127.0.0.1:{port}/health", timeout=0.25)
                if resp.status_code == 200:
                    health_ok = True
                    break
            except Exception:
                pass
            time.sleep(0.1)

        if not health_ok:
            raise RuntimeError("/health not ready within 2 seconds")

        print(json.dumps({"ok": True, "port": port, "health": "OK"}, ensure_ascii=False))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"SMOKE_RENDER_STARTUP_FAIL: {exc}")
        sys.exit(1)
