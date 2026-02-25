from __future__ import annotations

import json
import os
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


def timed_get(url: str, timeout: float = 5.0) -> tuple[requests.Response, float]:
    t0 = time.perf_counter()
    resp = requests.get(url, timeout=timeout)
    return resp, time.perf_counter() - t0


def timed_post(url: str, payload: dict[str, object], timeout: float = 10.0) -> tuple[requests.Response, float]:
    t0 = time.perf_counter()
    resp = requests.post(url, json=payload, timeout=timeout)
    return resp, time.perf_counter() - t0


def main() -> None:
    port = free_port()
    env = dict(**os.environ)
    env.setdefault("SCIBABEL_ENV", "production")
    env.setdefault("EVIDENCE_ENABLED", "false")

    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=str(BACKEND_DIR),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        deadline = time.time() + 10
        while time.time() < deadline:
            try:
                r = requests.get(f"http://127.0.0.1:{port}/health", timeout=0.25)
                if r.status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(0.1)
        else:
            raise RuntimeError("backend did not become healthy")

        base = f"http://127.0.0.1:{port}"
        health, t_health = timed_get(f"{base}/health", timeout=3)
        ready, t_ready = timed_get(f"{base}/ready", timeout=3)
        ann, t_ann = timed_post(
            f"{base}/annotate",
            {
                "text": "We optimize sparse attention under distribution shift.",
                "src": "auto",
                "tgt": "PM",
                "max_terms": 6,
            },
            timeout=8,
        )

        out = {
            "health_status": health.status_code,
            "ready_status": ready.status_code,
            "annotate_status": ann.status_code,
            "health_sec": round(t_health, 4),
            "ready_sec": round(t_ready, 4),
            "annotate_sec": round(t_ann, 4),
        }
        print(json.dumps(out, ensure_ascii=False))

        if t_health >= 1.0:
            raise RuntimeError(f"health too slow: {t_health:.3f}s")
        if t_ready >= 1.0:
            raise RuntimeError(f"ready too slow: {t_ready:.3f}s")
        if t_ann >= 5.0:
            raise RuntimeError(f"annotate too slow: {t_ann:.3f}s")
        if health.status_code != 200:
            raise RuntimeError(f"health status {health.status_code}")
        if ready.status_code != 200:
            raise RuntimeError(f"ready status {ready.status_code}")
        if ann.status_code not in {200, 503}:
            raise RuntimeError(f"annotate unexpected status {ann.status_code}")
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
        print(f"SMOKE_RENDER_BEHAVIOR_FAIL: {exc}")
        sys.exit(1)
