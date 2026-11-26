"""Shared pieces of the experiment drivers (they run inside the master namespace)."""
from __future__ import annotations

import json
import os
import platform
import statistics
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MASTER_IP = os.environ.get("HEDICES_IP", "127.0.0.1")


def closed_loop(run_one, make_input, concurrency: int, *, duration_s: float | None = None,
                n: int | None = None, start: int = 0, stop=None, inflight: dict | None = None):
    """``concurrency`` clients; each submits one workflow instance and waits for it.

    Stops after ``duration_s`` seconds (warm-up) or ``n`` completed instances, or once ``stop()`` is true (no new
    instance is submitted; the ones in flight finish).  ``inflight``, when given, holds index -> wall start of the
    instances in flight, and ``inflight["_done"]`` the wall time of the latest completion (for a caller watching).
    Returns (records, next_index); a record is (index, wall_start, latency_s).
    """
    lock = threading.Lock()
    nxt = [start]
    done: list = []
    errors: list = []
    deadline = time.monotonic() + duration_s if duration_s is not None else None

    def client():
        while True:
            with lock:
                if errors or (n is not None and nxt[0] - start >= n):
                    return
                if deadline is not None and time.monotonic() >= deadline:
                    return
                if stop is not None and stop():
                    return
                i = nxt[0]
                nxt[0] += 1
            args, kwargs = make_input(i)
            wall, t0 = time.time(), time.perf_counter()
            if inflight is not None:
                with lock:
                    inflight[i] = wall
            try:
                run_one(args, kwargs)
            except Exception as exc:  # surface the first failure
                with lock:
                    errors.append(exc)
                    if inflight is not None:
                        inflight.pop(i, None)
                return
            with lock:
                done.append((i, wall, time.perf_counter() - t0))
                if inflight is not None:
                    inflight.pop(i, None)
                    inflight["_done"] = time.time()

    threads = [threading.Thread(target=client, daemon=True) for _ in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]
    return sorted(done), nxt[0]


def summary(lat: list[float]) -> dict:
    s = sorted(lat)
    if not s:
        return {}
    pct = lambda q: s[min(len(s) - 1, int(round(q * (len(s) - 1))))]
    return {"n": len(s), "mean": statistics.fmean(s), "median": pct(0.5), "p95": pct(0.95),
            "stdev": statistics.pstdev(s), "min": s[0], "max": s[-1]}


def env_info() -> dict:
    import numpy

    info = {"python": platform.python_version(), "numpy": numpy.__version__,
            "node": os.environ.get("HEDICES_NODE"), "host": platform.node(),
            **{k: v for k, v in os.environ.items() if k.startswith("HEDICES_")}}
    for mod in ("torch", "onnxruntime", "ray", "dask", "distributed", "grpc"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            pass
    return info


def save(path: str | Path, obj: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, default=float))
    tmp.replace(path)
    print(f"saved {path}", flush=True)
