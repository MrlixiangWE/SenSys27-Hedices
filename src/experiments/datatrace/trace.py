"""Stage-call data tracing (see sitecustomize.py)."""
from __future__ import annotations

import functools
import hashlib
import importlib.abc
import importlib.machinery
import inspect
import json
import os
import pickle
import sys
import threading
import time

MODULES = {"apps.common.w1", "apps.common.w2", "apps.common.w3", "apps.common.w4"}
_tls = threading.local()
_lock = threading.Lock()
_fh = None


def _is_data(obj, depth: int = 0) -> bool:
    """Application data (arrays, buffers, scalars, strings and containers of them); models and
    runtime sessions passed to a stage are node-resident state, not data the call moves."""
    if obj is None or isinstance(obj, (bool, int, float, str, bytes, bytearray, memoryview)):
        return True
    if type(obj).__module__ == "numpy":
        return True
    if depth < 4 and isinstance(obj, (list, tuple)):
        return all(_is_data(x, depth + 1) for x in obj)
    if depth < 4 and isinstance(obj, dict):
        return all(_is_data(v, depth + 1) for v in obj.values())
    return False


def _digest(obj) -> tuple[str, int]:
    if not _is_data(obj):
        return "", 0
    # protocol 4: under 5 an array pickles differently when it is read-only (as a framework hands
    # a deserialized input to the consumer) than when writable (as its producer returned it)
    try:
        blob = pickle.dumps(obj, protocol=4)
    except Exception:
        return "", 0
    return hashlib.blake2b(blob, digest_size=8).hexdigest(), len(blob)


def _out():
    global _fh
    if _fh is None:
        d = os.environ["HEDICES_DATATRACE"]
        os.makedirs(d, exist_ok=True)
        node = os.environ.get("HEDICES_NODE", "unknown")
        _fh = open(os.path.join(d, f"{node}-{os.getpid()}.jsonl"), "a", buffering=1)
    return _fh


def _wrap(name, fn):
    @functools.wraps(fn)
    def traced(*args, **kwargs):
        if getattr(_tls, "depth", 0):  # a shared function called by another: not a stage call
            return fn(*args, **kwargs)
        _tls.depth = 1
        try:
            t = time.time()
            out = fn(*args, **kwargs)
        finally:
            _tls.depth = 0
        rec = {"t": t, "n": os.environ.get("HEDICES_NODE", "unknown"), "f": name,
               "in": [_digest(a) for a in args] + [_digest(v) for v in kwargs.values()],
               "out": _digest(out)}
        with _lock:
            _out().write(json.dumps(rec) + "\n")
        return out
    return traced


def _instrument(mod) -> None:
    for name, fn in list(vars(mod).items()):
        if inspect.isfunction(fn) and fn.__module__ == mod.__name__ and not name.startswith("_") \
                and name not in ("inputs",):
            setattr(mod, name, _wrap(f"{mod.__name__}.{name}", fn))


class _Loader(importlib.abc.Loader):
    def __init__(self, inner):
        self.inner = inner

    def create_module(self, spec):
        return self.inner.create_module(spec)

    def exec_module(self, module):
        self.inner.exec_module(module)
        _instrument(module)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname not in MODULES:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None and spec.loader is not None:
            spec.loader = _Loader(spec.loader)
        return spec


def install() -> None:
    if not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
    for m in MODULES & set(sys.modules):  # already imported (e.g. by a preload)
        _instrument(sys.modules[m])
