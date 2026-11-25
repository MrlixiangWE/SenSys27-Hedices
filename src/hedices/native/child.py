"""Executor process of the native pool: runs stage calls handed over by its Branch."""
from __future__ import annotations

import pickle
import sys
import time
import traceback


def main(name: str) -> None:
    from hedices.native import _hedx

    _hedx.die_with_parent()
    from hedices import runtime
    from hedices.package import load_package

    for lib in ("numpy", "onnxruntime", "torch"):  # imported once, before the first call
        try:
            __import__(lib)
        except ImportError:
            pass
    runtime.install_onnxruntime_adapter()
    from hedices import accel, package

    package.VALUE_HOOK = accel.accelerate
    ch = _hedx.Channel(name)
    fns: dict = {}
    while True:
        req = ch.recv()
        if req is None:
            return
        try:
            msg = pickle.loads(req)
            if msg[0] == "ping":
                rep = ("ok",)
            else:
                _, h, blob, args, kwargs, threads = msg
                fn = fns.get(h)
                if fn is None:
                    fn = fns[h] = load_package(blob)
                runtime.set_native_threads(threads)
                t0 = time.perf_counter()
                out = fn(*args, **kwargs)
                rep = ("ok", out, time.perf_counter() - t0)
            data = pickle.dumps(rep, protocol=5)
        except Exception:
            data = pickle.dumps(("error", traceback.format_exc()))
        ch.reply(data)


if __name__ == "__main__":
    main(sys.argv[1])
