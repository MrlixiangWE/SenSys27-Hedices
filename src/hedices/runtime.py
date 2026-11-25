"""Per-call execution context that Branch sets for native libraries (Alg. 2 line 13)."""
from __future__ import annotations

import sys
import threading

_tls = threading.local()


def native_threads() -> int | None:
    """Native-library threads t_v granted to the running call (None: library default)."""
    return getattr(_tls, "threads", None)


def set_native_threads(t: int | None) -> None:
    _tls.threads = t
    setter = getattr(sys.modules.get("torch"), "set_num_threads", None)  # None while torch imports
    if t is not None and setter is not None:
        setter(t)  # OpenMP ICV of the calling thread


def install_onnxruntime_adapter() -> None:
    """ONNX Runtime fixes its intra-op thread count when a session is created.  Sessions made on
    a Branch run each call with the threads granted to it: one underlying session per thread
    count, created lazily from the same model and options."""
    try:
        import onnxruntime as ort
    except ImportError:
        return
    if getattr(ort.InferenceSession, "_hedices_adapter", False):
        return
    base = ort.InferenceSession

    class InferenceSession(base):
        _hedices_adapter = True

        def __init__(self, path_or_bytes, sess_options=None, providers=None, *args, **kwargs):
            self._h_args = (path_or_bytes, providers, args, kwargs)
            self._h_opts = sess_options
            self._h_by_threads: dict[int, object] = {}
            self._h_lock = threading.Lock()
            super().__init__(path_or_bytes, sess_options, providers, *args, **kwargs)

        def _session_for(self, t: int):
            with self._h_lock:
                s = self._h_by_threads.get(t)
                if s is None:
                    path, providers, args, kwargs = self._h_args
                    opts = ort.SessionOptions()
                    if self._h_opts is not None:
                        opts.graph_optimization_level = self._h_opts.graph_optimization_level
                        opts.inter_op_num_threads = self._h_opts.inter_op_num_threads
                    opts.intra_op_num_threads = t
                    s = base(path, opts, providers, *args, **kwargs)
                    self._h_by_threads[t] = s
                return s

        def run(self, output_names, input_feed, run_options=None):
            t = native_threads()
            if t is None:
                return super().run(output_names, input_feed, run_options)
            return self._session_for(t).run(output_names, input_feed, run_options)

    ort.InferenceSession = InferenceSession
