"""Branch-side inference optimization of the models stages load (part of the Branch runtime).

A model a stage builds through a node-resident asset handle stays resident on the worker, so the
Branch can afford a one-time compilation step for it: on the first call, the convolutional
feature extractor (``backbone``), or a model without one as a whole, is traced with the real
input, frozen and passed through TorchScript's inference passes (operator fusion, constant
folding, prepacked weights), and kept only if it reproduces the eager output.  Later calls with
the same input shape run the optimized graph; any other shape, or a module that cannot be
traced, runs eagerly.  Numerics are unchanged (no quantization or precision change).

With BATCH set (Branch --norely-batch), calls that reach one resident model while it is busy are
coalesced: @NoRely lets the Branch overlap the independent invocations of a stage, and those that
queue on the model run as one batch along dimension 0 when it frees, then get their own rows back.
Opportunistic, no waiting window: a call that finds the model idle runs at once, so a lone call
pays nothing.  Calls that do not overlap (stages without @NoRely run one at a time) never batch.
Inference is per-sample independent (eval mode), so a call's rows equal its unbatched result.
"""
from __future__ import annotations

import sys
import threading
import warnings

BATCH = False  # coalesce overlapping calls of a resident model (Branch --norely-batch)
MAX_BATCH = 64  # samples per coalesced run
# native threads of a coalesced run: the calls it serves wait for it and leave their cores idle, so
# the run takes the worker's CPU-backed slots (set by the Branch); 0 keeps the leader's own grant
BATCH_THREADS = 0
_lock = threading.Lock()
_seen: set = set()
_whole: dict = {}  # id(model) -> its whole-model LazyTrace
_cls: list = []  # the torch-dependent wrapper class, made on first use


class _Call:
    __slots__ = ("x", "n", "out", "err", "lead", "ev")

    def __init__(self, x):
        self.x, self.n = x, int(x.shape[0])
        self.out = self.err = None
        self.lead = False
        self.ev = threading.Event()


class _Coalescer:
    """Opportunistic batching of the calls of one model: the caller that finds it idle runs; the
    calls arriving meanwhile queue, and when a run ends the queued calls with the same per-sample
    shape run as the next batch, led by the first of them."""

    def __init__(self, run):
        self.run = run
        self.cv = threading.Lock()
        self.q: list = []
        self.busy = False

    def __call__(self, x):
        c = _Call(x)
        with self.cv:
            self.q.append(c)
            if not self.busy:
                self.busy = c.lead = True
        if not c.lead:
            c.ev.wait()
        if c.lead:
            self._lead()
        if c.err is not None:
            raise c.err
        return c.out

    def _lead(self):
        with self.cv:
            head = self.q[0]
            key = (tuple(head.x.shape[1:]), head.x.dtype)
            batch, rest, total = [], [], 0
            for c in self.q:
                if (tuple(c.x.shape[1:]), c.x.dtype) == key and total + c.n <= MAX_BATCH:
                    batch.append(c)
                    total += c.n
                else:
                    rest.append(c)
            self.q = rest
        try:
            t = sys.modules["torch"]
            if len(batch) > 1 and BATCH_THREADS:
                t.set_num_threads(BATCH_THREADS)
            outs = _split(self.run(batch[0].x if len(batch) == 1 else t.cat([c.x for c in batch])),
                          [c.n for c in batch])
            for c, o in zip(batch, outs):
                c.out = o
        except Exception as e:  # every caller of the run sees the failure
            for c in batch:
                c.err = e
        with self.cv:
            nxt = self.q[0] if self.q else None
            if nxt is None:
                self.busy = False
            else:
                nxt.lead = True  # the next batch runs on the thread of its first call
        for c in batch:
            if not c.lead:
                c.ev.set()
        if nxt is not None:
            nxt.ev.set()


def _split(out, sizes):
    """The rows of each call: tensors split along dimension 0, containers element by element."""
    t = sys.modules["torch"]
    if len(sizes) == 1:
        return [out]
    if isinstance(out, t.Tensor):
        return list(t.split(out, sizes))
    if isinstance(out, dict):
        parts = {k: _split(v, sizes) for k, v in out.items()}
        return [type(out)((k, parts[k][i]) for k in out) for i in range(len(sizes))]
    if isinstance(out, (list, tuple)):
        parts = [_split(v, sizes) for v in out]
        return [type(out)(p[i] for p in parts) for i in range(len(sizes))]
    raise TypeError(f"cannot split a {type(out).__name__} per call")


def _lazy_trace_class(t):
    if _cls:
        return _cls[0]

    class LazyTrace(t.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
            self._graphs: dict = {}
            self._failed = False
            self._glock = threading.Lock()
            self._co = _Coalescer(self._forward1)

        def forward(self, x):
            if BATCH and isinstance(x, t.Tensor) and x.dim() > 0 and not t.is_grad_enabled():
                return self._co(x)
            return self._forward1(x)

        def _forward1(self, x):
            if self._failed or not isinstance(x, t.Tensor) or t.is_grad_enabled():
                return self.inner(x)
            key = (tuple(x.shape), x.dtype)
            g = self._graphs.get(key)
            if g is None:
                with self._glock:
                    g = self._graphs.get(key)
                    if g is None:
                        try:
                            with warnings.catch_warnings(), t.no_grad():
                                warnings.simplefilter("ignore")
                                g = t.jit.optimize_for_inference(
                                    t.jit.freeze(t.jit.trace(self.inner.eval(), (x,), strict=False)))
                                # the optimized graph must reproduce the eager result
                                if not _same(t, g(x), self.inner(x)):
                                    raise ValueError("traced graph differs from eager")
                        except Exception:
                            self._failed = True
                            return self.inner(x)
                        self._graphs[key] = g
            return g(x)

    _cls.append(LazyTrace)
    return LazyTrace


def _same(t, a, b) -> bool:
    if isinstance(a, t.Tensor) and isinstance(b, t.Tensor):
        return a.shape == b.shape and bool(t.allclose(a, b, rtol=1e-4, atol=1e-5))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(t, a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(t, x, y) for x, y in zip(a, b))
    return False


def accelerate(obj):
    """Install lazy tracing on a resident inference model; other objects pass through.  A model
    with a ``backbone`` gets only its feature extractor traced (detection heads vary in shape per
    input); a model without one is traced whole."""
    if id(obj) in _seen:
        return _whole.get(id(obj), obj)
    t = sys.modules.get("torch")
    # torch may be half-imported by another thread: no Module class yet means nothing to do now
    module = getattr(getattr(t, "nn", None), "Module", None)
    if module is None or not isinstance(obj, module) or obj.training:
        return obj
    with _lock:
        if id(obj) in _seen:
            return _whole.get(id(obj), obj)
        lazy = _lazy_trace_class(t)
        if isinstance(obj, lazy):
            return obj
        inner = getattr(obj, "backbone", None)
        if isinstance(inner, t.nn.Module):
            if not isinstance(inner, lazy):
                obj.backbone = lazy(inner)
        else:
            _whole[id(obj)] = lazy(obj)
        _seen.add(id(obj))
    return _whole.get(id(obj), obj)
