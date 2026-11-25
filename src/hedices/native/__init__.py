"""Native executor pool of the Branch: @NoRely calls of Python-bound stages in pre-started
executor processes, coupled through shared-memory channels (hedices/native/_hedx.cpp)."""
from __future__ import annotations

import os
import pickle
import queue
import subprocess
import sys
import threading
import time
import uuid

from . import _hedx

CAPACITY = 64 << 20  # bytes per channel (shared memory is committed only as it is touched)


class Executor:
    def __init__(self, tag: str, k: int):
        self.name = f"/hedx-{tag}-{k}"
        self.ch = _hedx.Channel(self.name, CAPACITY, True)
        self.proc = subprocess.Popen([sys.executable, "-m", "hedices.native.child", self.name],
                                     stdin=subprocess.DEVNULL)
        self.loaded: set = set()

    def call(self, msg):
        rep = pickle.loads(self.ch.call(pickle.dumps(msg, protocol=5), self.proc.pid))
        if rep[0] == "error":
            raise RuntimeError(f"executor {self.name}: {rep[1]}")
        return rep[1:]

    def stop(self):
        self.ch.close()
        try:
            self.proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.proc.kill()


class NativePool:
    """``width`` executors shared by the @NoRely stages of one Branch.  Callers hold CPU slots
    from the Branch ledger, whose total never exceeds ``width``, so an executor is free for every
    admitted call.  A stage's function package is sent to an executor once; the first use of a
    stage primes every executor with it, so model state is built before calls are timed."""

    def __init__(self, width: int):
        tag = f"{os.getpid()}-{uuid.uuid4().hex[:6]}"
        self.execs = [Executor(tag, k) for k in range(max(1, width))]
        self.free: queue.Queue = queue.Queue()
        for e in self.execs:
            self.free.put(e)
        self._primed: set = set()
        self._prime_lock = threading.Lock()
        for e in self.execs:  # executors import the runtime libraries before the first call
            e.call(("ping",))

    def _run(self, e: Executor, h, blob, args, kwargs, threads):
        msg = ("exec", h, None if h in e.loaded else blob, args, kwargs, threads)
        out, exec_s = e.call(msg)
        e.loaded.add(h)
        return out, exec_s

    def prime(self, h, blob, args, kwargs, threads) -> None:
        if h in self._primed:
            return
        with self._prime_lock:
            if h in self._primed:
                return
            got = [self.free.get() for _ in self.execs]
            try:
                ts = [threading.Thread(target=self._run, args=(e, h, blob, args, kwargs, threads))
                      for e in got]
                for t in ts:
                    t.start()
                for t in ts:
                    t.join()
            finally:
                for e in got:
                    self.free.put(e)
            self._primed.add(h)

    def run(self, h, blob, args, kwargs, threads):
        e = self.free.get()
        try:
            return self._run(e, h, blob, args, kwargs, threads)
        finally:
            self.free.put(e)

    def stop(self):
        for e in self.execs:
            e.stop()
