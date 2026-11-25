"""Least-loaded dispatch over a pool of actor replicas.

`next(pool)` hands out the replica with the fewest calls in flight (ties in turn), and the call
made through it is counted until its result is ready.  A plain round robin sends every replica
the same share whatever its device: on a mixed cluster the calls queue behind the slow replicas
while the fast ones idle."""
import threading


class LeastLoaded:
    def __init__(self, actors):
        self.actors = list(actors)
        self._n = [0] * len(self.actors)
        self._turn = 0
        self._lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self._lock:
            k = len(self.actors)
            i = min(range(k), key=lambda j: (self._n[j], (j - self._turn) % k))
            self._n[i] += 1
            self._turn = (i + 1) % k
        return _Replica(self, i)

    def _done(self, i):
        with self._lock:
            self._n[i] -= 1


class _Replica:
    def __init__(self, pool, i):
        self._pool, self._i = pool, i

    def __getattr__(self, name):
        return _Method(self._pool, self._i, getattr(self._pool.actors[self._i], name))


class _Method:
    def __init__(self, pool, i, method):
        self._pool, self._i, self._method = pool, i, method

    def remote(self, *args, **kwargs):
        ref = self._method.remote(*args, **kwargs)
        ref.future().add_done_callback(lambda _f: self._pool._done(self._i))
        return ref
