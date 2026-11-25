"""Small pipeline used by the runtime tests."""
import time

import numpy as np

from hedices import NoRely, Stem


def _burn(ms: float) -> int:
    end = time.perf_counter() + ms / 1000
    x = 0
    while time.perf_counter() < end:
        x += 1
    return x


@Stem("edge", cpus=1, mem="256MB")
def load(req_id, n):
    return np.arange(n, dtype=np.float64)


@Stem("edge", cpus=2, mem="256MB")
@NoRely(flag="req_id")
def heavy(req_id, x):
    _burn(40)
    return float(x.sum())


@Stem("cloud", cpus=1, mem="256MB")
def aggregate(req_id, a, b):
    return a + b


def pipeline(req_id, n):
    x = load(req_id, n)
    return aggregate(req_id, heavy(req_id, x), heavy(req_id, x))
