"""Branch: worker-side execution, function cache, object store and local @NoRely parallelism.

Paper Sec. 3.3, 4.4 and 4.5 (Algorithm 2).  One Branch runs per worker.  It
admits content-addressed function packages into an LRU cache bounded by a
memory budget, resolves hash references against that cache, resolves input
references against its object store or a peer's, executes stages, keeps
intermediate outputs for downstream stages, and reports cache deltas, queue
state and realized parallelism to Stem.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import itertools
import math
import multiprocessing as mp
import os
import signal
import statistics
import sys
import threading
import time
import traceback
from pathlib import Path

import cloudpickle

from . import runtime, wire
from .package import check_manifest, load_package

perf = time.perf_counter
# a peer that does not answer a fetch within this bound is treated as lost: the stage reports
# fetch_failed and the master recomputes the input elsewhere (worker-failure recovery, Sec. 4.6)
FETCH_TIMEOUT = float(os.environ.get("HEDICES_FETCH_TIMEOUT", "2.0"))

_CHILD_FNS: dict = {}
NOCACHE_STATE = "cold"  # --nocache-state
EVICT_STATE = "package"  # --evict-state
# --cache-evict: which entry a full function cache gives up.  lru (default): the least recently used;
# random: a uniformly random one other than the entry just admitted (seeded per worker).  Under a cyclic
# access of more functions than entries (a rotation over every worker) LRU evicts exactly the function needed
# next and never hits; random replacement keeps part of the cycle resident.
CACHE_EVICT = "lru"
SERIAL_SCOPE = "stage"  # --serial-scope
INLINE_KB = 64  # --inline-kb
CALL_PLANE = os.environ.get("HEDICES_CALL_PLANE", "1") == "1"
# fetch a peer's object over its data-plane stream (1, Sec. 4.4's input references) or
# through the control-plane RPC (0: one round trip per object per hop)
DATAPATH = os.environ.get("HEDICES_DATAPATH", "1") == "1"
IDLE_BORROW = 0  # --idle-borrow: 0 off, 1 latecomers within free cores, 2 latecomers keep their
# grant, 3 as 1 except serial stages (no @NoRely), which queue their successors, keep their grant
# (a call takes the whole budget b_v when the worker is idle; off by default: the paper's Sec. 4.5
# borrowing is the per-stage slot ledger, --slot-policy borrow, not per-call expansion)
# @NoRely stages profiled faster than this run in-process under --norely-mode native
NATIVE_MIN_S = float(os.environ.get("HEDICES_NATIVE_MIN_MS", "1.0")) / 1e3


def _child_exec(h, blob, args, kwargs):
    """Process-pool entry point: run a cached function in a child process."""
    fn = _CHILD_FNS.get(h)
    if fn is None:
        fn = _CHILD_FNS[h] = load_package(blob)
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0


def _noop(delay: float = 0.05) -> None:
    time.sleep(delay)


def bench_speed(rounds: int = 3) -> float:
    """Relative compute speed of this worker (higher is faster)."""
    import numpy as np

    a = np.random.default_rng(0).random((128, 128))
    best = float("inf")
    for _ in range(rounds):
        t0 = perf()
        for _ in range(20):
            a @ a
        x = 0
        for i in range(100_000):
            x += i
        best = min(best, perf() - t0)
    return 0.02 / best


def bench_core(rounds: int = 5) -> float:
    """Per-core speed probe (higher is faster), run once when the Branch starts: an interpreter loop
    and a small single-threaded matrix product, best of a few rounds.  A per-core score: the stages
    are mostly interpreter and few-thread native work, which the core count does not predict."""
    import numpy as np

    a = np.random.default_rng(0).random((64, 64))
    best = float("inf")
    for _ in range(rounds):
        t0 = perf()
        for _ in range(200):
            a @ a
        x = 0
        for i in range(200_000):
            x += i
        best = min(best, perf() - t0)
    return 0.02 / best


def bench_threads(parallel: int, rounds: int = 3) -> dict:
    """Native compute probe, run once when the Branch starts: the throughput (GFLOP/s) of a float32
    matrix product with 1, 2, 4 ... threads up to the worker's parallel width.  The heavy stages are
    multi-threaded native kernels: how fast a worker runs them depends on its cores, memory bandwidth
    and what shares its board, none of which a single-core score shows."""
    try:
        import torch
    except ImportError:
        return {}
    n = 384
    a, b = torch.rand(n, n), torch.rand(n, n)
    keep = torch.get_num_threads()
    out = {}
    ts = sorted({t for t in (1, 2, 4, 8, 16) if t <= parallel} | {parallel})
    try:
        for t in ts:
            torch.set_num_threads(t)
            torch.mm(a, b)  # warm the pool
            best = float("inf")
            for _ in range(rounds):
                t0 = perf()
                for _ in range(8):
                    torch.mm(a, b)
                best = min(best, perf() - t0)
            out[t] = 8 * 2 * n ** 3 / best / 1e9
    finally:
        torch.set_num_threads(keep)
    return out


def cgroup_cpu_capacity() -> float:
    """CPU cores available to this process: the tightest cpu.max on its cgroup path, and no more
    than the CPUs it may run on (cpuset)."""
    try:
        cores = float(len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        cores = float(os.cpu_count() or 1)
    return min(cores, _cgroup_quota_cores() or cores)


def _cgroup_quota_cores() -> float | None:
    try:
        rel = Path("/proc/self/cgroup").read_text().strip().split("::")[-1].lstrip("/")
        root = Path("/sys/fs/cgroup")
        p, best = root / rel, None
        while True:
            f = p / "cpu.max"
            if f.exists():
                quota, period = f.read_text().split()
                if quota != "max":
                    cap = int(quota) / int(period)
                    best = cap if best is None else min(best, cap)
            if p == root:
                break
            p = p.parent
        return best
    except Exception:
        return None


class CacheManager:
    """Content-addressed function cache F_i(t) with LRU eviction under a byte budget."""

    def __init__(self, budget_bytes: int, enabled: bool = True, max_entries: int = 0):
        self.budget = budget_bytes
        self.max_entries = max_entries  # 0: bounded by bytes only
        self.enabled = enabled
        self._lru: collections.OrderedDict[str, tuple] = collections.OrderedDict()
        self._bytes = 0
        self._events: list[tuple[str, str]] = []
        self._lock = threading.Lock()
        self.hits = self.misses = self.admits = self.evictions = 0
        self._evicted: set[str] = set()  # packages evicted here (EVICT_STATE=cold: their state went too)
        import random as _random
        self._rng = _random.Random(os.environ.get("HEDICES_NODE", "") + "/cache")

    def lookup(self, h: str):
        with self._lock:
            ent = self._lru.get(h)
            if ent is None:
                self.misses += 1
                return None
            self._lru.move_to_end(h)
            self.hits += 1
            return ent

    def admit(self, h: str, blob: bytes, manifest) -> tuple[tuple, list]:
        """Load a received package; returns ((fn, blob), cache events it caused)."""
        if self.enabled:
            with self._lock:
                if h in self._lru:  # resent against a stale view: keep the loaded entry
                    self._lru.move_to_end(h)
                    return self._lru[h], []
        check_manifest(manifest)
        # without the cache every dispatch redeploys and reloads the package; NOCACHE_STATE=cold
        # also rebuilds the state it initializes (models), =package keeps node-resident asset
        # handles, so only the deployment path (pack, transfer, parse, load) is repeated
        # an evicted package took the state it initialized with it (EVICT_STATE=cold: the cache is
        # bounded because what it holds occupies memory), so its redeployment rebuilds that state
        cold = (not self.enabled and NOCACHE_STATE == "cold") or (EVICT_STATE == "cold" and h in self._evicted)
        ent = (load_package(blob, cold=cold), blob)
        if not self.enabled:
            return ent, []
        events = []
        with self._lock:
            if h in self._lru:
                self._lru.move_to_end(h)
                return self._lru[h], []
            self._lru[h] = ent
            self._bytes += len(blob)
            self.admits += 1
            events.append(("+", h))
            while len(self._lru) > 1 and (self._bytes > self.budget or
                                          (self.max_entries and len(self._lru) > self.max_entries)):
                if CACHE_EVICT == "random" and len(self._lru) > 2:
                    old = self._rng.choice([k for k in self._lru if k != h])
                    _, old_blob = self._lru.pop(old)
                else:
                    old, (_, old_blob) = self._lru.popitem(last=False)
                self._evicted.add(old)
                self._bytes -= len(old_blob)
                self.evictions += 1
                events.append(("-", old))
            self._events.extend(events)
        return ent, events

    def flush(self) -> list:
        """Drop every cached function (the process itself stays warm)."""
        with self._lock:
            events = [("-", h) for h in self._lru]
            self._lru.clear()
            self._bytes = 0
            self._events.extend(events)
        return events

    def drain(self) -> list:
        with self._lock:
            events, self._events = self._events, []
        return events

    def restore(self, events: list) -> None:
        with self._lock:
            self._events = events + self._events

    def snapshot(self) -> list[str]:
        with self._lock:
            self._events.clear()
            return list(self._lru)

    def stats(self) -> dict:
        with self._lock:
            return {"entries": len(self._lru), "bytes": self._bytes, "hits": self.hits,
                    "misses": self.misses, "admits": self.admits, "evictions": self.evictions}


class ObjectStore:
    """Stage outputs held for downstream stages (and replicas fetched from peers)."""

    def __init__(self, budget_bytes: int):
        self.budget = budget_bytes
        self._data: collections.OrderedDict[str, bytes] = collections.OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

    def put(self, oid: str, data: bytes) -> None:
        with self._lock:
            if oid in self._data:
                self._data.move_to_end(oid)
                return
            self._data[oid] = data
            self._bytes += len(data)
            while self._bytes > self.budget and len(self._data) > 1:
                _, old = self._data.popitem(last=False)
                self._bytes -= len(old)

    def get(self, oid: str) -> bytes | None:
        with self._lock:
            data = self._data.get(oid)
            if data is not None:
                self._data.move_to_end(oid)
            return data

    def free(self, oids) -> None:
        with self._lock:
            for oid in oids:
                data = self._data.pop(oid, None)
                if data is not None:
                    self._bytes -= len(data)


class Gate:
    """Counting gate whose width can change while in use."""

    def __init__(self, width: int):
        self.width = width
        self._active = 0
        self._cv = threading.Condition()

    def __enter__(self):
        with self._cv:
            while self._active >= self.width:
                self._cv.wait()
            self._active += 1

    def __exit__(self, *exc):
        with self._cv:
            self._active -= 1
            self._cv.notify()

    def resize(self, width: int) -> None:
        with self._cv:
            self.width = max(1, width)
            self._cv.notify_all()


class SlotLedger:
    """Worker CPU slots C_i shared by co-located stages (Alg. 2 lines 3-4, 8-9, 13).

    Declared @Stem CPUs only decide placement; at run time each active stage v gets
    a budget b_v: its declared slots plus, under ``borrow``, a share of the slots
    that idle co-located stages leave unused.  Borrowed slots come back as running
    calls finish.  On first use each stage is profiled at 1, 2, 4, ... native threads
    (one call each, Alg. 2 line 6); t*_v is the largest thread count whose parallel
    efficiency stays >= ``efficiency``.  A call of v then runs with
    m_v = min(eligible instance groups, max(1, b_v // t*_v)) concurrent calls and
    t_v = b_v // m_v native threads, so m_v * t_v <= b_v.
      borrow     budgets as above (default ledger policy)
      partition  hard partition: b_v = declared slots, never borrowed
      libdefault borrow's concurrency, but native threads left at the library default
      demand     budgets from the stages with calls waiting or running now (a stage that ran a
                 moment ago but has nothing to do holds no share), and under a backlog of
                 independent (@NoRely) calls as many concurrent calls as the budget allows, each
                 with b_v // m_v threads: serial calls of a stage leave cores idle that its
                 queued, independent calls can use.  Serial stages are unchanged (m = 1).
    """

    GAIN = 0.10  # per doubling of native threads, the speedup that justifies the extra slots

    def __init__(self, capacity: int, policy: str = "borrow", idle_after: float = 0.25,
                 efficiency: float = 0.6, reps: int = 6, max_overlap: int = 0):
        self.capacity = max(1, int(capacity))
        self.policy = policy
        self.max_overlap = max_overlap  # demand: at most this many overlapped calls of a stage (0: budget)
        self.efficiency = efficiency
        self.probes = [t for t in (1, 2, 4, 8, 16) if t <= self.capacity]
        self.reps = reps
        self.profile: dict[str, dict[int, list]] = {}    # stage -> threads -> solo exec seconds
        self.tstar: dict[str, int] = {}
        # largest probed thread count that still shortens a call by GAIN per doubling: a call is
        # never granted threads that do not speed it up, which would only hold slots overlapped
        # calls of the stage could use
        self.tcap: dict[str, int] = {}
        self._built: set = set()
        self.queues: dict[str, collections.deque] = collections.defaultdict(collections.deque)
        self._tickets = itertools.count()
        self.idle_after = idle_after  # a stage is idle only after this long without calls
        self.last = {}
        self.declared: dict[str, int] = {}
        self.waiting = collections.Counter()
        self.running = collections.Counter()
        self.used = collections.Counter()
        self.total = 0
        self._cv = threading.Condition()

    def budget(self, v: str) -> int:
        own = min(self.declared[v], self.capacity)
        if self.policy == "partition":
            return own
        now = time.monotonic()
        recent = 0.0 if self.policy == "demand" else self.idle_after
        active = [s for s in self.declared if self.waiting[s] or self.running[s] or s == v
                  or now - self.last.get(s, -1e9) < recent]
        want = sum(min(self.declared[s], self.capacity) for s in active)
        idle = max(0, self.capacity - want)
        return own + idle * own // max(want, 1)

    def acquire(self, v: str, declared: int, serial: bool) -> tuple[int | None, int]:
        with self._cv:
            self.declared[v] = max(1, declared)
            self.waiting[v] += 1
            ticket = next(self._tickets)
            q = self.queues[v]
            q.append(ticket)  # calls of a stage start in arrival order (Alg. 2 line 3)
            while True:
                if q[0] != ticket:
                    self._cv.wait(self.idle_after)
                    continue
                b = self.budget(v)
                probe = self._next_probe(v)
                if probe is not None:  # profiling call: alone in its stage, t = probe threads
                    t = cost = probe
                    ok = self.running[v] == 0 and self.total + cost <= self.capacity
                else:
                    tstar = min(self.tstar.get(v, 1), b)
                    if serial:
                        m = 1
                    elif self.policy == "demand":  # a backlog of independent calls runs side by side
                        m = max(1, min(self.waiting[v] + self.running[v], b, self.max_overlap or b))
                    else:
                        m = max(1, min(self.waiting[v] + self.running[v], b // tstar))
                    if self.policy == "libdefault":
                        t, cost = None, 1
                        ok = self.running[v] < m and self.total < self.capacity
                    else:
                        t = cost = max(1, min(b // m, self.tcap.get(v, b)))
                        ok = self.used[v] + cost <= b and self.total + cost <= self.capacity
                        if self.max_overlap and not serial and self.policy == "demand":
                            ok = ok and self.running[v] < self.max_overlap
                if ok:
                    break
                self._cv.wait(self.idle_after)  # budgets also change when a stage turns idle
            q.popleft()
            self.waiting[v] -= 1
            self.running[v] += 1
            self.used[v] += cost
            self.total += cost
            self._cv.notify_all()
            return t, cost

    def _next_probe(self, v: str) -> int | None:
        if v in self.tstar:
            return None
        done = self.profile.setdefault(v, {})
        for t in self.probes:
            if len(done.get(t, ())) < self.reps:
                return t
        return None

    def record(self, v: str, probe: int | None, exec_s: float) -> None:
        """Profile result of a probing call; fixes t*_v once every probe has run."""
        with self._cv:
            if probe is None or v in self.tstar:
                return
            if v not in self._built:  # the first call builds the stage's state: not a sample
                self._built.add(v)
                return
            done = self.profile.setdefault(v, {})
            done.setdefault(probe, []).append(exec_s)
            if all(len(done.get(t, ())) >= self.reps for t in self.probes):
                # the 75th percentile of each probe's calls: a stage whose inputs make it bimodal
                # (W2 re-detects one frame in three) is compared on the calls that do the work
                med = {t: sorted(x)[len(x) * 3 // 4] for t, x in done.items()}  # P75
                ok = [t for t in self.probes if med[1] / max(med[t], 1e-9) / t >= self.efficiency]
                self.tstar[v] = max(ok) if ok else 1
                cap = self.probes[0]
                for prev, t in zip(self.probes, self.probes[1:]):
                    if med[t] > med[prev] * (1.0 - self.GAIN):
                        break
                    cap = t
                self.tcap[v] = max(cap, self.tstar[v])
            self._cv.notify_all()

    def solo_s(self, v: str) -> float:
        """Profiled single-thread time of v (infinite until its first probe has run)."""
        with self._cv:
            x = self.profile.get(v, {}).get(1)
            return max(x) if x else float("inf")

    def probing(self, v: str) -> bool:
        with self._cv:
            return v not in self.tstar

    def release(self, v: str, cost: int) -> None:
        with self._cv:
            self.running[v] -= 1
            self.last[v] = time.monotonic()
            self.used[v] -= cost
            self.total -= cost
            self._cv.notify_all()


class StageRuntime:
    """Per-stage execution state on one worker."""

    def __init__(self):
        self.serial = threading.Lock()        # stages without @NoRely run one at a time
        self.profile_lock = threading.Lock()
        self.mode: str | None = None          # 'thread' | 'process', chosen on first use
        self.gate: Gate | None = None
        self.pool: cf.Executor | None = None
        self._flags: dict[str, list] = {}
        self._flags_lock = threading.Lock()
        self._lock = threading.Lock()
        self._eids = itertools.count()
        self._running: dict[int, bool] = {}   # execution id -> ran alone so far
        self.solo_s: float | None = None      # median of recent executions that ran alone
        self._solo_hist: collections.deque = collections.deque(maxlen=16)
        self._built = False                   # the first execution builds the stage's state
        self.intervals: collections.deque = collections.deque(maxlen=64)
        self._since_adapt = 0

    def flag_lock(self, key: str):
        """Executions with the same flag value keep their order (Alg. 2 line 7)."""
        runtime = self

        class _Guard:
            def __enter__(self_):
                with runtime._flags_lock:
                    ent = runtime._flags.setdefault(key, [threading.Lock(), 0])
                    ent[1] += 1
                ent[0].acquire()
                self_.ent = ent

            def __exit__(self_, *exc):
                self_.ent[0].release()
                with runtime._flags_lock:
                    self_.ent[1] -= 1
                    if self_.ent[1] == 0:
                        runtime._flags.pop(key, None)

        return _Guard()

    def idle(self) -> bool:
        with self._lock:
            return not self._running

    def begin(self) -> int:
        with self._lock:
            eid = next(self._eids)
            alone = not self._running
            for k in self._running:
                self._running[k] = False
            self._running[eid] = alone
            return eid

    def end(self, eid: int, start: float, end: float) -> None:
        with self._lock:
            alone = self._running.pop(eid)
            first, self._built = not self._built, True
            if alone and not first:  # the state-building first call is not a solo sample
                self._solo_hist.append(end - start)
                self.solo_s = statistics.median(self._solo_hist)
            self.intervals.append((start, end))
            self._since_adapt += 1

    def rho(self) -> float:
        """Realized parallelism: work done (in solo-execution units) per unit of busy wall time."""
        with self._lock:
            iv = sorted(self.intervals)
            solo = self.solo_s
        if not iv or solo is None:
            return 1.0
        union, (cs, ce) = 0.0, iv[0]
        for s, e in iv[1:]:
            if s > ce:
                union += ce - cs
                cs, ce = s, e
            else:
                ce = max(ce, e)
        union += ce - cs
        return max(1.0, len(iv) * solo / union) if union > 0 else 1.0

    def overlap(self) -> float:
        """Fraction of busy time during which executions overlapped."""
        with self._lock:
            iv = sorted(self.intervals)
        if not iv:
            return 0.0
        total = sum(e - s for s, e in iv)
        union, (cs, ce) = 0.0, iv[0]
        for s, e in iv[1:]:
            if s > ce:
                union += ce - cs
                cs, ce = s, e
            else:
                ce = max(ce, e)
        union += ce - cs
        return 1.0 - union / total if total > 0 else 0.0

    def adapt(self) -> None:
        """Shrink the local pool when overlap brings no speedup (Alg. 2, local concurrency policy).
        Executions that never overlapped say nothing about the speedup of overlapping them."""
        if self.gate is None or self.gate.width <= 1 or self._since_adapt < 16:
            return
        self._since_adapt = 0
        if self.overlap() >= 0.2 and self.rho() < 1.15:
            self.gate.resize(self.gate.width - 1)


class Branch:
    def __init__(self, worker_id: str, master: str, listen: str = "0.0.0.0:0",
                 advertise_host: str = "127.0.0.1", tier: str = "edge", device_class: str = "edge",
                 cpus: float = 1.0, mem_mb: float = 1024.0, cache_mb: float = 64.0,
                 cache_enabled: bool = True, heartbeat_s: float = 0.5, speed: float | None = None,
                 norely_mode: str = "auto", object_mb: float = 256.0,
                 cpu_capacity: float | None = None, slot_policy: str | None = None, cache_entries: int = 0,
                 slot_efficiency: float = 0.6, accel: bool = True, slot_max_overlap: int = 0):
        self.worker_id = worker_id
        self.master = master
        self.listen = listen
        self.advertise_host = advertise_host
        self.tier = tier
        self.device_class = device_class
        self.cpus = float(cpus)
        self.mem_mb = float(mem_mb)
        self.heartbeat_s = heartbeat_s
        self.speed = speed
        self.probe = 0.0  # per-core speed probe, measured in start()
        self.probe_mt: dict = {}  # native throughput by thread count, measured in start()
        self.norely_mode = norely_mode
        self.capacity = cpu_capacity if cpu_capacity is not None else cgroup_cpu_capacity()
        self.cache = CacheManager(int(cache_mb * 1024 * 1024), cache_enabled, cache_entries)
        self.objects = ObjectStore(int(object_mb * 1024 * 1024))
        self.inline_limit = INLINE_KB * 1024  # small outputs return inline instead of being held
        self.slots_n = max(1, math.ceil(self.cpus))
        # parallel width backed by real CPU (never more than the slots)
        self.parallel = max(1, min(self.slots_n, int(self.capacity + 0.5)))
        self._slots = threading.BoundedSemaphore(self.slots_n)
        self._cores_busy = 0  # native threads held by the calls executing now
        self._cores_lock = threading.Lock()
        self._seen_at: dict = {}  # stage -> (last call start or end, its slot budget b_v)
        runtime.install_onnxruntime_adapter()
        if accel:  # resident models a stage loads are optimized for inference on first use
            from . import accel as _accel, package as _package

            _package.VALUE_HOOK = _accel.accelerate
        # worker-level slot ledger shared by co-located stages (None: fixed slot semaphore);
        # its slots C_i are the CPU-backed ones
        self.ledger = (SlotLedger(self.parallel, slot_policy, efficiency=slot_efficiency,
                                  max_overlap=slot_max_overlap)
                       if slot_policy else None)
        self._stages: dict[str, StageRuntime] = {}
        self._stages_lock = threading.Lock()
        self._peers: dict[str, object] = {}
        self._data_ports: dict[str, int] = {}  # peer gRPC address -> its data-plane port
        self._data = None                      # DataServer, started with the Branch
        self._calls = None                     # CallServer (call plane), started with the Branch
        self._data_client = None
        self._peers_lock = threading.Lock()
        self._inflight = 0
        self._inflight_lock = threading.Lock()
        self._stop = threading.Event()
        self._need_register = False
        self._mp = mp.get_context("spawn")
        self.native = None  # executor pool of --norely-mode native, started with the Branch
        self.addr = None

    # ---------------------------------------------------------------- lifecycle
    def start(self, register_timeout: float = 60.0):
        if self.speed is None:
            self.speed = bench_speed()
        self.probe = bench_core()
        self.probe_mt = bench_threads(self.parallel)
        if self.norely_mode == "native":
            from .native import NativePool

            self.native = NativePool(self.parallel)
        self._server, port = wire.serve("hedices.Branch", {"Call": self._rpc}, self.listen,
                                        max_workers=64)
        from .datapath import DataClient, DataServer

        self._data = DataServer(self.objects.get, self.listen.rsplit(":", 1)[0] or "0.0.0.0", 0)
        self._data_client = DataClient()
        self._calls = None
        if CALL_PLANE:  # dispatches from the Stem on a persistent stream (hedices/callplane.py)
            from .callplane import CallServer

            self._calls = CallServer(self._rpc, self.listen.rsplit(":", 1)[0] or "0.0.0.0", 0)
        self.addr = f"{self.advertise_host}:{port}"
        self._master_ch = wire.channel(self.master)
        self._master_call = wire.method(self._master_ch, "hedices.Master", "Call")
        deadline = time.monotonic() + register_timeout
        while True:
            try:
                self._register()
                break
            except Exception:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.2)
        self._hb = threading.Thread(target=self._heartbeat_loop, daemon=True, name="branch-hb")
        self._hb.start()
        return self

    def stop(self):
        self._stop.set()
        self._server.stop(0)
        if self._calls is not None:
            self._calls.stop()
        if self._data is not None:
            self._data.stop()
            self._data_client.close()
        for rt in list(self._stages.values()):
            if rt.pool is not None:
                rt.pool.shutdown(wait=False, cancel_futures=True)
        self._master_ch.close()
        if self.native is not None:
            self.native.stop()

    def _register(self):
        msg = {"op": "register", "worker_id": self.worker_id, "addr": self.addr, "tier": self.tier,
               "device_class": self.device_class, "cpus": self.cpus, "mem_mb": self.mem_mb,
               "speed": self.speed, "parallel": self.parallel, "cache": self.cache.snapshot(),
               "probe": self.probe, "probe_mt": self.probe_mt,
               "cache_entries": self.cache.max_entries, "cache_bytes": self.cache.budget}
        if self._calls is not None:
            msg["callport"] = self._calls.port
        self._master_call(wire.dumps(msg), timeout=5.0)
        self._need_register = False

    def _heartbeat_loop(self):
        while not self._stop.wait(self.heartbeat_s):
            if self._need_register:
                try:
                    self._register()
                except Exception:
                    continue
            events = self.cache.drain()
            with self._stages_lock:
                rho = {k: rt.rho() for k, rt in self._stages.items()}
                # local execution width of each @NoRely stage (Alg. 2 adapts it at run time)
                widths = {k: rt.gate.width for k, rt in self._stages.items() if rt.gate is not None}
            msg = {"op": "heartbeat", "worker_id": self.worker_id, "events": events,
                   "inflight": self._inflight, "rho": rho, "widths": widths, "ts": time.time()}
            try:
                rep = wire.loads(self._master_call(wire.dumps(msg), timeout=2.0))
                if rep.get("reregister"):
                    self._register()
            except Exception:
                self.cache.restore(events)
                self._need_register = True

    # ---------------------------------------------------------------- RPC
    def _rpc(self, req: bytes, ctx) -> bytes:
        t0 = perf()
        try:
            msg = wire.loads(req)
            op = msg.get("op")
            if op == "exec":
                from . import package as _pk
                _pk._LOADING.message = set()  # module sources parsed while serving this message (a chain)
                try:
                    return self._exec(msg, t0)
                finally:
                    _pk._LOADING.message = None
            if op == "get":
                data = self.objects.get(msg["oid"])
                return wire.dumps({"status": "ok", "data": data} if data is not None
                                  else {"status": "missing"})
            if op == "free":
                self.objects.free(msg["oids"])
                return wire.dumps({"status": "ok"})
            if op == "flush":
                return wire.dumps({"status": "ok", "events": self.cache.flush()})
            if op == "dataport":
                return wire.dumps({"status": "ok", "port": self._data.port if self._data else 0})
            if op == "ping":
                return wire.dumps({"status": "ok", "ts": time.time()})
            if op == "stats":
                return wire.dumps({"status": "ok", "cache": self.cache.stats(),
                                   "inflight": self._inflight})
            return wire.dumps({"status": "error", "error": f"unknown op {op!r}"})
        except Exception:
            return wire.dumps({"status": "error", "error": traceback.format_exc()})

    _serial_all = threading.Lock()

    def _stage(self, key: str) -> StageRuntime:
        with self._stages_lock:
            rt = self._stages.get(key)
            if rt is None:
                rt = self._stages[key] = StageRuntime()
                if SERIAL_SCOPE == "branch":
                    # stages without @NoRely execute serially on the worker: one at a time
                    # across all of them, not only per stage
                    rt.serial = self._serial_all
            return rt

    def _peer(self, addr: str):
        with self._peers_lock:
            call = self._peers.get(addr)
            if call is None:
                call = self._peers[addr] = wire.method(wire.channel(addr), "hedices.Branch", "Call")
            return call

    def _fetch(self, holder: str, oid: str):
        """An object from the peer that holds it: over its data-plane stream, else (a peer
        without one, or a failed stream) as a gRPC get."""
        if not DATAPATH:
            try:
                rep = wire.loads(self._peer(holder)(
                    wire.dumps({"op": "get", "oid": oid}), timeout=FETCH_TIMEOUT))
                return rep.get("data")
            except Exception:
                return None
        port = self._data_ports.get(holder)
        if port is None:
            try:
                rep = wire.loads(self._peer(holder)(wire.dumps({"op": "dataport"}), timeout=FETCH_TIMEOUT))
                port = int(rep.get("port") or 0)
            except Exception:
                port = 0
            self._data_ports[holder] = port
        if port:
            try:
                return self._data_client.fetch(holder.rsplit(":", 1)[0], port, oid, FETCH_TIMEOUT)
            except Exception:
                return None  # an unreachable holder: the input is lost (recomputed elsewhere)
        try:
            rep = wire.loads(self._peer(holder)(
                wire.dumps({"op": "get", "oid": oid}), timeout=FETCH_TIMEOUT))
            return rep.get("data")
        except Exception:
            return None

    def _resolve_inputs(self, obj):
        """Replace input references with values: local store first, then the holding peer."""
        values: dict[str, object] = {}
        fetched, failed, nbytes = [], [], 0

        def get(ref: wire.ObjRef):
            nonlocal nbytes
            if ref.oid in values:
                return values[ref.oid]
            data = self.objects.get(ref.oid)
            if data is None and ref.holder != self.addr:
                data = self._fetch(ref.holder, ref.oid)
                if data is not None:
                    self.objects.put(ref.oid, data)
                    fetched.append(ref.oid)
                    nbytes += len(data)
            if data is None:
                failed.append((ref.oid, ref.holder))
                return None
            values[ref.oid] = wire.loads(data)
            return values[ref.oid]

        out = wire.map_refs(obj, get)
        return out, fetched, failed, nbytes

    def _exec(self, msg: dict, t0: float) -> bytes:
        with self._inflight_lock:
            self._inflight += 1
        try:
            events: list = []
            loaded: dict[str, tuple] = {}
            missing: list[str] = []
            t_pkg = perf()
            for h in msg["fns"]:
                if h in msg["packages"]:
                    blob, manifest = msg["packages"][h]
                    ent, ev = self.cache.admit(h, blob, manifest)
                    loaded[h] = ent
                    events.extend(ev)
                else:
                    ent = self.cache.lookup(h) if self.cache.enabled else None
                    if ent is None:
                        missing.append(h)
                    else:
                        loaded[h] = ent
            pkg_load_s = perf() - t_pkg
            if missing:  # stale reported view: ask Stem for the packages (repair path)
                return wire.dumps({"status": "miss", "missing": missing, "events": events,
                                   "server_s": perf() - t0})
            fn, blob = loaded[msg["entry"]]
            inputs = wire.loads(msg["inputs"])
            parse_s = perf() - t0
            t_f = perf()
            (args, kwargs), fetched, failed, fbytes = self._resolve_inputs(inputs)
            fetch_s = perf() - t_f
            if failed:
                return wire.dumps({"status": "fetch_failed", "failed": failed, "events": events,
                                   "replicas": fetched, "server_s": perf() - t0})
            out, exec_s, queue_s, path, rt = self._execute(
                msg["stage"], msg["entry"], fn, blob, args, kwargs,
                msg.get("norely"), msg.get("flag_value"), float(msg.get("cpus", 1.0)),
                guest=bool(msg.get("guest")))
            data = wire.dumps(out)
            rep = {"status": "ok", "events": events, "replicas": fetched, "parse_s": parse_s,
                   "pkg_load_s": pkg_load_s, "fetch_s": fetch_s, "fetch_bytes": fbytes,
                   "queue_s": queue_s, "exec_s": exec_s, "rho": rt.rho(), "path": path}
            if msg.get("keep") and len(data) > self.inline_limit:
                self.objects.put(msg["oid"], data)
                rep["kept"], rep["kept_bytes"] = msg["oid"], len(data)
            else:
                rep["result"] = data
            rep["server_s"] = perf() - t0
            nxt = msg.get("next")
            if nxt is not None:  # chained successor: starts here on the output just produced
                if "kept" not in rep:
                    self.objects.put(msg["oid"], data)
                    rep["replicas"] = [*rep.get("replicas", ()), msg["oid"]]
                t_c = perf()
                rep["next"] = wire.loads(self._exec(nxt, t_c))
                rep["chain_s"] = perf() - t_c
            return wire.dumps(rep)
        except Exception:
            return wire.dumps({"status": "error", "error": traceback.format_exc(),
                               "server_s": perf() - t0})
        finally:
            with self._inflight_lock:
                self._inflight -= 1

    # a stage counts as idle after this long without a call (0: borrowing ignores recent stages;
    # 0.25 s, as the SlotLedger, left W4 no faster on the testbed)
    BORROW_HOLD_S = float(os.environ.get("HEDICES_BORROW_HOLD", "0"))
    GUEST_HOLD_S = float(os.environ.get("HEDICES_GUEST_HOLD", "1.0"))

    def _grant(self, t: int, key: str | None = None, budget: int | None = None,
               serial: bool = False, guest: bool = False) -> int:
        """Native threads for a call about to start (§4.5): a call that starts while nothing else
        executes on the worker borrows the idle slots, i.e. all cores except the budgets of the
        other stages active within BORROW_HOLD_S; otherwise it keeps its grant (mode 1: within
        the cores not already held).  Returned with _ungrant when the call ends."""
        with self._cores_lock:
            now = perf()
            if guest:
                # an edge-intent stage on a cloud worker borrows only the idle cores that the
                # stages active there within GUEST_HOLD_S do not hold: cloud stages keep theirs
                held = sum(b for k, (seen, b) in self._seen_at.items()
                           if k != key and now - seen < self.GUEST_HOLD_S)
                t = max(1, min(max(t, self.parallel - held), self.parallel - self._cores_busy))
            elif IDLE_BORROW and self._cores_busy == 0:
                held = sum(b for k, (seen, b) in self._seen_at.items()
                           if k != key and now - seen < self.BORROW_HOLD_S)
                t = max(t, self.parallel - held)
            elif IDLE_BORROW == 1 or (IDLE_BORROW == 3 and not serial):
                t = max(1, min(t, self.parallel - self._cores_busy))
            if key is not None:
                self._seen_at[key] = (now, budget or t)
            self._cores_busy += t
        runtime.set_native_threads(t)
        return t

    def _ungrant(self, t: int, key: str | None = None) -> None:
        with self._cores_lock:
            self._cores_busy -= t
            if key is not None:
                self._seen_at[key] = (perf(), self._seen_at.get(key, (0.0, t))[1])

    def _run(self, rt: StageRuntime, call):
        eid = rt.begin()
        s = perf()
        try:
            out = call()
        finally:
            e = perf()
            rt.end(eid, s, e)
        return out, s, e

    def _execute_ledger(self, rt, key, h, fn, blob, args, kwargs, norely, flag_value, cpus):
        t_arr = perf()
        serial = not norely or self.norely_mode == "off"
        # native mode: @NoRely calls run in the executor pool, off the Branch's interpreter lock;
        # a stage whose single-thread profile is below NATIVE_MIN_S stays in-process, where the
        # hand-over would cost more than the call (Alg. 2 line 6: the profile picks the pool)
        native = self.native is not None and not serial and self.ledger.solo_s(key) >= NATIVE_MIN_S
        if native:
            self.native.prime(h, blob, args, kwargs, 1)
        with (rt.serial if serial else rt.flag_lock(repr(flag_value))):
            probing = self.ledger.probing(key)
            t, cost = self.ledger.acquire(key, math.ceil(cpus), serial)
            try:
                queue_s = perf() - t_arr
                if native:
                    out, s, e = self._run(rt, lambda: self.native.run(h, blob, args, kwargs, t)[0])
                else:
                    runtime.set_native_threads(t)
                    out, s, e = self._run(rt, lambda: fn(*args, **kwargs))
                if probing:
                    self.ledger.record(key, cost, e - s)
            finally:
                self.ledger.release(key, cost)
        return out, e - s, queue_s, f"ledger-{self.ledger.policy}{'-native' if native else ''}-t{t}", rt

    def _execute(self, key, h, fn, blob, args, kwargs, norely, flag_value, cpus, guest=False):
        rt = self._stage(key)
        if self.ledger is not None:
            return self._execute_ledger(rt, key, h, fn, blob, args, kwargs, norely, flag_value, cpus)
        t_arr = perf()
        budget = max(1, min(math.ceil(cpus), int(self.capacity + 0.5)))  # b_v, CPU-backed
        if not norely or self.norely_mode == "off":
            # Alg. 2 lines 1-4: executions follow DAG order, rho = 1; one call holds b_v threads.
            with rt.serial, self._slots:
                queue_s = perf() - t_arr
                t = self._grant(budget, key, budget, serial=True, guest=guest)
                try:
                    out, s, e = self._run(rt, lambda: fn(*args, **kwargs))
                finally:
                    self._ungrant(t, key)
            return out, e - s, queue_s, "serial", rt

        with rt.flag_lock(repr(flag_value)):
            if rt.mode is None:
                with rt.profile_lock:
                    if rt.mode is None:
                        # Alg. 2 line 6: profile on first use, pick thread or process pool.
                        with self._slots:
                            queue_s = perf() - t_arr
                            c0 = time.process_time()
                            out, s, e = self._run(rt, lambda: fn(*args, **kwargs))
                            c1 = time.process_time()
                        cpu_ratio = (c1 - c0) / max(e - s, 1e-9)
                        mode = self.norely_mode if self.norely_mode in ("thread", "process") \
                            else ("process" if cpu_ratio >= 0.5 else "thread")
                        # Alg. 2 line 8: m = min(R_v^CPU, |G|), bounded by CPU-backed width.
                        width = max(1, min(math.ceil(cpus), self.parallel))
                        rt.gate = Gate(width)
                        if mode == "process" and width > 1:
                            rt.pool = cf.ProcessPoolExecutor(max_workers=width, mp_context=self._mp)
                            cf.wait([rt.pool.submit(_noop) for _ in range(width)])
                        rt.mode = mode
                        return out, e - s, queue_s, f"norely-{mode}-profile", rt
            with rt.gate, self._slots:
                queue_s = perf() - t_arr
                # m_v * t_v <= b_v: overlapped calls split the stage's CPU budget; a call that
                # starts alone takes all of it, so the solo time rho is measured against is the
                # serial (no @NoRely) execution and overlap is kept only if it beats that
                t = self._grant(budget if rt.idle() else max(1, budget // rt.gate.width), key, budget, guest=guest)
                try:
                    if rt.pool is not None:
                        out, s, e = self._run(rt, lambda: rt.pool.submit(
                            _child_exec, h, blob, args, kwargs).result()[0])
                    else:
                        out, s, e = self._run(rt, lambda: fn(*args, **kwargs))
                finally:
                    self._ungrant(t, key)
            rt.adapt()
            return out, e - s, queue_s, f"norely-{rt.mode}-w{rt.gate.width}", rt


def main(argv=None):
    p = argparse.ArgumentParser(description="Hedices Branch (worker daemon)")
    p.add_argument("--id", required=True)
    p.add_argument("--master", required=True, help="host:port of the Stem master")
    p.add_argument("--listen", default="0.0.0.0:0")
    p.add_argument("--advertise-host", default="127.0.0.1")
    p.add_argument("--tier", default="edge", choices=["cloud", "edge"])
    p.add_argument("--device-class", default="edge")
    p.add_argument("--cpus", type=float, default=float(os.cpu_count() or 1))
    p.add_argument("--mem-mb", type=float, default=1024.0)
    p.add_argument("--cache-mb", type=float, default=64.0)
    p.add_argument("--cache-entries", type=int, default=0, help="function-cache capacity in entries (0: bytes only)")
    p.add_argument("--object-mb", type=float, default=256.0)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--norely-batch", action="store_true",
                   help="coalesce the overlapping @NoRely calls that queue on one resident model "
                        "into a batch (hedices/accel.py BATCH; needs the model rewrite of accel)")
    p.add_argument("--norely-batch-threads", type=int, default=-1,
                   help="native threads of a coalesced run: -1 the worker's slots, 0 the leading "
                        "call's grant, N exactly N")
    p.add_argument("--no-accel", action="store_true",
                   help="leave resident models as loaded; by default hedices/accel.py rewrites a "
                        "convolutional backbone (or a model without one) for inference once, on the "
                        "worker, and keeps it only if it reproduces the eager output")
    p.add_argument("--switch-ms", type=float, default=None,
                   help="interpreter switch interval in ms (Python's default 5): how long a thread back "
                        "from I/O (a helper's compile, a peer fetch) waits for the lock a running call holds")
    p.add_argument("--compile-procs", type=int, default=-1,
                   help="helper interpreters compiling received module sources (-1: one per 4 slots, 0: none)")
    p.add_argument("--nocache-state", default="cold", choices=["cold", "package"],
                   help="--no-cache: rebuild the state a package initializes on every dispatch "
                        "(cold) or keep node-resident asset handles (package)")
    p.add_argument("--evict-state", default="package", choices=["cold", "package"],
                   help="a package evicted from the function cache releases the state it initialized "
                        "(cold: its redeployment rebuilds it) or leaves asset handles node-resident (package)")
    p.add_argument("--cache-evict", default="lru", choices=["lru", "random"],
                   help="entry a full function cache evicts: least recently used, or a random one")
    p.add_argument("--heartbeat", type=float, default=0.5)
    p.add_argument("--speed", type=float, default=None)
    p.add_argument("--cpu-capacity", type=float, default=None,
                   help="CPU cores backing this worker (default: from its cgroup)")
    p.add_argument("--norely-mode", action="append", default=None,
                   choices=["auto", "thread", "process", "native", "off"],
                   help="may repeat; the last value wins")
    p.add_argument("--inline-kb", type=float, default=64,
                   help="outputs up to this size return inline; larger ones stay here as references")
    p.add_argument("--idle-borrow", type=int, default=0, choices=[0, 1, 2, 3],
                   help="a call starting on an otherwise idle worker takes all its cores "
                        "(off by default; the paper's Sec. 4.5 borrowing is the slot ledger)")
    p.add_argument("--serial-scope", default="stage", choices=["stage", "branch"],
                   help="serial execution of stages without @NoRely: per stage, or one at a time "
                        "across the worker")
    p.add_argument("--slot-efficiency", type=float, default=0.6,
                   help="ledger: smallest parallel efficiency at which a call takes more native "
                        "threads instead of leaving the slots to overlapped calls")
    p.add_argument("--slot-policy", default=None, choices=["borrow", "partition", "libdefault", "demand"],
                   help="worker-level slot ledger for co-located stages (default: fixed slots)")
    p.add_argument("--slot-max-overlap", type=int, default=0,
                   help="ledger demand: at most this many overlapped @NoRely calls of a stage (0: its budget)")
    a = p.parse_args(argv)
    # native thread pools (OpenMP, and the Arm Compute Library behind oneDNN, which sizes its
    # pool once at start-up) span the worker's CPU-backed slots; each call is then granted its
    # t_v threads by the Branch (Alg. 2 line 13), not by the inherited process environment
    cap = a.cpu_capacity if a.cpu_capacity is not None else cgroup_cpu_capacity()
    os.environ["OMP_NUM_THREADS"] = str(max(1, int(cap + 0.5)))
    os.environ.pop("OPENBLAS_NUM_THREADS", None)
    global NOCACHE_STATE, EVICT_STATE, SERIAL_SCOPE, INLINE_KB, IDLE_BORROW, CACHE_EVICT
    IDLE_BORROW = a.idle_borrow
    NOCACHE_STATE = a.nocache_state
    EVICT_STATE = a.evict_state
    CACHE_EVICT = a.cache_evict
    SERIAL_SCOPE = a.serial_scope
    INLINE_KB = a.inline_kb
    mode = a.norely_mode[-1] if a.norely_mode else "auto"
    if a.norely_batch and not a.no_accel:
        from . import accel as _accel

        _accel.BATCH = True
        # threads of a coalesced run: -1 the worker's CPU-backed slots, 0 the leading call's grant
        _accel.BATCH_THREADS = (max(1, int(cap + 0.5)) if a.norely_batch_threads < 0
                                else a.norely_batch_threads)
    br = Branch(a.id, a.master, a.listen, a.advertise_host, a.tier, a.device_class, a.cpus,
                a.mem_mb, a.cache_mb, not a.no_cache, a.heartbeat, a.speed, mode, a.object_mb,
                a.cpu_capacity, a.slot_policy, cache_entries=a.cache_entries,
                slot_efficiency=a.slot_efficiency, accel=not a.no_accel,
                slot_max_overlap=a.slot_max_overlap)
    # helper interpreters that compile received module sources (hedices/package.py): one per 4
    # CPU-backed slots, so a package load does not hold the Branch's interpreter lock while calls
    # already run here (--compile-procs overrides; 0 compiles in the Branch process)
    if a.switch_ms is not None:
        sys.setswitchinterval(a.switch_ms / 1e3)
    from . import package as _package
    _package.set_compilers(a.compile_procs if a.compile_procs >= 0 else max(1, br.parallel // 4))
    _package.COMPILE_CONTENDED = lambda: br._inflight > 1  # the loading call itself is in flight
    br.start()
    print(f"branch {a.id} serving on {br.addr} (speed={br.speed:.2f}, capacity={br.capacity:.2f}, "
          f"parallel={br.parallel})", flush=True)
    done = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: done.set())
    signal.signal(signal.SIGINT, lambda *_: done.set())
    done.wait()
    br.stop()


if __name__ == "__main__":
    main()
