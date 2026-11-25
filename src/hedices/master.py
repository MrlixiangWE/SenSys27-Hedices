"""Stem: master-side orchestration (paper Sec. 3.2, 4.1-4.4, 4.6).

The master canonicalizes workflows, keeps a state store of reported worker
state (cache views, capacities, queues, realized parallelism), runs
StemSched, packages function-granular payloads and dispatches them over
persistent gRPC channels.

Payloads carry hash references for cached functions and *input references*
for data: an intermediate stage output stays in the object store of the
Branch that produced it, and the next stage fetches it from the nearest
replica (same worker, same tier, any).  Only workflow outputs return to the
master.  A worker failure is a scheduling event: unfinished stages go back to
the ready queue, and outputs that lived only on the failed worker are
recomputed from their inputs.
"""
from __future__ import annotations

import collections
import gc
import concurrent.futures as cf
import itertools
import json
import os
import sys
import queue
import threading
import time
import traceback
from dataclasses import dataclass, field

import grpc

from . import callplane, wire
from .api import Param, Ref, Workflow, walk
from .package import FunctionPackage, build_package, build_packages
from .stemsched import MODES, NOCACHE_PLACE, PROBE, StemSched

# DAG neighbours whose cache presence the alignment term rewards (ALIGN_NEAR): "both" (predecessors
# and successors) or "pred" (predecessors only: a stage is drawn to where its inputs come from)
NEAR_SIDE = os.environ.get("HEDICES_NEAR_SIDE", "both")
# warm-up touch rounds (the first builds each stage's state on a worker, the rest measure it)
WARM_ROUNDS = int(os.environ.get("HEDICES_WARM_ROUNDS", "4"))
# how far down a chain of single-dependency successors StemSched's lookahead reaches (1: the direct ones)
LOOK_DEPTH = int(os.environ.get("HEDICES_LOOK_DEPTH", "3"))
# chaining is a decision of StemSched's cost model (the successor is placed on the predecessor's
# worker when starting there is cheapest); RR and static have no cost model and dispatch every
# stage on its own.  1: they too chain a successor their policy happens to put on the same worker
CHAIN_POLICY = os.environ.get("HEDICES_CHAIN_POLICY", "0") == "1"
# (full only: keeping a chain on its worker is the alignment's preference for the layout the chain formed;
# alpha = 0 has no alignment and chains a successor only where it scores best)
# a successor rides on its predecessor's dispatch when StemSched scores the predecessor's worker within this
# margin of its best worker: the decision is taken while the predecessor still runs there (its contention and
# queue weigh on that worker), and the successor placed on its own afterwards often lands on the same worker
# anyway, then paying a round trip of its own (0: chain only when that worker scores best)
CHAIN_MARGIN_S = float(os.environ.get("HEDICES_CHAIN_MARGIN_MS", "0")) / 1e3
# the stable placement the alignment reward forms (Sec. 4.3): once the cache views no longer change (sigma = 0)
# and a successor has ridden on its predecessor's dispatch nearly every time (observed fraction >= ALIGN_STICKY),
# full keeps that chain without scoring it again -- the aligned layout is reused, not re-derived.  alpha = 0 has
# no alignment and keeps scoring every placement (0: off)
ALIGN_STICKY = float(os.environ.get("HEDICES_ALIGN_STICKY", "0"))

# edge-intent stages placed on cloud workers do not borrow the idle cloud cores
GUEST = os.environ.get("HEDICES_GUEST", "1") == "1"
# packager processes: a deployment's packaging runs beside the engine instead of holding it
# (and the interpreter) while other dispatches and completions wait; 0 packages in place
PACKERS = int(os.environ.get("HEDICES_PACKERS", "0"))
# learn the throughput of inline inputs per worker and price master-held inputs with it (0: link rate)
PAY_BW = os.environ.get("HEDICES_PAY_BW", "1") == "1"
COST_CAL = os.environ.get("HEDICES_COST_CAL", "0") == "1"  # see stemsched.COST_CAL
# stemsched.CULT_REPROBE: a call after the estimate went stale restarts the worker's samples of the stage
REPROBE = os.environ.get("HEDICES_CULT_REPROBE", "0") == "1"
REPROBE_S = float(os.environ.get("HEDICES_CULT_STALE_S", "10"))


def _pack_job(module: str, name: str):
    """Packager process: package the stage ``module.name`` as a deployment does."""
    import importlib
    spec = getattr(importlib.import_module(module), name).__hedices_stage__
    p = build_packages(spec)[0]
    return p.hash, p.blob, p.manifest

perf = time.perf_counter
PENDING, READY, INFLIGHT, DONE = "pending", "ready", "inflight", "done"


@dataclass(eq=False)
class WorkerState:
    id: str
    addr: str
    tier: str
    device_class: str
    cpus: float
    mem_mb: float
    speed: float
    parallel: float = 1.0                         # CPU-backed parallel width of the worker
    probe: float = 0.0                            # per-core speed the Branch probed at start (0: none)
    lat_fixed: float | None = None                # dispatch round trip over the first full window (PROBE)
    probe_mt: dict = field(default_factory=dict)  # native GFLOP/s by thread count, probed at start
    view: set = field(default_factory=set)        # reported cache view F^_i(t)
    alive: bool = True
    last_hb: float = 0.0
    inflight: dict = field(default_factory=dict)   # dispatch id -> Dispatch (reserved work)
    rho: dict = field(default_factory=dict)        # stage -> realized parallelism rho_i(v)
    exec_ewma: dict = field(default_factory=dict)  # stage -> T_exec_i(v)
    exec_hist: dict = field(default_factory=dict)  # stage -> recent execution samples
    exec_real: dict = field(default_factory=dict)  # stage -> samples from measured (non-warm) calls
    exec_at: dict = field(default_factory=dict)    # stage -> time of the last measured call
    warm_seen: set = field(default_factory=set)    # stages whose first warm touch ran here
    queue_bias: dict = field(default_factory=dict) # stage -> measured less predicted T_queue
    cost_bias: dict = field(default_factory=dict)  # stage -> measured less predicted dispatch cost
    bw: float | None = None                        # B_i, effective transfer+parse throughput
    lat: float | None = None                       # dispatch network round trip (T_comm latency)
    lat_hist: object = None                        # recent round-trip samples
    widths: dict = field(default_factory=dict)     # stage -> local execution width on the worker
    fetch_bw: float | None = None                  # measured peer-fetch throughput into the worker
    pay_bw: float | None = None                    # measured throughput of inline inputs sent with a dispatch
    cap_entries: int = 0                           # cache capacity reported by the Branch
    cap_bytes: int = 0
    lru: dict = field(default_factory=dict)        # cached hashes, least recently used first
    channel: object = None
    call: object = None                            # gRPC method, or the call stream below
    stream: object = None                          # callplane.CallClient
    dispatched: int = 0


@dataclass(eq=False)
class Dispatch:
    id: int
    inst: "Instance"
    node: int
    worker: str
    t_ready: float
    t_sched: float = 0.0
    t_sent: float = 0.0
    sched_s: float = 0.0
    pack_s: float = 0.0
    est_exec: float = 0.0
    cpus: float = 1.0
    payload_bytes: int = 0
    pkg_bytes: int = 0
    input_bytes: int = 0
    refs: int = 0
    data_refs: int = 0
    pkgs_sent: int = 0
    sent: tuple = ()                               # hashes of the packages this dispatch carries
    sigma: float = 0.0
    alpha: float = 0.0
    relaxed: bool = False
    scores: dict = field(default_factory=dict)   # placement score terms per candidate (diagnostics)
    repair: bool = False
    state: str = "inflight"
    future: object = None
    chain: "Dispatch | None" = None              # successor sent along to run on the same worker
    chained: bool = False                          # this dispatch rode on its predecessor's
    est_queue: float | None = None                 # T_queue the placement predicted


class Instance:
    def __init__(self, iid: int, wf: Workflow, params: dict, tag, pin: dict | None = None):
        self.id = iid
        self.wf = wf
        self.params = params
        self.tag = tag
        self.pin = pin or {}                       # node -> worker id (warm-up touches)
        self.picked: dict[int, str] = {}           # node -> worker a stateful policy (RR, static) chose early
        self.ran: dict[int, str] = {}              # node -> worker it was dispatched to
        self.state = {n.id: PENDING for n in wf.nodes}
        self.results: dict[int, object] = {}       # workflow outputs held by the master
        self.locs: dict[int, list] = {}            # node -> [(worker id, oid)] replicas
        self.sizes: dict[int, int] = {}            # node -> serialized output size (bytes)
        self.param_bytes: dict[str, int] = {}
        self.t_submit = perf()
        self.wall_submit = time.time()
        self.t_done: float | None = None
        self.requeued = 0
        self.future: cf.Future = cf.Future()
        self.future.instance = self

    def oid(self, node: int) -> str:
        return f"{self.id}:{node}"


class Master:
    CHAIN_DEPTH = int(os.environ.get("HEDICES_CHAIN_DEPTH", "4"))  # stages riding on one dispatch

    def __init__(self, listen: str = "0.0.0.0:7700", mode: str = "full", alpha_max: float = 0.5,
                 window: int = 50, hb_timeout: float = 1.5, dispatch_timeout: float = 300.0,
                 default_bw: float = 12.5e6, default_exec: float = 0.05):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.listen = listen
        self.mode = mode
        # package on every deployment (paper cost model) instead of reusing serialized packages
        self.pack_on_dispatch = os.environ.get("HEDICES_PACK_ON_DISPATCH", "1") == "1"
        # successor chaining: a stage whose placement falls on its predecessor's worker travels
        # with the predecessor's dispatch and starts there without a round trip to the master
        self.chaining = os.environ.get("HEDICES_CHAIN", "1") == "1"
        self._packer = None
        if PACKERS > 0 and self.pack_on_dispatch:
            import multiprocessing
            self._packer = cf.ProcessPoolExecutor(PACKERS, mp_context=multiprocessing.get_context("spawn"))
        self._node_size: dict = {}  # (workflow, node) -> last output size, for chained placement
        self._around_cache: dict = {}  # (workflow, node) -> DAG neighbourhood for placement
        # interpreter switch interval of the master process: completions arrive on gRPC threads
        # while another thread packages or schedules; a short interval hands the interpreter
        # over sooner (HEDICES_SWITCH_MS, Python's default is 5 ms)
        if os.environ.get("HEDICES_SWITCH_MS"):
            sys.setswitchinterval(float(os.environ["HEDICES_SWITCH_MS"]) / 1e3)
        self.sched = StemSched(mode, alpha_max, window, default_bw, default_exec)
        self.hb_timeout = hb_timeout
        self.dispatch_timeout = dispatch_timeout
        self._lock = threading.RLock()
        self._changed = threading.Condition(self._lock)
        self._q: queue.Queue = queue.Queue()
        self._workers: dict[str, WorkerState] = {}
        self._ready: collections.deque = collections.deque()
        self._active: dict[int, Instance] = {}
        self._pkgs: dict = {}                      # StageSpec -> [FunctionPackage] (U_v)
        self._wf_hashes: dict[str, frozenset] = {}  # workflow -> H_j
        self._ids = itertools.count(1)
        self._iids = itertools.count(1)
        self._stop = threading.Event()
        self.records: list[dict] = []
        self.instances: list[dict] = []
        self.events: list[dict] = []
        self.port = None

    # ------------------------------------------------------------ lifecycle
    def start(self):
        self._server, self.port = wire.serve("hedices.Master", {"Call": self._rpc}, self.listen,
                                             max_workers=16)
        self._threads = [threading.Thread(target=self._engine, daemon=True, name="stem-engine"),
                         threading.Thread(target=self._monitor, daemon=True, name="stem-monitor")]
        for t in self._threads:
            t.start()
        return self

    def stop(self):
        self._stop.set()
        self._q.put(("kick",))
        for t in self._threads:
            t.join(timeout=2)
        self._server.stop(0)
        if self._packer is not None:
            self._packer.shutdown(wait=False, cancel_futures=True)
        with self._lock:
            for w in self._workers.values():
                if w.channel is not None:
                    w.channel.close()
                if w.stream is not None:
                    w.stream.close()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    def wait_workers(self, n: int, timeout: float = 60.0) -> None:
        deadline = time.monotonic() + timeout
        with self._changed:
            while sum(w.alive for w in self._workers.values()) < n:
                left = deadline - time.monotonic()
                if left <= 0:
                    raise TimeoutError(f"only {sum(w.alive for w in self._workers.values())}/{n} workers joined")
                self._changed.wait(left)

    # ------------------------------------------------------------ public API
    def register(self, wf: Workflow) -> None:
        with self._lock:
            if wf.name in self._wf_hashes:
                return
            hashes = set()
            for spec in wf.stages():
                if spec not in self._pkgs:
                    self._pkgs[spec] = build_packages(spec)
                    self.sched.pkg_size.update({p.hash: p.size for p in self._pkgs[spec]})
                    if self._packer is not None:  # the packagers import the stage's modules once
                        self._pack_async(spec)
                hashes |= {p.hash for p in self._pkgs[spec]}
            self._wf_hashes[wf.name] = frozenset(hashes)

    def packages(self, wf: Workflow) -> dict[str, FunctionPackage]:
        self.register(wf)
        with self._lock:
            return {spec.short_name: self._pkgs[spec][0] for spec in wf.stages()}

    def submit(self, wf: Workflow, *args, tag=None, pin: dict | None = None, **kwargs) -> cf.Future:
        self.register(wf)
        inst = Instance(next(self._iids), wf, wf.bind(args, kwargs), tag, pin)
        self._q.put(("submit", inst))
        return inst.future

    def run(self, wf: Workflow, *args, timeout: float | None = None, **kwargs):
        return self.submit(wf, *args, **kwargs).result(timeout)

    def warm(self, wf: Workflow, args, kwargs=None, rounds: int | None = None, timeout: float = 600) -> int:
        """Execute every stage on every worker its placement intent admits (warm state).  The
        first touch of a stage on a worker builds its state; later rounds measure T_exec there.
        In probed mode they are the probing phase: StemSched fixes its parameters at the end."""
        rounds = rounds or WARM_ROUNDS
        kwargs = kwargs or {}
        runs = 0
        with self._lock:
            workers = [w for w in self._workers.values() if w.alive]
        for _ in range(rounds):
            for w in workers:
                pin = {n.id: w.id for n in wf.nodes if StemSched._intent(n.spec, w)}
                if pin:
                    self.submit(wf, *args, pin=pin, tag="warm", **kwargs).result(timeout)
                    runs += 1
        if PROBE:
            with self._lock:
                self.sched.freeze([w for w in self._workers.values() if w.alive],
                                  {n.spec.name: max(1, int(round(n.spec.cpus))) for n in wf.nodes})
        return runs

    def flush_caches(self) -> int:
        """Empty every worker's function cache; later placements rebuild it (cold cache, warm process)."""
        with self._lock:
            workers = [w for w in self._workers.values() if w.alive]
        n = 0
        for w in workers:
            rep = wire.loads(w.call(wire.dumps({"op": "flush"}), timeout=30.0))
            with self._lock:
                self._apply_events(w, rep.get("events", ()))
            n += len(rep.get("events", ()))
        self._event("flush", dropped=n)
        return n

    def flush_worker(self, wid: str) -> int:
        """Empty one worker's function cache (its process and loaded models stay warm)."""
        w = self._workers[wid]
        rep = wire.loads(w.call(wire.dumps({"op": "flush"}), timeout=30.0))
        with self._lock:
            self._apply_events(w, rep.get("events", ()))
        return len(rep.get("events", ()))

    def worker_info(self) -> list[dict]:
        with self._lock:
            return [{"id": w.id, "tier": w.tier, "device_class": w.device_class, "cpus": w.cpus,
                     "parallel": w.parallel, "speed": w.speed, "alive": w.alive, "view": len(w.view),
                     "bw": w.bw, "dispatched": w.dispatched, "inflight": len(w.inflight),
                     "rho": dict(w.rho)} for w in self._workers.values()]

    def reset_metrics(self) -> None:
        with self._lock:
            self.records.clear()
            self.instances.clear()
            self.events.clear()
        # Finished instances, dispatches and their futures form reference cycles that only the
        # cyclic collector frees; left to a full collection triggered under load it stalls every
        # thread of the master for ~100 ms.  Collect now, between phases, and move the survivors
        # (long-lived master state) out of the collector's scans.
        gc.collect()
        gc.freeze()

    def metrics(self) -> dict:
        with self._lock:
            return {"records": list(self.records), "instances": list(self.instances),
                    "events": list(self.events)}

    def dump_metrics(self, prefix: str) -> None:
        m = self.metrics()
        for key in ("records", "instances"):
            with open(f"{prefix}.{key}.jsonl", "w") as fh:
                for row in m[key]:
                    fh.write(json.dumps(row) + "\n")
        with open(f"{prefix}.events.json", "w") as fh:
            json.dump(m["events"], fh, indent=1)

    # ------------------------------------------------------------ RPC from Branches
    def _rpc(self, req: bytes, ctx) -> bytes:
        try:
            msg = wire.loads(req)
            with self._lock:
                if msg.get("op") == "register":
                    return wire.dumps(self._on_register(msg))
                if msg.get("op") == "heartbeat":
                    return wire.dumps(self._on_heartbeat(msg))
            return wire.dumps({"status": "error", "error": "unknown op"})
        except Exception:
            traceback.print_exc()
            return wire.dumps({"status": "error", "error": traceback.format_exc()})

    def _on_register(self, msg: dict) -> dict:
        wid = msg["worker_id"]
        w = self._workers.get(wid)
        if w is None:
            w = WorkerState(wid, msg["addr"], msg["tier"], msg["device_class"],
                            float(msg["cpus"]), float(msg["mem_mb"]), float(msg["speed"]))
            self._workers[wid] = w
        elif w.alive:
            self._mark_dead(w, "re-register")  # the Branch restarted: its in-flight work is gone
        w.addr, w.tier, w.device_class = msg["addr"], msg["tier"], msg["device_class"]
        w.cpus, w.mem_mb, w.speed = float(msg["cpus"]), float(msg["mem_mb"]), float(msg["speed"])
        w.parallel = float(msg.get("parallel", w.cpus))
        w.probe = float(msg.get("probe") or 0.0)
        w.probe_mt = {int(k): float(v) for k, v in (msg.get("probe_mt") or {}).items()}
        if w.channel is not None:
            w.channel.close()
        w.channel = wire.channel(w.addr)
        w.call = wire.method(w.channel, "hedices.Branch", "Call")
        if w.stream is not None:
            w.stream.close()
            w.stream = None
        if msg.get("callport"):  # dispatches on the worker's call stream (hedices/callplane.py)
            try:
                w.stream = callplane.CallClient(w.addr.rsplit(":", 1)[0], int(msg["callport"]))
                w.call = w.stream
            except OSError:
                w.stream = None
        w.view = set(msg.get("cache", ()))
        w.lru = dict.fromkeys(msg.get("cache", ()))
        w.cap_entries, w.cap_bytes = int(msg.get("cache_entries", 0)), int(msg.get("cache_bytes", 0))
        w.alive = True
        w.last_hb = time.monotonic()
        self._event("worker_join", worker=wid, device_class=w.device_class, cached=len(w.view))
        self._changed.notify_all()
        self._q.put(("kick",))
        return {"status": "ok"}

    def _on_heartbeat(self, msg: dict) -> dict:
        w = self._workers.get(msg["worker_id"])
        if w is None or not w.alive:
            return {"status": "ok", "reregister": True}
        self._apply_events(w, msg.get("events", ()))
        w.rho.update(msg.get("rho", {}))
        w.widths.update(msg.get("widths", {}))
        w.last_hb = time.monotonic()
        return {"status": "ok"}

    @staticmethod
    def _apply_events(w: WorkerState, events) -> None:
        for op, h in events:
            if op == "+":
                w.view.add(h)
                w.lru.setdefault(h, None)
            else:
                w.view.discard(h)
                w.lru.pop(h, None)

    def _event(self, kind: str, **kw) -> None:
        self.events.append({"kind": kind, "t": perf(), "wall": time.time(), **kw})

    # ------------------------------------------------------------ failure handling
    def _monitor(self) -> None:
        while not self._stop.wait(0.1):
            now = time.monotonic()
            with self._lock:
                for w in self._workers.values():
                    if w.alive and now - w.last_hb > self.hb_timeout:
                        self._mark_dead(w, "heartbeat timeout")
                        self._q.put(("kick",))

    def _mark_dead(self, w: WorkerState, cause: str) -> None:
        if not w.alive:
            return
        w.alive = False
        w.view = set()  # stale until the Branch re-registers with its cache contents
        affected = list(w.inflight.values())
        w.inflight.clear()
        for d in affected:
            d.state = "cancelled"
            if d.future is not None:
                d.future.cancel()
            if d.inst.state.get(d.node) == INFLIGHT:
                d.inst.state[d.node] = PENDING
                d.inst.requeued += 1
        for inst in list(self._active.values()):
            for n, reps in inst.locs.items():
                inst.locs[n] = [(wid, oid) for wid, oid in reps if wid != w.id]
            self._recover(inst)
        self._event("worker_dead", worker=w.id, cause=cause,
                    affected=[[d.inst.id, d.node] for d in affected])
        self._changed.notify_all()

    def _alive(self, wid: str) -> bool:
        w = self._workers.get(wid)
        return w is not None and w.alive

    def _available(self, inst: Instance, n: int) -> bool:
        return n in inst.results or any(self._alive(wid) for wid, _ in inst.locs.get(n, ()))

    def _try_ready(self, inst: Instance, n: int) -> None:
        if inst.state[n] != PENDING:
            return
        deps = inst.wf.nodes[n].deps
        if all(inst.state[d] == DONE and self._available(inst, d) for d in deps):
            inst.state[n] = READY
            self._ready.append((inst, n, perf(), None))

    def _recover(self, inst: Instance) -> None:
        """Re-execute finished stages whose outputs were lost but are still needed."""
        nodes = inst.wf.nodes
        stack = [d for n in nodes if inst.state[n.id] != DONE for d in n.deps
                 if inst.state[d] == DONE and not self._available(inst, d)]
        while stack:
            d = stack.pop()
            if inst.state[d] != DONE:
                continue
            inst.state[d] = PENDING
            inst.locs.pop(d, None)
            inst.results.pop(d, None)
            inst.requeued += 1
            stack.extend(dd for dd in nodes[d].deps
                         if inst.state[dd] == DONE and not self._available(inst, dd))
        for n in nodes:
            if inst.state[n.id] == READY and not all(
                    inst.state[d] == DONE and self._available(inst, d) for d in n.deps):
                inst.state[n.id] = PENDING
            self._try_ready(inst, n.id)

    # ------------------------------------------------------------ engine
    def _engine(self) -> None:
        while not self._stop.is_set():
            try:
                ev = self._q.get(timeout=0.05)
            except queue.Empty:
                ev = ("kick",)
            with self._lock:
                try:
                    if ev[0] == "submit":
                        inst = ev[1]
                        self._active[inst.id] = inst
                        for nid in inst.wf.roots:
                            self._try_ready(inst, nid)
                    elif ev[0] == "done":
                        self._on_done(*ev[1:])
                    self._drain_ready()
                except Exception:
                    traceback.print_exc()

    def _drain_ready(self) -> None:
        pending = collections.deque()
        while self._ready:
            item = self._ready.popleft()
            inst, n = item[0], item[1]
            if inst.future.done() or inst.state[n] != READY:
                continue
            if not self._dispatch(*item):
                pending.append(item)
        self._ready = pending

    def _input(self, inst: Instance, n: int, w: WorkerState):
        """Value (workflow output held here) or reference to the nearest live replica."""
        if n in inst.results:
            return inst.results[n]
        reps = [(wid, oid) for wid, oid in inst.locs.get(n, ()) if self._alive(wid)]
        local = [r for r in reps if r[0] == w.id]
        tier = [r for r in reps if self._workers[r[0]].tier == w.tier]
        wid, oid = (local or tier or reps)[0]
        return wire.ObjRef(oid, self._workers[wid].addr)

    def _input_sizes(self, inst: Instance, node, assume: dict | None = None) -> list:
        """[(bytes, holder worker ids or None for the master)] of the stage's inputs; ``assume``
        gives (bytes, holders) for inputs not produced yet (a chained predecessor's output)."""
        out = []

        def visit(x):
            if isinstance(x, Ref) and assume and x.node in assume:
                out.append(assume[x.node])
            elif isinstance(x, Param):
                if x.name not in inst.param_bytes:
                    inst.param_bytes[x.name] = len(wire.dumps(inst.params[x.name]))
                out.append((inst.param_bytes[x.name], None))
            elif isinstance(x, Ref):
                holders = None if x.node in inst.results else {wid for wid, _ in inst.locs.get(x.node, ())}
                out.append((inst.sizes.get(x.node, 0), holders))
            return x
        walk(node.args, visit)
        walk(node.kwargs, visit)
        return out

    def _dispatch(self, inst: Instance, nid: int, t_ready: float, force=None) -> bool:
        node = inst.wf.nodes[nid]
        spec = node.spec
        if not all(self._available(inst, d) for d in node.deps):
            inst.state[nid] = PENDING
            self._recover(inst)
            return True
        pkgs = self._pkgs[spec]
        workers = list(self._workers.values())
        t0 = perf()
        info: dict = {}
        sigma = 0.0
        w = None
        if force is not None:  # cache-miss repair: resend packages to the same worker
            w = self._workers.get(force[0])
            if w is None or not w.alive:
                w, force = None, None
        if w is None and nid in inst.pin and self._alive(inst.pin[nid]):
            w = self._workers[inst.pin[nid]]
        if w is None and nid in inst.picked:
            # RR / static already took their turn for this stage when its predecessor was sent
            # (see _chain): the stage goes where the policy said, the policy is not asked twice
            wid = inst.picked.pop(nid)
            if self._alive(wid):
                w = self._workers[wid]
        if w is None:
            sigma = self.sched.sigma(workers, self._wf_hashes[inst.wf.name])
            w, info = self.sched.place(spec, pkgs, workers, sigma, data=self._input_sizes(inst, node),
                                       after=inst.ran.get(max(node.deps)) if node.deps else None,
                                       **self._around(inst.wf, nid))
            if w is None:
                return False
        inst.ran[nid] = w.id
        sched_s = perf() - t0
        t1 = perf()
        msg, d = self._prepare(inst, nid, w, t_ready, t0, sched_s, sigma, info, force, workers)
        # chaining belongs to the Hedices runtime: the external policies (rr, static) do not chain; nocache is the
        # Hedices runtime without its cache and keeps it -- placed by the static rule, a successor rides on its
        # predecessor's dispatch when the rule itself puts it on the same worker (_chain asks the rule, no
        # StemSched override of the placement)
        if force is None and self.chaining and (CHAIN_POLICY or self.mode not in ("rr", "static")):
            chain = self._chain(inst, nid, w, sigma, workers, d)
            if chain is not None:
                msg["next"], d.chain = chain
        if force is None and not inst.pin:
            self.sched.observe(workers)
        building = [v for m in self._messages(msg) for v in m["packages"].values() if isinstance(v, cf.Future)]
        if not building:
            self._send(w, d, msg, t1)
            return True
        gate, left = threading.Lock(), [len(building)]

        def built(_f):
            with gate:
                left[0] -= 1
                if left[0]:
                    return
            self._send(w, d, msg, t1)
        for f in building:
            f.add_done_callback(built)
        return True

    @staticmethod
    def _messages(msg: dict):
        while msg is not None:
            yield msg
            msg = msg.get("next")

    def _pack_async(self, spec) -> cf.Future:
        try:
            fut = self._packer.submit(_pack_job, spec.fn.__module__, spec.fn.__name__)
        except Exception as exc:  # pool gone: the send packages in place
            fut = cf.Future()
            fut.set_exception(exc)
        fut.spec = spec
        return fut

    def _send(self, w: WorkerState, d: Dispatch, msg: dict, t1: float) -> None:
        """Serialize the exec message (with packages built in the packagers) and send it."""
        for m in self._messages(msg):
            pk = m["packages"]
            for h, v in list(pk.items()):
                if isinstance(v, cf.Future):
                    try:
                        got, blob, manifest = v.result()
                        if got != h:
                            raise ValueError("packager built a different package")
                    except Exception:
                        fresh = build_packages(v.spec)[0]
                        blob, manifest = fresh.blob, fresh.manifest
                    pk[h] = (blob, manifest)
        payload = wire.dumps(msg)
        with self._lock:
            if d.state != "inflight":  # its worker failed while the packages were built
                return
            d.pack_s, d.payload_bytes = perf() - t1, len(payload)
            d.t_sent = perf()
            fut = w.call.future(payload, timeout=self.dispatch_timeout)
            d.future = fut
        fut.add_done_callback(lambda f, d=d: self._completed(d, f, perf()))
        return True

    def _prepare(self, inst: Instance, nid: int, w: WorkerState, t_ready: float, t0: float,
                 sched_s: float, sigma: float, info: dict, force, workers,
                 chained_from: frozenset | None = None) -> tuple[dict, Dispatch]:
        """Build the exec message of stage nid for worker w and account the dispatch."""
        node = inst.wf.nodes[nid]
        spec = node.spec
        pkgs = self._pkgs[spec]
        # Packager: cached functions travel as hash references, misses as packages;
        # stage inputs travel as references to Branch-held outputs.
        forced = force[1] if force is not None else ()
        include, refs = {}, 0
        for p in pkgs:
            if self.mode != "nocache" and p.hash in w.view and p.hash not in forced:
                refs += 1
            elif self.pack_on_dispatch and p is pkgs[0]:
                # Sec. 4.4: a deployment packages the function (introspection,
                # by-value serialization, manifest) for the target; the content hash is stable
                if self._packer is not None:
                    include[p.hash] = self._pack_async(spec)  # resolved before sending
                else:
                    fresh = build_packages(spec)[0]
                    include[fresh.hash] = (fresh.blob, fresh.manifest)
            else:
                include[p.hash] = (p.blob, p.manifest)
        data_refs = 0

        def bind(x):
            nonlocal data_refs
            if isinstance(x, Param):
                return inst.params[x.name]
            if chained_from and x.node in chained_from:  # produced just before on the same worker
                data_refs += 1
                return wire.ObjRef(inst.oid(x.node), w.addr)
            v = self._input(inst, x.node, w)
            data_refs += isinstance(v, wire.ObjRef)
            return v

        args = walk(node.args, bind)
        kwargs = walk(node.kwargs, bind)
        inputs = wire.dumps((args, kwargs))
        flag_value = None
        if spec.norely_flag:
            bound = spec.signature.bind_partial(*args, **kwargs).arguments
            flag_value = bound.get(spec.norely_flag)
        did = next(self._ids)
        msg = {
            "op": "exec", "dispatch_id": did, "instance": inst.id, "stage": spec.name,
            # dependency packages are installed before the function package that imports them
            "entry": pkgs[0].hash, "fns": [p.hash for p in pkgs[1:]] + [pkgs[0].hash], "packages": include,
            "inputs": inputs, "norely": spec.norely_flag, "flag_value": flag_value,
            "cpus": spec.cpus, "keep": nid not in inst.wf.output_nodes, "oid": inst.oid(nid),
            # an edge-intent stage on a cloud worker runs within its declared CPUs (HEDICES_GUEST)
            "guest": GUEST and spec.target == "edge" and w.tier == "cloud",
        }
        sizes = {p.hash: p.size for p in pkgs}  # a package still being built has its registered size
        pkg_bytes = sum(sizes[h] if isinstance(v, cf.Future) else len(v[0]) for h, v in include.items())
        d = Dispatch(did, inst, nid, w.id, t_ready, t_sched=t0, sched_s=sched_s,
                     est_exec=self.sched.exec_estimate(w, spec, workers),
                     cpus=min(spec.cpus, w.parallel), payload_bytes=len(inputs) + pkg_bytes,
                     pkg_bytes=pkg_bytes, input_bytes=len(inputs),
                     refs=refs, data_refs=data_refs, pkgs_sent=len(include), sent=tuple(include), sigma=sigma,
                     alpha=info.get("alpha", 0.0), relaxed=info.get("relaxed", False),
                     scores=info.get("scores", {}),
                     repair=force is not None, chained=chained_from is not None,
                     est_queue=info.get("scores", {}).get(w.id, {}).get("queue"))
        inst.state[nid] = INFLIGHT
        now_use = perf()
        for p in pkgs:  # the Branch's LRU order as the master sees it (value: last use)
            w.lru.pop(p.hash, None)
            w.lru[p.hash] = now_use
        w.inflight[did] = d
        w.dispatched += 1
        return msg, d

    def _chain(self, inst: Instance, nid: int, w: WorkerState, sigma: float, workers,
               pred: Dispatch, ahead: frozenset = frozenset(), up: frozenset = frozenset()):
        """A successor of nid whose other inputs are ready, if StemSched places it on w when it
        sees the saving of starting there: no dispatch round trip and the predecessor's output
        already local.  Returns (its exec message, its dispatch) or None."""
        up = up | {nid}  # stages of this chain: their outputs are produced on w before c starts
        for c in inst.wf.children[nid]:
            node = inst.wf.nodes[c]
            if inst.state[c] != PENDING or any(
                    dd not in up and not (inst.state[dd] == DONE and self._available(inst, dd))
                    for dd in node.deps):
                continue
            info: dict = {}
            t0 = perf()
            if c in inst.pin:
                if inst.pin[c] != w.id:
                    return None
            elif self.mode in ("rr", "static") or (self.mode == "nocache" and NOCACHE_PLACE == "static"):
                # the dispatch path is the same for every placement policy: a successor rides on its
                # predecessor's dispatch whenever the policy puts it on the same worker.  RR and
                # static keep state per placement, so the policy is asked once per stage and its
                # answer kept for the stage's own dispatch when it is another worker.
                wid = inst.picked.get(c)
                if wid is None:
                    best, info = self.sched.place(node.spec, self._pkgs[node.spec], workers, sigma,
                                                  after=(w.id if max(node.deps) in up else inst.ran.get(max(node.deps))))
                    if best is None:
                        return None
                    wid = best.id
                if wid != w.id:
                    inst.picked[c] = wid
                    return None
                inst.picked.pop(c, None)
            elif (ALIGN_STICKY and self.mode == "full" and sigma <= 0.0
                  and self.sched._co.get(node.spec.name, 0.0) >= ALIGN_STICKY and w.alive):
                info = {"alpha": 0.0, "relaxed": False, "scores": {}, "sticky": True}
            else:
                data = self._input_sizes(inst, node, assume={
                    u: (self._node_size.get((inst.wf.name, u), 0), {w.id}) for u in up})
                best, info = self.sched.place(node.spec, self._pkgs[node.spec], workers, sigma,
                                              data=data, chain=(w.id, ahead | {pred.id}),
                                              **self._around(inst.wf, c))
                if best is None:
                    return None
                if best.id != w.id:
                    sc = info.get("scores") or {}
                    if not (CHAIN_MARGIN_S and self.mode == "full" and w.id in sc and best.id in sc
                            and sc[w.id]["score"] - sc[best.id]["score"] <= CHAIN_MARGIN_S):
                        return None
            inst.ran[c] = w.id
            msg, d = self._prepare(inst, c, w, perf(), t0, perf() - t0, sigma, info, None, workers,
                                   chained_from=up)
            if len(ahead) + 1 < self.CHAIN_DEPTH:  # the successor's own successor, further on
                more = self._chain(inst, c, w, sigma, workers, d, ahead | {pred.id}, up)
                if more is not None:
                    msg["next"], d.chain = more
            return msg, d
        return None

    def _around(self, wf: Workflow, nid: int) -> dict:
        """The stage's DAG neighbourhood for StemSched: its neighbours' function packages (near)
        and the successors that can chain on its worker (look)."""
        key = (wf.name, nid)
        got = self._around_cache.get(key)
        if got is None:
            node = wf.nodes[nid]
            side = node.deps if NEAR_SIDE == "pred" else node.deps | set(wf.children[nid])
            near = [(self._pkgs[wf.nodes[i].spec][0].hash, wf.nodes[i].spec.name) for i in sorted(side)
                    if wf.nodes[i].spec is not node.spec and wf.nodes[i].spec in self._pkgs]
            look, front = [], [nid]
            for _ in range(LOOK_DEPTH):  # the successors that would run on its worker, level by level
                front = [c for p in front for c in wf.children[p] if wf.nodes[c].deps <= {p}]
                look += [wf.nodes[c].spec for c in front]
            # the stage's chain segment: the stages it forms a single-input chain with, up and down
            # (what one worker runs back to back when the chain stays on it)
            # as (predecessors nearest first, successors level by level): StemSched counts, per
            # worker, the contiguous run of them that worker may execute
            up, down, front, cur = [], [], [nid], nid
            while front:
                front = [c for p in front for c in wf.children[p] if wf.nodes[c].deps <= {p}]
                if front:
                    down.append([wf.nodes[c].spec for c in front])
            while len(wf.nodes[cur].deps) == 1:
                cur = next(iter(wf.nodes[cur].deps))
                up.append(wf.nodes[cur].spec)
            segment = (up, down)
            got = self._around_cache[key] = {"near": near, "look": look, "segment": segment}
        return got

    def _unchain(self, d: Dispatch) -> None:
        """The predecessor did not complete: its chained successor never ran."""
        c, d.chain = d.chain, None
        if c is None or c.state != "inflight":
            return
        self._unchain(c)  # and whatever rode on it
        c.state = "cancelled"
        w = self._workers.get(c.worker)
        if w is not None:
            w.inflight.pop(c.id, None)
        if c.inst.state.get(c.node) == INFLIGHT:
            c.inst.state[c.node] = PENDING

    def _completed(self, d: Dispatch, fut, t_done: float) -> None:
        """gRPC completion callback: handle the result and dispatch successors in place."""
        with self._lock:
            try:
                self._on_done(d, fut, t_done)
                self._drain_ready()
            except Exception:
                traceback.print_exc()

    def _on_done(self, d: Dispatch, fut, t_done: float) -> None:
        if d.state != "inflight":
            return
        try:
            rep = wire.loads(fut.result())
        except Exception as exc:
            self._unchain(d)
            w = self._workers.get(d.worker)
            if w is not None:
                w.inflight.pop(d.id, None)
            inst = d.inst
            d.state = "failed"
            code = exc.code() if isinstance(exc, grpc.RpcError) else type(exc).__name__
            self._event("dispatch_failed", worker=d.worker, dispatch=d.id, code=str(code))
            if inst.state[d.node] == INFLIGHT:
                inst.state[d.node] = PENDING
                inst.requeued += 1
            if w is not None and code == grpc.StatusCode.UNAVAILABLE:
                self._mark_dead(w, f"rpc {code}")
            self._recover(inst)
            return
        self._handle(d, rep, t_done)

    def _handle(self, d: Dispatch, rep: dict, t_done: float) -> None:
        if d.state != "inflight":
            return
        w = self._workers.get(d.worker)
        if w is not None:
            w.inflight.pop(d.id, None)
        inst = d.inst
        if d.chain is not None and (rep.get("status") != "ok" or "next" not in rep):
            self._unchain(d)
        if w is not None:
            self._apply_events(w, rep.get("events", ()))
            self._add_replicas(inst, w, rep.get("replicas", ()))
        status = rep.get("status")
        if status == "miss":
            d.state = "miss"
            if w is not None:
                w.view.difference_update(rep["missing"])
            self._record(d, rep, t_done, "miss")
            inst.state[d.node] = READY
            self._ready.appendleft((inst, d.node, d.t_ready, (d.worker, set(rep["missing"]))))
            return
        if status == "fetch_failed":
            d.state = "fetch_failed"
            lost = {(oid, holder) for oid, holder in rep["failed"]}
            for n, reps in inst.locs.items():
                inst.locs[n] = [(wid, oid) for wid, oid in reps
                                if (oid, self._workers[wid].addr) not in lost]
            self._record(d, rep, t_done, "fetch_failed")
            inst.state[d.node] = PENDING
            inst.requeued += 1
            self._recover(inst)
            return
        if status != "ok":
            d.state = "error"
            self._record(d, rep, t_done, "error")
            self._fail(inst, RuntimeError(
                f"stage {inst.wf.nodes[d.node].spec.short_name} failed on {d.worker}:\n{rep.get('error')}"))
            return
        d.state = "done"
        self._learn(w, d, rep, t_done)
        self._record(d, rep, t_done, "ok")
        if inst.future.done():
            self._unchain(d)
            return
        inst.state[d.node] = DONE
        if "result" in rep:
            inst.results[d.node] = wire.loads(rep["result"])
            inst.sizes[d.node] = len(rep["result"])
        else:
            inst.locs.setdefault(d.node, []).append((d.worker, rep["kept"]))
            inst.sizes[d.node] = rep.get("kept_bytes", 0)
        self._node_size[(inst.wf.name, d.node)] = inst.sizes[d.node]
        for child in inst.wf.children[d.node]:
            self._try_ready(inst, child)
        if d.chain is not None:
            c, d.chain = d.chain, None
            nxt = rep["next"]
            # it started on the worker, after its predecessor and before its own successors
            c.t_sent = t_done - nxt.get("server_s", 0.0) - nxt.get("chain_s", 0.0)
            self._handle(c, nxt, t_done)
            return
        if all(s == DONE for s in inst.state.values()):
            self._complete(inst)

    @staticmethod
    def _add_replicas(inst: Instance, w: WorkerState, oids) -> None:
        for oid in oids:
            iid, n = oid.split(":")
            if int(iid) == inst.id:
                reps = inst.locs.setdefault(int(n), [])
                if (w.id, oid) not in reps:
                    reps.append((w.id, oid))

    def _learn(self, w: WorkerState | None, d: Dispatch, rep: dict, t_done: float) -> None:
        if w is None:
            return
        a = 0.3
        key = d.inst.wf.nodes[d.node].spec.name
        ex = rep["exec_s"]
        # T_exec_i(v): from recent executions on this worker (see below); contention is priced by
        # T_queue rather than inflating T_exec (which would count it twice and stick to past
        # congestion).  Warm-up
        # touches are not samples.  Calls that carried their package are: node-local model
        # handles outlive cache evictions, so their execution is the stage's own (a worker whose
        # cache thrashes would otherwise never be learned); the rare first build of a model on a
        # node is an outlier the low quantile ignores.  A warm touch samples T_exec on every
        # eligible worker except the first touch, which builds the stage's state.
        first_touch = d.inst.tag == "warm" and key not in w.warm_seen
        if d.inst.tag == "warm":
            w.warm_seen.add(key)
        if not first_touch:
            hist = w.exec_hist.setdefault(key, collections.deque(maxlen=20))
            if (REPROBE and self.mode == "full" and d.inst.tag != "warm"
                    and t_done - w.exec_at.get(key, t_done) > REPROBE_S and ex < w.exec_ewma.get(key, 0.0)):
                # a re-probe that finds w faster than its stale estimate: those samples (forming layout)
                # no longer describe w; a slower one joins them as usual (a stage whose calls differ
                # in kind would otherwise be judged by one call)
                hist.clear()
            hist.append(ex)
            if d.inst.tag != "warm":
                w.exec_real[key] = w.exec_real.get(key, 0) + 1
                self.sched.note_exec(key, ex)
                w.exec_at[key] = t_done
            # expected work per call: the mean, winsorized at the 80th percentile.  The mean keeps
            # stages whose calls differ in kind (W2 re-detects one frame in three) at their real
            # average; capping drops the occasional co-run spike, which T_queue prices instead.
            cap = sorted(hist)[int(0.8 * (len(hist) - 1))]
            w.exec_ewma[key] = sum(min(x, cap) for x in hist) / len(hist)
        if rep.get("rho"):
            w.rho[key] = rep["rho"]
        # T_queue calibration: the Branch reports the time the call actually waited; the
        # difference to the placement's prediction (calls whose cost varies by input make the
        # in-flight work estimate low) is carried as a per-stage correction on this worker
        if d.est_queue is not None and d.inst.tag != "warm":
            err = rep.get("queue_s", 0.0) - d.est_queue
            w.queue_bias[key] = (1 - a) * w.queue_bias.get(key, 0.0) + a * err
        # effective throughput of fetching inputs from peers into w (link and runtime included)
        fb, fs = rep.get("fetch_bytes", 0), rep.get("fetch_s", 0.0)
        if fb >= 64 * 1024 and fs > 0:
            w.fetch_bw = fb / fs if w.fetch_bw is None else (1 - a) * w.fetch_bw + a * fb / fs
        # T_comm latency of a dispatch to w: round trip outside the Branch less the payload's time
        # on the wire (priced separately by bytes); every dispatch, warm touches included, samples it
        # The propagation part is the low quantile of recent samples (as a minimum-RTT filter):
        # queueing in either endpoint's runtime only ever adds to a sample.
        node = d.inst.wf.nodes[d.node]
        if len(node.deps) == 1:  # chain segments (StemSched.CHAIN_SEG_MIN): does this stage ride its predecessor
            self.sched.note_chain(node.spec.name, d.chained)
        if COST_CAL and d.inst.tag != "warm" and not d.chained:
            raw = (d.scores.get(w.id) or {}).get("raw")
            if raw is not None:
                err = (t_done - d.t_sent) - rep.get("chain_s", 0.0) - raw
                b = (1 - a) * w.cost_bias.get(key, 0.0) + a * err
                w.cost_bias[key] = max(-0.5 * raw, b)
        if d.chained:  # no round trip of its own
            return
        link = (self.sched.SAME_BW if w.tier == self.sched.MASTER_TIER else self.sched.CROSS_BW)
        sample = max(0.0, (t_done - d.t_sent) - rep.get("server_s", 0.0) - rep.get("chain_s", 0.0)
                     - d.payload_bytes / link)
        if w.lat_hist is None:
            w.lat_hist = collections.deque(maxlen=20)
        w.lat_hist.append(sample)
        ordered = sorted(w.lat_hist)
        w.lat = ordered[int(0.2 * (len(ordered) - 1))]
        if w.lat_fixed is None and len(w.lat_hist) == w.lat_hist.maxlen:
            w.lat_fixed = w.lat
        # inline inputs travel with the dispatch from the master: their effective throughput is
        # the round trip beyond the propagation part and the server time (sizeable payloads only)
        if PAY_BW and d.payload_bytes - d.pkg_bytes >= 8192:
            extra = (t_done - d.t_sent) - rep.get("server_s", 0.0) - rep.get("chain_s", 0.0) - w.lat
            if extra > 1e-4:
                s = (d.payload_bytes - d.pkg_bytes) / extra
                w.pay_bw = s if w.pay_bw is None else (1 - a) * w.pay_bw + a * s
        if d.pkg_bytes > 0:
            transfer = max(0.0, (t_done - d.t_sent) - rep["server_s"] - rep.get("chain_s", 0.0))
            share = d.pkg_bytes / max(d.payload_bytes, 1)
            # effective deployment throughput: packaging, transfer share and parsing
            sample = d.pkg_bytes / max(d.pack_s + transfer * share + rep.get("pkg_load_s", 0.0), 1e-5)
            w.bw = sample if w.bw is None else (1 - a) * w.bw + a * sample

    def _record(self, d: Dispatch, rep: dict, t_done: float, status: str) -> None:
        w = self._workers.get(d.worker)
        rtt = t_done - d.t_sent - rep.get("chain_s", 0.0)  # less the chained successor's run
        server = rep.get("server_s", 0.0)
        self.records.append({
            "wall": time.time(), "mode": self.mode, "workflow": d.inst.wf.name,
            "instance": d.inst.id, "tag": d.inst.tag, "dispatch": d.id,
            "stage": d.inst.wf.nodes[d.node].spec.short_name, "node": d.node,
            "worker": d.worker, "device_class": w.device_class if w else None, "status": status,
            "refs": d.refs, "data_refs": d.data_refs, "pkgs_sent": d.pkgs_sent,
            "pkg_bytes": d.pkg_bytes, "input_bytes": d.input_bytes, "payload_bytes": d.payload_bytes,
            "meta_bytes": d.payload_bytes - d.pkg_bytes - d.input_bytes,
            "output_bytes": len(rep.get("result", b"")), "kept_bytes": rep.get("kept_bytes", 0),
            "fetch_bytes": rep.get("fetch_bytes", 0),
            "ready_wait_s": d.t_sched - d.t_ready, "sched_s": d.sched_s, "pack_s": d.pack_s,
            "rtt_s": rtt, "transfer_s": max(0.0, rtt - server), "server_s": server, "chained": d.chained,
            "parse_s": rep.get("parse_s", 0.0), "fetch_s": rep.get("fetch_s", 0.0),
            "queue_s": rep.get("queue_s", 0.0), "exec_s": rep.get("exec_s", 0.0),
            "sigma": d.sigma, "alpha": d.alpha, "rho": rep.get("rho"), "path": rep.get("path"),
            "relaxed": d.relaxed, "repair": d.repair, "t_done": t_done, "scores": d.scores,
        })

    def _release(self, inst: Instance) -> None:
        """Tell Branches to drop this instance's intermediate outputs."""
        by_worker: dict[str, list] = collections.defaultdict(list)
        for reps in inst.locs.values():
            for wid, oid in reps:
                by_worker[wid].append(oid)
        for wid, oids in by_worker.items():
            w = self._workers.get(wid)
            if w is not None and w.alive:
                try:
                    w.call.future(wire.dumps({"op": "free", "oids": oids}), timeout=10)
                except Exception:
                    pass

    def _fail(self, inst: Instance, exc: Exception) -> None:
        self._active.pop(inst.id, None)
        self._release(inst)
        if not inst.future.done():
            inst.future.set_exception(exc)

    def _complete(self, inst: Instance) -> None:
        inst.t_done = perf()
        self._active.pop(inst.id, None)
        self._release(inst)
        try:
            result = inst.wf.resolve(inst.wf.output, inst.params, inst.results)
        except Exception as exc:
            inst.future.set_exception(exc)
            return
        if inst.tag != "warm":
            self.instances.append({
                "instance": inst.id, "workflow": inst.wf.name, "tag": inst.tag, "mode": self.mode,
                "wall_submit": inst.wall_submit, "t_submit": inst.t_submit, "t_done": inst.t_done,
                "e2e_s": inst.t_done - inst.t_submit, "requeued": inst.requeued,
            })
        inst.future.set_result(result)
