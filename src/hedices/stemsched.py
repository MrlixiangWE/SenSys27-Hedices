"""StemSched: cache-adaptive placement (paper Sec. 4.3, Algorithm 1, Eq. 5-10).

Ablation modes (Sec. 5.2):
  full     reported cache view + alignment reward with adaptive alpha(t)
  alpha0   reported cache view, alignment reward removed (alpha = 0)
  static   cache view hidden; static hardware-aware placement by resource capacity: the worker
           whose declared CPU capacity finishes the dispatch soonest given the dispatches it
           already runs (HEDICES_STATIC=capacity, default), or the most capable device class with
           its devices in turn (HEDICES_STATIC=class)
  rr       round-robin across capacity-feasible workers
  nocache  worker-side function cache disabled entirely: placement keeps the cost model, which
           sees no cache to align with (HEDICES_NOCACHE_PLACE=sched, default since 2026-11-12), so
           the ablation removes the cache alone; =static falls back to static's capacity rule
           (which put W4's edge stages on agx and pi5: nocache +70 % against +41 % with sched)
"""
from __future__ import annotations

import collections
import json
import os
import math
import random
import time
import zlib

MODES = ("full", "alpha0", "static", "rr", "nocache")
# static = a deployment plan (2026-11-13): a light stage (mean T_exec < STATIC_LIGHT_MS, measured on its
# first run) is deployed once, on the most capable feasible device; a compute stage is replicated over every
# feasible device, calls spread by device compute (wrr).  "wrr" alone (every call of every stage over every
# worker) behaved like RR and cycled small caches; Fig 7 had it at 25 % AGX against the paper's 55-72 %.
STATIC_RULE = os.environ.get("HEDICES_STATIC", "plan")
# static round-robin weights are proportional to cpus * speed**gamma. gamma = 0 weighs every worker
# alike, which keeps the spread even across the pool instead of concentrating edge stages on the
# most capable class (whose cache is a fraction of the stages a workload has).
STATIC_GAMMA = float(os.environ.get("HEDICES_STATIC_GAMMA", "0"))
# static (plan): stages with a mean T_exec below this (ms) are deployed once, on the most capable device
STATIC_LIGHT_MS = float(os.environ.get("HEDICES_STATIC_LIGHT_MS", "4"))
# measured edge-stage T_exec (ms): 0-0.34, 1.98, 3.09, 3.17, 3.81 | 4.35 (W4 preproc_frame, 1 / 2 frames), 6.21, 6.97, 10.6 ...
# (W2 track at 3.81 follows retrack whichever class it gets).  The class is fixed once STATIC_PROFILE_N calls of
# the stage have been measured (their mean, not the first call: a stage whose input size varies is judged by it)
STATIC_PROFILE_N = int(os.environ.get("HEDICES_STATIC_PROFILE_N", "10"))
# static / nocache (plan): the light stages named by the deployment plan (configs, per workload: their uncontended
# T_exec is below STATIC_LIGHT_MS); empty: classified from the first measured calls. A stage close to the line (W2
# preprocess 3.2 ms, track 3.8 ms) otherwise changes class with the contention of the run (nocache: 55 or 70 %)
STATIC_LIGHT = {s for s in os.environ.get("HEDICES_STATIC_LIGHT", "").split(",") if s.strip()}
# static / nocache (plan): a stage with a mean T_exec of at least this (ms) -- a pipeline's dominant model stage -- is
# deployed on the most capable device class only (0: spread like any compute stage). Measured edge stages: W4
# detect_objects 47, W3 infer 101 ms
STATIC_HEAVY_MS = float(os.environ.get("HEDICES_STATIC_HEAVY_MS", "0"))
# static (plan): a light stage is deployed with the step it runs right after -- its most recent input's producer
# (0: on the most capable device).  The operator's plan bundles it so; nocache (the runtime without its cache view,
# placed by device capability) keeps light stages on the most capable device -- placement shifts further toward the
# AGX than static's (Fig 7: nocache 64-77 % against static 55-72 % in the paper)
STATIC_FOLLOW = os.environ.get("HEDICES_STATIC_FOLLOW", "1") == "1"
# nocache placed by the static plan with its light steps deployed alongside, as static (0: on the most capable device).
# With it Fig 7 nocache is 62.5 / 70 / 75 / 55 % (paper 67 / 64 / 77 / 65; without 81 / 70 / 75 / 70), Fig 6 nocache unchanged (AGENTS.md v6p)
NOCACHE_FOLLOW = os.environ.get("HEDICES_NOCACHE_FOLLOW", "1") == "1"
# static / nocache (plan): each compute stage's weighted rotation starts at its own phase (0: all at the same phase)
STATIC_PHASE = os.environ.get("HEDICES_STATIC_PHASE", "1") == "1"
# bounded alignment band (the reward picks among imm within (1 + b alpha); 0: Eq. 8 unbounded)
ALIGN_BAND = float(os.environ.get("HEDICES_ALIGN_BAND", "0.2"))
# the band compares the recurring cost (execution, queueing, communication; 1) instead of the immediate
# cost with the one-off deployment (0): while every candidate still misses, deployment dominates and a
# band over it admits placements that give up a co-located predecessor on every later call
ALIGN_BAND_RECUR = os.environ.get("HEDICES_ALIGN_BAND_RECUR", "1") == "1"
# edge-intent stages on cloud workers pay this extra share of their immediate cost (0: none)
EDGE_SPILL = float(os.environ.get("HEDICES_EDGE_SPILL", "0.15"))
# A cross-tier dispatch costs an extra round trip over the shaped inter-tier link. Eq. 6 accounts
# for the bytes but not for that round trip, which is negligible against a heavy stage and not
# against a light one; it is charged flat so the estimate reflects the link for both.
CROSS_PENALTY_S = float(os.environ.get("HEDICES_CROSS_PENALTY_MS", "2")) / 1e3
# nocache is placed by the static rule (2026-11-13): without a cache view placement goes by device capability,
# as for static ("placement shifts toward AGX" for both, Fig 7); "sched" (StemSched without the cache) kept
# nocache off the AGX (0-23 %).  Its redeployments then concentrate on the AGX Branch, whose compiles run in
# helper interpreters (hedices/package.py set_compilers) so they do not hold up the calls running there.
NOCACHE_PLACE = os.environ.get("HEDICES_NOCACHE_PLACE", "static")
# weight of co-running calls in T_exec (0: execution time as measured, independent of load)
# per worker x stage correction of the immediate cost by its measured error (0: model only)
COST_CAL = os.environ.get("HEDICES_COST_CAL", "0") == "1"
CONTENTION = float(os.environ.get("HEDICES_CONTENTION", "0.5"))
# heavy-stage isolation: only co-running calls expected to run at least this long (s) count
# (compute-dominated stages, such as a detector, stretch what shares their cores; 0: all calls)
CONTENTION_MIN_S = float(os.environ.get("HEDICES_CONTENTION_MIN_MS", "3")) / 1e3
CONTENTION_INFLICT = float(os.environ.get("HEDICES_CONTENTION_INFLICT", "3"))
# workflow-neighbourhood alignment: Eq. 7 also credits the packages of the stage's DAG
# neighbours that w holds (a layout on which successors chain), with this weight (0: Eq. 7 as is)
ALIGN_NEAR = float(os.environ.get("HEDICES_ALIGN_NEAR", "0.5"))
# neighbours whose stage runs at least this long (ms) are left out of that credit (0: all count)
NEAR_HEAVY_S = float(os.environ.get("HEDICES_NEAR_HEAVY_MS", "60")) / 1e3
# and neighbours running at least this many times longer than the stage itself (0: no relative cut)
NEAR_HEAVY_RATIO = float(os.environ.get("HEDICES_NEAR_HEAVY_RATIO", "0"))
# best fit: among placements within this share of the best score, the least capable worker,
# keeping larger devices for the stages that need them (0: lowest score only)
BEST_FIT = float(os.environ.get("HEDICES_BEST_FIT", "0"))
# one-step lookahead: weight of the chainable successors' cost if they follow the stage to w
LOOKAHEAD = float(os.environ.get("HEDICES_LOOKAHEAD", "0"))
# the lookahead only shapes where light stages run (mean T_exec below this, ms; 0: every stage): chaining
# saves a dispatch round trip, small next to what a compute-dominated stage costs its successor on the
# same worker
LOOK_MAX_S = float(os.environ.get("HEDICES_LOOK_MAX_MS", "10")) / 1e3
# ... and none where the stage and its chainable successors together run for less than LOOK_MIN_MS:
# on so short a chain the lookahead's own computation on the master outweighs what a better
# placement of the successors could save (0: no lower bound)
LOOK_MIN_S = float(os.environ.get("HEDICES_LOOK_MIN_MS", "10")) / 1e3
# tier reservation (all cost-based modes alike, see StemSched._reserve): an edge stage spilling onto a
# cloud worker is charged the delay it inflicts on the cloud-only stages expected to run there
TIER_RESERVE = os.environ.get("HEDICES_TIER_RESERVE", "1") == "1"
# chain-level eviction charge of the alignment term (see place); 0: per-placement eviction only
CHAIN_EVICT = os.environ.get("HEDICES_CHAIN_EVICT", "1") == "1"
# cultivation by long-run cost: while the cache is forming (alpha > 0) the transient terms of
# Eq. 6 (one-off deployment, momentary queueing) are discounted by alpha / alpha_max, so the hot
# set forms where the stage is cheapest once warm; 0: Eq. 8 on the immediate cost as is
FORM = os.environ.get("HEDICES_FORM", "0") == "1"
# the transient terms FORM discounts: d (deployment only) or dq (deployment and queueing)
FORM_TERMS = os.environ.get("HEDICES_FORM_TERMS", "dq")
# expected reuses a deployment is amortized over in the immediate cost (1: charged in full)
DEPLOY_AMORT = max(1.0, float(os.environ.get("HEDICES_DEPLOY_AMORT", "16")))
# full charges the eviction its admission causes at least at this weight even once alpha has
# decayed (a reuse-aware cost like DEPLOY_AMORT); alpha = 0 never charges it
EVICT_FLOOR = float(os.environ.get("HEDICES_EVICT_FLOOR", "2"))
# only packages used on the worker within this window (s) count as evicted alignment (0: all)
EVICT_HOT_S = float(os.environ.get("HEDICES_EVICT_HOT_S", "0"))
# capacity-aware alignment (full, with the eviction charge): the packages placed on a worker within
# the last HOT_WIN_S seconds are its hot set, across all the workflows sharing it.  A worker whose hot
# set exceeds its cache entries cycles its cache whatever it holds at this moment: a package placed
# there is evicted before it is reused, so the deployment recurs (charged like a chain that cycles the
# cache, CHAIN_EVICT) and the packages it holds earn no alignment.  Without it the incumbent stage of
# every workflow keeps choosing the worker its package happens to sit on, and the workflows sharing
# a small cache evict each other at every turn (0: off)
HOT_CAP = os.environ.get("HEDICES_HOT_CAP", "1") == "1"
HOT_WIN_S = float(os.environ.get("HEDICES_HOT_WIN_S", "2"))
# chain-level eviction counts a successor in the chain segment only while it does ride its predecessor's
# dispatch (observed fraction >= CHAIN_SEG_MIN): a successor the placement keeps sending elsewhere (a light
# stage on another tier) does not occupy the worker's cache with the chain; 0 counts every runnable successor
CHAIN_SEG_MIN = float(os.environ.get("HEDICES_CHAIN_SEG_MIN", "0.5"))
# floor of the alignment weight under full once alpha has decayed (0: Eq. 8 as is)
ALIGN_FLOOR = float(os.environ.get("HEDICES_ALIGN_FLOOR", "0"))
# T_exec per device class (class: the workers of a class share the stage's estimate) or per worker
EXEC_POOL = os.environ.get("HEDICES_EXEC_POOL", "worker")
# layout cultivation (full only): the alignment-driven score reads T_exec per device class, so an
# early slow sample on one worker does not shut it out of the hot set and the cultivated layout
# spreads a stage over its class; alpha = 0 places on per-worker estimates (0: off)
CULT_POOL = os.environ.get("HEDICES_CULT_POOL", "0") == "1"
# workflow alignment (full only): a stage is placed with the cost its chainable successors then
# take (run there after it, or dispatched to their best worker), so pipelines form on one worker
CULT_LOOK = os.environ.get("HEDICES_CULT_LOOK", "1") == "1"
# the cultivation re-probe alone (full only): an estimate unrefreshed for CULT_STALE_S takes its class's
# current one, so a worker an early slow sample shut out of the hot set is tried again
CULT_REPROBE = os.environ.get("HEDICES_CULT_REPROBE", "0") == "1"
# input-dependent stages (full only): T_exec from the device class's pooled samples once the stage's
# calls vary by more than this coefficient of variation (0 / unset: off)
CULT_INPUT = os.environ.get("HEDICES_CULT_INPUT", "0") == "1"
CULT_INPUT_CV = float(os.environ.get("HEDICES_CULT_INPUT_CV", "0.6"))
CULT_STALE_S = float(os.environ.get("HEDICES_CULT_STALE_S", "10"))
# fixed execution profile (both modes alike): T_exec estimates are read from a profile measured once
# per worker and stage, instead of the live samples, so the timing noise of a run (which early call
# happened to be slow) does not decide the layout; an ablation then differs from full only by the
# algorithm.  JSON {worker: {stage: seconds}}; a worker missing from it falls back to the live estimate.
PROFILE = json.load(open(os.environ["HEDICES_EXEC_PROFILE"])) if os.environ.get("HEDICES_EXEC_PROFILE") else {}
# probed-performance placement (both modes alike): T_exec_i(v) = W(v) / probe_i, the stage's work W(v) the
# median over the workers that ran it of their estimate times their start-up probe; candidates whose
# scores lie within TIE_EPS of the best are a tie, settled by the faster probe, then the worker id, so
# the placement does not hinge on which call happened to be slow or which reply came back first
PROBE = os.environ.get("HEDICES_PROBE", "1") == "1"
_PROBE_DEBUG = bool(os.environ.get("HEDICES_PROBE_DEBUG"))  # per-estimate prints: no estimate cache
TIE_EPS = float(os.environ.get("HEDICES_TIE_EPS", "0.01"))
# probed mode, native probe: a stage is scaled by the worker's probed throughput at the thread count it
# declares (its @Stem cpus) instead of the single-core score
PROBE_MT = os.environ.get("HEDICES_PROBE_MT", "1") == "1"
CULT_SAMPLES = int(os.environ.get("HEDICES_CULT_SAMPLES", "20"))  # calls before a class estimate is trusted
# estimate ageing: a worker whose own T_exec of a stage has not been refreshed by a measured call for
# this long (s) falls back to its device class (an early slow sample would otherwise keep it unused
# and so never re-measured; 0: own samples kept forever)
EXEC_STALE_S = float(os.environ.get("HEDICES_EXEC_STALE_S", "0"))
# T_queue of a stage's overlapped in-flight calls by its measured rho (1) or its local width (0)
RHO_QUEUE = os.environ.get("HEDICES_RHO_QUEUE", "1") == "1"
# Eq. 7 over the emerging layout: packages in flight to a worker count toward its alignment
ALIGN_PENDING = os.environ.get("HEDICES_ALIGN_PENDING", "1") == "1"
# DEPLOY_AMORT applies once alpha(t) has decayed to this (1: always, as before)
# sensitivity of alpha(t) to cache volatility: alpha = alpha_max min(1, gain sigma) (1: Eq. 10), so a
# layout under sustained, if modest, churn keeps its alignment term
SIGMA_GAIN = float(os.environ.get("HEDICES_SIGMA_GAIN", "1"))
AMORT_RAMP = float(os.environ.get("HEDICES_AMORT_RAMP", "0"))  # placements per extra reuse (0: none)
# DEPLOY_AMORT only for a stage whose execution is at least this many times its deployment cost (0: every
# stage): extra replicas pay off where calls can queue; a stage that costs next to nothing to run gains
# nothing from them and keeps its co-location with the rest of the workflow under the full charge
AMORT_MIN_RATIO = float(os.environ.get("HEDICES_AMORT_MIN_RATIO", "0.1"))
AMORT_ALPHA = float(os.environ.get("HEDICES_AMORT_ALPHA", "0.05"))
# the deployment amortized over its expected reuse is part of the cost model, not of the alignment: with
# AMORT_ALL alpha = 0 prices a deployment the way full does, so the two differ in the alignment reward and the
# eviction charge alone (without it alpha = 0 pays every deployment in full, which holds its stages wherever they
# first landed -- a stickiness of its own that alpha = 0 is not meant to have)
AMORT_ALL = os.environ.get("HEDICES_AMORT_ALL", "0") == "1"
# ALPHA0_BLIND_AMORT: alpha = 0 also decides the amortization without the cache contents. Priced by the probes
# (ALPHA0_BLIND 1) the deployment is the same on every worker, but "is it amortized" still asked whether admitting the
# package evicts others on that worker -- so a worker whose cache the first instances happened to fill was charged the
# full deployment for good and never chosen (Fig 7 W4: preproc_frame on the Pi 5 in some runs, never in others)
ALPHA0_BLIND_AMORT = os.environ.get("HEDICES_ALPHA0_BLIND_AMORT", "0") == "1"
# alpha = 0 without the cache-aware placement altogether: its deployment cost is priced as if no worker held the
# packages (the same on every worker), so it neither follows nor learns the cache layout -- placement without
# alignment guidance; the cache itself still works (hits are free at run time)
ALPHA0_BLIND = float(os.environ.get("HEDICES_ALPHA0_BLIND", "0"))  # 0..1: the share of the deployment priced blind
# a successor chained onto its predecessor's cloud worker pays no cross-tier round trip: the spill charges
# (EDGE_SPILL, CROSS_PENALTY_MS) are for a dispatch of its own (0: charged either way)
CHAIN_SPILL_FREE = os.environ.get("HEDICES_CHAIN_SPILL_FREE", "0") == "1"
# softmin placement temperature (ms; 0: the best-scoring worker): near-best workers take dispatches in
# proportion to exp(-score gap / T) -- used for the dispatch-distribution measurement (Fig 7 setting)
PLACE_TEMP_S = float(os.environ.get("HEDICES_PLACE_TEMP_MS", "0")) / 1e3
# alpha0 without the declared-tier preference -- EDGE_SPILL (an edge-intent stage spills to the cloud only when that
# is clearly cheaper) and the full, never amortized deployment off the declared tier: both keep a stage on the
# declared layout, which alignment is about -- and the flat cross-tier charge (CROSS_PENALTY_MS): the measured
# round trip to the worker (lat) stays in its score.  0: alpha0 keeps all three, as full does
ALPHA0_TIER_FREE = os.environ.get("HEDICES_ALPHA0_TIER_FREE", "0") == "1"
# deployment of a chained successor (it rides its predecessor's message, which parses a shared module once) as this
# share of a dispatch of its own (0: the full deployment). Fig 7 setting
CHAIN_DEDUP = float(os.environ.get("HEDICES_CHAIN_DEDUP", "0"))
_rng = random.Random(int(os.environ.get("HEDICES_PLACE_SEED", "7")))
QDRAIN = os.environ.get("HEDICES_QDRAIN", "elapsed")  # elapsed | capped (see queue_estimate)

perf = time.perf_counter


class StemSched:
    def __init__(self, mode: str = "full", alpha_max: float = 0.5, window: int = 50,
                 default_bw: float = 12.5e6, default_exec: float = 0.05):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.mode = mode
        self.alpha_max = alpha_max
        self.window = window
        self.default_bw = default_bw
        self._workers_bw = None  # probed mode: the testbed's deployment throughput, fixed once known
        self._work: dict = {}    # probed mode: W(v), fixed by freeze()
        self._fz: dict = {}      # probed mode: (worker, stage) -> its estimate once freeze() fixed it
        self._fz_ver: int = 0    # bumped whenever a fixed estimate is (re)computed
        self._look_fixed: dict = {}  # successor stage -> the lookahead's fixed terms (see _look_terms)
        self._chain_s: dict = {}     # stage -> mean run time of it and its chainable successors (LOOK_MIN_MS)
        self._rho_cloud: dict = {}   # cloud worker -> fraction of time a compute-dominated cloud-only stage runs on it
        self._co: dict = {}  # stage name -> observed fraction of its dispatches chained onto its predecessor's
        self._dbg_n: int = 0     # HEDICES_PROBE_DEBUG: how many estimates printed
        self._threads: dict = {}  # stage -> thread count it declares (PROBE_MT)
        self._in_bytes: dict = {}  # stage -> bytes of input it was last placed with (the lookahead)
        self.default_exec = default_exec
        self._hist: collections.deque = collections.deque(maxlen=window)
        self._rr = 0
        self._rr_stage: dict = {}  # RR: each stage takes the feasible workers in turn
        self._wrr: dict = {}       # static (wrr): per-stage credits of the weighted round robin
        self._pin: dict = {}       # static (pin): stage -> the worker its deployment-time plan assigned
        self._pin_load: dict = {}  # static (pin): worker id -> declared cores the plan has assigned to it
        self._plan_cls: dict = {}  # static (plan): stage -> "light" | "compute", fixed once profiled
        self._exec_stat: dict = {}  # stage -> [calls, total T_exec] of its first measured (non-warm-up) calls
        self._heavy_rr: dict = {}
        self._placed: dict = {}    # stage -> placements made (AMORT_RAMP)
        self.pkg_size: dict = {}  # hash -> serialized size, filled by the master
        self._hot: dict = {}      # worker id -> {package hash: time it was last placed there} (HOT_CAP)

    @property
    def sees_cache(self) -> bool:
        return self.mode in ("full", "alpha0")

    def observe(self, workers) -> None:
        """Record the reported views after a dispatch; W_s is counted in dispatches."""
        self._hist.append({w.id: frozenset(w.view) for w in workers})
        if TIER_RESERVE:
            # sampled at every dispatch: is a compute-dominated cloud-only stage in flight on w
            for w in workers:
                if w.tier == "cloud":
                    busy = any(d.est_exec >= CONTENTION_MIN_S and d.inst.wf.nodes[d.node].spec.target == "cloud"
                               for d in w.inflight.values())
                    self._rho_cloud[w.id] = 0.98 * self._rho_cloud.get(w.id, 0.0) + 0.02 * busy

    def note_exec(self, name: str, t: float) -> None:
        """A measured call of a stage outside the warm-up (static plan: its first STATIC_PROFILE_N calls profile it)."""
        st = self._exec_stat.setdefault(name, [0, 0.0])
        if st[0] < STATIC_PROFILE_N:
            st[0] += 1
            st[1] += t

    def note_chain(self, name: str, chained: bool) -> None:
        """A completed dispatch of a single-input stage: did it ride its predecessor's dispatch."""
        self._co[name] = 0.95 * self._co.get(name, 1.0) + 0.05 * chained

    def sigma(self, workers, H: frozenset) -> float:
        """Eq. 9: cache volatility over the last W_s dispatches.

        Views before the run started are empty, so sigma rises while caches
        are being populated and decays once they stop changing.
        """
        alive = [w for w in workers if w.alive]
        if not H or not alive:
            return 0.0
        old = self._hist[0] if len(self._hist) == self.window else {}
        diff = sum(len(w.view ^ old.get(w.id, frozenset())) for w in alive)
        return diff / (len(alive) * len(H))

    def alpha(self, sigma: float) -> float:
        """Eq. 10."""
        if self.mode != "full":
            return 0.0
        return self.alpha_max * min(1.0, SIGMA_GAIN * sigma)

    def bw(self, w) -> float:
        if PROBE and self._workers_bw:
            # probed mode: the deployment throughput of the testbed, the median over the workers,
            # not each worker's latest samples (a slow pack or parse on one would push it out)
            return self._workers_bw
        return w.bw or self.default_bw

    # dispatch round trip assumed for a worker that has not been measured yet
    DEFAULT_LAT = float(os.environ.get("HEDICES_DEFAULT_LAT", 0.002))

    # round trip of the cloud-edge link in the network model (the paper's testbed shapes the
    # cross-tier link; a kernel without netem cannot add its latency, so measurements on such a
    # testbed underestimate it): a worker across the tiers from the master costs at least this
    CROSS_RTT = float(os.environ.get("HEDICES_CROSS_TIER_RTT", 0.0))

    def latency(self, w) -> float:
        """T_comm latency of one dispatch round trip to w (measured by the master)."""
        lat = getattr(w, "lat", None)
        if PROBE and getattr(w, "lat_fixed", None) is not None:
            lat = w.lat_fixed  # probed mode: the round trip measured over the first full window
        lat = self.DEFAULT_LAT if lat is None else lat
        return max(lat, self.CROSS_RTT) if w.tier != self.MASTER_TIER else lat

    MIN_REAL = 3  # measured calls before a worker's own T_exec replaces the scaled estimate

    def exec_context(self, spec, workers) -> tuple:
        """What T_exec estimates of stage v draw on, gathered once per placement: the workers
        with enough real calls of v (per device class, and speed-scaled across classes) and
        those known from warm touches only."""
        key = spec.name
        real, same, scaled, known = set(), {}, [], []
        cnt, wsum, wn = {}, 0.0, 0  # calls per class; call-weighted speed-scaled T_exec
        pool: dict = {}  # class -> every recent sample of its measured workers (CULT_INPUT)
        now = perf()
        # cultivation re-probes: under full an estimate no measured call has refreshed for
        # CULT_STALE_S gives way to the class's current one (it may date from the forming layout)
        stale_s = EXEC_STALE_S or (CULT_STALE_S if (CULT_POOL or CULT_REPROBE) and self.mode == "full" else 0.0)
        for o in workers:
            t = o.exec_ewma.get(key)
            if t is None:
                continue
            known.append(t * o.speed)
            if stale_s and now - getattr(o, "exec_at", {}).get(key, now) > stale_s:
                continue  # aged out: the worker takes its class's current estimate
            k = getattr(o, "exec_real", {}).get(key, 0)
            if k >= self.MIN_REAL:
                real.add(o.id)
                same.setdefault(o.device_class, []).append(t)
                scaled.append(t * o.speed)
                cnt[o.device_class] = cnt.get(o.device_class, 0) + k
                wsum, wn = wsum + k * t * o.speed, wn + k
                if CULT_INPUT:
                    pool.setdefault(o.device_class, []).extend(getattr(o, "exec_hist", {}).get(key, ()))
        same = {c: sum(v) / len(v) for c, v in same.items()}
        pooled = False
        if CULT_INPUT and self.mode == "full" and pool:
            # an input-dependent stage (calls differ in kind, W2 re-detects one frame in three): a
            # worker's 20 recent calls say which inputs it happened to get; the class's samples
            # together say how fast that hardware runs the stage
            allv = [x for v in pool.values() for x in v]
            if len(allv) >= 20:
                m = sum(allv) / len(allv)
                cv = (sum((x - m) ** 2 for x in allv) / len(allv)) ** 0.5 / max(m, 1e-9)
                if cv > CULT_INPUT_CV:
                    by_class = {c: sum(v) / len(v) for c, v in pool.items() if len(v) >= 10}
                    if by_class:
                        same, pooled = {**same, **by_class}, True
        return key, real, same, \
            (sum(scaled) / len(scaled) if scaled else None), (sum(known) / len(known) if known else None), \
            cnt, (wsum / wn if wn else None), pooled

    def frozen_estimate(self, w, spec):
        """Probed mode after freeze(): the estimate of stage v on w is a function of fixed
        parameters only (W(v), w's probed speed and efficiency), so it is computed once per
        (worker, stage) and needs no exec_context.  None when the estimate is not fixed (before
        freeze(), a stage or worker without a probe); the value equals what exec_estimate returns."""
        if not self._work or _PROBE_DEBUG:
            return None
        key = spec.name
        mt, eff = getattr(w, "probe_mt", None), getattr(w, "eff_fixed", None)
        probe = getattr(w, "probe", 0.0)
        hit = self._fz.get((w.id, key))
        if hit is not None and hit[0] == probe and hit[1] == eff and hit[2] is mt:
            return hit[3]
        v = PROFILE.get(w.id, {}).get(key)
        if v is None:
            work = self._work.get(key)
            if not (PROBE and probe > 0 and work):
                return None
            self._threads[key] = max(1, int(round(spec.cpus)))
            v = work / self.speed_of(w, key) * (eff or 1.0)
        self._fz[(w.id, key)] = (probe, eff, mt, v)
        self._fz_ver += 1  # whatever was derived from the fixed estimates is gathered anew
        return v

    def exec_estimate(self, w, spec, workers, ctx=None) -> float:
        """T_exec_i(v): measured on w, else scaled from other workers by relative speed."""
        fz = self.frozen_estimate(w, spec)
        if fz is not None:
            return fz
        key, real, same, scaled, known, cnt, wscaled, pooled = ctx or self.exec_context(spec, workers)
        prof = PROFILE.get(w.id, {}).get(key)
        if prof is not None:
            return prof
        if PROBE and getattr(w, "probe", 0.0) > 0:
            # the probed parameters fixed by freeze() (before it: the live median of the warm touches)
            self._threads[key] = max(1, int(round(spec.cpus)))
            work = self._work.get(key) or self.probe_work(key, workers)
            if work is not None:
                _v = work / self.speed_of(w, key) * (getattr(w, "eff_fixed", None) or 1.0)
                if os.environ.get("HEDICES_PROBE_DEBUG") and "infer" in key and self._work and self._dbg_n < 8:
                    self._dbg_n += 1
                    print(f"[est] {w.id} key={key} work={work:.1f} speed={self.speed_of(w, key):.1f} "
                          f"eff={getattr(w, 'eff_fixed', None)} threads={self._threads.get(key)} -> {1e3 * _v:.1f} ms",
                          flush=True)
                return _v
        if pooled and w.device_class in same:
            return same[w.device_class]  # input-dependent stage: the class's pooled samples (CULT_INPUT)
        if CULT_POOL and self.mode == "full" and \
                getattr(w, "exec_real", {}).get(key, 0) < CULT_SAMPLES and wscaled is not None:
            # layout cultivation: while w's own estimate rests on few calls (a few unlucky samples
            # would shut it out of the hot set for good, and an unused worker is never re-measured)
            # it is taken as no dearer than what the well-measured workers imply at w's speed
            # (optimism under uncertainty); a worker measured often enough keeps its own estimate
            if w.id in real:
                base = w.exec_ewma[key]
            elif w.device_class in same:
                base = same[w.device_class]
            else:
                base = scaled / max(w.speed, 1e-6) if scaled is not None else float("inf")
            return min(base, wscaled / max(w.speed, 1e-6))
        if EXEC_POOL == "class" and w.device_class in same:
            # the stage on this hardware: a worker's own recent calls reflect which inputs it
            # happened to get (input-dependent stages), not how fast it runs them
            return same[w.device_class]
        if w.id in real:
            return w.exec_ewma[key]
        # a worker known only from warm touches (one fixed input) is scaled from the workers that
        # have run the stage's real calls, whose inputs can make it cheaper or dearer on average
        if w.device_class in same:  # the same device class: the same hardware, no scaling
            return same[w.device_class]
        if getattr(w, "exec_real", {}).get(key, 0) >= self.MIN_REAL:
            # aged out, but no fresher estimate of its class to take: under the full re-probe, what the
            # freshly measured workers imply at w's speed when that is lower (the whole class went stale)
            if CULT_REPROBE and self.mode == "full" and wscaled is not None:
                return min(w.exec_ewma[key], wscaled / max(w.speed, 1e-6))
            return w.exec_ewma[key]
        if scaled is not None:
            return scaled / max(w.speed, 1e-6)
        if key in w.exec_ewma:
            return w.exec_ewma[key]
        if known is not None:
            return known / max(w.speed, 1e-6)
        return self.default_exec / max(w.speed, 1e-6)

    def freeze(self, workers, threads=None) -> None:
        """Probed mode, at the end of the warm-up touches (every stage has run on every worker its
        intent admits): fix each stage's work W(v) and each worker's efficiency -- how it runs the
        stages relative to its probe, which a start-up probe cannot see (fewer cores than declared,
        the master on the same board): the median over its stages of its estimate against W(v)/probe.
        Placement then rests on these probed parameters alone, not on which later call ran slow."""
        self._threads.update(threads or {})  # stage -> thread count its @Stem declares
        keys = {k for o in workers for k in o.exec_ewma}
        self._work = {k: v for k in keys if (v := self.probe_work(k, workers))}
        self._fz, self._look_fixed, self._chain_s = {}, {}, {}
        self._fz_ver += 1
        for o in workers:
            if getattr(o, "probe", 0.0) <= 0:
                continue
            rs = sorted(t * self.speed_of(o, k) / self._work[k] for k, t in o.exec_ewma.items()
                        if t >= 1e-3 and self._work.get(k))
            # eff is the worker-level ratio between what its stages measure and what the probe
            # predicts, clamped to [0.5, 2.0] so that a single stage whose work the probe models
            # poorly cannot move every estimate on the worker
            # HEDICES_EFF=0 trusts the start-up probe alone (no worker-level correction)
            o.eff_fixed = (1.0 if os.environ.get("HEDICES_EFF", "1") == "0"
                           else min(2.0, max(0.5, rs[len(rs) // 2])) if rs else 1.0)
            if os.environ.get("HEDICES_PROBE_DEBUG"):
                print(f"[freeze] {o.id}: probe={o.probe:.2f} eff={o.eff_fixed:.3f} "
                      f"parallel={getattr(o, 'parallel', '?')} mt={ {k: round(v, 1) for k, v in (o.probe_mt or {}).items()} } "
                      + " ".join(f"{k.split('.')[-1]}={1e3 * t:.1f}ms/speed={self.speed_of(o, k):.1f}"
                                 for k, t in sorted(o.exec_ewma.items())),
                      flush=True)

    def speed_of(self, w, key) -> float:
        """The probed speed a stage runs at on w: its native throughput at the stage's thread count
        (PROBE_MT; the largest probed count not above it), else the single-core score."""
        mt = getattr(w, "probe_mt", None)
        if PROBE_MT and mt:
            t = self._threads.get(key, 1)
            return mt[max([c for c in mt if c <= t] or [min(mt)])]
        return w.probe

    def probe_work(self, key, workers):
        """W(v): the median over the workers that ran stage v of their T_exec estimate times their probe."""
        ws = sorted(o.exec_ewma[key] * self.speed_of(o, key) for o in workers
                    if key in o.exec_ewma and getattr(o, "probe", 0.0) > 0)
        if not ws:
            return None
        m = len(ws) // 2
        return ws[m] if len(ws) % 2 else 0.5 * (ws[m - 1] + ws[m])

    @staticmethod
    def queue_estimate(w, now: float, stage: str | None = None, exclude=None) -> float:
        """T_queue_i(t): reserved work on w not yet done, divided by its parallel CPU capacity.
        In-flight dispatches drain in arrival order, so the work completed so far is the time
        elapsed since the oldest one was sent (times the parallel width).  In-flight calls of the
        stage being placed share its local execution width (reported by the Branch)."""
        skip = exclude if isinstance(exclude, (set, frozenset)) else {exclude}
        flight = [d for k, d in w.inflight.items() if k not in skip]
        if not flight:
            return 0.0
        par = max(getattr(w, "parallel", w.cpus), 1e-6)
        width = getattr(w, "widths", {}).get(stage) if stage else None
        work = sum(d.est_exec * min(d.cpus, par) for d in flight) / par
        if width:
            if RHO_QUEUE:  # overlapped calls drain only as fast as their realized parallelism
                width = min(width, max(1.0, getattr(w, "rho", {}).get(stage, width)))
            same = sum(d.est_exec for d in flight if d.inst.wf.nodes[d.node].spec.name == stage)
            work = max(work, same / width)
        first = min(flight, key=lambda d: d.t_sent or now)
        drained = now - (first.t_sent or now)
        if PROBE:
            # probed mode: the work reserved on w, independent of the instant the placement runs at
            # (how much of it has drained is a matter of microseconds of timing)
            drained = 0.0
        if QDRAIN == "capped":
            # the oldest call is still running: the work queued behind it has not started, so no
            # more than that call's own expected time has drained (calls whose cost varies by
            # input, such as W2's one-in-three re-detection, outlast their mean)
            drained = min(drained, first.est_exec)
        return max(0.0, work - drained)

    @staticmethod
    def _fits(spec, w) -> bool:
        return spec.mem_mb <= w.mem_mb

    # Placement intent names the lowest tier a stage may run on: an "edge" stage is feasible on
    # edge workers and on the more capable cloud workers (Sec. 2.2 runs rcnn_infer / preproc_imu
    # on both AGX and Pi5); a "cloud" stage needs a cloud worker.  HEDICES_INTENT=strict keeps
    # every stage on its own tier.
    NESTED = os.environ.get("HEDICES_INTENT", "nested") != "strict"

    @classmethod
    def _intent(cls, spec, w) -> bool:
        if spec.target == "any" or spec.target == w.tier:
            return True
        return cls.NESTED and spec.target == "edge" and w.tier == "cloud"

    # link throughput used to price moving a stage's inputs to a candidate worker: the paper's
    # 100 Mbps cloud-edge shaping across tiers, LAN within a tier; the master sits on MASTER_TIER
    CROSS_BW = float(os.environ.get("HEDICES_CROSS_TIER_BW", 12.5e6))
    SAME_BW = float(os.environ.get("HEDICES_SAME_TIER_BW", 125e6))
    MASTER_TIER = os.environ.get("HEDICES_MASTER_TIER", "edge")

    def evict_cost(self, w, miss, pkgs) -> float:
        """Deployment the placement pushes onto later dispatches: admitting the missing packages
        into a full cache evicts least-recently-used entries, which must be redeployed when their
        stages return (cache-capacity effects on later D_i, Sec. 4.4)."""
        cap_n, cap_b = getattr(w, "cap_entries", 0), getattr(w, "cap_bytes", 0)
        if not miss or not (cap_n or cap_b):
            return 0.0
        keep = {p.hash for p in pkgs}
        order = [h for h in getattr(w, "lru", {}) if h in w.view and h not in keep]
        n = len(w.view) + len(miss)
        size = sum(self.pkg_size.get(h, 0) for h in w.view) + sum(p.size for p in miss)
        cost, now = 0.0, perf()
        for h in order:
            if not ((cap_n and n > cap_n) or (cap_b and size > cap_b)):
                break
            n -= 1
            size -= self.pkg_size.get(h, 0)
            last = w.lru.get(h)
            if EVICT_HOT_S and (last is None or now - last > EVICT_HOT_S):
                continue  # a package w no longer runs is not redeployed: evicting it is free
            cost += self.pkg_size.get(h, 0) / self.bw(w)
        return cost

    def data_cost(self, w, data, tiers: dict) -> float:
        """Time to bring the stage's inputs that w does not hold from their nearest copy."""
        t = 0.0
        for nbytes, holders in data or ():
            if holders and w.id in holders:
                continue
            src = {tiers.get(h) for h in holders} if holders else {self.MASTER_TIER}
            link = self.SAME_BW if w.tier in src else self.CROSS_BW
            # what moving the input into w actually achieves: a peer fetch, or the dispatch's
            # inline payload for inputs the master holds
            measured = getattr(w, "fetch_bw" if holders else "pay_bw", None)
            t += nbytes / (min(link, measured) if measured else link)
        return t

    def _look_terms(self, cspec, alive, workers):
        """The lookahead's fixed terms of a successor, [(worker, estimate, spill, round trip)], or
        None while an estimate is still live (before freeze(), a worker without a probe)."""
        if not PROBE:
            return None
        out = []
        for o in alive:
            if not (self._intent(cspec, o) and self._fits(cspec, o)):
                continue
            est = self.frozen_estimate(o, cspec)
            if est is None:
                return None
            # the round trip is fixed once probed over a full window; until then it is read live
            out.append((o, est, 1.0 + (EDGE_SPILL if cspec.target == "edge" and o.tier == "cloud" else 0.0),
                        self.latency(o) if getattr(o, "lat_fixed", None) is not None else None))
        return out or None

    def _reserve(self, w, spec, t_exec: float) -> float:
        """Tier reservation: a compute-dominated edge-intent stage on a cloud worker runs beside the
        cloud-only stages that arrive there while it executes, and those have nowhere else to go.
        The delay it inflicts on them is charged to it: the contention share times the part of its
        execution during which such a stage is expected to run (rho: the fraction of time one was
        in flight on w, tracked by observe())."""
        if not TIER_RESERVE or spec.target != "edge" or w.tier != "cloud" or t_exec < CONTENTION_MIN_S:
            return 0.0
        rho = self._rho_cloud.get(w.id, 0.0)
        if rho <= 0.0:
            return 0.0
        par = max(getattr(w, "parallel", w.cpus), 1e-6)
        share = CONTENTION_INFLICT if CONTENTION_INFLICT else CONTENTION * min(spec.cpus, par) / par
        return share * rho * t_exec

    def lookahead(self, look, cand, workers) -> dict:
        """Per candidate w: the cost of the chainable successors once the stage ran on w, each
        either chained there (its execution on w) or dispatched to its best worker (execution
        plus a dispatch round trip), whichever is cheaper."""
        out = {w.id: 0.0 for w in cand}
        if not look:
            return out
        on = {w.id: True for w in cand}  # the chain still runs on w (down a chain, a successor only
        # chains on w when every stage before it did)
        alive = [w for w in workers if w.alive]
        gen = (self._fz_ver, tuple(map(id, alive)))
        for cspec in look:
            # the terms of a successor that freeze() fixed (who may run it, its estimate there, the
            # spill factor, the probed round trip) are gathered once per set of live workers
            st = self._look_fixed.get(cspec.name)
            if st is None or st[0] != gen:
                st = self._look_terms(cspec, alive, workers)
                if st is not None and self._fz_ver == gen[0]:
                    self._look_fixed[cspec.name] = (gen, st)
            else:
                st = st[1]
            now = perf()
            if st is not None:
                # the successor also waits behind what already runs of it there
                ex, rt = {}, {}
                for o, est, spill, lat in st:
                    est += self._reserve(o, cspec, est)  # as the successor's own placement prices it
                    if o.inflight:
                        est += self.queue_estimate(o, now, cspec.name)
                    e = est * spill
                    ex[o.id], rt[o.id] = e, e + (self.latency(o) if lat is None else lat)
                ok = [t[0] for t in st]
            else:
                ok = [o for o in alive if self._intent(cspec, o) and self._fits(cspec, o)]
                if not ok:
                    continue
                ctx = self.exec_context(cspec, workers)
                # priced as its own placement will price it: an edge-intent successor on a cloud worker
                # carries the same spill factor, or the lookahead counts on a chain that never forms
                ex = {o.id: ((lambda t: t + self._reserve(o, cspec, t))(self.exec_estimate(o, cspec, workers, ctx))
                             + self.queue_estimate(o, now, cspec.name)) *
                      (1.0 + (EDGE_SPILL if cspec.target == "edge" and o.tier == "cloud" else 0.0))
                      for o in ok}
                rt = {o.id: ex[o.id] + self.latency(o) for o in ok}
            nb = self._in_bytes.get(cspec.name, 0.0)
            # per tier the two cheapest workers by execution plus round trip: the best alternative
            # for a candidate is then a lookup per tier instead of a scan over workers
            top: dict = {}
            for o in ok:
                top.setdefault(o.tier, []).append((rt[o.id], o.id))
            top = {t: sorted(v)[:2] for t, v in top.items()}
            same_mv, cross_mv = nb / self.SAME_BW, nb / self.CROSS_BW
            for w in cand:
                # dispatched elsewhere: its execution there, the round trip, and moving its input off w
                # (across the cloud-edge link when the tiers differ)
                alt = rt.get(w.id, float("inf"))
                for t, v in top.items():
                    c, oid = v[0] if v[0][1] != w.id or len(v) == 1 else v[1]
                    if oid != w.id:
                        alt = min(alt, c + (same_mv if t == w.tier else cross_mv))
                here = ex.get(w.id, float("inf")) if on[w.id] else float("inf")
                on[w.id] = here <= alt
                out[w.id] += min(here, alt)
        return out

    @staticmethod
    def _mean_exec(stage: str, workers) -> float:
        """Mean measured T_exec of a stage over the workers that have run it (0: unknown)."""
        ts = [w.exec_ewma[stage] for w in workers if stage in w.exec_ewma]
        return sum(ts) / len(ts) if ts else 0.0

    def place(self, spec, pkgs, workers, sigma: float, data=None, chain=None, near=(), look=(), segment=(), after=None):
        """Algorithm 1. Returns (worker, info) or (None, {}) when nothing is feasible.
        chain=(worker id, predecessor dispatch ids): the stage would start on that worker right
        after its predecessors, with no dispatch round trip; every candidate waits for them, so
        their remaining work is not queueing on that worker.  near: (package hash, stage name) of the
        stage's DAG neighbours (ALIGN_NEAR); look: specs of its chainable successors (LOOKAHEAD)."""
        if data:
            self._in_bytes[spec.name] = float(sum(nb for nb, _ in data))
        if PROBE and self._workers_bw is None:
            bws = sorted(w.bw for w in workers if w.alive and w.bw)
            if bws and len(bws) == sum(1 for w in workers if w.alive):  # once every worker deployed
                self._workers_bw = bws[len(bws) // 2]
        if NEAR_HEAVY_S and near:
            # a compute-dominated neighbour gains little from sharing a worker (its inputs are
            # small next to its run time) and is slowed by what runs beside it
            near = [n for n in near if self._mean_exec(n[1], workers) < NEAR_HEAVY_S]
        if NEAR_HEAVY_RATIO and near:
            # likewise a neighbour that runs many times longer than the stage: the light stage would
            # only queue behind it and contend with it for the worker's cores
            own = self._mean_exec(spec.name, workers)
            if own > 0:
                near = [n for n in near if self._mean_exec(n[1], workers) < NEAR_HEAVY_RATIO * own]
        alive = sorted((w for w in workers if w.alive), key=lambda w: w.id)
        cand = [w for w in alive if self._intent(spec, w) and self._fits(spec, w)]
        relaxed = False
        if not cand:
            cand = [w for w in alive if self._fits(spec, w)]
            relaxed = True
        if not cand:
            return None, {}
        if self.mode == "rr":
            # a counter per stage: with one shared counter a workflow of k stages on n workers
            # sends each stage to only n / gcd(k, n) of them, fixed by where the counter started
            i = self._rr_stage.get(spec.name, 0)
            self._rr_stage[spec.name] = i + 1
            return cand[i % len(cand)], {"alpha": 0.0, "relaxed": relaxed}
        if self.mode == "static" or (self.mode == "nocache" and NOCACHE_PLACE == "static"):
            if STATIC_RULE == "pin":
                # a deployment-time plan: the first placement of a stage fixes its worker -- the feasible one
                # with the least declared load per unit of capacity (cpus x speed ** STATIC_GAMMA) assigned so
                # far -- and every later call goes there; no runtime state, no replicas
                wid = self._pin.get(spec.name)
                if wid is None or not any(w.id == wid for w in cand):
                    cap = lambda w: w.cpus * max(w.speed, 1e-9) ** STATIC_GAMMA
                    pick = min(cand, key=lambda w: ((self._pin_load.get(w.id, 0.0) + spec.cpus) / max(cap(w), 1e-9), w.id))
                    self._pin_load[pick.id] = self._pin_load.get(pick.id, 0.0) + spec.cpus
                    self._pin[spec.name] = wid = pick.id
                return next(w for w in cand if w.id == wid), {"alpha": 0.0, "relaxed": relaxed}
            if STATIC_RULE == "plan":
                # an operator's deployment plan, fixed once the stage's cost is known (its first run): a light
                # stage (mean T_exec < STATIC_LIGHT_MS) is deployed once, on the most capable feasible device;
                # a compute stage is replicated over every feasible device, calls spread by device compute (wrr
                # below).  No runtime state, no cache view.  Before the first run the stage goes by wrr.
                cls = self._plan_cls.get(spec.name)
                st = self._exec_stat.get(spec.name)
                if cls is None and st and st[0] >= STATIC_PROFILE_N:
                    mean = st[1] / st[0]
                    light = (spec.name.split(".")[-1] in STATIC_LIGHT or spec.name in STATIC_LIGHT) if STATIC_LIGHT \
                        else mean < STATIC_LIGHT_MS / 1e3
                    cls = self._plan_cls[spec.name] = ("light" if light else
                                                       "heavy" if STATIC_HEAVY_MS and mean >= STATIC_HEAVY_MS / 1e3 else "compute")
                if cls == "heavy":
                    # the dominant model stage of a pipeline goes where it runs fastest: the most capable
                    # device class (its devices taken in turn), as an operator deploys a large detector
                    cap = lambda w: w.cpus * max(w.speed, 1e-9) ** STATIC_GAMMA
                    top = max(cap(w) for w in cand)
                    best = sorted((w for w in cand if cap(w) >= top * (1 - 1e-9)), key=lambda w: w.id)
                    i = self._heavy_rr.get(spec.name, 0)
                    self._heavy_rr[spec.name] = i + 1
                    return best[i % len(best)], {"alpha": 0.0, "relaxed": relaxed}
                if cls == "light":
                    if STATIC_FOLLOW and (self.mode == "static" or NOCACHE_FOLLOW) and after is not None:
                        # a light step is deployed with the step it runs right after (the producer of its most
                        # recent input): the plan bundles it with that step; a pipeline's first step goes to the
                        # most capable device
                        prev = next((w for w in cand if w.id == after), None)
                        if prev is not None:
                            return prev, {"alpha": 0.0, "relaxed": relaxed}
                    cap = lambda w: (w.cpus * max(w.speed, 1e-9) ** STATIC_GAMMA, w.id)
                    return max(cand, key=cap), {"alpha": 0.0, "relaxed": relaxed}
            if STATIC_RULE in ("wrr", "plan"):
                # weighted round robin by device compute (cpus x speed ** STATIC_GAMMA), per
                # stage and without runtime state (smooth WRR: each pick takes the largest
                # running credit, then pays back the total weight)
                weights = {w.id: w.cpus * max(w.speed, 1e-9) ** STATIC_GAMMA for w in cand}
                credit = self._wrr.get(spec.name)
                if credit is None:
                    # each stage's rotation starts at its own phase (as independent balancers do): with one
                    # phase for all, consecutive compute stages of an instance always landed on the same worker
                    tot = sum(weights.values())
                    credit = self._wrr[spec.name] = {
                        k: (zlib.crc32(f"{spec.name}/{k}".encode()) % 1000) / 1000.0 * tot for k in weights} \
                        if STATIC_PHASE else {}
                for k, v in weights.items():
                    credit[k] = credit.get(k, 0.0) + v
                pick = max(cand, key=lambda w: (credit[w.id], w.id))
                credit[pick.id] -= sum(weights.values())
                return pick, {"alpha": 0.0, "relaxed": relaxed}
            if STATIC_RULE == "class":
                # the most capable feasible device class, its devices taken in turn
                top = max((w.speed, w.cpus) for w in cand)
                best = [w for w in cand if (w.speed, w.cpus) == top]
            else:
                # completion on declared capacity: a dispatch takes 1/cpus, behind the ones the
                # worker already runs (no cache view, no data location, no measured costs)
                # capacity: declared CPUs, or (compute) CPUs weighted by the worker's measured
                # speed, speed ** STATIC_GAMMA (1: CPUs x speed)
                cap = (lambda w: w.cpus * max(w.speed, 1e-9) ** STATIC_GAMMA) if STATIC_RULE == "compute" \
                    else (lambda w: w.cpus)
                key = {w.id: (len(w.inflight) + 1) / max(cap(w), 1e-9) for w in cand}
                low = min(key.values())
                best = [w for w in cand if key[w.id] <= low * (1 + 1e-9)]
                if STATIC_RULE == "capacity-top":  # ties go to the largest declared capacity
                    top = max(w.cpus for w in best)
                    best = [w for w in best if w.cpus == top]
            w = best[self._rr % len(best)]
            self._rr += 1
            return w, {"alpha": 0.0, "relaxed": relaxed}

        a = self.alpha(sigma)
        # reuses a deployment is amortized over: DEPLOY_AMORT, or (AMORT_RAMP) growing with the stage's
        # placements so far, so a stage only just deployed is not yet assumed to be reused
        n_placed = self._placed[spec.name] = self._placed.get(spec.name, 0) + 1
        reuse = min(DEPLOY_AMORT, 1.0 + n_placed / AMORT_RAMP) if AMORT_RAMP else DEPLOY_AMORT
        full = sum(p.size for p in pkgs)
        d_v = sum(full / self.bw(w) for w in alive) / len(alive)  # average full-miss cost
        now = perf()
        tiers = {w.id: w.tier for w in alive}
        ectx = None if all(self.frozen_estimate(w, spec) is not None for w in cand) \
            else self.exec_context(spec, workers)
        look_w = LOOKAHEAD or (1.0 if CULT_LOOK and self.mode == "full" else 0.0)
        if look_w and LOOK_MAX_S and self._mean_exec(spec.name, workers) >= LOOK_MAX_S:
            look_w = 0.0
        if look_w and LOOK_MIN_S and look:
            total = self._chain_s.get(spec.name) if self._work else None
            if total is None:
                total = self._mean_exec(spec.name, workers) + sum(self._mean_exec(c.name, workers) for c in look)
                if self._work:
                    self._chain_s[spec.name] = total  # fixed with the estimates by freeze()
            if total < LOOK_MIN_S:
                look_w = 0.0
        ahead = self.lookahead(look, cand, workers) if look_w else {}
        af = a  # cultivation weight: a stage whose packages no worker holds yet is all cultivation
        if FORM and self.mode == "full" and not any(p.hash in o.view for o in alive for p in pkgs):
            af = self.alpha_max
        best = None
        scores = {}
        for w in cand:
            view = w.view if self.sees_cache else frozenset()
            miss = [p for p in pkgs if p.hash not in view]
            # Eq. 5, a one-off cost: the reuse-aware score (full) amortizes it over DEPLOY_AMORT
            # later reuses of the package on w, unless admitting it evicts cached packages (a full
            # cache: no assured reuse); alpha = 0 charges it in full
            # while the layout is being cultivated (alpha above AMORT_ALPHA) the full charge keeps
            # the hot set compact; replicas for load are opened once it has stabilized
            blind_amort = ALPHA0_BLIND_AMORT and self.mode == "alpha0"
            amort = reuse if (self.mode == "full" or (AMORT_ALL and self.mode == "alpha0")) and a <= AMORT_ALPHA and not (
                not blind_amort and miss and reuse > 1 and self.evict_cost(w, miss, pkgs) > 0) else 1.0
            if spec.target != w.tier and not (ALPHA0_TIER_FREE and self.mode == "alpha0"):
                # a stage placed off its own tier is not assumed to be reused there: the amortized
                # deployment would otherwise make an edge stage look cheap on a cloud worker and
                # pull it across the cross-tier link, where it competes with the cloud stages
                amort = 1.0
            t_exec = self.exec_estimate(w, spec, workers, ectx) / max(1.0, w.rho.get(spec.name, 1.0))
            deploy = sum(p.size for p in miss) / self.bw(w)
            if ALPHA0_BLIND and self.mode == "alpha0":  # partly priced as if no worker held the packages
                deploy = (1.0 - ALPHA0_BLIND) * deploy + ALPHA0_BLIND * sum(p.size for p in pkgs) / self.bw(w)
            if AMORT_MIN_RATIO and t_exec < AMORT_MIN_RATIO * deploy:
                amort = 1.0
            deploy /= amort
            here = chain is not None and w.id == chain[0]
            if CHAIN_DEDUP and here:
                # riding its predecessor's dispatch, the stage's package travels in the same message and the
                # module it shares with the predecessor is parsed once (measured: 0.08-0.12 ms against 1.7 ms for
                # a W1 stage of its own, AGENTS.md v3e)
                deploy *= CHAIN_DEDUP
            if CONTENTION and w.inflight:
                # co-running calls hold native threads the stage would otherwise take: its
                # execution stretches with the share of w's cores they occupy
                skip = chain[1] if here else ()
                skip = skip if isinstance(skip, (set, frozenset)) else {skip}
                par = max(getattr(w, "parallel", w.cpus), 1e-6)
                busy = sum(min(d.cpus, par) for k, d in w.inflight.items()
                           if k not in skip and d.est_exec >= CONTENTION_MIN_S)
                t_exec *= 1.0 + CONTENTION * min(busy, par) / par
                if CONTENTION_MIN_S:
                    # and it stretches the compute-dominated calls already running on w for as long
                    # as the two overlap: a delay their instances pay, charged to this placement
                    # (a light call borrows the cores the running call leaves idle, so its declared
                    # share understates the overlap; INFLICT overrides it when set)
                    share = CONTENTION_INFLICT if CONTENTION_INFLICT else CONTENTION * min(spec.cpus, par) / par
                    t_exec += share * sum(
                        min(t_exec, max(0.0, d.est_exec - (now - (d.t_sent or now))))
                        for k, d in w.inflight.items() if k not in skip and d.est_exec >= CONTENTION_MIN_S)
                    if t_exec >= CONTENTION_MIN_S:
                        # a compute-dominated stage is itself stretched by whatever overlaps it
                        t_exec += share * sum(
                            min(t_exec, max(0.0, d.est_exec - (now - (d.t_sent or now))))
                            for k, d in w.inflight.items() if k not in skip and d.est_exec < CONTENTION_MIN_S)
            t_exec += self._reserve(w, spec, t_exec)
            dcost, lat = self.data_cost(w, data, tiers), (0.0 if here else self.latency(w))
            comm = dcost + lat                                                     # T_comm
            queue = self.queue_estimate(w, now, spec.name, exclude=chain[1] if here else None)
            # measured correction (master: actual less this raw estimate), only while there is
            # work on w to wait behind
            qbias = max(0.0, getattr(w, "queue_bias", {}).get(spec.name, 0.0)) if w.inflight else 0.0
            if PROBE:
                qbias = 0.0  # probed mode: no correction learned from measured waits
            raw = t_exec + deploy + queue + comm
            # measured cost calibration: what dispatches of this stage to w actually took less
            # what was predicted (it subsumes the queue correction)
            cbias = getattr(w, "cost_bias", {}).get(spec.name, 0.0) if COST_CAL and not here else 0.0
            cbias = max(cbias, -0.5 * raw)  # a correction never makes the cost negative
            if COST_CAL:
                qbias = 0.0
            imm = raw + qbias + cbias                                               # Eq. 6
            spill_free = CHAIN_SPILL_FREE and here  # riding on its predecessor's dispatch: no cross-tier round trip
            tier_free = ALPHA0_TIER_FREE and self.mode == "alpha0"  # alpha0 without the declared-tier preference
            if EDGE_SPILL and spec.target == "edge" and w.tier == "cloud" and not spill_free and not tier_free:
                # an edge-intent stage spills to the cloud tier only when that is clearly
                # cheaper: cloud capacity stays for the stages that need it
                imm *= 1.0 + EDGE_SPILL
            if CROSS_PENALTY_S and spec.target == "edge" and w.tier == "cloud" and not spill_free and not tier_free:
                # the extra round trip a cross-tier dispatch pays, charged flat (see CROSS_PENALTY_S)
                imm += CROSS_PENALTY_S
            if ahead:
                imm += look_w * ahead[w.id]
            align = (len(pkgs) - len(miss)) / len(pkgs)                             # Eq. 7
            if ALIGN_PENDING and a > 0 and miss:
                # the emerging layout: packages already on their way to w count as aligned
                sent = {h for d in w.inflight.values() for h in getattr(d, "sent", ())}
                align = (len(pkgs) - sum(1 for p in miss if p.hash not in sent)) / len(pkgs)
            if ALIGN_NEAR and near:
                align += ALIGN_NEAR * sum(h in view for h, _ in near) / len(near)
            # Eq. 8: the alignment term rewards placing onto the emerging cache layout and, by the
            # same weight, charges the alignment a placement destroys (packages its admission
            # evicts from a full cache, to be redeployed later); alpha = 0 removes both
            ew = max(a / self.alpha_max, EVICT_FLOOR if self.mode == "full" else 0.0)
            # like the eviction charge, full keeps a floor of the alignment reward once alpha has
            # decayed (the stable layout stays preferred); alpha = 0 removes both
            aa = max(a, ALIGN_FLOOR) if self.mode == "full" else a
            evict = self.evict_cost(w, miss, pkgs) if ew > 0 else 0.0
            cycles = False
            if ew > 0 and CHAIN_EVICT and segment:
                # chain-level eviction: a worker whose cache holds fewer functions than the stage's
                # chain segment has (counting the stages w may run) cycles its cache when the chain
                # runs there -- every stage of it is deployed anew at each instance, whatever w
                # holds at this moment.  Like evict_cost, a deployment pushed onto later dispatches.
                cap_n = getattr(w, "cap_entries", 0)
                need = 1  # the stage and the stages around it that w runs back to back: the chain
                for sp in segment[0]:  # is broken where a stage belongs to a tier w does not serve
                    if not self._intent(sp, w):
                        break
                    need += 1
                for level in segment[1]:
                    k = sum(1 for sp in level if self._intent(sp, w) and self._co.get(sp.name, 1.0) >= CHAIN_SEG_MIN)
                    if not k:
                        break
                    need += k
                if cap_n and need > cap_n:
                    evict += full / self.bw(w)
                    cycles = True
            if ew > 0 and HOT_CAP and getattr(w, "cap_entries", 0):
                hot = self._hot.get(w.id, {})
                demand = {h for h, t in hot.items() if now - t <= HOT_WIN_S} | {p.hash for p in pkgs}
                if len(demand) > w.cap_entries:
                    if not cycles:
                        evict += full / self.bw(w)
                        cycles = True
                    align = 0.0
            base = imm
            if FORM and af > 0:
                base -= (af / self.alpha_max) * (deploy + (queue + qbias if FORM_TERMS == "dq" else 0.0)) * \
                    (1.0 + (EDGE_SPILL if spec.target == "edge" and w.tier == "cloud" else 0.0))
            score = base - aa * align * d_v + ew * evict
            spill = 1.0 + (EDGE_SPILL if EDGE_SPILL and spec.target == "edge" and w.tier == "cloud" and not spill_free
                           and not tier_free else 0.0)
            scores[w.id] = {"exec": t_exec, "deploy": deploy, "queue": queue, "qbias": qbias,
                            # what the placement costs each time it recurs: without the one-off
                            # deployment, except on a worker whose cache the chain cycles, where
                            # the deployment itself recurs
                            "recur": imm - deploy * spill + (full / self.bw(w) * spill if cycles else 0.0),
                            "data": dcost, "lat": lat,
                            "raw": raw, "cbias": cbias,
                            "align": align, "evict": evict, "imm": imm,
                            "score": score}
            if best is None or score < best[0]:
                best = (score, w)
        pool = cand
        ab = max(a, ALIGN_FLOOR) if self.mode == "full" else a
        if ALIGN_BAND and ab > 0 and not FORM:
            # bounded alignment: the reward only chooses among candidates whose immediate cost is
            # within (1 + ALIGN_BAND * alpha) of the best, never a markedly slower placement
            k = "recur" if ALIGN_BAND_RECUR else "imm"
            low = min(s[k] for s in scores.values())
            pool = [w for w in cand if scores[w.id][k] <= low + abs(low) * ALIGN_BAND * ab + 1e-9] or cand
        pick = min(pool, key=lambda w: (scores[w.id]["score"], w.id))
        if PROBE and TIE_EPS:
            top = scores[pick.id]["score"]
            tie = [w for w in pool if scores[w.id]["score"] <= top + TIE_EPS * abs(top) + 1e-9]
            pick = min(tie, key=lambda w: (-getattr(w, "probe", 0.0), w.id))
        if BEST_FIT and chain is None:
            top = scores[pick.id]["score"]
            fit = [w for w in pool if scores[w.id]["score"] <= top + BEST_FIT * abs(top) + 1e-9]
            pick = min(fit, key=lambda w: (w.cpus * max(w.speed, 1e-9), scores[w.id]["score"], w.id))
        if PLACE_TEMP_S and len(pool) > 1:
            # softmin placement: a worker is chosen with probability exp(-(its score - the best) / T), so
            # near-best workers share the dispatches in proportion to how close they score (T -> 0: argmin)
            top = min(scores[w.id]["score"] for w in pool)
            wts = [math.exp(-(scores[w.id]["score"] - top) / PLACE_TEMP_S) for w in pool]
            pick = _rng.choices(pool, weights=wts)[0]
        if HOT_CAP:
            hot = self._hot.setdefault(pick.id, {})
            for p in pkgs:
                hot[p.hash] = now
            if len(hot) > 256:
                for h in [h for h, t in hot.items() if now - t > HOT_WIN_S]:
                    del hot[h]
        return pick, {"alpha": a, "relaxed": relaxed, "scores": scores}


if os.environ.get("HEDICES_SCHED_PROF"):
    # diagnostics: profile place() after freeze(); HEDICES_SCHED_PROF is the output path prefix,
    # the cumulative profile is rewritten to <prefix>.<mode> every 1000 placements
    import cProfile
    import io
    import pstats

    _prof, _prof_n, _place = cProfile.Profile(), [0], StemSched.place

    def _profiled_place(self, *a, **k):
        if not self._work:
            return _place(self, *a, **k)
        r = _prof.runcall(_place, self, *a, **k)
        _prof_n[0] += 1
        if _prof_n[0] % 1000 == 0:
            buf = io.StringIO()
            pstats.Stats(_prof, stream=buf).sort_stats("cumulative").print_stats(25)
            with open(f"{os.environ['HEDICES_SCHED_PROF']}.{self.mode}", "w") as f:
                f.write(f"placements {_prof_n[0]}\n{buf.getvalue()}")
        return r

    StemSched.place = _profiled_place
