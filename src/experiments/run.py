"""Experiment driver; runs inside the master namespace.

  steady   warm-up, then N instances at fixed concurrency (Fig. 6/7/9/10/11, Table 4)
  mixed    q instances of each of W1-W4 arrive together, C = 4q (Fig. 12)
  failure  steady load while the orchestrator makes a worker unreachable (Fig. 13)
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import threading
import time

from experiments.apps_systems import APP_SYSTEMS, app_module, make_inputs
from experiments.common import closed_loop, env_info, save, summary
from experiments.systems import SYSTEMS
from hedices.workloads.registry import WORKLOADS

SUITE = {"suite": "legacy"}


def workload(system: str, name: str):
    """(module, make_input) of a workload: the benchmark-suite application of the system under
    test (--suite apps) or the harness workload modules (--suite legacy)."""
    if SUITE["suite"] == "apps":
        return app_module(system, name), make_inputs(name)
    mod = WORKLOADS[name]
    return mod, pregen(mod)


def _profile_threads(path: str) -> None:
    import atexit
    import cProfile
    import pstats

    profs = []

    def start(*_):
        pr = cProfile.Profile()
        profs.append(pr)
        pr.enable()
        threading.setprofile(None)

    def dump():
        st = None
        for pr in profs:
            pr.disable()
            st = pstats.Stats(pr) if st is None else (st.add(pr) or st)
        if st is not None:
            st.dump_stats(path)

    threading.setprofile(start)
    start()
    atexit.register(dump)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--system", default="hedices", choices=sorted(APP_SYSTEMS))
    p.add_argument("--suite", default="apps", choices=["apps", "legacy"])
    p.add_argument("--phase", default="steady", choices=["steady", "mixed", "failure"])
    p.add_argument("--workload", default="W1")
    p.add_argument("--mode", default="full")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--warmup", type=float, default=60.0)
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--keep-warmup", action="store_true")
    p.add_argument("--bursts", default="1,2,3,4,5,6,7,8")
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--duration", type=float, default=40.0)
    p.add_argument("--status", default="", help="failure: file to publish busy workers to")
    p.add_argument("--hb-timeout", type=float, default=1.5, help="Hedices worker-failure detection: 3 missed 0.5 s heartbeats")
    p.add_argument("--recover-timeout", type=float, default=120.0,
                   help="failure phase: per-instance bound after which an instance counts as not recovered")
    # 300 s (2026-11-14): the Kubernetes baselines are measured once their replica set has converged
    # (Knative scaled out, every KubeEdge replica pulled and ready); with 0 the measurement started at
    # whatever point the scale-out / image pulls had reached (Knative W3 one infer replica: 381-683 ms,
    # else 200 ms).  Systems without a scaling replica set (Hedices, Ray, Dask) are not affected.
    p.add_argument("--settle-max", type=float, default=300.0,
                   help="extra warm-up until a scaling replica set converges (s; 0: none)")
    p.add_argument("--label", default="")
    p.add_argument("--cold-cache", action="store_true",
                   help="hedices: empty function caches after the warm touch")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    SUITE["suite"] = a.suite
    if os.environ.get("HEDICES_PROFILE"):  # diagnostics: per-thread cProfile, merged at exit
        _profile_threads(os.environ["HEDICES_PROFILE"])
    sysobj = (APP_SYSTEMS if a.suite == "apps" else SYSTEMS)[a.system](a)
    out = {"system": a.system, "phase": a.phase, "mode": a.mode if a.system == "hedices" else None,
           "workload": a.workload, "label": a.label, "concurrency": a.concurrency,
           "warmup_s": a.warmup, "suite": a.suite, "env": env_info()}
    try:
        if a.phase == "steady":
            out.update(steady(sysobj, a))
        elif a.phase == "mixed":
            out.update(mixed(sysobj, a))
        else:
            out.update(failure(sysobj, a))
        out.update(sysobj.extra())
    finally:
        sysobj.close()
    save(a.out, out)


PREGEN = 32


def pregen(mod):
    """Inputs are generated before timing so the driver does not compete for master CPU."""
    inputs = [mod.make_input(i) for i in range(PREGEN)]
    return lambda i: inputs[i % PREGEN]


def warm_touch(s, mod, name, make_input) -> dict:
    t = time.time()
    runs = s.warm(mod, name, make_input(0))
    return {"warm_touch_runs": runs, "warm_touch_s": time.time() - t}


def settle(s, run_one, make_input, a, nxt: int) -> int:
    """Systems whose replica set scales with load (Knative autoscaler, pending pods) keep warming
    up at the measurement rate until it stops changing (bounded by --settle-max seconds)."""
    if not hasattr(s, "settled"):
        return nxt
    s.settled()
    deadline = time.monotonic() + a.settle_max
    while time.monotonic() < deadline:
        _, nxt = closed_loop(run_one, make_input, a.concurrency, duration_s=15, start=nxt)
        if s.settled():
            break
    return nxt


MIX_PHASE = int(os.environ.get("HEDICES_MIX_PHASE", "0"))
# warm-up opens with this many instances in sequence (concurrency 1), so the cache layout forms in a
# reproducible order rather than from the race of concurrent first instances (0: none)
WARM_SEQ = int(os.environ.get("HEDICES_WARM_SEQ", "0"))
# the warm-up's closed-loop concurrency when it differs from the measured one (configs e6 "warm_conc"): the cache
# layout forms under a burst, as a deployment that comes up under load does, and the measurement then runs at the
# configured concurrency on whatever layout each mode formed (0: the measured concurrency)
WARM_CONC = int(os.environ.get("HEDICES_WARM_CONC", "0"))
# keep the warm-up's dispatch records (the measured latency is still the --n instances after it): a placement
# distribution over the whole run from cold caches, including the cache-formation period (Fig 7)
KEEP_WARM_RECORDS = os.environ.get("HEDICES_KEEP_WARM_RECORDS", "0") == "1"
# HEDICES_DRIVER_GC=1: collect the warm-up's garbage and freeze the surviving objects before the measured window, so a
# full collection of the driver's heap (every in-flight instance waits for it) does not fall inside the window; the
# collections that do (generation 2, >= 20 ms) are recorded either way ("gc_pauses_s")
DRIVER_GC = os.environ.get("HEDICES_DRIVER_GC", "0") == "1"


class _GcPauses:
    def __init__(self):
        self.t0, self.pauses = None, []

    def __call__(self, phase, info):
        if info.get("generation") != 2:
            return
        if phase == "start":
            self.t0 = time.perf_counter()
        elif self.t0 is not None:
            d = time.perf_counter() - self.t0
            if d >= 0.02:
                self.pauses.append(round(d, 4))
            self.t0 = None
# the measured phase under a changing load (Fig 7 setting): closed-loop concurrency steps through these levels,
# LOAD_STEP_N instances each (0: --n split evenly); empty: the fixed --concurrency
LOAD_STEPS = [int(x) for x in os.environ.get("HEDICES_LOAD_STEPS", "").split(",") if x.strip()]
LOAD_STEP_N = int(os.environ.get("HEDICES_LOAD_STEP_N", "0"))


def mix_workload(i: int) -> str:
    """Workload of MIX instance i: W1..W4 round-robin, or (HEDICES_MIX_PHASE=K) phases of K instances
    alternating between the application pairs W1+W2 and W3+W4 (the active set shifts over time)."""
    names = list(WORKLOADS)
    if not MIX_PHASE:
        return names[i % len(names)]
    pair = names[:2] if (i // MIX_PHASE) % 2 == 0 else names[2:]
    return pair[i % 2]


def mixed_stream(s, a):
    """--workload MIX: instances of W1..W4 co-deployed (see mix_workload)."""
    names = list(WORKLOADS)
    parts = {k: workload(a.system, k) for k in names}
    runners = {k: s.runner(mod, k) for k, (mod, _) in parts.items()}

    def make_input(i):
        k = mix_workload(i)
        args, kw = parts[k][1](i // len(names))
        return (k, args), kw

    def run_one(args, kw, timeout=900):
        k, real = args
        return runners[k](real, kw, timeout=timeout)
    return run_one, make_input


def counted(run_one, label: str, total: int | None = None, every_s: float = 10.0):
    """run_one that prints how many instances completed, at most every every_s seconds (a live progress line
    in the run log; the instances themselves are untouched)."""
    lock, n, last = threading.Lock(), [0], [time.monotonic()]

    def run(args, kw, **kws):
        r = run_one(args, kw, **kws)
        with lock:
            n[0] += 1
            now = time.monotonic()
            if now - last[0] >= every_s:
                last[0] = now
                print(f"progress: {label} {n[0]}" + (f"/{total}" if total else "") + " instances", flush=True)
        return r
    return run


def warm_retry(run_one, attempts: int = 30):
    """Warm-up instances are discarded: one refused by a stage replica whose Service routing is not up
    yet (the Kubernetes baselines right after a deployment, 2026-11-14: KubeEdge W2 / W4 refused the
    first calls of a replica in four runs) is retried after 1 s instead of failing the run.  Any other
    error, and every error of a measured instance, still fails it."""
    def run(args, kw, **kws):
        for k in range(attempts):
            try:
                return run_one(args, kw, **kws)
            except Exception as exc:  # noqa: BLE001
                if "Connection" not in type(exc).__name__ or k == attempts - 1:
                    raise
                time.sleep(1.0)
    return run


def steady(s, a) -> dict:
    if a.workload == "MIX":
        run_one, make_input = mixed_stream(s, a)
        res = {}
        if not a.keep_warmup:  # co-deployed applications: warm every one, as for a single one
            for k in WORKLOADS:
                mod_k, mk = workload(a.system, k)
                res.setdefault("warm_touch", {})[k] = warm_touch(s, mod_k, k, mk)
    else:
        mod, make_input = workload(a.system, a.workload)
        run_one = s.runner(mod, a.workload)
        res = warm_touch(s, mod, a.workload, make_input) if not a.keep_warmup else {}
    t0 = time.time()
    nxt = 0
    warm_one = warm_retry(run_one)
    if WARM_SEQ:  # the cache layout forms from a fixed sequence of instances run one at a time
        _, nxt = closed_loop(warm_one, make_input, 1, n=WARM_SEQ)
        res["warm_seq"] = WARM_SEQ
        print(f"warm-up: {WARM_SEQ} sequential instances", flush=True)
    if WARM_CONC:
        res["warm_conc"] = WARM_CONC
    print(f"warm-up: {a.warmup:.0f} s at concurrency {WARM_CONC or a.concurrency}", flush=True)
    _, nxt = closed_loop(counted(warm_one, "warm-up"), make_input, WARM_CONC or a.concurrency, duration_s=a.warmup,
                         start=nxt)
    if not a.keep_warmup:  # a deployment session (Table 4) is measured from the deployment on
        nxt = settle(s, warm_one, make_input, a, nxt)
    res["warmup_window"] = [t0, time.time()]
    if not a.keep_warmup and not KEEP_WARM_RECORDS:
        s.reset()
    res["keep_warm_records"] = KEEP_WARM_RECORDS
    import gc
    if DRIVER_GC:
        gc.collect()
        gc.freeze()
        res["driver_gc"] = "collected and frozen before the window"
    print(f"measuring: {a.n} instances at concurrency {a.concurrency}", flush=True)
    gcp = _GcPauses()
    gc.callbacks.append(gcp)
    t1 = time.time()
    if LOAD_STEPS:  # a load that changes over the run: closed-loop concurrency steps through LOAD_STEPS
        recs = []
        for c in LOAD_STEPS:
            r, nxt = closed_loop(run_one, make_input, c, n=LOAD_STEP_N or max(1, a.n // len(LOAD_STEPS)), start=nxt)
            recs += r
        res["load_steps"] = LOAD_STEPS
    else:
        recs, _ = closed_loop(counted(run_one, "measured", a.n), make_input, a.concurrency, n=a.n, start=nxt)
    res["measure_window"] = [t1, time.time()]
    gc.callbacks.remove(gcp)
    res["gc_pauses_s"] = gcp.pauses
    res["latency_s"] = [r[2] for r in recs]
    res["summary"] = summary(res["latency_s"])
    if a.workload == "MIX":  # instance i runs mix_workload(i)
        names = list(WORKLOADS)
        res["per_workload"] = {k: {"latency_s": v, **summary(v)} for k in names
                               if (v := [r[2] for r in recs if mix_workload(r[0]) == k])}
        print(f"{a.system} MIX {a.mode}: " + ", ".join(
            f"{k} mean={x['mean']*1e3:.1f}" for k, x in res["per_workload"].items()), flush=True)
    if hasattr(s, "packages") and a.workload != "MIX":
        res["packages"] = s.packages(mod, a.workload)
    print(f"{a.system} {a.workload} {a.mode}: mean={res['summary']['mean']*1e3:.1f} ms "
          f"p95={res['summary']['p95']*1e3:.1f} ms n={res['summary']['n']}", flush=True)
    return res


def mixed(s, a) -> dict:
    s.pool_div = len(WORKLOADS)  # Ray: the four applications share the cluster's actor CPUs
    mods = {k: workload(a.system, k) for k in WORKLOADS}
    runners = {k: s.runner(mod, k) for k, (mod, _) in mods.items()}
    inputs = {k: mk for k, (_, mk) in mods.items()}
    for k, (mod, _) in mods.items():
        warm_touch(s, mod, k, inputs[k])
    idx = {k: 0 for k in runners}
    failed: dict = {}  # workload -> instances not completed within --recover-timeout
    pool = cf.ThreadPoolExecutor(max_workers=64)

    def burst(q: int) -> dict:
        jobs = []
        for k in runners:
            for _ in range(q):
                args, kw = inputs[k](idx[k]); idx[k] += 1
                jobs.append((k, args, kw))

        def one(job):
            k, args, kw = job
            t = time.perf_counter()
            try:
                runners[k](args, kw, timeout=a.recover_timeout)
            except Exception as exc:  # not completed within the bound: censored at the bound
                failed[k] = failed.get(k, 0) + 1
                print(f"{a.system} {k} instance failed: {type(exc).__name__}", flush=True)
                return k, max(time.perf_counter() - t, a.recover_timeout)
            return k, time.perf_counter() - t

        lat = {k: [] for k in runners}
        for k, dt in pool.map(one, jobs):
            lat[k].append(dt)
        return lat

    t0 = time.time()
    end = time.monotonic() + a.warmup
    while time.monotonic() < end:
        burst(1)
    if hasattr(s, "settled"):
        # the four applications are rolled out together here (one pod per stage, then every replica), which can
        # take longer than a single application's: the bursts start on a converged cluster, waited for up to twice
        # --settle-max (a burst on a cluster still replacing pods loses instances and stops the measurement)
        s.settled()
        last, stop = time.monotonic(), time.monotonic() + 2 * a.settle_max
        while time.monotonic() < stop:
            burst(1)
            if time.monotonic() - last >= 15:
                last = time.monotonic()
                if s.settled():
                    break
    res = {"warmup_window": [t0, time.time()], "bursts": {}}
    s.reset()
    for q in [int(x) for x in a.bursts.split(",")]:
        lat = {k: [] for k in runners}
        failed.clear()
        for _ in range(a.rounds):
            for k, v in burst(q).items():
                lat[k].extend(v)
        res["bursts"][q] = {k: {"latency_s": v, "failed": failed.get(k, 0), **summary(v)}
                            for k, v in lat.items()}
        print(f"{a.system} q={q}: " + ", ".join(
            f"{k} p95={res['bursts'][q][k]['p95']*1e3:.0f}ms" for k in runners)
            + (f" failed={dict(failed)}" if failed else ""), flush=True)
        if failed:  # instances lost to a degraded cluster: higher burst levels are not measured
            res["stopped_at_q"] = q
            break
    pool.shutdown()
    return res


def failure(s, a) -> dict:
    mod, make_input = workload(a.system, a.workload)
    run_one = s.runner(mod, a.workload)
    warm_touch(s, mod, a.workload, make_input)
    t0 = time.time()
    _, nxt = closed_loop(warm_retry(run_one), make_input, a.concurrency, duration_s=a.warmup)
    nxt = settle(s, warm_retry(run_one), make_input, a, nxt)
    s.reset()
    stop = threading.Event()

    def publish():
        while not stop.wait(0.2):
            try:
                busy = sorted(s.busy())
            except Exception:
                busy = []
            with open(a.status + ".tmp", "w") as fh:
                json.dump({"t": time.time(), "busy": busy}, fh)
            import os
            os.replace(a.status + ".tmp", a.status)

    th = threading.Thread(target=publish, daemon=True)
    th.start()
    t1 = time.time()
    failed: list = []

    def timed(args, kw):
        # an instance the system does not recover within the bound counts as failed at the bound
        # (recorded, not fatal: the other instances still measure recovery)
        try:
            run_one(args, kw, timeout=a.recover_timeout)
        except Exception as exc:
            failed.append([time.time(), repr(exc)[:200]])
    recs, _ = closed_loop(timed, make_input, a.concurrency, duration_s=a.duration, start=nxt)
    stop.set()
    return {"warmup_window": [t0, t1], "measure_window": [t1, time.time()],
            "instances_wall": [[i, w, dt] for i, w, dt in recs], "failed": failed,
            "recover_timeout_s": a.recover_timeout}


if __name__ == "__main__":
    main()
