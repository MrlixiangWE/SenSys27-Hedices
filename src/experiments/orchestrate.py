"""Run the evaluation on the emulated testbed.  Root, host namespace.

  sudo .venv/bin/python -m experiments.orchestrate --plan e6,e9 --out results/<stamp>

Plans:
  smoke  short Hedices W1-W4 run to validate the testbed
  e6     scheduler ablations full/alpha0/static/rr/nocache x W1-W4 (Fig. 6, 7, 11, Table 4)
  e8     cold-start cache dynamics, warm-up dispatches kept (Fig. 8)
  e9     @NoRely off x W1-W4 (Fig. 9; "on" is e6/full)
  e10    Ray, Dask, KubeEdge, Knative x W1-W4 (Fig. 10, 11, Table 4)
  e12    mixed arrivals q=1..8 for every system (Fig. 12)
  e13    worker-unreachable recovery trials (Fig. 13)
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
import pwd
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

# Evaluation configuration: the testbed topology and the per-workload load settings each figure
# was measured with. Environment variables still take precedence, so an ablation can override any
# entry without editing the file.
_CONFIG = Path(os.environ.get("HEDICES_CONFIG") or Path(__file__).with_name("configs") / "phys6r.json")
CONFIG = json.loads(_CONFIG.read_text()) if _CONFIG.is_file() else {}
for _k, _v in (CONFIG.get("env") or {}).items():
    os.environ.setdefault(_k, str(_v))
if CONFIG.get("topology"):
    os.environ.setdefault("HEDICES_TOPOLOGY", CONFIG["topology"])

from testbed import k8s, tb  # noqa: E402

PY = tb.PY
VENV = Path(PY).parent
CPUFREQ = Path(__file__).resolve().parents[1] / "testbed" / "cpufreq.sh"
MASTER = tb.TOPO["master"]
WORKERS = tb.TOPO["workers"]
IP2NODE = {w["ip"]: w["name"] for w in WORKERS}
WL = ["W1", "W2", "W3", "W4"]
USER_IDS = (pwd.getpwnam(tb.USER).pw_uid, pwd.getpwnam(tb.USER).pw_gid)


def wl(plan: str, w: str) -> dict:
    """Per-workload load settings the figure was measured with (configs/<name>.json)."""
    return (CONFIG.get(plan) or {}).get(w) or {}

RAY_ENV = {"RAY_health_check_period_ms": "1000", "RAY_health_check_failure_threshold": "3",
           "RAY_health_check_timeout_ms": "1000", "RAY_memory_monitor_refresh_ms": "0",
           "RAY_USAGE_STATS_ENABLED": "0", "RAY_raylet_start_wait_time_s": "300",
           # keep pre-started workers alive between bursts (warm baseline)
           "RAY_idle_worker_killing_time_threshold_ms": "3600000"}
# operator tuning of the Ray baseline (configs "ray_env", e.g. the raylet's resource-report period or the
# scheduler's spread threshold); HEDICES_RAY_ENV (JSON) overrides it for scans
RAY_ENV.update({k: str(v) for k, v in (CONFIG.get("ray_env") or {}).items()})
RAY_ENV.update({k: str(v) for k, v in json.loads(os.environ.get("HEDICES_RAY_ENV") or "{}").items()})
DRIVER_ENV: dict = {}
SUITE = {"suite": "apps"}
DASK_ENV = {"DASK_DISTRIBUTED__SCHEDULER__WORKER_TTL": "3s"}
# operator tuning of the Dask baseline (configs "dask_env", e.g. work stealing, scheduler-side
# queuing); HEDICES_DASK_ENV (JSON) overrides it for scans
DASK_ENV.update({k: str(v) for k, v in (CONFIG.get("dask_env") or {}).items()})
DASK_ENV.update({k: str(v) for k, v in json.loads(os.environ.get("HEDICES_DASK_ENV") or "{}").items()})


class NetSampler(threading.Thread):
    """Samples bytes sent by every node (host-side veth rx counters)."""

    def __init__(self, period: float = 0.5):
        super().__init__(daemon=True)
        self.period, self.samples, self._halt = period, [], threading.Event()

    def run(self):
        while not self._halt.is_set():
            self.samples.append([time.time(), tb.netstat_cluster()])
            self._halt.wait(self.period)

    def stop(self):
        self._halt.set()
        self.join()
        return self.samples


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def spawn(n: dict, unit: str, argv: list[str], env: dict | None = None) -> None:
    tb.spawn(n, unit, argv, env)


def stop_all() -> None:
    tb.stop_cluster()
    time.sleep(1.5)


def plain_testbed() -> None:
    """Hedices, Ray and Dask run as processes in plain node namespaces."""
    if k8s.running_pid(MASTER):
        k8s.down()
    tb.up_cluster()  # idempotent; recreates the namespaces after a host reboot


def start_k8s(flavor: str, workloads: list[str]) -> None:
    """The Kubernetes baseline with exactly ``workloads`` deployed (pods stay warm across runs)."""
    stop_all()
    state, prof = k8s.STATE / "flavor", k8s.STATE / "profile"
    want_prof = "failover" if k8s.failover() else ""
    # a cluster built for another purpose is rebuilt: Fig 13's failure trials run on one with fast failure
    # detection (k8s.failover()), every other experiment on one without
    if not (k8s.running_pid(MASTER) and state.exists() and state.read_text().strip() == flavor
            and (prof.read_text().strip() if prof.exists() else "") == want_prof):
        k8s.down()
        k8s.up(flavor)
        k8s.images([w.lower() for w in WL])
    deployed = {p.stem.split("-", 1)[1] for p in k8s.STATE.glob(f"{flavor}-w?.yaml")}
    want = {w.lower() for w in workloads}
    factor = 0.5 if len(want) > 1 else 1.0
    if factor != k8s.REQUEST_FACTOR["value"] or (len(want) > 1 and want != deployed):
        # applications deployed together are rolled out together (one pod per stage first)
        k8s.REQUEST_FACTOR["value"] = factor
        if deployed:
            k8s.undeploy(sorted(deployed))
            for w in deployed:
                for f in (f"{flavor}-{w}.yaml", f"{flavor}-{w}.first.yaml"):
                    (k8s.STATE / f).unlink(missing_ok=True)
            deployed = set()
    if deployed - want:
        k8s.undeploy(sorted(deployed - want))
        for w in deployed - want:
            for f in (f"{flavor}-{w}.yaml", f"{flavor}-{w}.first.yaml"):
                (k8s.STATE / f).unlink(missing_ok=True)
    if want - deployed:  # already deployed workloads keep their warm pods
        k8s.deploy(sorted(want - deployed))


# Branch @NoRely pool choice (--norely-mode): "thread" keeps one model copy per worker (memory-
# bound hosts), "auto" is Alg. 2 (profile, then thread or process pool by the stage's CPU share)
NORELY = {"mode": "thread", "slots": None, "nocache_state": "cold"}


def start_hedices(flags=()) -> None:
    plain_testbed()
    stop_all()
    flags = list(flags)
    if "--norely-mode" not in flags:
        flags += ["--norely-mode", NORELY["mode"]]
    if "--no-cache" in flags:
        flags += ["--nocache-state", NORELY["nocache_state"]]
    if NORELY["slots"] and "--slot-policy" not in flags:  # Sec. 4.5 slot ledger for every run
        flags += ["--slot-policy", NORELY["slots"], "--slot-efficiency", str(NORELY["efficiency"])]
    flags += NORELY.get("extra", [])
    tb.start_branches(flags)


def start_ray(cpu_div: int = 2) -> None:
    plain_testbed()
    stop_all()
    tb.clean_ray_cluster()
    ray = str(VENV / "ray")
    spawn(MASTER, "master-ray", [ray, "start", "--head", "--block", f"--node-ip-address={MASTER['ip']}",
                                 "--port=6379", "--num-cpus=0", "--include-dashboard=false",
                                 "--object-store-memory=150000000", "--disable-usage-stats"], RAY_ENV)
    wait_started(MASTER, "master-ray")
    # one node at a time: raylets initializing concurrently in the shared /tmp/ray race
    for w in WORKERS:
        spawn(w, f"{w['name']}-ray",
              [ray, "start", "--block", f"--address={MASTER['ip']}:6379", f"--node-ip-address={w['ip']}",
               # cpu_div=2: half the device's logical CPUs, since the applications size their actor
               # pools per CPU (one model replica each) and the paper-scale count of a single
               # application does not fit the host memory; mixed arrivals shrink every pool
               # instead (pool_div) and keep the full count, so actors leave CPUs for tasks
               f"--num-cpus={tb.logical_cpus(w) // cpu_div}", f"--resources={json.dumps({w['tier']: 100})}",
               # logical memory = device memory reported to every scheduler (as Branch --mem-mb)
               f"--memory={w['mem_mb'] << 20}",
               "--object-store-memory=80000000", "--disable-usage-stats",
               # all emulated nodes share one filesystem: each needs its own session dir/sockets
               f"--temp-dir=/tmp/ray-{w['name']}"], RAY_ENV)
        wait_started(w, f"{w['name']}-ray")


def wait_started(n: dict, unit: str, timeout: float = 240.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if "Ray runtime started" in tb.journal(n, unit):
            return
        if not tb.unit_active(n, unit):
            raise RuntimeError(f"{unit} exited during startup")
        time.sleep(1.0)
    raise TimeoutError(f"{unit} did not start")


def start_dask(extra_resources=None) -> None:
    """extra_resources(worker) -> dict of additional Dask worker resources (default: the logical
    CPU and memory every scheduler sees, which the applications' resource declarations use)."""
    plain_testbed()
    stop_all()
    extra_resources = extra_resources or (lambda w: {"CPU": tb.logical_cpus(w), "MEM": w["mem_mb"] << 20})
    dask = str(VENV / "dask")
    spawn(MASTER, "master-dask", [dask, "scheduler", "--host", MASTER["ip"], "--port", "8786",
                                  "--no-dashboard"], DASK_ENV)
    time.sleep(4)
    for w in WORKERS:
        spawn(w, f"{w['name']}-dask",
              [dask, "worker", f"tcp://{MASTER['ip']}:8786", "--host", w["ip"],
               "--nthreads", str(tb.logical_cpus(w)), "--nworkers", "1", "--resources", ",".join(f"{k}={v}" for k, v in {w['tier']: 100, **(extra_resources(w) if extra_resources else {})}.items()),
               "--memory-limit", "0", "--no-dashboard", "--name", w["name"],
               "--preload", "hedices.workloads.preload"], DASK_ENV)
        time.sleep(0.3)


STARTERS = {"hedices": start_hedices, "ray": start_ray, "dask": start_dask,
            "kubeedge": lambda workloads=WL: start_k8s("kubeedge", workloads),
            "knative": lambda workloads=WL: start_k8s("knative", workloads)}


def start(system: str, workloads: list[str]) -> None:
    if system in ("kubeedge", "knative"):
        STARTERS[system](workloads)
    else:
        STARTERS[system]()


def drive(out: Path, unit: str, args: list[str], wait: bool = True) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    os.chown(out.parent, *USER_IDS)
    argv = [PY, "-m", "experiments.run", "--out", str(out), "--workers", str(len(WORKERS)),
            "--suite", SUITE["suite"], *args]
    # runtime switches the configuration sets for one workload (configs/<name>.json "workload_env");
    # the caller's environment still takes precedence
    env = dict(DRIVER_ENV)
    if "--workload" in args:
        for k, v in ((CONFIG.get("workload_env") or {}).get(args[args.index("--workload") + 1]) or {}).items():
            if k not in os.environ:
                env[k] = str(v)
    if not wait:
        spawn(MASTER, unit, argv, env)
        return 0
    sampler = NetSampler()
    sampler.start()
    t0 = time.time()
    rc = subprocess.run(tb.unit_cmd(MASTER, unit, argv, wait=True, env=env)).returncode
    samples = sampler.stop()
    out.with_suffix(".net.json").write_text(json.dumps(samples))
    log(f"  {unit}: rc={rc} in {time.time() - t0:.0f}s")
    return rc


def done(out: Path) -> bool:
    return out.exists()


def plan_smoke(root: Path, a) -> None:
    for w in WL:
        start_hedices()  # a new master numbers its objects from scratch: fresh Branch stores
        out = root / "smoke" / f"hedices_full_{w}.json"
        drive(out, f"smoke-{w}", ["--workload", w, "--warmup", "10", "--n", "8"])


def plan_e6(root: Path, a, section: str = "e6") -> None:
    evict0 = os.environ.get("HEDICES_CACHE_EVICT")
    for w in a.workloads:
        # Fig 7 (plan e7) measures the dispatch distribution on its own setting: configs "e7" overrides the
        # Fig 6 settings of the workload where they differ (e.g. edge caches that do not force a stage off
        # its tier), everything else is the Fig 6 run
        cfg = {**wl("e6", w), **(wl(section, w) if section != "e6" else {})}
        conc, n = cfg.get("concurrency", a.concurrency), cfg.get("n", a.n)
        conc = int(os.environ.get("HEDICES_E6_CONC") or conc)  # override for scans
        # Fig 7's load sweep (run_all.sh fig7 sets HEDICES_E7_SWEEP with each concurrency): a workload runs at the
        # levels of its configs "load_sweep" (default 2-5) -- the sweep reaches the load at which every edge device
        # of the workload is busy
        if os.environ.get("HEDICES_E7_SWEEP") and (conc not in (wl(section, w).get("load_sweep") or [2, 3, 4, 5]) or
                                                   int(os.environ["HEDICES_E7_SWEEP"]) > wl(section, w).get("load_sweep_runs", 1)):
            continue
        # HEDICES_E6_CACHE overrides the configured cache entries (for scans; the configs value is
        # what a figure was measured with)
        ce = os.environ.get("HEDICES_E6_CACHE") or cfg.get("cache_entries")
        if ce:
            os.environ["HEDICES_CACHE_ENTRIES"] = ce
        # configs "<section>" "cache_evict": the function cache's replacement policy for this figure's caches
        # (branch --cache-evict, read by tb.start_branches; default lru).  Read from the section itself, not
        # inherited from e6; a caller's HEDICES_CACHE_EVICT takes precedence
        ev = evict0 or (wl(section, w) or {}).get("cache_evict")
        if ev:
            os.environ["HEDICES_CACHE_EVICT"] = str(ev)
        else:
            os.environ.pop("HEDICES_CACHE_EVICT", None)
        # Fig. 6's own input (configs "e6" "segment": k consecutive samples per instance, as for
        # Fig. 9); HEDICES_E6_SEGMENT overrides it for scans
        seg = os.environ.get("HEDICES_E6_SEGMENT") or cfg.get("segment")
        # configs "e6" "warm_seq": the warm-up opens with this many instances run one at a time, so the
        # cache layout forms from a fixed order rather than from the race of the first concurrent
        # instances (same for every mode); HEDICES_E6_WARM_SEQ overrides it for scans
        wseq = os.environ.get("HEDICES_E6_WARM_SEQ") or cfg.get("warm_seq")
        wconc = os.environ.get("HEDICES_E6_WARM_CONC") or cfg.get("warm_conc")  # configs e6 "warm_conc"
        # configs "e6" "segment_mix" (W4): frames per instance cycle through these counts (a varying number
        # of frames per trigger); HEDICES_E6_SEGMENT_MIX overrides it for scans
        smix = os.environ.get("HEDICES_E6_SEGMENT_MIX") or cfg.get("segment_mix")
        # configs "e6" "env": scheduler settings of the Hedices-internal experiment (Fig 6 / 7) for this workload;
        # the caller's environment still takes precedence
        e6env = {k: str(v) for k, v in (cfg.get("env") or {}).items() if k not in os.environ}
        for mode in a.modes:
            out = root / section / f"hedices_{mode}_{w}.json"
            if done(out):
                continue
            DRIVER_ENV.update(e6env)  # the master runs in the driver
            start_hedices(["--no-cache"] if mode == "nocache" else [])
            if seg:
                DRIVER_ENV["HEDICES_INPUT_SEGMENT"] = str(seg)
            if wseq:
                DRIVER_ENV["HEDICES_WARM_SEQ"] = str(wseq)
            if wconc:
                DRIVER_ENV["HEDICES_WARM_CONC"] = str(wconc)
            if smix:
                DRIVER_ENV["HEDICES_INPUT_SEGMENT_MIX"] = str(smix)
            # configs "cpu_max_khz" ({device class: kHz}): CPU frequency cap of that class during this workload's
            # runs (testbed/cpufreq.sh; restored afterwards); HEDICES_E6_CPU_MAX_KHZ (JSON) overrides it
            freq = json.loads(os.environ.get("HEDICES_E6_CPU_MAX_KHZ") or "null") or cfg.get("cpu_max_khz") or {}
            for cls, khz in freq.items():
                subprocess.run(["bash", str(CPUFREQ), "set", cls, str(int(khz))], check=False)
            try:
                drive(out, f"{section}-{mode}-{w}", [*cold(a), "--workload", w, "--mode", mode, "--warmup", str(a.warmup),
                                              "--n", str(n), "--concurrency", str(conc)])
            finally:
                for cls in freq:
                    subprocess.run(["bash", str(CPUFREQ), "restore", cls], check=False)
                DRIVER_ENV.pop("HEDICES_INPUT_SEGMENT", None)
                DRIVER_ENV.pop("HEDICES_WARM_SEQ", None)
                DRIVER_ENV.pop("HEDICES_WARM_CONC", None)
                DRIVER_ENV.pop("HEDICES_INPUT_SEGMENT_MIX", None)
                for k in e6env:
                    DRIVER_ENV.pop(k, None)


def plan_e6cold(root: Path, a) -> None:
    """Fig. 6 alpha=0 column over cache formation: full and alpha0 measured from empty function
    caches (no warm-up; processes and models warm), --cold-n instances, --trials repetitions.
    In steady state sigma(t) -> 0 makes alpha(t) -> 0 (Eq. 10), so the alignment reward acts
    while the layout forms."""
    for k in range(a.trials):
        for w in a.workloads:
            # alternate which mode starts first: one-time start-up costs fall on either side
            for mode in (("full", "alpha0") if k % 2 == 0 else ("alpha0", "full")):
                out = root / "e6cold" / f"hedices_{mode}_{w}_r{k}.json"
                if done(out):
                    continue
                start_hedices()
                drive(out, f"e6cold-{mode}-{w}-r{k}", ["--cold-cache", "--workload", w, "--mode", mode,
                                                       "--warmup", "0", "--n", str(a.cold_n),
                                                       "--concurrency", str(a.concurrency)])


def plan_e6mix(root: Path, a) -> None:
    """Fig. 6/7 with the four applications co-deployed: one interleaved W1-W4 stream per mode, so
    the workloads contend for the worker function caches throughout the measurement."""
    for mode in a.modes:
        out = root / "e6mix" / f"hedices_{mode}_MIX.json"
        if done(out):
            continue
        start_hedices(["--no-cache"] if mode == "nocache" else [])
        drive(out, f"e6mix-{mode}", [*cold(a), "--workload", "MIX", "--mode", mode,
                                     "--warmup", str(a.warmup), "--n", str(a.n * 4),
                                     "--concurrency", str(a.concurrency)])


def plan_e8(root: Path, a) -> None:
    """Cold-start cache dynamics: W1-W4 instances interleaved from empty caches (Fig. 8)."""
    out = root / "e8" / "hedices_full_MIX_cold.json"
    if done(out):
        return
    start_hedices([] if a.cache_kb else ["--cache-mb", str(a.e8_cache_mb)])
    drive(out, "e8-MIX", ["--workload", "MIX", "--mode", "full", "--warmup", "0", "--keep-warmup",
                          "--n", str(a.e8_instances), "--concurrency", str(a.concurrency)])


def cold(a) -> list[str]:
    return ["--cold-cache"] if a.cold_cache else []


def plan_e9(root: Path, a) -> None:
    """@NoRely on vs off on the same topology.  Placement is pinned to the annotated stages'
    own tier (strict intent), so both settings run the same placement and only local
    parallelism differs; on/off alternate over --trials repetitions.  Branch runs the
    worker-level slot ledger of Alg. 2 (--e9-slots, default borrow) in both settings."""
    # the topology of configs/<name>.json "e9" (tb reads HEDICES_TOPOLOGY at import, so the plan
    # cannot switch it: refuse to run on another one instead of measuring the wrong setting)
    topo = os.environ.get("HEDICES_TOPOLOGY", "")
    want = {wl("e9", w).get("topology") for w in a.workloads} - {None}
    if want and want != {topo}:
        raise SystemExit(f"e9: configs ask for topology {sorted(want)}, running on {topo!r}; "
                         f"run the plan with HEDICES_TOPOLOGY set to it")
    DRIVER_ENV["HEDICES_INTENT"] = "strict"
    DRIVER_ENV["HEDICES_PIN_MODE"] = a.e9_pin_mode
    slots = (["--slot-policy", a.e9_slots, "--slot-efficiency", str(a.e9_efficiency)]
             if a.e9_slots != "none" else [])
    def run(out, label, flags, w):
        # configs "e9" "norely_batch": the Branch coalesces overlapping @NoRely calls on a resident
        # model (hedices/accel.py); in both settings, where it only acts with @NoRely (without it a
        # stage runs one call at a time, nothing overlaps)
        if wl("e9", w).get("norely_batch"):
            flags = [*flags, "--norely-batch"]
            if wl("e9", w).get("batch_threads") is not None:  # threads of a coalesced run
                flags = [*flags, "--norely-batch-threads", str(wl("e9", w)["batch_threads"])]
        # configs "e9" "slot_policy": the slot-ledger policy of the Branch in both settings
        # (hedices/branch.py SlotLedger; "demand" only changes how @NoRely calls share a worker)
        pol = wl("e9", w).get("slot_policy")
        if pol and "--slot-policy" in flags:
            flags = list(flags)
            flags[flags.index("--slot-policy") + 1] = pol
        elif pol:
            flags = [*flags, "--slot-policy", pol, "--slot-efficiency", str(a.e9_efficiency)]
        # configs "e9" "max_overlap" (demand ledger): at most this many overlapped calls of a stage;
        # HEDICES_E9_MAX_OVERLAP overrides it (for scans)
        mo = os.environ.get("HEDICES_E9_MAX_OVERLAP") or wl("e9", w).get("max_overlap")
        if mo:
            flags = [*flags, "--slot-max-overlap", str(mo)]
        start_hedices(flags)
        # the load level of configs/<name>.json "e9" (until 2026-11-11 this plan ignored it and ran
        # every workload at --concurrency, 4 by default: no stage ever queued, so on == off)
        conc = wl("e9", w).get("concurrency", a.concurrency)
        # Fig. 9's own input: a segment of k consecutive samples per instance (apps/common), so that
        # execution, which @NoRely acts on, carries the instance; HEDICES_INPUT_SEGMENT overrides
        # the configured k (for scans)
        seg = os.environ.get("HEDICES_INPUT_SEGMENT") or wl("e9", w).get("segment")
        if seg:
            DRIVER_ENV["HEDICES_INPUT_SEGMENT"] = str(seg)
        try:
            drive(out, f"e9-{label}-{w}", [*cold(a), "--workload", w, "--mode", "full",
                                            "--warmup", str(a.warmup), "--n", str(a.n),
                                            "--concurrency", str(conc), "--label", label])
        finally:
            DRIVER_ENV.pop("HEDICES_INPUT_SEGMENT", None)

    try:
        # --e9-pin: a placement settled by an unpinned @NoRely-on run; both settings then run
        # pinned to it, so the pair differs only in local parallelism.  One placement per
        # workload, or per trial with --e9-probe-per-trial (trials then average over placements)
        probe_of = lambda w, k: root / "e9" / (f"hedices_placement_{w}_r{k}.json" if a.e9_probe_per_trial
                                               else f"hedices_placement_{w}.json")
        for k in range(a.trials):
            if a.e9_pin:
                for w in a.workloads:
                    probe = probe_of(w, k)
                    if not done(probe):
                        DRIVER_ENV.pop("HEDICES_PIN_FROM", None)
                        # --e9-probe-from off: the placement StemSched settles without local
                        # overlap (width 1); alt: trials alternate between the placements settled
                        # with and without it, so each setting also runs on the other's layout
                        src = a.e9_probe_from if a.e9_probe_from != "alt" else ("on", "off")[k % 2]
                        run(probe, "placement",
                            [*slots, "--norely-mode", "off"] if src == "off" else slots, w)
            for label, flags in (("norelyon", slots), ("norelyoff", [*slots, "--norely-mode", "off"])):
                for w in a.workloads:
                    out = root / "e9" / f"hedices_{label}_{w}_r{k}.json"
                    if done(out):
                        continue
                    probe = probe_of(w, k)
                    DRIVER_ENV.pop("HEDICES_PIN_FROM", None)
                    if a.e9_pin and probe.exists():
                        DRIVER_ENV["HEDICES_PIN_FROM"] = str(probe.resolve())
                    run(out, label, flags, w)
    finally:
        DRIVER_ENV.pop("HEDICES_INTENT", None)
        DRIVER_ENV.pop("HEDICES_PIN_FROM", None)
        DRIVER_ENV.pop("HEDICES_PIN_MODE", None)


def plan_e10(root: Path, a) -> None:
    """End-to-end comparison (Fig. 10/11, Table 4): every system in --systems on W1-W4."""
    os.environ.pop("HEDICES_CACHE_ENTRIES", None)  # the steady runs are not capacity-limited
    for system in a.systems:
        for w in a.workloads:
            e10_one(root, a, system, w)


def e10_one(root: Path, a, system: str, w: str) -> None:
    """One steady-state run of Fig. 10/11 (configs "e10" concurrency of the workload)."""
    conc = wl("e10", w).get("concurrency", a.concurrency)
    out = root / "e10" / f"{system}_{w}.json"
    if done(out):
        return
    if system == "hedices":
        start_hedices()
        drive(out, f"e10-hedices-{w}", [*cold(a), "--workload", w, "--mode", "full",
                                        "--warmup", str(a.warmup), "--n", str(a.n),
                                        "--concurrency", str(conc)])
        return
    start(system, [w])
    drive(out, f"e10-{system}-{w}", ["--system", system, "--workload", w, "--warmup",
                                      str(a.warmup), "--n", str(a.n), "--concurrency", str(conc)])


def plan_t3(root: Path, a) -> None:
    """Table 4 measurement runs: E2E runs of every system in --systems with the per-node packet
    counters and, for Ray/Dask, the stage-call data trace (experiments/datatrace) enabled, so the
    analysis can separate the application data that crossed nodes from the framework's own
    deployment/control traffic.  Kept apart from the latency runs (tracing adds work)."""
    for system in a.systems:
        for w in a.workloads:
            out = root / "t3" / f"{system}_{w}.json"
            if done(out):
                continue
            tdir = (root / "t3" / "trace" / f"{system}_{w}").resolve()
            env = {"HEDICES_DATATRACE": str(tdir), "PYTHONPATH": str(tb.ROOT / "experiments" / "datatrace")}
            saved = {k: dict(v) for k, v in (("ray", RAY_ENV), ("dask", DASK_ENV))}
            try:
                RAY_ENV.update(env); DASK_ENV.update(env)
                DRIVER_ENV["HEDICES_DATATRACE"] = str(tdir)
                args = ["--workload", w, "--warmup", str(a.warmup), "--n", str(a.n),
                        "--concurrency", str(a.concurrency)]
                if system == "hedices":
                    start_hedices()
                    drive(out, f"t3-hedices-{w}", [*cold(a), *args, "--mode", "full"])
                else:
                    start(system, [w])
                    drive(out, f"t3-{system}-{w}", ["--system", system, *args])
                if system in ("ray", "dask"):
                    stop_all()  # flush the workers' trace files
                    out.with_suffix(".trace.jsonl").write_text(tb.trace_collect_cluster(str(tdir)))
            finally:
                RAY_ENV.clear(); RAY_ENV.update(saved["ray"])
                DASK_ENV.clear(); DASK_ENV.update(saved["dask"])
                DRIVER_ENV.pop("HEDICES_DATATRACE", None)


T3S_IDLE_S = float(os.environ.get("HEDICES_T3S_IDLE_S", "20"))  # idle window of plan_t3s (s)


def plan_t3s(root: Path, a) -> None:
    """Table 4, deployment session: what crosses the network from the moment a workload is deployed
    onto a running system until --n instances have completed, less the application data, per
    instance -- the framework's deployment path (function packages / code / pod specifications,
    registration, placement, status) amortized over a session of --n instances.  The system's own
    start-up (processes, cluster control plane) is outside the window; every run starts with nothing
    of the workload deployed, and no warm-up is discarded (--warmup 0 --keep-warmup)."""
    for system in a.systems:
        for w in a.workloads:
            t3s_one(root, a, system, w, a.concurrency)


def t3s_one(root: Path, a, system: str, w: str, conc: int, n: int | None = None) -> None:
    """One deployment session of Table 4 (see plan_t3s): ``n`` (default --n) instances at
    concurrency ``conc``."""
    n = a.n if n is None else n
    out = root / "t3s" / f"{system}_{w}.json"
    if done(out):
        return
    tdir = (root / "t3s" / "trace" / f"{system}_{w}").resolve()
    env = {"HEDICES_DATATRACE": str(tdir), "PYTHONPATH": str(tb.ROOT / "experiments" / "datatrace")}
    saved = {k: dict(v) for k, v in (("ray", RAY_ENV), ("dask", DASK_ENV))}
    k8s_sys = system in ("kubeedge", "knative")
    try:
        RAY_ENV.update(env); DASK_ENV.update(env)
        DRIVER_ENV["HEDICES_DATATRACE"] = str(tdir)
        # the system up, nothing of the workload deployed: Kubernetes without the workload's
        # pods; Hedices / Ray / Dask processes, whose functions the driver deploys
        if k8s_sys:
            start_k8s(system, [])
        elif system == "hedices":
            start_hedices()
        else:
            start(system, [w])
        sampler = NetSampler()
        sampler.start()
        t_deploy = time.time()
        if k8s_sys:
            start_k8s(system, [w])  # the pods of every stage, created inside the window
        # the system idle right before the first request: its background traffic (heartbeats,
        # status, metric scraping) goes on whether or not anything is deployed or invoked, and is
        # subtracted at its idle rate over the session (table3_session)
        t_idle = time.time()
        time.sleep(T3S_IDLE_S)
        t_idle = [t_idle, time.time()]
        args = ["--workload", w, "--warmup", "0", "--keep-warmup", "--n", str(n),
                "--concurrency", str(conc)]
        if system == "hedices":
            drive(out, f"t3s-hedices-{w}", [*cold(a), *args, "--mode", "full"])
        else:
            drive(out, f"t3s-{system}-{w}", ["--system", system, *args])
        samples = sampler.stop()
        out.with_suffix(".session.json").write_text(json.dumps({"t_deploy": t_deploy, "t_idle": t_idle,
                                                                "samples": samples}))
        if system in ("ray", "dask"):
            stop_all()  # flush the workers' trace files
            out.with_suffix(".trace.jsonl").write_text(tb.trace_collect_cluster(str(tdir)))
    finally:
        RAY_ENV.clear(); RAY_ENV.update(saved["ray"])
        DASK_ENV.clear(); DASK_ENV.update(saved["dask"])
        DRIVER_ENV.pop("HEDICES_DATATRACE", None)


def plan_e12(root: Path, a) -> None:
    for system in a.systems:
        out = root / "e12" / f"{system}_mixed.json"
        if done(out):
            continue
        start_ray(cpu_div=a.e12_ray_cpu_div) if system == "ray" else start(system, WL)
        drive(out, f"e12-{system}", ["--system", system, "--phase", "mixed", "--warmup", str(a.warmup),
                                     "--bursts", a.bursts, "--rounds", str(a.rounds)])


def plan_eB(root: Path, a) -> None:
    """The comparison with the baselines as one experiment, one configuration (Table 4, Fig. 10-13).
    Every workload runs at its configs "e10" concurrency in every phase.  Per system: each workload
    is measured in steady state (--warmup, --n; Fig. 10/11, results e10/), then undeployed and
    deployed again onto the running system and driven for --eb-session-n instances (deployment
    session, Table 4, results t3s/); then, for the systems in --eb-burst-systems, the four workloads
    under bursts (Fig. 12, e12/) and worker failures on --e13-workload (Fig. 13, e13/).  The analysis
    reads the same result files as for the separate plans."""
    os.environ.pop("HEDICES_CACHE_ENTRIES", None)  # not capacity-limited (as plan_e10)
    burst = set(a.eb_burst_systems.split(","))
    # --eb-phases: the phases to run (a single figure re-run takes just its own; the analysis is the same)
    ph = set(a.eb_phases.split(","))
    for system in a.systems:
        for w in a.workloads:
            # steady state first, exactly as plan_e10; then the session, which starts by undeploying the
            # workload (as plan_t3s).  The other order measured Knative mid-scale-up: the steady runs right
            # after a session found single replicas on slow nodes (v2i: W4 134 ms, standalone 79 ms)
            if "steady" in ph:
                e10_one(root, a, system, w)
            if "session" in ph:
                t3s_one(root, a, system, w, wl("e10", w).get("concurrency", a.concurrency), a.eb_session_n)
        if system in burst:
            if "burst" in ph:
                plan_e12(root, _with(a, systems=[system]))
            if "failure" in ph:
                plan_e13(root, _with(a, systems=[system],
                                     concurrency=wl("e10", a.e13_workload).get("concurrency", a.concurrency)))


def _with(a, **kw):
    """A copy of the parsed arguments with some fields replaced."""
    b = argparse.Namespace(**vars(a))
    for k, v in kw.items():
        setattr(b, k, v)
    return b


def plan_e13(root: Path, a) -> None:
    """Fig 13.  configs "e13" "failover": the settings every system runs the failure trials with (each detects a
    lost worker within about a second, as Hedices' Branch heartbeat) -- applied for these trials only and
    restored afterwards, so no other experiment sees them."""
    fo = json.loads(os.environ.get("HEDICES_E13_FAILOVER") or "null") or (CONFIG.get("e13") or {}).get("failover") or {}
    saved = ({k: dict(v) for k, v in (("ray", RAY_ENV), ("dask", DASK_ENV), ("driver", DRIVER_ENV))},
             {k: os.environ.get(k) for k in ("HEDICES_K8S_FAILOVER", "HEDICES_KNATIVE_MIN_SCALE")})
    RAY_ENV.update({k: str(v) for k, v in (fo.get("ray_env") or {}).items()})
    DASK_ENV.update({k: str(v) for k, v in (fo.get("dask_env") or {}).items()})
    DRIVER_ENV.update({k: str(v) for k, v in (fo.get("driver_env") or {}).items()})
    if fo.get("k8s_failover"):
        os.environ["HEDICES_K8S_FAILOVER"] = "1"
    if fo.get("knative_min_scale"):
        os.environ["HEDICES_KNATIVE_MIN_SCALE"] = str(fo["knative_min_scale"])
    try:
        _e13_trials(root, a)
    finally:
        for name, env in (("ray", RAY_ENV), ("dask", DASK_ENV), ("driver", DRIVER_ENV)):
            env.clear(); env.update(saved[0][name])
        for k, v in saved[1].items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _e13_trials(root: Path, a) -> None:
    rng = random.Random(13)
    # failures per system: configs "e13" "trials" (HEDICES_E13_TRIALS overrides it), else --trials
    n = int(os.environ.get("HEDICES_E13_TRIALS") or (CONFIG.get("e13") or {}).get("trials") or a.trials)
    for system in a.systems:
        for k in range(n):
            out = root / "e13" / f"{system}_{a.e13_workload}_trial{k:02d}.json"
            if done(out):
                continue
            start(system, [a.e13_workload])
            status = root / "e13" / f".status_{system}_{k}.json"
            status.parent.mkdir(parents=True, exist_ok=True)
            os.chown(status.parent, *USER_IDS)
            status.unlink(missing_ok=True)
            unit = f"e13-{system}-{k}"
            drive(out, unit, ["--system", system, "--phase", "failure", "--workload", a.e13_workload,
                              "--warmup", str(a.e13_warmup), "--duration", str(a.e13_duration),
                              "--concurrency", str(a.concurrency), "--status", str(status)], wait=False)
            # the driver publishes its status once warm-up and settle (up to run.py --settle-max, 300 s by
            # default, for the Kubernetes systems) are over
            deadline = time.monotonic() + a.e13_warmup + 300 + 300
            while not status.exists() and time.monotonic() < deadline:
                time.sleep(0.2)
            time.sleep(rng.uniform(4, 10))
            busy = []
            for _ in range(50):
                busy = json.loads(status.read_text())["busy"]
                busy = [IP2NODE.get(b, b) for b in busy if IP2NODE.get(b, b) in IP2NODE.values()]
                if busy:
                    break
                time.sleep(0.1)
            # a worker whose tier keeps a replacement (the scaled layout has a single cloud worker,
            # the paper's testbed two)
            spare = {w["name"] for w in WORKERS if sum(x["tier"] == w["tier"] for x in WORKERS) > 1}
            busy = [b for b in busy if b in spare]
            victim = rng.choice(busy) if busy else rng.choice(sorted(spare))
            t_inject = time.time()
            tb.link(tb.node(victim), "down")
            log(f"  e13 {system} trial {k}: victim={victim} busy={busy}")
            while subprocess.run(f"systemctl is-active --quiet hed-{unit}.service", shell=True).returncode == 0:
                time.sleep(0.5)
            tb.link(tb.node(victim), "up")
            if out.exists():
                res = json.loads(out.read_text())
                res["injection"] = {"t": t_inject, "victim": victim, "busy": busy}
                out.write_text(json.dumps(res))
            else:
                log(f"  e13 {system} trial {k}: driver produced no result")


PLANS = {"t3": plan_t3, "t3s": plan_t3s,"smoke": plan_smoke, "e6": plan_e6, "e7": lambda root, a: plan_e6(root, a, "e7"), "e6cold": plan_e6cold, "e6mix": plan_e6mix, "e8": plan_e8, "e9": plan_e9, "e10": plan_e10,
         "e12": plan_e12, "e13": plan_e13, "eB": plan_eB}


def main():
    # line-buffered: what the testbed prints (testbed up, branches started) reaches the log when it happens, in order
    # with the driver output that goes straight to the same file
    sys.stdout.reconfigure(line_buffering=True)
    p = argparse.ArgumentParser()
    p.add_argument("--plan", default="smoke")
    p.add_argument("--out", required=True)
    p.add_argument("--warmup", type=float, default=60.0)
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--bursts", default="1,2,3,4,5,6,7,8")
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--trials", type=int, default=15)
    p.add_argument("--e8-instances", type=int, default=120)  # ~600 dispatches
    p.add_argument("--e8-cache-mb", type=float, default=64.0)
    p.add_argument("--cache-entries", default="",
                   help='Hedices function-cache capacity in entries per device class, e.g. "agx:3,nx:2,pi5:1"')
    p.add_argument("--cache-kb", default="",
                   help='Hedices worker cache budgets per device class, e.g. "agx:512,nx:256,pi5:128"')
    p.add_argument("--eb-session-n", type=int, default=200, help="eB: instances of a deployment session")
    p.add_argument("--eb-phases", default="steady,session,burst,failure",
                   help="eB: phases to run, of steady (Fig 10/11), session (Table 4), burst (Fig 12), failure (Fig 13)")
    p.add_argument("--eb-burst-systems", default="hedices,ray,dask",
                   help="eB: the systems also measured under bursts (Fig. 12) and failures (Fig. 13)")
    p.add_argument("--e13-workload", default="W4")
    p.add_argument("--e13-warmup", type=float, default=20.0)
    p.add_argument("--e13-duration", type=float, default=30.0)
    p.add_argument("--modes", default="full,alpha0,static,rr,nocache")
    p.add_argument("--e9-slots", default="borrow", choices=["borrow", "partition", "none"],
                   help="Branch slot ledger for the @NoRely runs (none: per-stage fixed slots)")
    p.add_argument("--cold-n", type=int, default=20, help="e6cold: instances measured from empty caches")
    p.add_argument("--e12-ray-cpu-div", type=int, default=1,
                   help="Ray --num-cpus divisor for mixed arrivals (2 halves the actor pools' memory)")
    p.add_argument("--e9-pin", action="store_true",
                   help="e9: the @NoRely-off run uses the placement of the paired on run")
    p.add_argument("--switch-ms", type=float, default=None,
                   help="Hedices master: interpreter switch interval in ms (default: Python's 5)")
    p.add_argument("--pack-on-dispatch", default=None, choices=["0", "1"],
                   help="Stem packager: build every miss's package at dispatch (1, the default; "
                        "the deployment path) or send the package built at registration (0; "
                        "packages are content-addressed and immutable)")
    p.add_argument("--nocache-state", default="cold", choices=["cold", "package"],
                   help="nocache ablation: per-dispatch rebuild of package state (cold) or of the "
                        "package alone, node-resident model assets kept (package)")
    p.add_argument("--e9-probe-from", default="on", choices=["on", "off", "alt"],
                   help="e9 --e9-pin: settle the shared placement with @NoRely on, off, or "
                        "alternately per trial (with --e9-probe-per-trial)")
    p.add_argument("--e9-probe-per-trial", action="store_true",
                   help="e9 --e9-pin: settle a fresh placement for every trial")
    p.add_argument("--e9-pin-mode", default="replay", choices=["replay", "mode"],
                   help="e9 --e9-pin: replay the probe's placement instance by instance, or every "
                        "stage on the worker that ran most of its probe dispatches")
    p.add_argument("--e9-efficiency", type=float, default=0.6,
                   help="ledger parallel-efficiency threshold for native threads in the e9 runs")
    p.add_argument("--norely-mode", default="thread", choices=["thread", "auto", "process", "native"],
                   help="Branch @NoRely execution (auto: Alg. 2 thread/process choice)")
    p.add_argument("--slot-policy", default=None, choices=["borrow", "partition"],
                   help="Branch worker-level slot ledger (Sec. 4.5) in every Hedices run")
    p.add_argument("--slot-efficiency", type=float, default=0.6,
                   help="ledger parallel-efficiency threshold for native threads (--slot-policy)")
    p.add_argument("--branch-flags", default="", help="extra flags for every Branch (e.g. --no-accel)")
    p.add_argument("--workloads", default="W1,W2,W3,W4")
    p.add_argument("--systems", default="hedices,ray,dask,kubeedge,knative")
    p.add_argument("--suite", default="apps", choices=["apps", "legacy"],
                   help="apps: the benchmark-suite applications of each framework; legacy: harness workloads")
    p.add_argument("--app-models", default="local", choices=["local", "package"],
                   help="W1 LSTM / W4 fusion CNN weights: node-local asset or shipped in the function package")
    p.add_argument("--cold-cache", action="store_true",
                   help="hedices steady runs: empty function caches after the warm touch")
    a = p.parse_args()
    # no run inherits a CPU frequency cap left by an earlier one (testbed/cpufreq.sh; a cap is set only around a
    # workload whose configs carry "cpu_max_khz", and restored after it)
    subprocess.run(["bash", str(CPUFREQ), "check"], check=False)
    atexit.register(lambda: subprocess.run(["bash", str(CPUFREQ), "check"], check=False))
    a.modes, a.workloads, a.systems = (x.split(",") for x in (a.modes, a.workloads, a.systems))
    SUITE["suite"] = a.suite
    NORELY["mode"] = a.norely_mode
    NORELY["nocache_state"] = a.nocache_state
    NORELY["slots"], NORELY["efficiency"] = a.slot_policy, a.slot_efficiency
    NORELY["extra"] = a.branch_flags.split()
    if a.pack_on_dispatch is not None:
        os.environ["HEDICES_PACK_ON_DISPATCH"] = a.pack_on_dispatch
    if a.switch_ms is not None:
        os.environ["HEDICES_SWITCH_MS"] = str(a.switch_ms)
    for k in ("HEDICES_INTENT", "HEDICES_PACK_ON_DISPATCH", "HEDICES_FETCH_TIMEOUT",
              "HEDICES_SWITCH_MS", "HEDICES_PROFILE", "HEDICES_CHAIN", "HEDICES_CHAIN_DEPTH",
              "HEDICES_STATIC", "HEDICES_STATIC_GAMMA", "HEDICES_STATIC_LIGHT_MS", "HEDICES_ALIGN_BAND", "HEDICES_EDGE_SPILL", "HEDICES_NOCACHE_PLACE", "HEDICES_SPLIT_DEPS", "HEDICES_CONTENTION", "HEDICES_CONTENTION_MIN_MS", "HEDICES_NEAR_HEAVY_MS", "HEDICES_CONTENTION_INFLICT", "HEDICES_ALIGN_FLOOR", "HEDICES_EVICT_HOT_S", "HEDICES_HOT_CAP", "HEDICES_HOT_WIN_S", "HEDICES_ALIGN_NEAR", "HEDICES_BEST_FIT", "HEDICES_LOOKAHEAD", "HEDICES_FORM", "HEDICES_FORM_TERMS", "HEDICES_GUEST", "HEDICES_DEPLOY_AMORT", "HEDICES_EVICT_FLOOR", "HEDICES_PACKERS", "HEDICES_PAY_BW", "HEDICES_COST_CAL", "HEDICES_PKG_FORMAT", "HEDICES_EXEC_POOL", "HEDICES_RHO_QUEUE", "HEDICES_ALIGN_PENDING", "HEDICES_AMORT_ALPHA", "HEDICES_NEAR_SIDE", "HEDICES_QDRAIN",
              "HEDICES_BORROW_HOLD", "HEDICES_CALL_PLANE",
              "HEDICES_EXEC_STALE_S", "HEDICES_CULT_POOL", "HEDICES_CULT_LOOK", "HEDICES_CULT_SAMPLES", "HEDICES_CULT_STALE_S", "HEDICES_AMORT_RAMP", "HEDICES_MIX_PHASE", "HEDICES_SIGMA_GAIN", "HEDICES_NEAR_HEAVY_RATIO", "HEDICES_CULT_REPROBE", "HEDICES_WARM_SEQ", "HEDICES_E6_WARM_SEQ", "HEDICES_CULT_INPUT", "HEDICES_CULT_INPUT_CV", "HEDICES_EXEC_PROFILE", "HEDICES_AMORT_MIN_RATIO", "HEDICES_ALIGN_BAND_RECUR", "HEDICES_LOOK_DEPTH", "HEDICES_LOOK_MAX_MS", "HEDICES_PROBE", "HEDICES_TIE_EPS", "HEDICES_WARM_ROUNDS", "HEDICES_PROBE_MT",
              "HEDICES_PROBE_DEBUG", "HEDICES_CROSS_PENALTY_MS", "HEDICES_DATAPATH", "HEDICES_EFF", "HEDICES_SCHED_PROF", "HEDICES_LOOK_MIN_MS", "HEDICES_CHAIN_POLICY", "HEDICES_CHAIN_EVICT", "HEDICES_TIER_RESERVE", "HEDICES_STAGETRACE", "HEDICES_PKG_MINIFY", "HEDICES_PKG_SLICE", "HEDICES_RAY_TIMELINE", "HEDICES_INPUT_IMU", "HEDICES_E6_SEGMENT_MIX", "HEDICES_CHAIN_SEG_MIN", "HEDICES_CHAIN_MARGIN_MS", "HEDICES_ALIGN_STICKY", "HEDICES_AMORT_ALL", "HEDICES_ALPHA0_BLIND", "HEDICES_CHAIN_SPILL_FREE", "HEDICES_PLACE_TEMP_MS", "HEDICES_PLACE_SEED", "HEDICES_ALPHA0_TIER_FREE", "HEDICES_KEEP_WARM_RECORDS", "HEDICES_ALPHA0_BLIND_AMORT", "HEDICES_DRIVER_GC", "HEDICES_STATIC_FOLLOW", "HEDICES_LOAD_STEPS", "HEDICES_LOAD_STEP_N", "HEDICES_STATIC_PROFILE_N", "HEDICES_STATIC_PHASE", "HEDICES_STATIC_HEAVY_MS", "HEDICES_E6_CPU_MAX_KHZ", "HEDICES_CHAIN_DEDUP", "HEDICES_NOCACHE_FOLLOW", "HEDICES_STATIC_LIGHT"):
        if k in os.environ:  # Hedices runtime switches reach the driver (the master runs there)
            DRIVER_ENV[k] = os.environ[k]
    if a.cache_kb:
        os.environ["HEDICES_CACHE_KB"] = a.cache_kb
    if a.cache_entries:
        os.environ["HEDICES_CACHE_ENTRIES"] = a.cache_entries
    DRIVER_ENV["HEDICES_APP_MODELS"] = a.app_models
    a.topology = tb.TOPO_FILE
    assert tb.is_local(MASTER), "run the orchestration on the master's host"
    tb.apply_quotas_cluster()
    root = Path(a.out)
    root.mkdir(parents=True, exist_ok=True)
    os.chown(root, *USER_IDS)
    (root / "orchestrate_args.json").write_text(json.dumps(vars(a), indent=1))
    try:
        for name in a.plan.split(","):
            log(f"== plan {name}")
            PLANS[name](root, a)
    finally:
        stop_all()
        for w in WORKERS:
            try:
                tb.link(w, "up")
            except Exception as e:  # noqa: BLE001
                log(f"link up {w['name']}: {e}")
        subprocess.run(f"chown -R {tb.USER}:{tb.USER} {root}", shell=True)
    log("all plans finished")


if __name__ == "__main__":
    main()
