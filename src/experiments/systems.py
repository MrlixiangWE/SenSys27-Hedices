"""System adapters: Hedices, Ray and Dask run the same stage functions and inputs.

Each adapter turns a workload module into ``run_one(args, kwargs) -> result``.
The stage DAG comes from the same Workflow trace for every system; stage
placement intent (cloud/edge) becomes a resource constraint in Ray and Dask.
User workload code is shipped by value in every system.  ``warm`` executes
every stage on every worker its intent admits, using each system's own
placement-pinning mechanism, so all systems start measurement warm.
"""
from __future__ import annotations

import collections
import itertools
import json
import os
import threading
import time
from pathlib import Path

from experiments.common import MASTER_IP
from hedices import Workflow

RAY_ADDR = f"{MASTER_IP}:6379"
DASK_ADDR = f"tcp://{MASTER_IP}:8786"


def pinned_placement(name: str):
    """HEDICES_PIN_FROM=<result.json>: the placement of that run for workflow ``name``, so paired
    measurements share it.  HEDICES_PIN_MODE=replay (default) returns one {node: worker} per
    measured instance of the reference run, in instance order, to be replayed cyclically;
    ``mode`` returns a single dict, every node on the worker that ran most of its dispatches."""
    path = os.environ.get("HEDICES_PIN_FROM")
    if not path:
        return None
    recs = [r for r in json.loads(Path(path).read_text())["records"]
            if r.get("workflow") == name and r["status"] == "ok" and r.get("tag") != "warm"]
    if os.environ.get("HEDICES_PIN_MODE", "replay") == "replay":
        by_inst: dict = {}
        for r in recs:
            by_inst.setdefault(r["instance"], {})[r["node"]] = r["worker"]
        return [by_inst[i] for i in sorted(by_inst)] or None
    by: dict = {}
    for r in recs:
        by.setdefault(r["node"], collections.Counter())[r["worker"]] += 1
    return {n: c.most_common(1)[0][0] for n, c in by.items()}


class HedicesSystem:
    name = "hedices"

    def __init__(self, a):
        from hedices.master import Master

        self.m = Master(listen=f"{MASTER_IP}:7700", mode=a.mode, hb_timeout=a.hb_timeout).start()
        self.m.wait_workers(a.workers, timeout=180)
        self.cold_cache = getattr(a, "cold_cache", False)
        self._wfs = {}

    def _wf(self, mod, name):
        if name not in self._wfs:
            self._wfs[name] = Workflow(mod.workflow, name=name)
            self.m.register(self._wfs[name])
        return self._wfs[name]

    def runner(self, mod, name):
        wf = self._wf(mod, name)
        pin = pinned_placement(name)
        if isinstance(pin, list):  # replay the reference run's placements instance by instance
            nxt, lock = itertools.count(), threading.Lock()

            def run(args, kw, timeout=900):
                with lock:
                    p = pin[next(nxt) % len(pin)]
                return self.m.run(wf, *args, timeout=timeout, pin=p, **kw)
            return run
        if pin:
            return lambda args, kw, timeout=900: self.m.run(wf, *args, timeout=timeout, pin=pin, **kw)
        return lambda args, kw, timeout=900: self.m.run(wf, *args, timeout=timeout, **kw)

    def warm(self, mod, name, inp) -> int:
        args, kw = inp
        runs = self.m.warm(self._wf(mod, name), args, kw)
        if self.cold_cache:
            # processes, imports and node-local assets stay warm; function caches are
            # rebuilt by the warm-up phase under the scheduling mode being measured
            self.m.flush_caches()
        return runs

    def packages(self, mod, name):
        return {k: p.size for k, p in self.m.packages(self._wf(mod, name)).items()}

    def busy(self) -> set[str]:
        return {w["id"] for w in self.m.worker_info() if w["inflight"] > 0 and w["alive"]}

    def reset(self):
        self.m.reset_metrics()

    def extra(self) -> dict:
        return {**self.m.metrics(), "workers": self.m.worker_info()}

    def close(self):
        self.m.stop()


class RaySystem:
    name = "ray"

    def __init__(self, a):
        import ray

        self.ray = ray
        ray.init(address=RAY_ADDR, logging_level="ERROR", log_to_driver=False)
        deadline = time.monotonic() + 300
        while sum(n["Alive"] for n in ray.nodes()) < a.workers + 1:
            if time.monotonic() > deadline:
                raise TimeoutError("ray workers did not join")
            time.sleep(0.5)
        self.nodes = {n["NodeID"]: n for n in ray.nodes() if n["Alive"]}
        self._remote = {}

    def _remotes(self, mod, name):
        if name not in self._remote:
            import ray
            import ray.cloudpickle as rcp

            rcp.register_pickle_by_value(mod)
            wf = Workflow(mod.workflow, name=name)
            remote = {}
            for spec in wf.stages():
                res = {spec.target: 1} if spec.target in ("cloud", "edge") else {}
                remote[spec] = ray.remote(num_cpus=1, resources=res, max_retries=5)(spec.fn)
            self._remote[name] = (wf, remote)
        return self._remote[name]

    def runner(self, mod, name, pin_node: str | None = None):
        import ray
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        wf, remote = self._remotes(mod, name)
        res_of = lambda nid: self.nodes[nid]["Resources"]

        def run(args, kw, timeout=900):
            params = wf.bind(args, kw)
            refs: dict = {}
            for node in wf.nodes:
                a_ = Workflow.resolve(node.args, params, refs)
                k_ = Workflow.resolve(node.kwargs, params, refs)
                fn = remote[node.spec]
                if pin_node and (node.spec.target == "any" or node.spec.target in res_of(pin_node)):
                    fn = fn.options(scheduling_strategy=NodeAffinitySchedulingStrategy(pin_node, soft=False))
                refs[node.id] = fn.remote(*a_, **k_)
            return ray.get(Workflow.resolve(wf.output, params, refs), timeout=timeout)

        return run

    def warm(self, mod, name, inp) -> int:
        args, kw = inp
        n = 0
        for nid, node in self.nodes.items():
            if node["Resources"].get("CPU"):
                self.runner(mod, name, pin_node=nid)(args, kw)
                n += 1
        return n

    def busy(self) -> set[str]:
        """Nodes with CPU in use, from the GCS resource view (no dashboard needed)."""
        try:
            from ray._private import state as rstate

            avail = rstate.available_resources_per_node()
            total = rstate.total_resources_per_node()
        except Exception:
            return set()
        return {self.nodes[nid]["NodeManagerAddress"] for nid, res in total.items()
                if nid in self.nodes and avail.get(nid, {}).get("CPU", 0.0) < res.get("CPU", 0.0)}

    def reset(self):
        pass

    def extra(self) -> dict:
        return {}

    def close(self):
        self.ray.shutdown()


class DaskSystem:
    name = "dask"

    def __init__(self, a):
        from distributed import Client

        self.c = Client(DASK_ADDR, timeout=60)
        self.c.wait_for_workers(a.workers, timeout=180)
        self._wfs = {}

    def _wf(self, mod, name):
        if name not in self._wfs:
            import cloudpickle

            cloudpickle.register_pickle_by_value(mod)
            self._wfs[name] = Workflow(mod.workflow, name=name)
        return self._wfs[name]

    def runner(self, mod, name, pin_worker: str | None = None):
        wf = self._wf(mod, name)
        c = self.c
        info = c.scheduler_info()["workers"]

        def run(args, kw, timeout=900):
            params = wf.bind(args, kw)
            futs: dict = {}
            for node in wf.nodes:
                a_ = Workflow.resolve(node.args, params, futs)
                k_ = Workflow.resolve(node.kwargs, params, futs)
                t = node.spec.target
                res = {t: 1} if t in ("cloud", "edge") else None
                opts = {}
                if pin_worker and (t == "any" or t in info[pin_worker].get("resources", {})):
                    opts = {"workers": [pin_worker], "allow_other_workers": False}
                futs[node.id] = c.submit(node.spec.fn, *a_, resources=res, pure=False, retries=5,
                                         **opts, **k_)
            return Workflow.resolve(wf.output, params, futs).result(timeout=timeout)

        return run

    def warm(self, mod, name, inp) -> int:
        args, kw = inp
        workers = list(self.c.scheduler_info()["workers"])
        for addr in workers:
            self.runner(mod, name, pin_worker=addr)(args, kw)
        return len(workers)

    def busy(self) -> set[str]:
        try:
            proc = self.c.processing()
        except Exception:
            return set()
        return {addr.split("//")[1].rsplit(":", 1)[0] for addr, keys in proc.items() if keys}

    def reset(self):
        pass

    def extra(self) -> dict:
        return {}

    def close(self):
        self.c.close()


SYSTEMS = {"hedices": HedicesSystem, "ray": RaySystem, "dask": DaskSystem}
