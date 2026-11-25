"""Adapters that drive the benchmark-suite applications (apps/<framework>/wN) unchanged.

Each adapter runs the set-up statements of the application's ``__main__`` block (Ray actor
replicas, Dask worker plugins, ...) against the testbed cluster, then calls the application's
own ``workflow`` once per workflow instance.  Only the cluster connection and the submission
loop come from the harness; stage code, placement declarations, model handling and inter-stage
data flow are the application's.

Harness-side substitutions (identical for every system):
  * dependency installation (Ray ``runtime_env["pip"]``, Dask ``PipInstall``) is skipped: the
    node environment already holds the pinned libraries.  Application code is still shipped by
    each system's own mechanism (Ray py_modules, Dask upload_file, images, function packages).
  * the Kubernetes drivers run on the master node outside a pod, so cluster DNS names of the
    stage services are resolved from the API server instead of CoreDNS.
"""
from __future__ import annotations

import ast
import importlib
import itertools
import inspect
import json
import os
import threading
import socket
import subprocess
import time
from pathlib import Path

from experiments.common import MASTER_IP, ROOT

RAY_ADDR = f"{MASTER_IP}:6379"
DASK_ADDR = f"tcp://{MASTER_IP}:8786"


def app_module(framework: str, workload: str):
    w = workload.lower()
    if framework in ("kubeedge", "knative"):
        return importlib.import_module(f"apps.{framework}.{w}.pipeline")
    return importlib.import_module(f"apps.{framework}.{w}")


def common_module(workload: str):
    return importlib.import_module(f"apps.common.{workload.lower()}")


def make_inputs(workload: str, pregen: int = 32):
    """Instance i gets the i-th input of the application's input generator, with a fresh req_id."""
    base = common_module(workload).inputs(pregen)

    def make(i):
        args = list(base[i % len(base)])
        args[0] = f"{workload.lower()}-{i}"
        return tuple(args), {}
    return make


def main_block(mod) -> list[ast.stmt]:
    tree = ast.parse(inspect.getsource(mod))
    for node in tree.body:
        if (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
                and getattr(node.test.left, "id", "") == "__name__"):
            return node.body
    raise ValueError(f"{mod.__name__} has no __main__ block")


def setup_and_pools(mod, skip) -> list[str]:
    """Execute the __main__ set-up statements (all statements before the submission comprehension,
    minus those ``skip`` rejects) in the module namespace; return the names the application passes
    to ``workflow`` ahead of the instance arguments."""
    body = main_block(mod)
    setup, pools = [], []
    for stmt in body:
        comp = getattr(stmt, "value", None)
        if isinstance(comp, ast.ListComp) and getattr(comp.elt.func, "id", "") == "workflow":
            pools = [a.id for a in comp.elt.args if isinstance(a, ast.Name)]
            break
        if not skip(stmt):
            setup.append(stmt)
    code = compile(ast.Module(body=setup, type_ignores=[]), mod.__file__, "exec")
    exec(code, mod.__dict__)
    return pools


def _calls(stmt, name: str) -> bool:
    src = ast.unparse(stmt)
    return src.startswith(name) or f"= {name}" in src


class AppHedices:
    name = "hedices"

    def __init__(self, a):
        from experiments.systems import HedicesSystem

        self.inner = HedicesSystem(a)

    def runner(self, mod, name):
        return self.inner.runner(mod, name)

    def warm(self, mod, name, inp):
        return self.inner.warm(mod, name, inp)

    def __getattr__(self, item):
        return getattr(self.inner, item)


class AppRay:
    name = "ray"

    def __init__(self, a):
        import ray

        self.ray = ray
        self.workers = a.workers
        self._pools = {}
        self.out = getattr(a, "out", "")

    def _connect(self, mod):
        if self.ray.is_initialized():
            return
        env = {k: v for k, v in mod.RUNTIME_ENV.items() if k != "pip"}
        self.ray.init(address=RAY_ADDR, runtime_env=env, logging_level="ERROR", log_to_driver=False)
        deadline = time.monotonic() + 300
        while sum(n["Alive"] for n in self.ray.nodes()) < self.workers + 1:
            if time.monotonic() > deadline:
                raise TimeoutError("ray workers did not join")
            time.sleep(0.5)
        self.nodes = {n["NodeID"]: n for n in self.ray.nodes() if n["Alive"]}

    def _placed_only(self, mod):
        """Wrap the application's replica-pool helper so the pool keeps only actors that were
        placed (an actor that cannot get its CPUs stays pending and would hold every call routed
        to it; a Kubernetes Service likewise routes only to ready pods)."""
        orig = mod.replicas
        ray = self.ray

        def replicas(*args, **kw):
            cyc = orig(*args, **kw)
            if hasattr(cyc, "actors"):  # the application's own dispatcher over its replicas
                pool = list(cyc.actors)
            else:  # a plain itertools.cycle
                first, pool = next(cyc), []
                pool.append(first)
                while (a := next(cyc)) is not first:
                    pool.append(a)
            refs = [a.__ray_ready__.remote() for a in pool]
            # wait for every replica the caller will actually keep: a pool trimmed to the actors
            # that happened to come up first serialises the calls routed to it (Ray W1 read 100 ms
            # instead of 27 ms when one of six replicas was admitted). Under mixed arrivals each
            # application keeps len(pool)//div, and the replicas beyond that never get their CPUs,
            # so waiting for all of them would idle for the full timeout in every warm-up.
            div = max(1, getattr(self, "pool_div", 1))
            need = max(1, len(pool) // div)
            done, _ = ray.wait(refs, num_returns=need,
                               timeout=float(os.environ.get("HEDICES_RAY_POOL_WAIT", "600")))
            done = set(done)
            placed = [a for a, r in zip(pool, refs) if r in done]
            print(f"[ray] {getattr(mod, '__name__', '?')}: {len(placed)}/{len(pool)} replicas placed"
                  f" (keep {need})", flush=True)
            # several applications sharing the cluster (mixed arrivals) each keep 1/k of their
            # placed replicas, so actor reservations leave CPUs for the applications' tasks;
            # replicas still waiting for a pinned node are dropped either way
            keep = placed[:need] if placed else pool
            for a in pool:
                if a not in keep:
                    ray.kill(a)
            return type(cyc)(keep) if hasattr(cyc, "actors") else itertools.cycle(keep)
        mod.__dict__["replicas"] = replicas

    def runner(self, mod, name):
        if name not in self._pools:
            self._connect(mod)
            if hasattr(mod, "replicas"):
                self._placed_only(mod)
            names = setup_and_pools(mod, skip=lambda s: _calls(s, "ray.init"))
            self._pools[name] = [mod.__dict__[n] for n in names]
        pools, ray = self._pools[name], self.ray
        return lambda args, kw, timeout=900: ray.get(mod.workflow(*pools, *args, **kw), timeout=timeout)

    def warm(self, mod, name, inp):
        return 0

    def busy(self):
        from experiments.systems import RaySystem

        return RaySystem.busy(self) if hasattr(self, "nodes") else set()

    def reset(self):
        pass

    def extra(self):
        # diagnostics (HEDICES_RAY_TIMELINE=1): the task timeline of the run -- node, worker and span of
        # every task and actor call -- next to the result file; Ray itself is not changed
        if os.environ.get("HEDICES_RAY_TIMELINE") == "1" and self.out and self.ray.is_initialized():
            try:
                self.ray.timeline(filename=self.out.rsplit(".json", 1)[0] + ".timeline.json")
            except Exception as exc:  # noqa: BLE001
                print(f"[ray] timeline: {exc}", flush=True)
        return {}

    def close(self):
        self.ray.shutdown()


class AppDask:
    name = "dask"

    def __init__(self, a):
        from distributed import Client

        self.c = Client(DASK_ADDR, timeout=60)
        try:  # a scaled-down topology has fewer workers than the paper's ten
            self.c.wait_for_workers(a.workers, timeout=60)
        except Exception as exc:
            print(f"[dask] wait_for_workers({a.workers}): {type(exc).__name__}", flush=True)
        info = self.c.scheduler_info()["workers"]
        print(f"[dask] {len(info)} workers: " + ", ".join(
            f"{w.get('name', '?')}(nthreads={w.get('nthreads')},"
            f"edge={w.get('resources', {}).get('edge', 0)})" for w in info.values()), flush=True)
        self._ready = set()

    def runner(self, mod, name):
        if name not in self._ready:
            mod.__dict__["client"] = self.c
            # deploy() ships apps.zip with client.upload_file; its PipInstall plugin is skipped
            mod.__dict__["PipInstall"] = lambda packages: _NoopPlugin()
            setup_and_pools(mod, skip=lambda s: _calls(s, "client = Client"))
            self._ready.add(name)
        c = self.c
        return lambda args, kw, timeout=900: mod.workflow(c, *args, **kw).result(timeout=timeout)

    def warm(self, mod, name, inp):
        return 0

    def busy(self):
        try:
            proc = self.c.processing()
        except Exception:
            return set()
        return {addr.split("//")[1].rsplit(":", 1)[0] for addr, keys in proc.items() if keys}

    def reset(self):
        pass

    def extra(self):
        return {}

    def close(self):
        self.c.close()


class _NoopPlugin:
    """Stands in for PipInstall (dependencies are preinstalled on the testbed nodes)."""

    def __new__(cls):
        from distributed import WorkerPlugin

        class Noop(WorkerPlugin):
            name = "pip-preinstalled"
        return Noop()


def _k8s_assets() -> Path:
    """The Kubernetes assets: HEDICES_K8S_ASSETS, else inside the checkout, else beside it."""
    if os.environ.get("HEDICES_K8S_ASSETS"):
        return Path(os.environ["HEDICES_K8S_ASSETS"])
    return next((d for d in (ROOT / "k8s-assets", ROOT.parent / "k8s-assets") if d.exists()),
                ROOT.parent / "k8s-assets")


def _cluster_services() -> dict[str, str]:
    """name.namespace.svc.cluster.local -> ClusterIP, following ExternalName services."""
    kube = _k8s_assets()
    out = subprocess.run(f"{kube}/k3s kubectl --kubeconfig {kube}/state/kubeconfig.yaml "
                         f"--server https://{MASTER_IP}:6443 get svc -A -o json",
                         shell=True, capture_output=True, text=True, check=True).stdout
    items = json.loads(out)["items"]
    fqdn = lambda s: f"{s['metadata']['name']}.{s['metadata']['namespace']}.svc.cluster.local"
    ips = {fqdn(s): s["spec"].get("clusterIP") for s in items if s["spec"].get("clusterIP") not in (None, "None")}
    for s in items:
        if s["spec"].get("type") == "ExternalName":
            ips[fqdn(s)] = ips.get(s["spec"]["externalName"])
    return {k: v for k, v in ips.items() if v}


class AppK8s:
    """KubeEdge / Knative: the application's pipeline driver against the stage services."""

    def __init__(self, a, name):
        self.name = name
        self.services = _cluster_services()
        real = socket.getaddrinfo

        def resolve(host, *rest, **kw):
            return real(self.services.get(host, host), *rest, **kw)
        socket.getaddrinfo = resolve

    def runner(self, mod, name):
        if os.environ.get("HEDICES_DATATRACE") and hasattr(mod, "session"):
            self._count_bodies(mod)  # Table 4: application data the driver moves per call
        if os.environ.get("HEDICES_STAGETRACE") and hasattr(mod, "session"):
            self._trace_stages(mod)  # diagnostics: per-call round trip, pod and execution time
        return lambda args, kw, timeout=900: mod.workflow(*args, **kw)

    def _trace_stages(self, mod) -> None:
        """Record, for every stage call of the driver, [stage, round trip ms, pod, execution ms]
        (pod and execution time from the X-Pod / X-Exec-Ms headers the stage server sets)."""
        self.stage_trace = []
        post, lock = mod.session.post, threading.Lock()

        def traced(url, data=None, **kw):
            t = time.perf_counter()
            r = post(url, data=data, **kw)
            rt = 1e3 * (time.perf_counter() - t)
            row = [url.rsplit("/", 1)[-1], round(rt, 3), r.headers.get("X-Pod", ""),
                   float(r.headers.get("X-Exec-Ms", "nan"))]
            with lock:
                self.stage_trace.append(row)
            return r
        mod.session.post = traced

    def _count_bodies(self, mod) -> None:
        self.app_bytes = [0, 0]  # request bodies, response bodies
        post, lock = mod.session.post, threading.Lock()

        def counted(url, data=None, **kw):
            with lock:  # sent even when the call then fails and the driver retries it
                self.app_bytes[0] += len(data or b"")
            r = post(url, data=data, **kw)
            with lock:
                self.app_bytes[1] += len(r.content)
            return r
        mod.session.post = counted

    WARM_READY_S = 180.0
    WARM_READY_RUNS = 10  # consecutive instances: the Service spreads them over every stage replica

    def warm(self, mod, name, inp):
        """Wait until every stage replica answers before any measured call: a Deployment can be
        Ready while the Service routing (EdgeMesh / Kourier) still refuses connections to one of its
        replicas for a few seconds (KubeEdge W4's imu-features in three runs, 2026-11-14; one
        successful instance is not enough, it may have reached another replica).  Instances end to
        end until WARM_READY_RUNS in a row succeed; a connection error restarts the count after 2 s,
        for up to WARM_READY_S.  None of them is part of any measurement.  Returns the attempts."""
        args, kw = inp
        deadline = time.monotonic() + self.WARM_READY_S
        n = ok = 0
        while ok < self.WARM_READY_RUNS:
            n += 1
            try:
                mod.workflow(*args, **kw)
                ok += 1
            except Exception as exc:  # noqa: BLE001
                if "Connection" not in type(exc).__name__ or time.monotonic() > deadline:
                    raise
                ok = 0
                time.sleep(2.0)
        return n

    def pods(self) -> tuple:
        kube = _k8s_assets()
        out = subprocess.run(f"{kube}/k3s kubectl --kubeconfig {kube}/state/kubeconfig.yaml "
                             f"--server https://{MASTER_IP}:6443 get pods --no-headers",
                             shell=True, capture_output=True, text=True).stdout
        return tuple(sorted(" ".join(l.split()[:3]) for l in out.splitlines()))

    def settled(self) -> bool:
        """Steady state: the stage pod set (names, readiness, phase) did not change since the
        previous check and no pod is starting or still unscheduled (Pending: e.g. waiting for the
        previous workload's pods to release their CPU requests) -- the autoscaler / scheduler has converged."""
        now = self.pods()
        prev, self._last_pods = getattr(self, "_last_pods", None), now
        starting = [p for p in now if p.split()[2] in ("Pending", "ContainerCreating", "Init", "Terminating")
                    or (p.split()[2] == "Running" and p.split()[1].split("/")[0] != p.split()[1].split("/")[1])]
        return now == prev and not starting

    def busy(self):
        """Nodes running the application's stage pods (every Ready replica takes requests under
        the closed-loop load; per-request in-flight state is not exposed by the cluster)."""
        kube = _k8s_assets()
        out = subprocess.run(f"{kube}/k3s kubectl --kubeconfig {kube}/state/kubeconfig.yaml "
                             f"--server https://{MASTER_IP}:6443 get pods -o wide --no-headers",
                             shell=True, capture_output=True, text=True).stdout
        return {f[6] for f in (l.split() for l in out.splitlines())
                if len(f) > 6 and f[2] == "Running" and f[1].split("/")[0] == f[1].split("/")[1]}

    def reset(self):
        if hasattr(self, "app_bytes"):
            self.app_bytes[:] = [0, 0]
        if hasattr(self, "stage_trace"):  # keep only the calls of the measured instances
            self.stage_trace.clear()

    def extra(self):
        out = {"app_bytes": list(self.app_bytes)} if hasattr(self, "app_bytes") else {}
        if hasattr(self, "stage_trace"):
            out["stage_trace"] = list(self.stage_trace)
        return out

    def close(self):
        pass


APP_SYSTEMS = {"hedices": AppHedices, "ray": AppRay, "dask": AppDask,
               "kubeedge": lambda a: AppK8s(a, "kubeedge"), "knative": lambda a: AppK8s(a, "knative")}
