#!/usr/bin/env python3
"""KubeEdge and Knative baselines on the emulated testbed.  Must run as root.

Every node of the topology becomes a privileged container placed in the node's systemd slice
(same CPU quota, 1 ms CFS period and memory cap the Hedices/Ray/Dask processes get), and the
container's network namespace is attached as the node's namespace, so node IPs, the 100 Mbps
cross-tier shaping, failure injection (host veth down) and the driver location are exactly the
ones the other systems use.

  knative : k3s server on the master node, k3s agents on all workers, Knative Serving + Kourier.
  kubeedge: k3s server on the master node, k3s agents on cloud workers, edgecore on edge
            workers, CloudCore + EdgeMesh (the KubeEdge service mesh) for cloud-edge services.

Images are served by a registry on the host (10.77.0.1:5000) acting as docker.io/gcr.io mirror;
once a node has pulled an image it keeps it (node-local image caching, IfNotPresent).

Multi-host topologies: every node container runs on its node's host (node subcommands below run
on that host over SSH), the registry and the cluster control run on the master's host, and the
node-container images reach the other hosts over HTTP from the master's host.

  k8s.py up knative|kubeedge      start node containers, wire the testbed, install the system
  k8s.py down                     remove node containers and their state
  k8s.py images [W1 ...]          build the stage images of the workloads and push them
  k8s.py deploy W [W ...]         apply the testbed manifests of the workloads, wait until ready
  k8s.py undeploy W [W ...]
  k8s.py kubectl ARGS...
  host-local (run on a node's host by tb.remote): write-config | node-up NODE ROLE | node-pid NODE | node-rm NODE |
                                     attach | put-edge-config NODE YAML | load-image IMAGE URL
"""
from __future__ import annotations

import argparse
import json
import os

import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from testbed import tb  # noqa: E402
from tools import gen_k8s  # noqa: E402

ASSETS = Path(os.environ.get("HEDICES_K8S_ASSETS",
                             tb.ROOT / "k8s-assets" if tb.HOSTS else Path(tb.HOME) / "k8s-assets"))
STATE = ASSETS / "state"
K3S_IMAGE = "rancher/k3s:v1.32.13-k3s1"
EDGE_IMAGE = "hedices-testbed/edgecore:v1.23.1"
HOST_IP = (tb.HOSTS[tb.TOPO["master"]["host"]]["bridge_ip"] if tb.HOSTS else tb.TOPO["bridge_ip"]).split("/")[0]
IMAGE_PORT = 8099  # node-container images for the other hosts (multi-host topologies)
REGISTRY = f"{HOST_IP}:5000"
TOKEN = "hedices-testbed"
KUBECONFIG = STATE / "kubeconfig.yaml"
# CPU units: k8s sees the 4 host cores inside every node container; nodes reserve all but
# logical_cpus/3 of them and stage requests are scaled by 1/3, so pods pack exactly as the
# declared CPUs pack on the paper's devices (AGX 12, NX 8, Pi5 4 cores).
CPU_SCALE = 3
VISIBLE_CPUS = os.cpu_count() or 4
# library versions of the stage images = the venv the other systems run
PINS = {"numpy": "2.5.3", "pillow": "12.3.0", "flask": "3.1.3", "werkzeug": "3.1.9",
        "onnxruntime": "1.31.0", "torch": "2.14.1+cpu", "torchvision": "0.29.1+cpu"}


def sh(cmd: str, check: bool = True, quiet: bool = False) -> str:
    r = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if check and r.returncode:
        raise RuntimeError(f"{cmd}\n{r.stdout}\n{r.stderr}")
    return r.stdout if quiet or not check else r.stdout


def kubectl(args: str, check: bool = True) -> str:
    return sh(f"{ASSETS}/k3s kubectl --kubeconfig {KUBECONFIG} --server https://{tb.TOPO['master']['ip']}:6443 "
              f"{args}", check=check)


def cname(n: dict) -> str:
    return f"hedk-{n['name']}"


def reserved_cpu(n: dict) -> str:
    alloc = tb.logical_cpus(n) / CPU_SCALE
    # the cores the node container sees: its host's, when the topology states them (a 4-core
    # Raspberry Pi host cannot reserve the master host's 8)
    seen = (tb.HOSTS.get(n.get("host"), {}) if tb.HOSTS else {}).get("cpus", VISIBLE_CPUS)
    return f"{max(seen - alloc, 0):.3f}"


# ---------------------------------------------------------------- node containers

def write_config() -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    (STATE / "registries.yaml").write_text(
        "mirrors:\n" + "".join(f'  {r}:\n    endpoint: ["http://{REGISTRY}"]\n' for r in ("docker.io", "gcr.io")))


def node_call(n: dict, sub: str, *args: str) -> str:
    """Run a node subcommand on the node's host (stdout)."""
    if tb.is_local(n):
        return LOCAL_OPS[sub](n, *args)
    return tb.on_host(n["host"], ["k8s", sub, n["name"], *args])["out"]


def running_pid(n: dict) -> int:
    if not tb.is_local(n):
        return int(node_call(n, "node-pid").strip() or 0)
    out = sh(f"docker inspect -f '{{{{.State.Running}}}} {{{{.State.Pid}}}}' {cname(n)}", check=False).split()
    return int(out[1]) if out and out[0] == "true" else 0


def node_labels(n: dict) -> dict:
    if n is tb.TOPO["master"] or "class" not in n:
        return {"hedices.io/role": "master"}
    return {"hedices.io/tier": n["tier"], "hedices.io/class": n["class"]}


def failover() -> bool:
    """Fig 13's fast failure detection: nodes report every second and are declared lost after 4 s (the default
    40 s), as Hedices' Branch heartbeat.  experiments.orchestrate sets it for the failure trials only, on a cluster
    built for them (the other experiments run on a cluster built without it)."""
    return os.environ.get("HEDICES_K8S_FAILOVER") == "1"


def node_up(n: dict, role: str) -> str:
    role, _, profile = role.partition(":")  # "agent:failover": built for the failure trials (failover())
    ff = profile == "failover"
    pid = edge_node(n) if role == "edge" else k3s_node(n, role, node_labels(n), ff)
    return str(pid)


def node_rm(n: dict) -> str:
    sh(f"docker rm -f {cname(n)}", check=False)
    sh(f"docker volume rm {cname(n)} {cname(n)}-id", check=False)
    (STATE / f"edge-{n['name']}" / "edgecore.yaml").unlink(missing_ok=True)
    return ""


def attach_local() -> None:
    """Wire this host's node containers into the testbed (their namespaces become the nodes')."""
    pids = {n["name"]: pid for n in tb.local_nodes() if (pid := running_pid(n))}
    tb.up(attach=pids)


def put_edge_config(n: dict, text: str) -> str:
    cfg = STATE / f"edge-{n['name']}"
    cfg.mkdir(parents=True, exist_ok=True)
    (cfg / "edgecore.yaml").write_text(text)
    return ""


def load_image(image: str, url: str, want: str = "") -> None:
    """Load IMAGE from URL unless the host already has it (that very build, when WANT gives its id)."""
    have = sh(f"docker image inspect -f '{{{{.Id}}}}' {image}", check=False).strip()
    if not have or (want and have != want):
        sh(f"curl -fsS {url} | docker load")


LOCAL_OPS = {"node-pid": lambda n: str(running_pid(n)), "node-up": node_up, "node-rm": node_rm,
             "put-edge-config": put_edge_config}


def image_server():
    """Serve saved node-container images to the other hosts (LAN address of the master's host)."""
    import functools
    import http.server
    import threading

    d = ASSETS / "images"
    d.mkdir(parents=True, exist_ok=True)
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(d))
    srv = http.server.ThreadingHTTPServer((tb.HOSTS[tb.LOCAL]["lan"], IMAGE_PORT), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def distribute_images(flavor: str) -> None:
    """The node-container images on every host that runs node containers."""
    if not tb.HOSTS:
        return
    hosts = {}
    for w in tb.TOPO["workers"]:
        if not tb.is_local(w):
            edge = flavor == "kubeedge" and w["tier"] == "edge"
            hosts.setdefault(w["host"], set()).add(EDGE_IMAGE if edge else K3S_IMAGE)
    files = {}
    for img in set().union(*hosts.values()) if hosts else ():
        f = ASSETS / "images" / (img.replace("/", "_").replace(":", "_") + ".tar")
        f.parent.mkdir(parents=True, exist_ok=True)
        if not f.exists():
            sh(f"docker save -o {f} {img}")
        files[img] = (f.name, sh(f"docker image inspect -f '{{{{.Id}}}}' {img}").strip())
    srv = image_server()
    try:
        for h, imgs in hosts.items():
            for img in imgs:
                tb.on_host(h, ["k8s", "load-image", img,
                               f"http://{tb.HOSTS[tb.LOCAL]['lan']}:{IMAGE_PORT}/{files[img][0]}", files[img][1]])
    finally:
        srv.shutdown()


def run_node(n: dict, image: str, argv: list[str], volumes: list[str], env: dict | None = None) -> int:
    if pid := running_pid(n):  # re-running `up` resumes an installation
        return pid
    sh(f"docker rm -f {cname(n)}", check=False)
    vols = " ".join(f"-v {v}" for v in volumes) + "".join(f" -e {k}={v}" for k, v in (env or {}).items())
    # wait for the testbed to plug eth0 in before the node agent starts
    boot = 'while [ ! -e /sys/class/net/eth0 ]; do sleep 0.2; done; sleep 1; exec "$@"'
    sh(f"docker run -d --name {cname(n)} --hostname {n['name']} --privileged --cgroupns=private "
       f"--network none --cgroup-parent={tb.slice_name(n)} --tmpfs /run --tmpfs /var/run "
       f"--restart no {vols} --entrypoint /bin/sh {image} -c '{boot}' boot {' '.join(argv)}")
    return int(sh(f"docker inspect -f '{{{{.State.Pid}}}}' {cname(n)}").strip())


def k3s_node(n: dict, role: str, labels: dict, ff: bool = False) -> int:
    vols = [f"{cname(n)}:/var/lib/rancher/k3s", f"{cname(n)}-id:/etc/rancher/node",
            f"{STATE}/registries.yaml:/etc/rancher/k3s/registries.yaml:ro",
            f"{ASSETS}/k3s-airgap-images-arm64.tar.zst:/var/lib/rancher/k3s/agent/images/airgap.tar.zst:ro"]
    common = [f"--node-name={n['name']}", f"--node-ip={n['ip']}", "--flannel-iface=eth0",
              *[f"--node-label={k}={v}" for k, v in labels.items()]]
    if "class" in n:
        common.append(f"--kubelet-arg=system-reserved=cpu={reserved_cpu(n)}")
    if ff:  # Fig 13: the node's status every second
        common.append("--kubelet-arg=node-status-update-frequency=1s")
    if role == "server":
        vols.append(f"{STATE}:/output")
        argv = ["/bin/k3s", "server", *common, f"--advertise-address={n['ip']}", f"--tls-san={n['ip']}",
                f"--token={TOKEN}", "--disable=traefik,servicelb,metrics-server,local-storage",
                "--write-kubeconfig=/output/kubeconfig.yaml", "--write-kubeconfig-mode=644"]
        if ff:  # Fig 13: a node silent for 4 s is declared lost (default 40 s), checked every second
            argv += ["--kube-controller-manager-arg=node-monitor-period=1s",
                     "--kube-controller-manager-arg=node-monitor-grace-period=4s"]
    else:
        argv = ["/bin/k3s", "agent", *common, f"--server=https://{tb.TOPO['master']['ip']}:6443",
                f"--token={TOKEN}"]
    # legacy iptables: the Orin NX kernels have the x_tables matches but no nft_compat, which the
    # nf_tables backend needs for them (kube-proxy / network policy rules use comment, mark, ...)
    return run_node(n, K3S_IMAGE, argv, vols, env={"IPTABLES_MODE": "legacy"})


def wait(pred, what: str, timeout: float = 600, every: float = 3) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if pred():
                return
        except Exception:
            pass
        time.sleep(every)
    raise TimeoutError(what)


def nodes_ready(names: list[str]) -> bool:
    out = kubectl("get nodes --no-headers", check=False)
    ready = {l.split()[0] for l in out.splitlines() if l.split()[1].startswith("Ready")}
    return set(names) <= ready


def up(flavor: str) -> None:
    write_config()
    tb.on_all_hosts(["k8s", "write-config"])
    if flavor == "kubeedge":
        build_edge_image()
    ensure_registry()
    distribute_images(flavor)
    if not running_pid(tb.TOPO["master"]):
        tb.down()  # the plain namespaces of the process-based systems
        tb.on_all_hosts(["down"], check=False)
    m = tb.TOPO["master"]
    prof = ":failover" if failover() else ""
    node_call(m, "node-up", "server" + prof)
    for w in tb.TOPO["workers"]:
        node_call(w, "node-up", ("edge" if flavor == "kubeedge" and w["tier"] == "edge" else "agent") + prof)
    attach_local()
    tb.on_all_hosts(["k8s", "attach"])
    wait(lambda: KUBECONFIG.exists() and "Ready" in kubectl("get nodes", check=False), "k3s server", 900)
    k3s_workers = [w["name"] for w in tb.TOPO["workers"] if flavor == "knative" or w["tier"] == "cloud"]
    wait(lambda: nodes_ready([m["name"], *k3s_workers]), "k3s agents", 600)
    if flavor == "knative":
        install_knative()
    else:
        install_kubeedge()
    (STATE / "flavor").write_text(flavor)
    (STATE / "profile").write_text("failover" if failover() else "")
    print(f"{flavor} up")


def down() -> None:
    for n in tb.nodes():
        node_call(n, "node-rm")
    tb.down()
    tb.on_all_hosts(["down"], check=False)
    for f in ("kubeconfig.yaml", "flavor"):
        (STATE / f).unlink(missing_ok=True)
    for f in [*STATE.glob("*-w?.yaml"), *STATE.glob("*-w?.first.yaml")]:  # deployed workloads
        f.unlink()
    print("k8s down")


# ---------------------------------------------------------------- Knative

def install_knative() -> None:
    kubectl(f"apply -f {ASSETS}/serving-crds.yaml")
    wait(lambda: "condition met" in kubectl("wait --for=condition=Established crd/services.serving.knative.dev "
                                          "--timeout=5s", check=False), "knative crds", 300)
    kubectl(f"apply -f {ASSETS}/serving-core.yaml")
    kubectl(f"apply -f {ASSETS}/kourier.yaml")
    # configuration changes go through Knative's validating webhook: wait until it serves
    wait(lambda: "condition met" in kubectl("wait --for=condition=Available deploy/webhook -n knative-serving "
                                            "--timeout=10s", check=False), "knative webhook", 900)
    patch = lambda args: wait(lambda: kubectl(args, check=False) and True, f"kubectl {args[:40]}", 300)
    patch("patch configmap/config-network -n knative-serving --type merge "
            "-p '{\"data\":{\"ingress-class\":\"kourier.ingress.networking.knative.dev\"}}'")
    # one warm pod per service and no scale to zero (paper Sec. 5.1); node selectors and
    # tolerations of the stage services need these feature flags
    # the autoscaler otherwise keeps Knative's defaults (target-burst-capacity 200: the activator
    # stays on the request path while burst capacity is short); HEDICES_KNATIVE_TBC overrides it,
    # e.g. 0 to take the activator off the path once warm pods are always present
    auto = {"enable-scale-to-zero": "false"}
    if os.environ.get("HEDICES_KNATIVE_TBC"):
        auto["target-burst-capacity"] = os.environ["HEDICES_KNATIVE_TBC"]
    patch("patch configmap/config-autoscaler -n knative-serving --type merge "
            f"-p '{json.dumps({'data': auto})}'")
    patch("patch configmap/config-features -n knative-serving --type merge "
            "-p '{\"data\":{\"kubernetes.podspec-nodeselector\":\"enabled\","
            "\"kubernetes.podspec-tolerations\":\"enabled\","
            # node affinity of profiled stages (tools/gen_k8s.py AVOID): rejected by the webhook otherwise
            "\"kubernetes.podspec-affinity\":\"enabled\"}}'")
    # stage images come from the local mirror; the controller would otherwise resolve their tags
    # against docker.io (deployment-time only, no effect on the request path)
    patch("patch configmap/config-deployment -n knative-serving --type merge "
            "-p '{\"data\":{\"registries-skipping-tag-resolving\":\"kind.local,ko.local,dev.local,docker.io,index.docker.io\"}}'")
    # the ingress gateway beside its clients (the drivers run on the master node)
    patch("patch deploy/3scale-kourier-gateway -n kourier-system --type merge "
          "-p '{\"spec\":{\"template\":{\"spec\":{\"nodeSelector\":{\"hedices.io/role\":\"master\"}}}}}'")
    for ns in ("knative-serving", "kourier-system"):
        wait(lambda ns=ns: "condition met" in kubectl(f"wait --for=condition=Available deploy --all -n {ns} "
                                                      "--timeout=10s", check=False), f"{ns} ready", 900)


# ---------------------------------------------------------------- KubeEdge

EDGEMESH_RELAY = "master"
# Cloud-to-edge service path.  "routed": the testbed's nodes share one flat L2 network, so the
# master routes each edge node's pod subnet to that node and kube-proxy serves the stage services
# (a flat-network KubeEdge deployment); "edgemesh": EdgeMesh agents proxy them (for networks the
# cloud cannot route into).  EdgeMesh is installed either way.
KUBEEDGE_PATH = os.environ.get("HEDICES_KUBEEDGE_PATH", "routed")


def edge_pod_cidr(n: dict) -> str:
    return f"10.88.{[w['name'] for w in edge_workers()].index(n['name']) + 1}.0/24"


def route_edge_pods() -> None:
    """Routes from the master node to every edge node's pod subnet (routed path)."""
    m = tb.TOPO["master"]
    for w in edge_workers():
        sh(f"ip -n {tb.ns(m)} route replace {edge_pod_cidr(w)} via {w['ip']}")


def edge_workers() -> list[dict]:
    return [w for w in tb.TOPO["workers"] if w["tier"] == "edge"]


def edge_node(n: dict) -> int:
    """edgecore + containerd node; it waits for its edgecore.yaml (written once CloudCore is up)."""
    cfg = STATE / f"edge-{n['name']}"
    cfg.mkdir(parents=True, exist_ok=True)
    (cfg / "edgecore.yaml").unlink(missing_ok=True)
    vols = [f"{cname(n)}:/var/lib/containerd", f"{cfg}:/etc/kubeedge/config"]
    return run_node(n, EDGE_IMAGE, ["/entry.sh"], vols, env={"POD_CIDR": edge_pod_cidr(n)})


def build_edge_image() -> None:
    ctx = tb.ROOT / "testbed" / "edgenode"
    shutil.copy(ASSETS / "kubeedge-v1.23.1-linux-arm64" / "edge" / "edgecore", ctx / "edgecore")
    # containerd of the edge nodes pulls docker.io images through the testbed registry
    (ctx / "hosts.toml").write_text(f'server = "https://registry-1.docker.io"\n[host."http://{REGISTRY}"]\n'
                                    f'  capabilities = ["pull", "resolve"]\n')
    if sh(f"docker image inspect {EDGE_IMAGE}", check=False).strip() in ("", "[]"):
        proxy = os.environ.get("http_proxy", "")
        args = f"--build-arg http_proxy={proxy} --build-arg https_proxy={proxy}" if proxy else ""
        sh(f"docker build --network host {args} -t {EDGE_IMAGE} {ctx}")
    (ctx / "edgecore").unlink()


def edgecore_config(n: dict, token: str) -> str:
    import yaml

    m = tb.TOPO["master"]["ip"]
    cfg = yaml.safe_load(sh(f"{ASSETS}/kubeedge-v1.23.1-linux-arm64/edge/edgecore --defaultconfig"))
    mod = cfg["modules"]
    hub = mod["edgeHub"]
    hub["httpServer"], hub["token"] = f"https://{m}:10002", token
    hub["websocket"]["server"], hub["quic"]["server"] = f"{m}:10000", f"{m}:10001"
    ed = mod["edged"]
    ed["hostnameOverride"], ed["nodeIP"] = n["name"], n["ip"]
    kc = ed["tailoredKubeletConfig"]
    kc["clusterDNS"] = ["169.254.96.16"]  # EdgeMesh DNS
    kc["systemReserved"] = {"cpu": reserved_cpu(n)}
    kc["cpuCFSQuotaPeriod"] = "100ms"
    mod["metaManager"]["metaServer"]["enable"] = True  # list-watch endpoint EdgeMesh needs
    # no devices on these nodes: the MQTT event bus and device twin are not deployed
    mod["eventBus"]["enable"] = mod["deviceTwin"]["enable"] = False
    if failover():  # Fig 13: the edge node's status and its hub heartbeat every second
        kc["nodeStatusUpdateFrequency"] = "1s"
        hub["heartbeat"] = 1
    return yaml.safe_dump(cfg, sort_keys=False)


def install_kubeedge() -> None:
    import yaml

    m = tb.TOPO["master"]
    keadm = ASSETS / "keadm-v1.23.1-linux-arm64" / "keadm" / "keadm"
    text = sh(f"{keadm} manifest --advertise-address={m['ip']} --kubeedge-version=v1.23.1 "
              f"--kube-config={KUBECONFIG} -d")
    docs = [d for d in yaml.safe_load_all(text) if d]
    keep = []
    cluster_scoped = {"ClusterRole", "ClusterRoleBinding", "CustomResourceDefinition", "Namespace",
                      "MutatingWebhookConfiguration", "ValidatingWebhookConfiguration"}
    for d in docs:
        if d["kind"] not in cluster_scoped:
            d["metadata"].setdefault("namespace", "kubeedge")  # what `keadm init` (helm) sets
        if d["kind"] == "DaemonSet" and "mosquitto" in d["metadata"]["name"]:
            continue  # event-bus broker for device apps, not used
        if d["kind"] == "Deployment" and d["metadata"]["name"] == "cloudcore":
            d["spec"]["template"]["spec"]["nodeSelector"] = {"hedices.io/role": "master"}
        keep.append(d)
    path = STATE / "cloudcore.yaml"
    path.write_text(yaml.safe_dump_all(keep, sort_keys=False))
    kubectl("create namespace kubeedge", check=False)
    kubectl(f"apply --server-side -f {path}")
    wait(lambda: "condition met" in kubectl("wait --for=condition=Available deploy/cloudcore -n kubeedge "
                                            "--timeout=10s", check=False), "cloudcore", 900)
    wait(lambda: kubectl("get secret tokensecret -n kubeedge -o jsonpath={.data.tokendata}", check=False).strip(),
         "cloudcore token", 300)
    import base64

    token = base64.b64decode(kubectl("get secret tokensecret -n kubeedge -o jsonpath={.data.tokendata}")).decode()
    for w in edge_workers():
        node_call(w, "put-edge-config", edgecore_config(w, token))
    wait(lambda: nodes_ready([w["name"] for w in edge_workers()]), "edge nodes", 900)
    for w in edge_workers():
        kubectl(f"label node {w['name']} --overwrite hedices.io/tier=edge hedices.io/class={w['class']}")
    if KUBEEDGE_PATH == "routed":
        # flat routing: each edge node's pod subnet has a route through that node, so stage
        # services resolve through the master's kube-proxy. The EdgeMesh overlay is not on the
        # data path, and installing it anyway is what made the six-node topology time out -- its
        # agents never reached ready and held the run for the full 900 s.
        route_edge_pods()
    else:
        # EdgeMesh (stock v1.17.0 agent manifests): relay on the master, fresh PSK
        kubectl('label services kubernetes --overwrite service.edgemesh.kubeedge.io/service-proxy-name=""')
        em = ASSETS / "edgemesh"
        docs = []
        for f in sorted(em.glob("0*.yaml")):
            docs += [d for d in yaml.safe_load_all(f.read_text()) if d]
        psk = base64.b64encode(os.urandom(32)).decode()
        for d in docs:
            if d["kind"] == "ConfigMap" and d["metadata"]["name"] == "edgemesh-agent-cfg":
                c = yaml.safe_load(d["data"]["edgemesh-agent.yaml"])
                c["modules"]["edgeTunnel"]["relayNodes"] = [{"nodeName": EDGEMESH_RELAY, "advertiseAddress": [m["ip"]]}]
                d["data"]["edgemesh-agent.yaml"] = yaml.safe_dump(c, sort_keys=False)
            if d["kind"] == "ConfigMap" and d["metadata"]["name"] == "edgemesh-agent-psk":
                d["data"]["psk"] = psk
        path = STATE / "edgemesh.yaml"
        path.write_text(yaml.safe_dump_all(docs, sort_keys=False))
        kubectl(f"apply -f {path}")
        want = len(tb.TOPO["workers"]) + 1
        wait(lambda: kubectl("get ds edgemesh-agent -n kubeedge -o jsonpath={.status.numberReady}").strip() == str(want),
             "edgemesh agents", 900)


# ---------------------------------------------------------------- stage images

def ensure_registry() -> None:
    """The testbed registry (docker.io/gcr.io mirror holding every image the clusters pull)."""
    if "true" not in sh("docker inspect -f '{{.State.Running}}' hed-registry", check=False):
        sh("docker rm -f hed-registry", check=False)
        sh(f"docker run -d --name hed-registry --restart always -p 5000:5000 "
           f"-v {ASSETS}/registry:/var/lib/registry registry:2")

def dockerfile(flavor: str, w: str) -> str:
    """The Dockerfile tools/gen_k8s.py emits, with library versions pinned to the venv and the
    wheels installed from a local wheelhouse (no network on the testbed)."""
    out = []
    for t, (pkgs, assets) in gen_k8s.IMAGES[w].items():
        pins = " ".join(f"{p}=={PINS[p.lower()]}" if p.lower() in PINS else p for p in pkgs)
        if t == "base":
            out += ["FROM python:3.12-slim AS base", "WORKDIR /app"]
        else:
            out += [f"FROM base AS {t}"]
        out.append(f"RUN --mount=type=bind,source=wheels,target=/wheels "
                   f"pip install --no-cache-dir --no-index --find-links /wheels {pins}")
        if t == "base":
            out.append("COPY apps/ apps/")
        if assets:
            out.append(f"COPY {' '.join('assets/' + a for a in assets)} assets/")
        if t == "base":
            out += ["ENV PORT=8080", "EXPOSE 8080", f'CMD ["python", "-m", "apps.{flavor}.{w}.stages"]']
        out.append("")
    return "\n".join(out)


def images(workloads: list[str]) -> None:
    flavor = (STATE / "flavor").read_text().strip()
    ctx = STATE / "build"
    shutil.rmtree(ctx, ignore_errors=True)
    ctx.mkdir(parents=True)
    shutil.copytree(tb.ROOT / "apps", ctx / "apps", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ASSETS / "wheels", ctx / "wheels")
    (ctx / "assets").mkdir()
    for w in workloads:
        for _, assets in gen_k8s.IMAGES[w].values():
            for a in assets:
                if not (ctx / "assets" / a).exists():
                    shutil.copy(tb.ROOT / "assets" / a, ctx / "assets" / a)
    for w in workloads:
        (ctx / f"Dockerfile.{w}").write_text(dockerfile(flavor, w))
        for t in gen_k8s.IMAGES[w]:
            tag = f"localhost:5000/hedices-bench/{w}-{t}:latest"
            sh(f"DOCKER_BUILDKIT=1 docker build -q -f {ctx}/Dockerfile.{w} --target {t} -t {tag} {ctx}")
            sh(f"docker push -q {tag}")
            print(f"pushed {tag}", flush=True)
    # the images now live in the registry: drop the build cache and the dangling layers each rebuild leaves
    # (about 9 GB a cluster start on the master's root disk, which holds kubelet's image filesystem:
    # imagefs.available < 15 % evicts pods)
    sh("docker builder prune -af > /dev/null 2>&1; docker image prune -f > /dev/null 2>&1", check=False)


# ---------------------------------------------------------------- testbed manifests

# Requests of applications deployed side by side (mixed arrivals): the scaled cluster has one
# cloud device where the paper's has two, so the reservations of all four applications at once
# do not fit; they are halved so every stage holds a pod (physical CPU quotas are unchanged).
REQUEST_FACTOR = {"value": 1.0}


def scale_mem(v: str) -> str:
    units = {"Ki": 1 << 10, "Mi": 1 << 20, "Gi": 1 << 30}
    unit = next((u for u in units if v.endswith(u)), None)
    n = float(v[:-2]) * units[unit] if unit else float(v)
    return f"{int(n * REQUEST_FACTOR['value']) >> 20}Mi"


def scale_cpu(v) -> str:
    return f"{int(round(float(v) * 1000 / CPU_SCALE * REQUEST_FACTOR['value']))}m"


def testbed_manifest(flavor: str, w: str) -> str:
    """tools/gen_k8s.py manifests adapted to the topology.  KubeEdge: the paper manifests' replica
    counts (4 edge / 2 cloud for 8 edge / 2 cloud devices), at most one per node of the stage's
    tier on a scaled layout; replicas the scheduler cannot fit stay Pending.  Knative: one warm pod per service with
    scale-to-zero disabled (paper Sec. 5.1), scaled out by the Knative autoscaler under load.
    CPU requests in the scaled units above.  Everything else is unchanged."""
    import yaml

    per_tier = {t: sum(1 for x in tb.TOPO["workers"] if x["tier"] == t) for t in ("cloud", "edge")}
    text = gen_k8s.kubeedge(w) if flavor == "kubeedge" else gen_k8s.knative(w)
    docs = [d for d in yaml.safe_load_all(text) if d]
    for d in docs:
        if flavor == "kubeedge" and d["kind"] == "Service" and KUBEEDGE_PATH == "edgemesh":
            # cloud-edge services are served by EdgeMesh; kube-proxy on the cloud nodes skips them
            d["metadata"].setdefault("labels", {})["service.kubernetes.io/service-proxy-name"] = "edgemesh"
        if d["kind"] not in ("Deployment", "Service") or "template" not in d.get("spec", {}):
            continue
        spec = d["spec"]["template"]["spec"]
        tier = spec["nodeSelector"]["hedices.io/tier"]
        if d["kind"] == "Deployment":
            # the paper manifests' replica counts (4 edge / 2 cloud for the 8 edge / 2 cloud
            # devices), at most one per node of the tier on a smaller layout
            d["spec"]["replicas"] = min(d["spec"]["replicas"], per_tier[tier])
        else:
            # one warm pod per service (Fig 13's failure trials may keep more: HEDICES_KNATIVE_MIN_SCALE)
            d["spec"]["template"]["metadata"]["annotations"]["autoscaling.knative.dev/min-scale"] = \
                os.environ.get("HEDICES_KNATIVE_MIN_SCALE", "1")
        for c in spec["containers"]:
            req = c["resources"]["requests"]
            req["cpu"] = scale_cpu(req["cpu"])
            if REQUEST_FACTOR["value"] != 1.0:  # side-by-side deployments: memory as well
                req["memory"] = scale_mem(req["memory"])
    return yaml.safe_dump_all(docs, sort_keys=False)


def deploy(workloads: list[str]) -> None:
    """Apply the workloads' manifests.  KubeEdge Deployments first start with one replica each, so
    every stage holds a pod before the remaining replicas compete for capacity (several workloads
    deployed together do not fit every replica)."""
    flavor = (STATE / "flavor").read_text().strip()
    targets = {}
    for w in workloads:
        path = STATE / f"{flavor}-{w}.yaml"
        text = testbed_manifest(flavor, w)
        path.write_text(text)
        if flavor == "kubeedge":
            import yaml

            docs = list(yaml.safe_load_all(text))
            for d in docs:
                if d["kind"] == "Deployment":
                    targets[d["metadata"]["name"]] = d["spec"]["replicas"]
                    d["spec"]["replicas"] = 1
            first = STATE / f"{flavor}-{w}.first.yaml"
            first.write_text(yaml.safe_dump_all(docs, sort_keys=False))
            kubectl(f"apply -f {first}")
        else:
            kubectl(f"apply -f {path}")
    if targets:
        wait(lambda: not [l for l in kubectl("get deploy --no-headers").splitlines()
                          if l.split()[1].startswith("0/")], "one pod per stage", 1800, 5)
        for name, n in targets.items():
            kubectl(f"scale deploy/{name} --replicas={n}")
    if flavor == "knative":
        jp = "'{range .items[*]}{.status.conditions[?(@.type==\"Ready\")].status}{\"\\n\"}{end}'"
        wait(lambda: set(kubectl(f"get ksvc -o jsonpath={jp}").split()) == {"True"},
             "knative services ready", 1800, 5)
    # every stage has a ready pod (unschedulable KubeEdge replicas stay Pending)
    wait(lambda: not [l for l in kubectl("get deploy --no-headers").splitlines()
                      if l.split()[1].startswith("0/")], "stage pods ready", 1800, 5)
    print(kubectl("get pods -o wide"))


def undeploy(workloads: list[str]) -> None:
    flavor = (STATE / "flavor").read_text().strip()
    for w in workloads:
        kubectl(f"delete -f {STATE / f'{flavor}-{w}.yaml'} --ignore-not-found --wait=true", check=False)


def main() -> None:
    if os.geteuid() != 0:
        raise SystemExit("run as root")
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("up"); u.add_argument("flavor", choices=["knative", "kubeedge"])
    sub.add_parser("down")
    for name in ("images", "deploy", "undeploy"):
        s = sub.add_parser(name); s.add_argument("workloads", nargs="*", default=["W1", "W2", "W3", "W4"])
    k = sub.add_parser("kubectl"); k.add_argument("args", nargs=argparse.REMAINDER)
    sub.add_parser("write-config"); sub.add_parser("attach")
    for name in ("node-pid", "node-rm"):
        sub.add_parser(name).add_argument("node")
    nu = sub.add_parser("node-up"); nu.add_argument("node"); nu.add_argument("role")
    pe = sub.add_parser("put-edge-config"); pe.add_argument("node"); pe.add_argument("text")
    li = sub.add_parser("load-image"); li.add_argument("image"); li.add_argument("url"); li.add_argument("want", nargs="?", default="")
    a = p.parse_args()
    if a.cmd in ("node-pid", "node-rm", "node-up", "put-edge-config"):
        n = tb.node(a.node)
        extra = [a.role] if a.cmd == "node-up" else [a.text] if a.cmd == "put-edge-config" else []
        print(LOCAL_OPS[a.cmd](n, *extra))
        return
    if a.cmd == "write-config":
        return write_config()
    if a.cmd == "attach":
        return attach_local()
    if a.cmd == "load-image":
        return load_image(a.image, a.url, a.want)
    ws = [w.lower() for w in getattr(a, "workloads", [])]
    if a.cmd == "up":
        up(a.flavor)
    elif a.cmd == "down":
        down()
    elif a.cmd == "images":
        images(ws)
    elif a.cmd == "deploy":
        deploy(ws)
    elif a.cmd == "undeploy":
        undeploy(ws)
    else:
        print(kubectl(" ".join(a.args), check=False))


if __name__ == "__main__":
    main()
