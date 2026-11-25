#!/usr/bin/env python3
"""The paper's heterogeneous cloud-edge testbed: nodes as network namespaces.

Each node gets its own network namespace on one bridge; every process of a
node runs in that namespace inside the node's systemd slice, which carries
the node's CPU quota (and cpuset).  Traffic between the cloud and edge tiers
is shaped with htb + netem (paper: 100 Mbps).  Must run as root.

A topology with "hosts" spans several machines: every node names its host,
each host bridges its own nodes, and the bridges are joined by a VXLAN overlay
into one L2 segment.  Commands for nodes on other hosts run there over SSH (key
TOPO["ssh_key"], passwordless sudo); the master's host runs the orchestration.

  tb.py up | down | status
  tb.py start-branches [--mode M]     start one Hedices Branch per worker
  tb.py stop [PATTERN]                stop hed-* units (default: all)
  tb.py kill NODE                     SIGKILL every unit of NODE (failure injection)
  tb.py run NODE --unit U -- CMD...   run CMD as the user inside NODE (waits, streams output)
  tb.py spawn NODE --unit U -- CMD... start CMD as a background unit inside NODE
  tb.py link NODE up|down             set the node's link (failure injection)
  tb.py active UNIT | journal UNIT | netstat | clean-ray   (queries used by the host agent)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pwd
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOPO_FILE = os.environ.get("HEDICES_TOPOLOGY", "topology.json")
TOPO = json.loads((ROOT / "testbed" / TOPO_FILE).read_text())
# the unprivileged user that owns the checkout runs every node process
USER = os.environ.get("SUDO_USER") or ROOT.owner()
HOME = pwd.getpwnam(USER).pw_dir
PY = str(ROOT / ".venv" / "bin" / "python")
MASTER_PORT = 7700
BRANCH_PORT = 7800


def sh(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, shell=True, check=check, text=True, capture_output=True)


HOSTS: dict = TOPO.get("hosts", {})
SSH_KEY = TOPO.get("ssh_key", "")


def _local_host() -> str | None:
    if not HOSTS:
        return None
    if os.environ.get("HEDICES_HOST"):
        return os.environ["HEDICES_HOST"]
    addrs = sh("hostname -I", check=False).stdout.split()
    for name, h in HOSTS.items():
        if h["lan"] in addrs:
            return name
    raise SystemExit("this machine is not a host of the topology")


LOCAL = _local_host()


# Logical CPUs of each device class on the paper's testbed (AGX Orin, Orin NX 16GB, Pi 5).  Every
# system's scheduler sees these, so stage CPU declarations pack as they would on the real devices;
# the physical CPU a node actually gets is its cpu_quota.
CLASS_CPUS = {"agx": 12, "nx": 8, "pi5": 4}


def logical_cpus(n: dict) -> int:
    return n.get("cpus") or CLASS_CPUS[n["class"]]


def nodes() -> list[dict]:
    return [TOPO["master"], *TOPO["workers"]]


def node(name: str) -> dict:
    for n in nodes():
        if n["name"] == name:
            return n
    raise SystemExit(f"unknown node {name}")


def is_local(n: dict) -> bool:
    return not HOSTS or n["host"] == LOCAL


def local_nodes() -> list[dict]:
    return [n for n in nodes() if is_local(n)]


def bridge_ip() -> str:
    return HOSTS[LOCAL]["bridge_ip"] if HOSTS else TOPO["bridge_ip"]


def lan_iface(ip: str) -> str:
    out = sh("ip -o -4 addr show", check=False).stdout
    for line in out.splitlines():
        f = line.split()
        if f[3].split("/")[0] == ip:
            return f[1]
    raise SystemExit(f"no interface with {ip}")


# ---------------------------------------------------------------- host agent client

def remote(host: str, argv: list[str], timeout: float = 900.0) -> dict:
    """Run ``tb.py argv`` (``k8s.py argv[1:]`` for argv[0] == "k8s") as root on another host over
    SSH (one multiplexed connection per host): {"rc", "out", "err"}."""
    h = HOSTS[host]
    script = ["k8s.py", *argv[1:]] if argv[:1] == ["k8s"] else ["tb.py", *argv]
    cmd = ["sudo", "-n", "env", f"HEDICES_TOPOLOGY={TOPO_FILE}", PY, str(ROOT / "testbed" / script[0]),
           *script[1:]]
    ssh = ["ssh", "-i", SSH_KEY, "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
           "-o", "ControlMaster=auto", "-o", "ControlPath=/tmp/hedcm-%r@%h", "-o", "ControlPersist=30m",
           "-o", "ServerAliveInterval=15", f"{h['user']}@{h['lan']}", shlex.join(cmd)]
    r = subprocess.run(ssh, text=True, capture_output=True, timeout=timeout)
    return {"rc": r.returncode, "out": r.stdout, "err": r.stderr}


def on_host(host: str | None, argv: list[str], check: bool = True, timeout: float = 900.0) -> dict:
    if host is None or host == LOCAL:
        r = subprocess.run([PY, str(ROOT / "testbed" / "tb.py"), *argv], text=True, capture_output=True,
                           timeout=timeout)
        res = {"rc": r.returncode, "out": r.stdout, "err": r.stderr}
    else:
        res = remote(host, argv, timeout)
    if check and res["rc"]:
        raise RuntimeError(f"{host}: tb.py {' '.join(argv)} -> rc={res['rc']}\n{res['out']}\n{res['err']}")
    return res


def other_hosts() -> list[str]:
    return [h for h in HOSTS if h != LOCAL]


def on_all_hosts(argv: list[str], check: bool = True) -> dict[str, dict]:
    """Run ``tb.py argv`` on every other host concurrently (the local host is the caller's job)."""
    res: dict[str, dict] = {}

    def one(h):
        try:
            res[h] = on_host(h, argv, check=check)
        except Exception as e:  # noqa: BLE001
            res[h] = {"rc": -1, "out": "", "err": repr(e)}
    ts = [threading.Thread(target=one, args=(h,)) for h in other_hosts()]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    bad = {h: r for h, r in res.items() if r["rc"]}
    if check and bad:
        raise RuntimeError(f"tb.py {' '.join(argv)} failed on {bad}")
    return res


# ---------------------------------------------------------------- cluster-wide operations

def up_cluster() -> None:
    up()
    on_all_hosts(["up"])


def down_cluster() -> None:
    """Tear the testbed down on every host of the topology (namespaces, overlay, bridge)."""
    down()
    on_all_hosts(["down"], check=False)


def stop_cluster(pattern: str = "hed-*") -> None:
    stop(pattern)
    on_all_hosts(["stop", pattern], check=False)


def apply_quotas_cluster() -> None:
    apply_quotas()
    on_all_hosts(["quotas"])


def clean_ray_cluster() -> None:
    sh("rm -rf /tmp/ray /tmp/ray-*", check=False)
    on_all_hosts(["clean-ray"], check=False)


def spawn(n: dict, unit: str, argv: list[str], env: dict | None = None) -> None:
    if is_local(n):
        subprocess.run(unit_cmd(n, unit, argv, wait=False, env=env), check=True, capture_output=True)
    else:
        # options before the node name: everything after it is the command
        on_host(n["host"], ["spawn", "--unit", unit, *[f"--env={k}={v}" for k, v in (env or {}).items()],
                            n["name"], "--", *argv])


def unit_active(n: dict, unit: str) -> bool:
    if is_local(n):
        return sh(f"systemctl is-active --quiet hed-{unit}.service", check=False).returncode == 0
    return on_host(n["host"], ["active", unit], check=False)["rc"] == 0


def journal(n: dict, unit: str) -> str:
    if is_local(n):
        return sh(f"journalctl -u hed-{unit} --since '-5min' -o cat --no-pager", check=False).stdout
    return on_host(n["host"], ["journal", unit], check=False)["out"]


def link(n: dict, state: str) -> None:
    if is_local(n):
        sh(f"ip link set {host_if(n)} {state}")
    else:
        on_host(n["host"], ["link", n["name"], state])


def kill_node(n: dict) -> None:
    if is_local(n):
        sh(f"systemctl kill --signal=SIGKILL 'hed-{n['name']}-*.service'", check=False)
    else:
        on_host(n["host"], ["kill", n["name"]], check=False)


_NS_PID: dict[str, str] = {}


def _retrans_segs(name: str) -> int | None:
    """TCP segments the node's namespace has retransmitted (read through a process inside it)."""
    for _ in range(2):
        pid = _NS_PID.get(name)
        if pid is None:
            out = sh(f"ip netns pids hed-{name}", check=False).stdout.split()
            if not out:
                return None
            pid = _NS_PID[name] = out[0]
        try:
            lines = Path(f"/proc/{pid}/net/snmp").read_text().splitlines()
        except OSError:
            _NS_PID.pop(name, None)
            continue
        tcp = [l.split() for l in lines if l.startswith("Tcp:")]
        return int(tcp[1][tcp[0].index("RetransSegs")])
    return None


def netstat_local() -> dict[str, int]:
    """Bytes each local node has sent (host-side veth rx counters), under "<node>#p" the packets
    they took (to separate per-packet header bytes from payload), and under "<node>#r" the TCP
    segments its namespace retransmitted (bytes the network, not the system, made it resend)."""
    row = {}
    for path in glob.glob("/sys/class/net/hv-*/statistics/rx_bytes"):
        name = path.split("/")[4][3:]
        row[name] = int(Path(path).read_text())
        try:
            row[name + "#p"] = int(Path(path).with_name("rx_packets").read_text())
        except OSError:
            pass
        r = _retrans_segs(name)
        if r is not None:
            row[name + "#r"] = r
    return row


def trace_collect_local(d: str) -> str:
    """Concatenate and remove the stage-call traces this host's processes wrote under ``d``."""
    out = []
    for f in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
        out.append(Path(f).read_text())
        os.unlink(f)
    return "".join(out)


def trace_collect_cluster(d: str) -> str:
    text = trace_collect_local(d)
    for r in on_all_hosts(["trace-collect", d], check=False).values():
        if r["rc"] == 0:
            text += r["out"]
    return text


def netstat_cluster() -> dict[str, int]:
    row = netstat_local()
    for r in on_all_hosts(["netstat"], check=False).values():
        if r["rc"] == 0:
            row.update(json.loads(r["out"]))
    return row


def host_if(n: dict) -> str:
    return f"hv-{n['name']}"[:15]


def ns(n: dict) -> str:
    return f"hed-{n['name']}"


def slice_name(n: dict) -> str:
    return f"hed-{n['name'].replace('-', '')}.slice"


def other_tier(n: dict) -> str:
    return TOPO["tiers"]["edge" if n["tier"] == "cloud" else "cloud"]


def same_netns(name: str, pid: int) -> bool:
    try:
        return os.stat(f"/run/netns/{name}").st_ino == os.stat(f"/proc/{pid}/ns/net").st_ino
    except OSError:
        return False


def _kernel_has(qdisc: str) -> bool:
    """Whether tc can create ``qdisc`` (probed on a scratch dummy device)."""
    sh("ip link add hedprobe0 type dummy", check=False)
    args = "htb default 1" if qdisc == "htb" else "netem delay 1ms"
    ok = sh(f"tc qdisc add dev hedprobe0 root handle 1: {args}", check=False).returncode == 0
    sh("ip link del hedprobe0", check=False)
    return ok


HAS_HTB = HAS_NETEM = None


def shaper(tc: str, dev: str, field: str, net: str, shape: dict) -> None:
    sh(f"{tc} qdisc add dev {dev} root handle 1: htb default 10")
    sh(f"{tc} class add dev {dev} parent 1: classid 1:10 htb rate 10gbit")
    sh(f"{tc} class add dev {dev} parent 1: classid 1:20 htb rate {shape['rate']} ceil {shape['rate']}")
    if HAS_NETEM and shape.get("delay"):
        sh(f"{tc} qdisc add dev {dev} parent 1:20 handle 20: netem delay {shape['delay']}")
    sh(f"{tc} filter add dev {dev} parent 1: protocol ip prio 1 u32 match ip {field} {net} flowid 1:20")


def up(attach: dict[str, int] | None = None) -> None:
    """attach: node name -> pid whose network namespace becomes the node's namespace (the
    container nodes of the Kubernetes baselines); other nodes get a fresh namespace."""
    for mod in ("veth", "bridge", *(("vxlan",) if HOSTS else ())):
        sh(f"modprobe {mod}")
    if attach:
        # Kubernetes nodes: pod -> Service -> pod on the same node (e.g. the Kourier gateway
        # reaching its controller on the master) needs bridged traffic through iptables; k3s
        # cannot load the module from inside its node container on every host kernel
        sh("modprobe br_netfilter", check=False)
        sh("sysctl -qw net.bridge.bridge-nf-call-iptables=1", check=False)
    for mod in ("sch_netem", "sch_htb", "cls_u32"):  # built in, a module, or absent
        if sh(f"modprobe {mod}", check=False).returncode:
            # a kernel built without it: the module compiled out of tree for this kernel
            ko = ROOT / "kmod" / mod.split("_")[-1] / f"{mod}.ko"
            if ko.exists() and mod not in sh("lsmod", check=False).stdout:
                sh(f"insmod {ko}", check=False)
    global HAS_HTB, HAS_NETEM
    HAS_HTB, HAS_NETEM = _kernel_has("htb"), _kernel_has("netem")
    shaped = any(n["tier"] == "cloud" for n in local_nodes())
    if shaped and not HAS_HTB:
        raise SystemExit("cloud-tier nodes need a kernel with htb/u32 to shape the cross-tier links")
    br = TOPO["bridge"]
    if sh(f"ip link show {br}", check=False).returncode != 0:
        sh(f"ip link add {br} type bridge")
        sh(f"ip addr add {bridge_ip()} dev {br}")
    sh(f"ip link set {br} up")
    vx = TOPO.get("vxlan")
    mtu = vx["mtu"] if HOSTS and vx else None
    if HOSTS and vx and sh(f"ip link show {vx['name']}", check=False).returncode != 0:
        # overlay joining the hosts' bridges into one L2 segment (head-end replication to peers)
        lan = HOSTS[LOCAL]["lan"]
        sh(f"ip link add {vx['name']} type vxlan id {vx['id']} dstport {vx['port']} local {lan} "
           f"dev {lan_iface(lan)}")
        for h in other_hosts():
            sh(f"bridge fdb append 00:00:00:00:00:00 dev {vx['name']} dst {HOSTS[h]['lan']}")
        sh(f"ip link set {vx['name']} mtu {mtu} master {br} up")
    # once br_netfilter is loaded (k3s loads it) bridged frames traverse the host FORWARD chain,
    # whose policy Docker sets to DROP: keep node-to-node traffic on the bridge forwarded
    chain = "DOCKER-USER" if sh("iptables -S DOCKER-USER", check=False).returncode == 0 else "FORWARD"
    if sh(f"iptables -C {chain} -i {br} -o {br} -j ACCEPT", check=False).returncode:
        sh(f"iptables -I {chain} -i {br} -o {br} -j ACCEPT")
    shape = TOPO["cross_tier"]
    for n in local_nodes():
        name, hif = ns(n), host_if(n)
        if sh(f"ip netns list | grep -qw {name}", check=False).returncode == 0:
            if not attach or n["name"] not in attach or same_netns(name, attach[n["name"]]):
                continue
            # the node's container was recreated: drop the stale namespace and wire the new one
            sh(f"ip link del {hif}", check=False)
            sh(f"ip netns del {name}", check=False)
        if attach and n["name"] in attach:
            sh(f"ip netns attach {name} {attach[n['name']]}")
        else:
            sh(f"ip netns add {name}")
        # the host end of a removed container's veth lingers until the kernel tears its namespace
        # down: wait for the name to be free
        for _ in range(50):
            if sh(f"ip link show {hif}", check=False).returncode:
                break
            sh(f"ip link del {hif}", check=False)
            time.sleep(0.2)
        sh(f"ip link add {hif} type veth peer name eth0 netns {name}")
        if mtu:
            sh(f"ip link set {hif} mtu {mtu}")
            sh(f"ip -n {name} link set eth0 mtu {mtu}")
        sh(f"ip link set {hif} master {br} up")
        sh(f"ip -n {name} addr add {n['ip']}/16 dev eth0")
        sh(f"ip -n {name} link set eth0 up")
        sh(f"ip -n {name} link set lo up")
        if TOPO.get("rto_min"):
            # the testbed is one LAN (sub-ms RTTs): TCP's 200 ms minimum retransmission timeout is a
            # WAN default that turns one burst loss at a switch port into a 200 ms stall
            sh(f"ip -n {name} route replace 10.77.0.0/16 dev eth0 proto kernel scope link src {n['ip']} "
               f"rto_min {TOPO['rto_min']}")
        # cross-tier traffic: 100 Mbps (+ delay where the kernel has netem); same-tier traffic is
        # not shaped.  Egress of the node (by destination tier) and, on the host side of its veth,
        # ingress (by source tier), so a link is shaped even when the peer's kernel cannot shape
        # (the Orin NX kernels have no htb/u32; every cross-tier path has a cloud-tier end).
        # (shaped at the cloud-tier end only, so no path is shaped twice where an edge host can
        # shape too)
        if HAS_HTB and n["tier"] == "cloud":
            shaper(f"ip netns exec {name} tc", "eth0", "dst", other_tier(n), shape)
            shaper("tc", hif, "src", other_tier(n), shape)
        rate = HOSTS.get(LOCAL, {}).get("link_rate") if HOSTS else None
        if rate and HAS_HTB:
            # the host's own uplink (e.g. a device on a 100 Mbps port): all of the node's traffic,
            # both directions
            for tc, dev in ((f"ip netns exec {name} tc", "eth0"), ("tc", hif)):
                sh(f"{tc} qdisc add dev {dev} root handle 1: htb default 10")
                sh(f"{tc} class add dev {dev} parent 1: classid 1:10 htb rate {rate} ceil {rate}")
        if attach and n["name"] in attach:
            sh(f"ip -n {name} route add default via {bridge_ip().split('/')[0]}")
    apply_quotas()
    print("testbed up:", ", ".join(f"{n['name']}={n['ip']}" for n in local_nodes()))


def testbed_memory_mb() -> int:
    """Memory all emulated nodes may use together: 85% of host RAM, the rest stays with the host
    (sshd, the registry); exceeding it OOM-kills inside the testbed instead of swapping the host."""
    total_kb = int(next(l for l in Path("/proc/meminfo").read_text().splitlines()
                        if l.startswith("MemTotal")).split()[1])
    return int(os.environ.get("HEDICES_TESTBED_MEM_MB", total_kb // 1024 * 85 // 100))


def apply_quotas() -> None:
    sh(f"systemctl set-property --runtime hed.slice MemoryMax={testbed_memory_mb()}M MemorySwapMax=0")
    for n in local_nodes():
        quota = int(round(n["cpu_quota"] * 100))
        # per-node memory is bounded only by the testbed-wide cap above: the paper's devices have
        # 8-32 GB each, which a single host cannot partition
        # a node that owns whole cores (cpuset) needs no bandwidth quota on top: CFS bandwidth
        # control would only add throttling jitter to bursts of its threads
        cpu = (f"AllowedCPUs={n['cpuset']} CPUQuota=" if n.get("cpuset")
               else f"CPUQuota={quota}% CPUQuotaPeriodSec=1ms")
        sh(f"systemctl set-property --runtime {slice_name(n)} {cpu} MemoryMax=infinity MemorySwapMax=0")


def stop(pattern: str = "hed-*") -> None:
    sh(f"systemctl stop '{pattern}.service'", check=False)
    sh("systemctl reset-failed 'hed-*'", check=False)


def down() -> None:
    stop()
    for n in local_nodes():
        sh(f"ip netns del {ns(n)}", check=False)
    if TOPO.get("vxlan"):
        sh(f"ip link del {TOPO['vxlan']['name']}", check=False)
    sh(f"ip link del {TOPO['bridge']}", check=False)
    print("testbed down")


def status() -> None:
    print(sh("ip netns list", check=False).stdout.strip() or "(no netns)")
    print(sh("systemctl list-units --no-legend 'hed-*'", check=False).stdout.strip() or "(no units)")


def unit_cmd(n: dict, unit: str, argv: list[str], wait: bool, env: dict | None = None) -> list[str]:
    cmd = ["systemd-run", f"--unit=hed-{unit}", "--collect", f"--slice={slice_name(n)}",
           f"--property=NetworkNamespacePath=/run/netns/{ns(n)}", f"--uid={USER}", f"--gid={USER}",
           f"--working-directory={ROOT}", f"--setenv=HOME={HOME}",
           # native thread pools are left to each system's own default (Ray sizes them to a task's
           # num_cpus, Dask and the libraries to the node's cores, the Hedices Branch to its slots)
           "--setenv=PYTHONUNBUFFERED=1",
           f"--setenv=HEDICES_NODE={n['name']}", f"--setenv=HEDICES_IP={n['ip']}",
           # node-local model/data assets, wherever a framework unpacks the application code
           f"--setenv=HEDICES_ASSETS={ROOT / 'assets'}"]
    for k, v in (env or {}).items():
        cmd.append(f"--setenv={k}={v}")
    if wait:
        cmd += ["--wait", "--pipe", "--quiet"]
    return cmd + argv


def cache_budget_mb(w: dict) -> float | None:
    """Per-device-class function-cache budget, HEDICES_CACHE_KB="agx:512,nx:256,pi5:128" (KB)."""
    spec = os.environ.get("HEDICES_CACHE_KB", "")
    budgets = dict(item.split(":") for item in spec.split(",") if ":" in item)
    return float(budgets[w["class"]]) / 1024 if w["class"] in budgets else None


def start_branches(extra: list[str]) -> None:
    master = TOPO["master"]
    for w in TOPO["workers"]:
        budget = cache_budget_mb(w)
        cache = ["--cache-mb", f"{budget:.4f}"] if budget is not None and "--cache-mb" not in extra else []
        entries = dict(i.split(":") for i in os.environ.get("HEDICES_CACHE_ENTRIES", "").split(",") if ":" in i)
        # an entry names a device class or one worker ("nx:5,nx-3:4"); the worker's own entry wins
        n_entries = entries.get(w["name"], entries.get(w["class"]))
        if n_entries is not None and "--cache-entries" not in extra:
            cache += ["--cache-entries", n_entries]
        # HEDICES_CACHE_EVICT=random: the function cache's replacement policy (branch --cache-evict; default lru)
        if os.environ.get("HEDICES_CACHE_EVICT") and "--cache-evict" not in extra:
            cache += ["--cache-evict", os.environ["HEDICES_CACHE_EVICT"]]
        argv = [PY, "-m", "hedices.branch", "--id", w["name"], *cache,
                "--master", f"{master['ip']}:{MASTER_PORT}", "--listen", f"0.0.0.0:{BRANCH_PORT}",
                "--advertise-host", w["ip"], "--tier", w["tier"], "--device-class", w["class"],
                "--cpus", str(logical_cpus(w)), "--mem-mb", str(w["mem_mb"]),
                "--speed", str(w["cpu_quota"]), "--norely-mode", "thread", *extra]
        spawn(w, f"{w['name']}-branch", argv)
        time.sleep(0.2)
    print(f"started {len(TOPO['workers'])} branches")


def main() -> None:
    if os.geteuid() != 0:
        raise SystemExit("run as root")
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("up"); sub.add_parser("down"); sub.add_parser("down-cluster"); sub.add_parser("status")
    sb = sub.add_parser("start-branches"); sb.add_argument("extra", nargs="*")
    st = sub.add_parser("stop"); st.add_argument("pattern", nargs="?", default="hed-*")
    k = sub.add_parser("kill"); k.add_argument("node")
    sub.add_parser("quotas"); sub.add_parser("netstat"); sub.add_parser("clean-ray")
    tc = sub.add_parser("trace-collect"); tc.add_argument("dir")
    lk = sub.add_parser("link"); lk.add_argument("node"); lk.add_argument("state", choices=["up", "down"])
    for name in ("active", "journal"):
        sub.add_parser(name).add_argument("unit")
    for name in ("run", "spawn"):
        r = sub.add_parser(name)
        r.add_argument("node"); r.add_argument("--unit", required=True)
        r.add_argument("--env", action="append", default=[])
        r.add_argument("argv", nargs=argparse.REMAINDER)
    a = p.parse_args()
    if a.cmd == "up":
        up()
    elif a.cmd == "down":
        down()
    elif a.cmd == "down-cluster":
        down_cluster()
    elif a.cmd == "status":
        status()
    elif a.cmd == "start-branches":
        start_branches(a.extra)
    elif a.cmd == "stop":
        stop(a.pattern)
    elif a.cmd == "kill":
        sh(f"systemctl kill --signal=SIGKILL 'hed-{a.node}-*.service'", check=False)
    elif a.cmd == "quotas":
        apply_quotas()
    elif a.cmd == "trace-collect":
        print(trace_collect_local(a.dir), end="")
    elif a.cmd == "netstat":
        print(json.dumps(netstat_local()))
    elif a.cmd == "clean-ray":
        sh("rm -rf /tmp/ray /tmp/ray-*", check=False)
    elif a.cmd == "link":
        sh(f"ip link set {host_if(node(a.node))} {a.state}")
    elif a.cmd == "active":
        sys.exit(sh(f"systemctl is-active --quiet hed-{a.unit}.service", check=False).returncode)
    elif a.cmd == "journal":
        print(sh(f"journalctl -u hed-{a.unit} --since '-5min' -o cat --no-pager", check=False).stdout)
    else:
        argv = a.argv[1:] if a.argv[:1] == ["--"] else a.argv
        env = dict(e.split("=", 1) for e in a.env)
        res = subprocess.run(unit_cmd(node(a.node), a.unit, argv, wait=a.cmd == "run", env=env))
        sys.exit(res.returncode)


if __name__ == "__main__":
    main()
