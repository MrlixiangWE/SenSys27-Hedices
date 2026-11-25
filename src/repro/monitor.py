#!/usr/bin/env python3
"""repro/monitor.py -- a live terminal view of a run of repro/run_all.sh.

Redraws in place every two seconds: the step being run, elapsed time and an estimate of the remaining time, the
CPU / memory / temperature of every host of the testbed, and the last lines of the run's log. Standard library
only. run_all.sh starts it when its output is a terminal (NO_MONITOR=1 turns it off); it can also be attached to
a run started elsewhere:

    python3 repro/monitor.py --out results/ae [--pid PID]
"""
import argparse
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time

# expected minutes per step of run_all.sh (rounded up from our runs); multiplied by ROUNDS where marked
MINUTES = {"sync": 1, "fig6": 20, "fig7": 60, "fig8": 2, "fig9": 55, "fig12": 50, "fig13": 85,
           "fig10": 60, "table3": 85, "table4": 85, "bexp": 190, "summary": 1}
PER_ROUND = {"fig6", "fig10", "bexp"}
HEXP = ["fig6", "fig7", "fig8", "fig9"]


def stat_script(interval=2.0):
    """The sampling loop run on a host: one block per interval, ended by "@@".  Only shell builtins inside the loop
    (read, echo, case), so sampling twice a second costs the host no process spawns; the pause is a read with a
    timeout on a pipe nobody writes to."""
    return (
        "hostname; IF=$(ip route show default 2>/dev/null | awk '{print $5; exit}'); C=/sys/fs/cgroup/hed.slice; "
        "TZ=(/sys/class/thermal/thermal_zone*/temp); exec 3<> <(:); "
        "while :; do "
        "read -r l < /proc/stat; echo \"$l\"; "
        "while read -r k v _; do case $k in MemTotal:|MemAvailable:) echo \"$k $v\";; esac; done < /proc/meminfo; "
        "m=0; for z in \"${TZ[@]}\"; do t=0; read -r t < \"$z\" 2>/dev/null; [ \"${t:-0}\" -gt \"$m\" ] 2>/dev/null && m=$t; done; "
        "[ \"$m\" -gt 0 ] && echo \"$m\"; "
        "cur=; lim=; ina=; read -r cur < $C/memory.current 2>/dev/null; read -r lim < $C/memory.max 2>/dev/null; "
        "if [ -r $C/memory.stat ]; then while read -r k v; do [ \"$k\" = inactive_file ] && { ina=$v; break; }; done < $C/memory.stat; fi; "
        "echo \"CG $cur $lim inactive_file $ina\"; "
        "if [ -n \"$IF\" ]; then read -r rx < /sys/class/net/$IF/statistics/rx_bytes; read -r tx < /sys/class/net/$IF/statistics/tx_bytes; "
        "echo \"NET ${EPOCHREALTIME:-$(date +%s.%N)} $rx $tx\"; fi; "
        f"echo @@; read -t {interval:g} -u 3; done")


STAT = stat_script(2.0)


def expected(step, rounds):
    if step == "hexp":
        return sum(expected(s, rounds) for s in HEXP)
    if step == "all":
        return sum(expected(s, rounds) for s in ("sync", "hexp", "bexp", "summary"))
    return MINUTES.get(step, 5) * (rounds if step in PER_ROUND else 1)


class Host(threading.Thread):
    """Samples one host every `interval` seconds (the local one directly, the others over one ssh session each)."""

    def __init__(self, target, key, interval=2.0):
        super().__init__(daemon=True)
        self.target, self.key = target, key
        self.script = stat_script(interval)
        self.name_ = target.split("@")[-1] if target else socket.gethostname()
        self.cpu = self.mem = self.temp = None
        self.budget = None  # (working set, limit) of the testbed's memory slice in GB: what the experiment may use
        self.mt = 0
        self.prev = None
        self.net, self.net_prev = None, None  # (rx, tx) bytes per second on the uplink interface
        self.seen = 0.0

    def feed(self, lines):
        for ln in lines:
            if ln.startswith("cpu "):
                v = [int(x) for x in ln.split()[1:]]
                idle, total = v[3] + (v[4] if len(v) > 4 else 0), sum(v)
                if self.prev:
                    dt = total - self.prev[1]
                    self.cpu = 100.0 * (1 - (idle - self.prev[0]) / dt) if dt > 0 else self.cpu
                self.prev = (idle, total)
            elif ln.startswith("MemTotal:"):
                self.mt = int(ln.split()[1])
            elif ln.startswith("MemAvailable:"):
                self.mem = (self.mt - int(ln.split()[1])) / 1048576, self.mt / 1048576
            elif ln.startswith("CG"):
                # hed.slice (testbed/tb.py caps it at 85 % of RAM): working set = usage less inactive file cache
                f = ln.split()
                try:
                    cur, lim = int(f[1]), int(f[2])
                    inact = int(f[4]) if len(f) > 4 and f[3] == "inactive_file" else 0
                    self.budget = (max(0, cur - inact) / 1073741824, lim / 1073741824)
                except (IndexError, ValueError):  # no slice yet, or no limit ("max")
                    self.budget = None
            elif ln.strip().isdigit():
                self.temp = int(ln) / 1000.0
            elif ln.startswith("NET "):
                try:
                    t, rx, tx = (float(x) for x in ln.split()[1:4])
                    if self.net_prev and t > self.net_prev[0]:
                        dt = t - self.net_prev[0]
                        self.net = (max(0.0, (rx - self.net_prev[1]) / dt), max(0.0, (tx - self.net_prev[2]) / dt))
                    self.net_prev = (t, rx, tx)
                except ValueError:
                    pass
        self.seen = time.time()

    def run(self):
        while True:
            if self.target:
                cmd = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-o", "ServerAliveInterval=10"]
                if self.key and os.path.exists(self.key):
                    cmd += ["-i", self.key]
                cmd += [self.target, "bash -c " + shlex.quote(self.script)]
            else:
                cmd = ["bash", "-c", self.script]
            try:
                p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
                                     text=True)
                first, block = True, []
                for ln in p.stdout:
                    ln = ln.rstrip("\n")
                    if first:
                        self.name_, first = getattr(self, "label", None) or ln.strip() or self.name_, False
                    elif ln == "@@":
                        self.feed(block)
                        block = []
                    else:
                        block.append(ln)
            except OSError:
                pass
            time.sleep(5)


def labels():
    """LAN address -> "host (workers)" from the testbed topology; the local host is the master."""
    path = os.path.join("testbed", os.environ.get("HEDICES_TOPOLOGY") or "topology_phys6r_nd.json")
    try:
        import json
        topo = json.load(open(path))
    except (OSError, ValueError):
        return {}
    workers = {}
    for w in topo.get("workers", []):
        workers.setdefault(w.get("host"), []).append(w.get("name"))
    out = {}
    for name, h in topo.get("hosts", {}).items():
        ws = ", ".join(x for x in workers.get(name, []) if x)
        out[h.get("lan")] = f"{name} ({ws})" if ws else name
    return out


def hms(s):
    s = max(0, int(s))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def bar(frac, width):
    n = int(round(max(0.0, min(1.0, frac)) * width))
    return "█" * n + "·" * (width - n)


def read_progress(path):
    """Lines '<epoch> start|end <step> [rc]' written by run_all.sh."""
    events = []
    try:
        with open(path) as f:
            for ln in f:
                p = ln.split()
                if len(p) >= 3:
                    events.append((float(p[0]), p[1], p[2], p[3] if len(p) > 3 else ""))
    except OSError:
        pass
    return events


def tail(path, n):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 8192))
            lines = f.read().decode(errors="replace").splitlines()
        return [ln for ln in lines if ln.strip()][-n:]
    except OSError:
        return []


def alive(pid):
    if not pid:
        return True
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def draw(args, hosts, t0):
    cols = shutil.get_terminal_size((100, 30)).columns
    out = []
    events = read_progress(os.path.join(args.out, ".progress"))
    steps = args.steps.split() if args.steps else []
    done = {s for _, k, s, _ in events if k == "end"}
    cur = next((s for _, k, s, _ in reversed(events) if k == "start" and s not in done), None)
    start = events[0][0] if events else t0
    now = time.time()
    cur_t0 = next((t for t, k, s, _ in reversed(events) if k == "start" and s == cur), now)
    left = 0.0
    for s in steps:
        if s in done:
            continue
        exp = expected(s, args.rounds) * 60
        left += max(exp - (now - cur_t0), 60) if s == cur else exp
    total = (now - start) + left
    out.append(f" Hedices artifact · {os.path.basename(os.path.normpath(args.out))}"
               f"{'':>4}{time.strftime('%H:%M:%S')}")
    out.append(" " + "─" * (cols - 2))
    out.append(f" step     {cur or ('done' if steps and set(steps) <= done else 'starting')}"
               f"   ({len(done)}/{len(steps) or '?'} steps done; in this step {hms(now - cur_t0) if cur else '-'})")
    out.append(f" elapsed  {hms(now - start)}    remaining ≈ {hms(left)}    finish ≈ "
               f"{time.strftime('%H:%M', time.localtime(now + left))}")
    w = max(10, min(50, cols - 30))
    out.append(f" progress [{bar((now - start) / total if total else 0, w)}] {100 * (now - start) / total if total else 0:4.0f} %")
    out.append(" " + "─" * (cols - 2))
    out.append(f" {'host':<23}{'cpu':>6}  {'':<{min(20, w)}}{'memory':>16}{'temp':>9}")
    for h in hosts:
        stale = time.time() - h.seen > 15
        cpu = "  --" if h.cpu is None or stale else f"{h.cpu:5.0f}%"
        mem = "--" if not h.mem or stale else f"{h.mem[0]:.1f}/{h.mem[1]:.1f} GB"
        temp = "--" if h.temp is None or stale else f"{h.temp:.0f} °C"
        b = bar((h.cpu or 0) / 100, min(20, w)) if h.cpu is not None and not stale else " " * min(20, w)
        out.append(f" {h.name_[:22]:<23}{cpu:>6}  {b}{mem:>16}{temp:>9}")
    out.append(" " + "─" * (cols - 2))
    for ln in tail(args.log, args.lines):
        out.append(" " + ln)
    rows = shutil.get_terminal_size((100, 30)).lines
    out = [ln[:cols - 1] for ln in out][: rows - 1]
    sys.stdout.write("\033[H" + "\n".join(ln + "\033[K" for ln in out) + "\033[J")
    sys.stdout.flush()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="the run's result directory (REPRO_OUT)")
    p.add_argument("--log", help="the run's log (default <out>/run.log)")
    p.add_argument("--steps", default="", help="the steps of the run, in order")
    p.add_argument("--rounds", type=int, default=int(os.environ.get("ROUNDS", "2")))
    p.add_argument("--pid", type=int, default=0, help="exit when this process (the run) has ended")
    p.add_argument("--hosts", default=os.environ.get("REPRO_HOSTS") or "repro/hosts")
    p.add_argument("--key", default=os.environ.get("REPRO_SSH_KEY") or os.path.expanduser("~/.ssh/id_hedices"))
    p.add_argument("--lines", type=int, default=6)
    args = p.parse_args()
    args.log = args.log or os.path.join(args.out, "run.log")
    hf = args.hosts if os.path.exists(args.hosts) else "repro/hosts.example"
    targets = [None]
    try:
        targets += [ln.strip() for ln in open(hf) if ln.strip() and not ln.startswith("#")]
    except OSError:
        pass
    hosts = [Host(t, args.key) for t in targets]
    names = labels()
    local = next((v for k, v in names.items() if v.startswith(socket.gethostname())), None)
    for h in hosts:
        lab = names.get(h.target.split("@")[-1]) if h.target else (local or socket.gethostname())
        h.label = (lab + (" master" if not h.target else "")) if lab else None
    for h in hosts:
        h.start()
    t0 = time.time()

    def restore(*_):
        sys.stdout.write("\033[?25h\n")
        sys.stdout.flush()
        os._exit(130)
    signal.signal(signal.SIGTERM, restore)
    sys.stdout.write("\033[?25l\033[2J")
    try:
        while alive(args.pid):
            draw(args, hosts, t0)
            time.sleep(2)
        draw(args, hosts, t0)
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\033[?25h\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
