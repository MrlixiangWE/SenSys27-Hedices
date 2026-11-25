import subprocess
import sys

import pytest

from hedices.master import Master

WORKERS = [("cloud-0", "cloud", "agx", 2), ("edge-0", "edge", "nx", 2), ("edge-1", "edge", "pi5", 2)]


def spawn(port, wid, tier, cls, cpus, extra=()):
    return subprocess.Popen(
        [sys.executable, "-m", "hedices.branch", "--id", wid, "--master", f"127.0.0.1:{port}",
         "--listen", "127.0.0.1:0", "--tier", tier, "--device-class", cls, "--cpus", str(cpus),
         "--mem-mb", "1024", "--heartbeat", "0.2", *extra],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


@pytest.fixture
def cluster(request):
    mode = getattr(request, "param", "full")
    m = Master(listen="127.0.0.1:0", mode=mode, hb_timeout=1.0).start()
    extra = ("--no-cache",) if mode == "nocache" else ()
    procs = {wid: spawn(m.port, wid, tier, cls, cpus, extra) for wid, tier, cls, cpus in WORKERS}
    try:
        m.wait_workers(len(WORKERS), timeout=60)
        yield m, procs
    finally:
        m.stop()
        for p in procs.values():
            p.terminate()
        for p in procs.values():
            p.wait(10)
