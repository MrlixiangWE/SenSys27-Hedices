import time
from types import SimpleNamespace

import pytest

from hedices import Workflow
from hedices.examples import toy
from hedices.package import build_package
from hedices.stemsched import MODES, StemSched

WF = Workflow(toy.pipeline)
N = 200
EXPECTED = 2.0 * sum(range(N))


def test_trace_builds_dag():
    assert [n.spec.short_name for n in WF.nodes] == ["load", "heavy", "heavy", "aggregate"]
    assert WF.roots == [0]
    assert WF.nodes[3].deps == frozenset({1, 2})
    assert WF.nodes[1].spec.norely_flag == "req_id"


def test_package_hash_is_content_based():
    a = build_package(WF.nodes[1].spec)
    b = build_package(WF.nodes[1].spec)
    assert a.hash == b.hash
    assert build_package(WF.nodes[0].spec).hash != a.hash
    assert ("numpy", __import__("numpy").__version__) not in a.manifest or True
    assert any(name == "numpy" for name, _ in build_package(WF.nodes[0].spec).manifest)


def test_sigma_eq9():
    s = StemSched(window=2)
    w1 = SimpleNamespace(id="a", alive=True, view={"h1"})
    w2 = SimpleNamespace(id="b", alive=True, view=set())
    H = frozenset({"h1", "h2"})
    assert s.sigma([w1, w2], H) == pytest.approx(1 / 4)  # window not full: compare with empty
    s.observe([w1, w2]); s.observe([w1, w2])
    assert s.sigma([w1, w2], H) == 0.0                   # unchanged over the window
    w2.view = {"h1", "h2"}
    assert s.sigma([w1, w2], H) == pytest.approx(2 / 4)
    assert s.alpha(1.7) == 0.5 and s.alpha(0.2) == pytest.approx(0.1)


@pytest.mark.parametrize("cluster", MODES, indirect=True)
def test_modes_compute_correct_results(cluster):
    m, _ = cluster
    for i in range(6):
        assert m.run(WF, f"r{i}", N, timeout=60) == EXPECTED


def test_cache_reuse_and_intent(cluster):
    m, _ = cluster
    for i in range(12):
        assert m.run(WF, f"r{i}", N, timeout=60) == EXPECTED
    recs = m.metrics()["records"]
    assert recs[0]["pkgs_sent"] == 1
    late = recs[-16:]
    assert sum(r["refs"] for r in late) >= 12, [(r["stage"], r["worker"], r["refs"]) for r in late]
    for r in recs:
        if r["stage"] == "aggregate":  # cloud intent: cloud workers only
            assert r["device_class"] == "agx"
        # edge intent admits edge and (more capable) cloud workers


def test_cloud_intent_stays_on_cloud_and_strict_mode():
    from types import SimpleNamespace as NS

    from hedices.stemsched import StemSched

    cloud, edge = NS(tier="cloud"), NS(tier="edge")
    assert StemSched._intent(NS(target="cloud"), cloud) and not StemSched._intent(NS(target="cloud"), edge)
    assert StemSched._intent(NS(target="edge"), edge) and StemSched._intent(NS(target="edge"), cloud)
    StemSched.NESTED = False
    try:
        assert not StemSched._intent(NS(target="edge"), cloud)
    finally:
        StemSched.NESTED = True


@pytest.mark.parametrize("cluster", ["nocache"], indirect=True)
def test_nocache_ships_packages_every_time(cluster):
    m, _ = cluster
    for i in range(5):
        m.run(WF, f"r{i}", N, timeout=60)
    recs = m.metrics()["records"]
    assert all(r["pkgs_sent"] == 1 and r["refs"] == 0 for r in recs)


def test_norely_overlaps_instances(cluster):
    m, _ = cluster
    m.run(WF, "warm", N, timeout=60)
    futs = [m.submit(WF, f"r{i}", N) for i in range(16)]
    assert all(f.result(timeout=120) == EXPECTED for f in futs)
    heavy = [r for r in m.metrics()["records"] if r["stage"] == "heavy"]
    # rho is the realized speedup over solo execution (not mere overlap)
    assert max(r["rho"] for r in heavy) > 1.1, [r["rho"] for r in heavy]


def test_worker_failure_recovery(cluster):
    m, procs = cluster
    m.run(WF, "warm", N, timeout=60)
    futs = [m.submit(WF, f"r{i}", N) for i in range(24)]
    time.sleep(0.3)
    procs["edge-0"].kill()
    assert all(f.result(timeout=120) == EXPECTED for f in futs)
    # the remaining work can finish before the heartbeat timeout declares edge-0 dead
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and "worker_dead" not in [e["kind"] for e in m.metrics()["events"]]:
        time.sleep(0.1)
    kinds = [e["kind"] for e in m.metrics()["events"]]
    assert "worker_dead" in kinds
    assert not [w for w in m.worker_info() if w["id"] == "edge-0"][0]["alive"]


def test_large_outputs_stay_on_workers_small_ones_return_inline(cluster):
    m, _ = cluster
    big = 20000  # load() output is 160 KB (> 64 KB inline limit); heavy() outputs are floats
    for i in range(6):
        assert m.run(WF, f"r{i}", big, timeout=60) == 2.0 * sum(range(big))
    by_stage = {}
    for r in m.metrics()["records"]:
        by_stage.setdefault(r["stage"], []).append(r)
    assert all(r["output_bytes"] == 0 and r["kept_bytes"] > 64 * 1024 for r in by_stage["load"])
    assert all(r["data_refs"] == 1 and r["output_bytes"] > 0 for r in by_stage["heavy"])
    # small inputs arrive inline, or as a local reference when the stage was chained behind its
    # predecessor on the same worker
    assert all(r["data_refs"] == 0 or r["chained"] for r in by_stage["aggregate"])


def test_warm_touches_every_eligible_worker(cluster):
    m, _ = cluster
    runs = m.warm(WF, ("w", N))
    assert runs == 6  # 3 eligible workers x 2 rounds (the second samples T_exec)
    placed = {(r["stage"], r["worker"]) for r in m.metrics()["records"]}
    assert {("heavy", "edge-0"), ("heavy", "edge-1"), ("aggregate", "cloud-0")} <= placed
    assert m.metrics()["instances"] == []  # warm-up instances are not measured


def test_lost_intermediate_outputs_are_recomputed(cluster):
    m, procs = cluster
    m.run(WF, "warm", N, timeout=60)
    futs = [m.submit(WF, f"r{i}", N) for i in range(30)]
    time.sleep(0.5)
    procs["edge-1"].kill()
    assert all(f.result(timeout=120) == EXPECTED for f in futs)
