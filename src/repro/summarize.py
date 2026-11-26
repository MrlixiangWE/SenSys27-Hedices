"""repro/summarize.py RESULTS: paper-vs-measured table of a repro/run_all.sh run.

Fig 6 and Fig 10 are computed from the per-round result files (fig6_r*/e6, fig10_r*/e10); the other
figures from the report each step's experiments.analyze wrote (<step>/report/summary.json).  Missing
steps are reported as such.  The reference values are the paper's (AGENTS.md section 4)."""
import glob
import json
import os
import statistics
import sys

R = sys.argv[1] if len(sys.argv) > 1 else "results/repro"
WL = ["W1", "W2", "W3", "W4"]
MODES = ["alpha0", "static", "rr", "nocache"]
PAPER = {
    "fig6": {"alpha0": [4, 7, 6, 6], "static": [10, 18, 12, 12], "rr": [24, 26, 14, 19], "nocache": [47, 47, 26, 32]},
    "fig10": {"ray": [1.55, 1.47, 1.41, 1.44], "dask": [1.77, 1.69, 1.58, 1.62],
              "kubeedge": [1.71, 1.65, 1.53, 1.57], "knative": [1.88, 1.79, 1.67, 1.71]},
    "fig9": [16, 17, 18, 13],
    "table3": {"hedices": [12.4, 18.3, 24.1, 19.1], "ray": [27.8, 36.7, 42.3, 35.8], "dask": [33.1, 42.8, 49.0, 41.7],
               "kubeedge": [39.2, 47.1, 52.8, 45.9], "knative": [31.0, 40.9, 47.6, 38.8]},
}


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def mean_ms(path):
    r = load(path)
    return 1e3 * r["summary"]["mean"] if r and r.get("summary") else None


def report(step):
    return load(os.path.join(R, step, "report", "summary.json"))


def first_report(*steps):
    """The report of the first step directory that has one (the per-figure steps, or the bexp campaign)."""
    for s in steps:
        rep = report(s)
        if rep:
            return rep
    return None


def fmt(v, nd=1):
    return "-" if v is None else f"{v:.{nd}f}"

SYS = ["hedices", "ray", "dask", "kubeedge", "knative"]
SYS_NAME = {"hedices": "Hedices", "ray": "Ray", "dask": "Dask", "kubeedge": "KubeEdge", "knative": "Knative"}
MODE_NAME = {"full": "full", "alpha0": "α=0", "static": "static", "rr": "RR", "nocache": "nocache"}
# Fig 7, share of edge-stage dispatches on the AGX, % (the paper's heatmap, AGX column)
PAPER_FIG7 = {"full": [6.33, 8.77, 9.27, 6.34], "alpha0": [19.08, 11.56, 11.84, 11.35], "static": [61.31, 60.16, 72.22, 55.23],
              "rr": [20.86, 21.80, 19.69, 22.35], "nocache": [66.88, 64.02, 76.61, 64.74]}
PAPER_FIG9 = [16.4, 16.9, 18.1, 13.4]


def table(head, rows, first=10, width=16):
    """An aligned table: a label column, then one right-aligned column per heading."""
    print("   " + " " * first + "".join(f"{h:>{width}}" for h in head))
    for label, cells in rows:
        print("   " + f"{label:<{first}}" + "".join(f"{c:>{width}}" for c in cells))


WL_NAME = {"W1": "W1 Recognition", "W2": "W2 Tracking", "W3": "W3 Detection", "W4": "W4 Fusion"}


def _rolling(v, k):
    h = k // 2
    return [sum(v[max(0, i - h): i + h + 1]) / len(v[max(0, i - h): i + h + 1]) for i in range(len(v))]


def _p95_nearest(v):
    s = sorted(v)
    return s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))] if s else None


def _tukey_keep(v, k=3.0):
    if len(v) < 4:
        return list(v)
    s = sorted(v)
    q1, q3 = s[int(0.25 * (len(s) - 1))], s[int(round(0.75 * (len(s) - 1)))]
    return [x for x in v if x <= q3 + k * (q3 - q1)]


def cell(v, ref=None, nd=1, refnd=None, sign=False, unit=""):
    """'value (paper)' with fixed decimals; '-' for a missing value."""
    if v is None:
        txt = "-"
    else:
        txt = f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"
        txt += unit
    if ref is not None:
        rnd = nd if refnd is None else refnd
        txt += f" ({ref:+.{rnd}f}{unit})" if sign else f" ({ref:.{rnd}f}{unit})"
    return txt



def fig6():
    rounds = sorted(glob.glob(os.path.join(R, "fig6_r*")))
    print(f"\n== Fig 6: mean latency increase over full, % ({f'rounds: {len(rounds)}; ' if rounds else ''}paper in brackets)")
    if not rounds:
        print("   not run")
        return
    rows = []
    for i, w in enumerate(WL):
        full = [mean_ms(os.path.join(d, "e6", f"hedices_full_{w}.json")) for d in rounds]
        cells = [" / ".join(fmt(f) for f in full)]
        for m in MODES:
            inc = []
            for d, f in zip(rounds, full):
                v = mean_ms(os.path.join(d, "e6", f"hedices_{m}_{w}.json"))
                if v is not None and f:
                    inc.append(100 * (v - f) / f)
            cells.append((" / ".join(f"{x:+.1f}" for x in inc) or "-") + f" ({PAPER['fig6'][m][i]:+d})")
        rows.append((w, cells))
    wide = max(16, max(len(c) for _, cs in rows for c in cs) + 2)
    table(["full (ms)"] + [MODE_NAME[m] for m in MODES], rows, first=4, width=wide)


def fig10():
    rounds = sorted(glob.glob(os.path.join(R, "fig10_r*"))) or sorted(glob.glob(os.path.join(R, "bexp_r*")))
    print("\n== Fig 10: end-to-end latency relative to Hedices, the bar labels"
          + (f" (median of {len(rounds)} round{'s' if len(rounds) != 1 else ''}; paper in brackets)" if rounds else " (paper in brackets)"))
    if not rounds:
        print("   not run")
        return
    med = {}
    for s in ["hedices", *PAPER["fig10"]]:
        for w in WL:
            v = [x for x in (mean_ms(os.path.join(d, "e10", f"{s}_{w}.json")) for d in rounds) if x is not None]
            med[(s, w)] = statistics.median(v) if v else None
    rows = []
    for i, w in enumerate(WL):
        h = med[("hedices", w)]
        rows.append((w, [f"1.00× ({h:.1f} ms)" if h else "-"]
                     + [cell(med[(s, w)] / h if med[(s, w)] and h else None, ref[i], nd=2, unit="×") for s, ref in PAPER["fig10"].items()]))
    table(["Hedices"] + [SYS_NAME[s] for s in PAPER["fig10"]], rows, first=4, width=16)


def fig11():
    rep = first_report("fig10_r1", "bexp_r1")
    d = (rep or {}).get("fig11_breakdown_ms")
    print("\n== Fig 11: latency breakdown, ms -- one block per panel, a row per bar")
    if not d or not d.get("hedices"):
        print("   not run")
        return
    parts = ["Pack", "Parse", "Transfer", "Control", "Execute"]
    for w in WL:
        h = d["hedices"].get(w)
        if not h:
            continue
        rows = []
        for s in ["hedices", "ray", "dask", "kubeedge", "knative"]:
            x = (d.get(s) or {}).get(w)
            if x:
                rows.append((SYS_NAME[s], [f"{x[k]:.2f}" for k in parts] + [f"{sum(x.values()):.1f}"]))
        print(f"   {WL_NAME[w]}   (Hedices Transfer {100 * h['Transfer'] / (sum(h.values()) or 1):.1f} % of the bar; paper about 5 %)")
        table(parts + ["Total"], rows, first=10, width=10)

def fig7():
    # Fig 7's own run only: its report, or (a run under way, an older report) its measurement files directly
    rep = load(os.path.join(R, "fig7", "report", "summary.json")) or {}
    try:
        cfg = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments", "configs",
                                          "phys6r.json"))).get("e7", {})
    except (OSError, ValueError):
        cfg = {}
    shares = rep.get("fig7_dispatch_share_pct") or {}
    pooled = (rep.get("fig7_dispatch_share_pooled_pct") or {}).get("share") or {}
    sweep = (rep.get("fig7_dispatch_share_pooled_pct") or {}).get("sweep") or []
    edge = {"W1": ("load_signal", "preproc", "recognize", "decode_states"),
            "W2": ("read_frame", "preprocess", "retrack", "track", "track_update"),
            "W3": ("load_image", "preprocess", "infer"),
            "W4": ("preproc_frame", "detect_objects", "to_boxes", "smooth_imu", "imu_features")}
    if not shares:
        for m in ["full", "alpha0", "static", "rr", "nocache"]:
            for w in WL:
                r = load(os.path.join(R, "fig7", "e7", f"hedices_{m}_{w}.json")) or {}
                rec = [x for x in r.get("records", []) if x.get("status") == "ok" and x["stage"].split(".")[-1] in edge[w]]
                if rec:
                    shares[f"{m}/{w}"] = {"agx": 100 * sum(1 for x in rec if x.get("worker") == "agx-0") / len(rec)}
    if not pooled:  # a report written before the pooled shares were part of it, or none yet: pool the sweep here
        dirs = sorted(glob.glob(os.path.join(R, "fig7", "c*")))
        # a run under way (no report yet): a workload's pooled share only once its whole sweep is there -- the low
        # levels alone leave the AGX idle and would read 0 where the figure will show the share over every level
        def whole(m, w):
            if rep:
                return True
            c = cfg.get(w) or {}
            need = [f"c{k}" + (f"r{r}" if r > 1 else "") for k in (c.get("load_sweep") or [2, 3, 4, 5])
                    for r in range(1, int(c.get("load_sweep_runs", 1)) + 1)]
            return all(os.path.exists(os.path.join(R, "fig7", d, "e7", f"hedices_{m}_{w}.json")) for d in need)
        for m in ["full", "alpha0"]:
            for w in WL:
                if not whole(m, w):
                    continue
                a = n = 0
                for d in dirs:
                    r = load(os.path.join(d, "e7", f"hedices_{m}_{w}.json")) or {}
                    rec = [x for x in r.get("records", []) if x.get("status") == "ok" and x["stage"].split(".")[-1] in edge[w]]
                    a += sum(1 for x in rec if x.get("worker") == "agx-0"); n += len(rec)
                if n:
                    pooled[f"{m}/{w}"] = {"agx": 100 * a / n}
        sweep = [os.path.basename(d) for d in dirs]
    # the sweep's levels: from the run when it is complete, else (under way) the levels configured for it
    lv = sorted({int(d[1:].split("r")[0]) for d in sweep if d[1:].split("r")[0].isdigit()})
    if not rep:
        lv = sorted({k for w in WL for k in ((cfg.get(w) or {}).get("load_sweep") or [2, 3, 4, 5])})
    print("\n== Fig 7: share of edge-stage dispatches on the AGX, % -- the AGX column of each panel (paper in brackets)")
    if lv and len(lv) > 1:
        print(f"   full and α=0 pooled over the load sweep, concurrency {lv[0]}–{lv[-1]}")
    if not shares and not pooled:
        print("   not run")
        return
    modes = ["full", "alpha0", "static", "rr", "nocache"]
    # full and α=0 are the load sweep's numbers (the plot's); a single run of them is not what the figure shows
    use_pool = bool(pooled or glob.glob(os.path.join(R, "fig7", "c*")) or os.path.isdir(os.path.join(R, "fig7", "e7")))
    src = {m: pooled if m in ("full", "alpha0") and use_pool else shares for m in modes}
    # laid out as the figure: a row per workload, the five panels from left to right
    rows = [(w, [cell((src[m].get(f"{m}/{w}") or {}).get("agx"), PAPER_FIG7[m][i], nd=2) for m in modes])
            for i, w in enumerate(WL)]
    table([MODE_NAME[m] for m in modes], rows, first=4, width=15)


def fig8():
    """The plotted curves: sigma and alpha smoothed over the dispatches (experiments.analyze's rule), the phases."""
    rep = report("fig8")
    print("\n== Fig 8: cache instability σ(t) and alignment weight α(t) over the dispatches (cold start)")
    r = load(os.path.join(R, "fig8", "e8", "hedices_full_MIX_cold.json"))
    if not r:
        f = (rep or {}).get("fig8")
        print("   not run" if not isinstance(f, dict) else "   " + json.dumps(f))
        return
    recs = sorted((x for x in r["records"] if not x["repair"]), key=lambda x: x["dispatch"])
    sig, alp = [x["sigma"] for x in recs], [x["alpha"] for x in recs]
    n = len(recs)
    k = max(9, n // 25) | 1
    ss, sa = _rolling(sig, k), _rolling(alp, k)
    ipk = max(range(n), key=lambda i: sa[i])
    t1 = next((i for i in range(ipk, n) if sa[i] < 0.9 * sa[ipk]), ipk)
    speak = max(ss) or 1.0
    t2 = next((i for i in range(t1, n) if all(x < 0.15 * speak for x in ss[i:])), n)
    table(["cultivation", "decay", "exploitation"], [("dispatches", [f"0–{t1}", f"{t1}–{t2}", f"{t2}–{n}"])], first=12, width=14)
    print()
    table(["σ(t)", "α(t)"], [("peak", [f"{max(ss):.2f}", f"{max(sa):.2f}"]),
                             ("last 50", [f"{statistics.fmean(ss[-50:]):.2f}", f"{statistics.fmean(sa[-50:]):.2f}"]),
                             ("raw max", [f"{max(sig):.2f}", f"{max(alp):.2f}"])], first=12, width=10)
    print(f"   curves: moving average over {k} dispatches, as plotted")


def fig9():
    """From the runs (fig9/e9) as they are written: per workload the groups measured both without and with @NoRely
    so far, the means over them and the reduction -- the bars and labels of the plot (experiments.analyze's rule)."""
    print("\n== Fig 9: end-to-end latency without / with @NoRely, and the reduction (paper in brackets)")
    d = os.path.join(R, "fig9", "e9")
    rows, ngroups = [], 0
    for i, w in enumerate(WL):
        def runs(label):
            out = {}
            for f in glob.glob(os.path.join(d, f"hedices_{label}_{w}_r*.json")):
                if not f.endswith(".net.json"):
                    v = mean_ms(f)
                    if v is not None:
                        out[os.path.basename(f).rsplit("_r", 1)[1][:-5]] = v
            return out
        off, on = runs("norelyoff"), runs("norelyon")
        both = sorted(set(off) & set(on))
        if not both:
            if off or on:
                rows.append((w, ["-", "-", "-"]))
            continue
        ngroups = max(ngroups, len(both))
        a, b = statistics.fmean(off[k] for k in both), statistics.fmean(on[k] for k in both)
        rows.append((w, [f"{a:.1f} ms", f"{b:.1f} ms", cell(100 * (a - b) / a, PAPER_FIG9[i], nd=1, unit=" %")]))
    if not rows:
        rep = report("fig9")
        f = (rep or {}).get("fig9")
        if not isinstance(f, dict):
            print("   not run")
            return
        rows = [(w, [f"{f['off_ms'][i]:.1f} ms", f"{f['on_ms'][i]:.1f} ms", cell(f["reduction_pct"][i], PAPER_FIG9[i], nd=1, unit=" %")])
                for i, w in enumerate(WL)]
    table(["@NoRely off", "@NoRely on", "reduction"], rows, first=4, width=19)
    if ngroups:
        print(f"   mean over {ngroups} group{'s' if ngroups != 1 else ''} measured both ways")


def fig12():
    """The plotted points: per panel, the P95 at each burst size q (far outliers dropped, experiments.analyze's
    rule) divided by the largest P95 in that panel."""
    d = None
    for step in ("fig12", "bexp_r1"):
        res = {s: load(os.path.join(R, step, "e12", f"{s}_mixed.json")) for s in SYS}
        res = {s: r for s, r in res.items() if r}
        if res:
            d = {}
            for s, r in res.items():
                d[s] = {}
                for w in WL:
                    d[s][w] = {}
                    for q, cellq in r["bursts"].items():
                        c = cellq.get(w) or {}
                        keep = _tukey_keep(c.get("latency_s") or [])
                        d[s][w][int(q)] = _p95_nearest(keep) if keep else c.get("p95")
            break
    print("\n== Fig 12: P95 latency at each burst size q, normalized by the largest P95 in the panel (the plotted points)")
    if not d:
        print("   not run")
        return
    qs = sorted({q for s in d for w in WL for q in d[s][w]})
    for w in WL:
        vals = [v for s in d for v in d[s][w].values() if v]
        if not vals:
            continue
        top = max(vals)
        print(f"   ({'abcd'[WL.index(w)]}) {WL_NAME[w]}   (1.00 = {1e3 * top:.0f} ms)")
        table([f"q={q}" for q in qs], [(SYS_NAME[s], [f"{d[s][w][q] / top:.2f}" if d[s][w].get(q) else "-" for q in qs])
                                      for s in SYS if s in d], first=10, width=7)


def _recovery(r):
    """One failure trial's recovery time, s (experiments.analyze's rule)."""
    t = r["injection"]["t"]
    ends = [w + dt for _, w, dt in r["instances_wall"] if w <= t < w + dt]
    val = max(ends) - t if ends else None
    done = sorted(w + dt for _, w, dt in r["instances_wall"])
    before = [b - a for a, b in zip(done, done[1:]) if b <= t]
    normal = sorted(before)[len(before) // 2] if before else 0.0
    after = [t] + [x for x in done if x > t]
    bound = float(r.get("recover_timeout_s", 120.0))
    stalls = [(b - a, b) for a, b in zip(after, after[1:]) if a <= t + bound]
    if stalls:
        gap, end = max(stalls)
        if gap > max(1.0, 10 * normal):
            val = max(val or 0.0, end - t)
    # an instance sent after the fault that was still unfinished when the load would have ended (the run's
    # measurement window; a run that takes several failures records each one's window) stalled on it: recovery
    # lasts until it completes.  With a run that stops its load at the end of the window this is the stall the
    # rule above sees in the completions; with a load that goes on it is not visible there.
    win = r["injection"].get("window_s", r["measure_window"][1] - t)
    lat = sorted(dt for _, w, dt in r["instances_wall"] if w + dt <= t)
    slow = max(1.0, 10 * lat[len(lat) // 2]) if lat else 1.0
    late = [w + dt for _, w, dt in r["instances_wall"] if t <= w <= t + bound and dt > slow and w + dt > t + win]
    if late:
        val = max(val or 0.0, max(late) - t)
    if any(f[0] >= t for f in r.get("failed", ())):
        val = max(val or 0.0, bound)
    return val


def fig13():
    # from the trials (fig13/e13, else bexp_r1/e13) as each is written, else the report
    d = {}
    for step in ("fig13", "bexp_r1"):
        for f in sorted(glob.glob(os.path.join(R, step, "e13", "*_trial*.json"))):
            r = load(f)
            if r and "injection" in r:
                v = _recovery(r)
                if v is not None:
                    d.setdefault(r["system"], []).append(v)
        if d:
            break
    if not d:
        d = (first_report("fig13", "bexp_r1") or {}).get("fig13_recovery_s")
    print("\n== Fig 13: recovery time after a worker failure, s -- where each curve crosses the dashed 0.5 line (median), "
          "its ends, and every failure")
    if not isinstance(d, dict):
        print("   not run")
        return
    rows = []
    for s in SYS:
        v = sorted(x for x in d.get(s, []) if x is not None)
        if v:
            rows.append((SYS_NAME[s], [f"{statistics.median(v):.2f}", f"{v[0]:.2f}", f"{v[-1]:.2f}", "  ".join(f"{x:.2f}" for x in v)]))
    head = ["median", "min", "max"]
    print("   " + " " * 10 + "".join(f"{h:>9}" for h in head) + "   every failure (sorted)")
    for label, c in rows:
        print("   " + f"{label:<10}" + "".join(f"{x:>9}" for x in c[:3]) + "   " + c[3])


def dask_stage_io(rnd, w):
    """Dask's application data, KB per instance, counted at the stage boundaries: the arguments and result of
    every stage call (its stage-call trace over the deployment session).  Dask runs the consecutive stages of a
    tier as one task, so the data between them never reaches the network counters; this counts it per stage,
    as the stage graph moves it."""
    tr, ses, res = (os.path.join(rnd, "t3s", f"dask_{w}{x}") for x in (".trace.jsonl", ".session.json", ".json"))
    try:
        s = json.load(open(ses))
        (ta, _), (tb, _) = s["samples"][0], s["samples"][-1]
        n = len(json.load(open(res))["latency_s"])
        calls = [json.loads(l) for l in open(tr) if l.strip()]
    except (OSError, ValueError, KeyError, IndexError):
        return None
    calls = [c for c in calls if c["out"][1] > 0 and ta <= c["t"] <= tb]  # stage calls (model loaders: no data)
    return sum(sum(sz for d, sz in c["in"] if d) + c["out"][1] for c in calls) / 1024 / max(n, 1)


def table3():
    # Table 4: all network traffic per instance in steady state (framework messages and the application data
    # that crosses nodes; e10 phase of every bexp round, median), and the ratio to Hedices
    # the bexp rounds, else the table3 step on its own (the same two phases), else the fig10 rounds (no session)
    rounds = sorted(glob.glob(os.path.join(R, "bexp_r*"))) or \
        [d for d in [os.path.join(R, "table3")] if os.path.isdir(os.path.join(d, "e10"))]
    reps = [r for r in (load(os.path.join(d, "report", "summary.json")) for d in rounds) if r]
    reps = reps or [r for r in [first_report("fig10_r1")] if r]
    wire = {}
    for rep in reps:
        for s, ws in ((rep.get("table3") or {}).get("measured") or {}).items():
            for w, x in ws.items():
                if x.get("wire_kb_per_instance") is not None:
                    wire.setdefault((s, w), []).append(x["wire_kb_per_instance"])
    for rnd in rounds:  # Dask: plus its application data counted at the stage boundaries (dask_stage_io)
        for w in WL:
            io = dask_stage_io(rnd, w)
            if io is not None:
                wire.setdefault(("dask_io", w), []).append(io)
    for w in WL:
        if ("dask", w) in wire and ("dask_io", w) in wire:
            wire[("dask", w)] = [statistics.median(wire[("dask", w)]) + statistics.median(wire[("dask_io", w)])]
    print("\n== Table 4: network traffic per instance in steady state, KB, and its ratio to Hedices (paper in brackets)"
          "\n   Dask: plus its application data counted at the stage boundaries (its fused tier tasks keep it off the wire)")
    if not wire:
        print("   not run")
        return
    # laid out as the paper's table: a row per workload, a column per system
    names = {"W1": "W1 Recognition", "W2": "W2 Tracking", "W3": "W3 Detection", "W4": "W4 Fusion"}
    systems = list(PAPER["table3"])
    val = {(s, w): statistics.median(wire[(s, w)]) if (s, w) in wire else None for s in systems for w in WL}
    if all(v is None for v in val.values()):  # only Dask's stage data so far (a run under way): nothing to show yet
        print("   not run")
        return
    rows = [(names[w], [cell(val[(s, w)], PAPER["table3"][s][i], nd=1) for s in systems]) for i, w in enumerate(WL)]
    table([SYS_NAME[s] for s in systems], rows, first=15, width=14)
    print("\n   ratio to Hedices (paper in brackets)")
    ratio = [(names[w], [cell(val[(s, w)] / val[("hedices", w)] if val[(s, w)] is not None and val[("hedices", w)] else None,
                              PAPER["table3"][s][i] / PAPER["table3"]["hedices"][i], nd=1, unit="×") for s in systems[1:]])
             for i, w in enumerate(WL)]
    table([SYS_NAME[s] for s in systems[1:]], ratio, first=15, width=14)
    rep = first_report("bexp_r1", "table3")
    d = (rep or {}).get("table3_session")
    print("\n   framework traffic only, over a deployment session, KB (paper in brackets)")
    if not isinstance(d, dict):
        print("   not run")
        return
    table([SYS_NAME[s] for s in systems],
          [(names[w], [cell((d.get(s, {}).get(w) or {}).get("deploy_session_kb_per_instance"), PAPER["table3"][s][i], nd=1)
                       for s in systems]) for i, w in enumerate(WL)], first=15, width=14)


if __name__ == "__main__":
    print(f"results: {R}")
    for fn in (fig6, fig7, fig8, fig9, fig10, fig11, table3, fig12, fig13):  # the paper's order
        fn()
