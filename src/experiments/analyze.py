"""Turn raw results into the paper's figures/tables and a comparison summary.

  python -m experiments.analyze results/<run>

Writes <run>/report/{fig*.png, summary.json, summary.md}.  Everything reported
is computed from the measured JSON; systems that were not run are marked so.
"""
from __future__ import annotations

import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import logging  # noqa: E402

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)  # Times New Roman falls back to Liberation Serif
import matplotlib.pyplot as plt  # noqa: E402

# HEDICES_FIG_DPI / HEDICES_REPORT_DIR: the live dashboard renders sharper copies of the plots into a cache of its own;
# unset, the report is written to <run>/report at 300 dpi.
DPI = int(os.environ.get("HEDICES_FIG_DPI") or 300)
# HEDICES_FIG_FORMATS: extra vector copies next to each PNG (e.g. "pdf"); fonts embedded as TrueType (no Type 3)
FORMATS = [f.strip().lower() for f in (os.environ.get("HEDICES_FIG_FORMATS") or "").split(",") if f.strip()]

ROOT = Path(__file__).resolve().parent.parent
WL = ["W1", "W2", "W3", "W4"]
MODES = ["full", "alpha0", "static", "rr", "nocache"]
CLASSES = ["nx", "pi5", "agx"]
# the paper's figures: one five-colour palette for the five modes (Fig 6) and the five systems (Fig 10-13)
PAL5 = ["#e87d7d", "#78be7a", "#f2a65a", "#6ea8e6", "#c89ac8"]
BASELINES = ["ray", "dask", "kubeedge", "knative"]
SYSTEMS = ["hedices", *BASELINES]
SYS_COLOR = dict(zip(SYSTEMS, PAL5))
SYS_LABEL = {"hedices": "Hedices", "ray": "Ray", "dask": "Dask", "kubeedge": "KubeEdge", "knative": "Knative"}
SYS_SHORT = {"hedices": "Hed", "ray": "Ray", "dask": "Dsk", "kubeedge": "KbE", "knative": "Knv"}
SYS_MARK = {"hedices": "o", "ray": "s", "dask": "^", "kubeedge": "d", "knative": "v"}
INK, INK2 = "#000000", "#2a2a2a"
EDGE, ERR = "#5f5f5f", "#444444"  # bar outlines, error bars

# Paper (SenSys '27 final) numbers, for side-by-side comparison only.
PAPER = {
    "fig6_increase_pct": {"alpha0": [3.9, 6.2], "static": [9.5, 16.8], "rr": [14.8, 27.6],
                          "nocache": [24.8, 48.9]},
    "fig9_reduction_pct": [13.4, 18.1],
    "fig10_norm": {"ray": [1.41, 1.55], "dask": [1.58, 1.77], "kubeedge": [1.53, 1.71],
                   "knative": [1.67, 1.88]},
    "fig10_avg_reduction_pct": {"ray": 31.9, "dask": 39.9, "kubeedge": 38.1, "knative": 43.3},
    "table3_kb": {"hedices": [12.4, 18.3, 24.1, 19.1], "ray": [27.8, 36.7, 42.3, 35.8],
                  "dask": [33.1, 42.8, 49.0, 41.7], "kubeedge": [39.2, 47.1, 52.8, 45.9],
                  "knative": [31.0, 40.9, 47.6, 38.8]},
}

# Figure style of the paper: single-column figures (about 3.3 in wide) in Times for the bar / heatmap / line figures
# (Fig 6-11), DejaVu Sans for Fig 12 / 13; open top and right spines, dotted light-grey grid behind the data.
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "text.color": INK,
    "axes.linewidth": 0.8, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#b0b0b0", "grid.linestyle": ":", "grid.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "legend.frameon": False, "mathtext.fontset": "stix", "hatch.linewidth": 0.5,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
SERIF = {"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
         "font.size": 8.2}
SANS = {"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 6.7}


def load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def mean_ms(res) -> float | None:
    return 1e3 * statistics.fmean(res["latency_s"]) if res and res.get("latency_s") else None


def pct(a, b):
    return None if a is None or b is None else 100.0 * (a - b) / b


# ---------------------------------------------------------------- plotting helpers
WL_TICK = {"W1": "W1\nRecog.", "W2": "W2\nTrack", "W3": "W3\nDetect", "W4": "W4\nFusion"}
WL_TITLE = {"W1": "W1 Recog.", "W2": "W2 Track", "W3": "W3 Detect", "W4": "W4 Fusion"}
WL_CAPTION = {"W1": "(a) W1 Recognition", "W2": "(b) W2 Tracking", "W3": "(c) W3 Detection", "W4": "(d) W4 Fusion"}
MODE_LABEL = {"full": "full", "alpha0": "α=0", "static": "static", "rr": "RR", "nocache": "nocache"}
MODE_COLORS = PAL5  # full, α=0, static, RR, nocache


def plain_log(ax, axis="y") -> None:
    """Log axis labelled 0.1 / 1 / 10 / 100 instead of 10^k mantissas."""
    from matplotlib.ticker import FuncFormatter, NullFormatter
    a = ax.yaxis if axis == "y" else ax.xaxis
    a.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    a.set_minor_formatter(NullFormatter())


def ci95_ms(res) -> float | None:
    """Half-width of the 95 % confidence interval of the mean latency, from the instances."""
    v = res.get("latency_s") if res else None
    return 1e3 * 1.96 * statistics.stdev(v) / len(v) ** 0.5 if v and len(v) > 1 else None


def grouped(ax, groups, series, values, colors, labels, err=None, width=0.78, ecolor=ERR, capsize=1.8):
    """Grouped bars (values[s][g]) in the paper's style -- adjacent bars with a thin grey outline, optional symmetric
    error bars; returns the bar x positions."""
    n = len(series)
    w = width / n
    pos = {}
    for i, s in enumerate(series):
        xs = [g + (i - (n - 1) / 2) * w for g in range(len(groups))]
        ys = [v if v is not None else 0 for v in values[s]]
        e = [x or 0 for x in err[s]] if err and s in err else None
        ax.bar(xs, ys, w, color=colors[i], label=labels[i], edgecolor=EDGE, linewidth=0.4, zorder=2,
               yerr=e, error_kw={"elinewidth": 0.8, "capsize": capsize, "capthick": 0.8, "ecolor": ecolor})
        pos[s] = xs
    ax.set_xticks(range(len(groups)), groups)
    ax.tick_params(axis="x", length=0)
    return pos


def top_legend(fig_or_ax, handles, labels, ncol, fontsize, y=1.0, **kw):
    """The paper's legend: one row of entries centred above the plot, no frame."""
    return fig_or_ax.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, y), ncol=ncol, fontsize=fontsize,
                            handlelength=kw.pop("handlelength", 1.2), handletextpad=0.35,
                            columnspacing=kw.pop("columnspacing", 0.7), borderaxespad=0.1, **kw)


def finish(fig, path):
    """Save the figure (no title or note on it: the caption carries them, as in the paper) and close it."""
    fig.savefig(path, dpi=DPI)
    for ext in FORMATS:
        if ext != "png":
            fig.savefig(Path(path).with_suffix("." + ext), dpi=DPI)
    plt.close(fig)


def rolling(v, k):
    """Centred moving average over k points (shorter at the ends)."""
    h = k // 2
    out = []
    for i in range(len(v)):
        seg = v[max(0, i - h): i + h + 1]
        out.append(sum(seg) / len(seg))
    return out


def p95_nearest(v):
    """Same rank rule as experiments.common.summary."""
    s = sorted(v)
    return s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))] if s else None


def tukey_keep(v, k=3.0):
    """Samples within Q3 + k*IQR (far outliers dropped; needs at least 4 samples)."""
    if len(v) < 4:
        return list(v)
    s = sorted(v)
    q1, q3 = s[int(0.25 * (len(s) - 1))], s[int(round(0.75 * (len(s) - 1)))]
    return [x for x in v if x <= q3 + k * (q3 - q1)]


def _edge_stages() -> dict:
    try:
        import importlib

        from hedices import Workflow

        out = {}
        for w in WL:
            mod = importlib.import_module(f"apps.hedices.{w.lower()}")
            out[w] = [sp.name for sp in Workflow(mod.workflow, name=w).stages() if sp.target == "edge"]
        return out
    except Exception:
        return {}


EDGE_STAGES = _edge_stages()


def e6cold(root: Path):
    """alpha0 vs full from empty caches (e6cold): per workload, the mean over repetitions of the
    per-repetition increase of alpha0 over full."""
    d = root / "e6cold"
    if not d.exists():
        return None
    res = {"full_ms": [], "alpha0_ms": [], "increase_pct": [], "reps": []}
    for w in WL:
        pairs = []
        for f in sorted(d.glob(f"hedices_full_{w}_r*.json")):
            if f.name.endswith(".net.json"):
                continue
            g = d / f.name.replace("_full_", "_alpha0_")
            a, b = mean_ms(load(f)), mean_ms(load(g))
            if a and b:
                pairs.append((a, b))
        res["reps"].append(len(pairs))
        res["full_ms"].append(statistics.fmean(a for a, _ in pairs) if pairs else None)
        res["alpha0_ms"].append(statistics.fmean(b for _, b in pairs) if pairs else None)
        res["increase_pct"].append(statistics.fmean(pct(b, a) for a, b in pairs) if pairs else None)
    return res


def e6_result(root: Path, m: str, w: str):
    """Fig. 6/7 run of mode m: the co-deployed stream (e6mix) restricted to workload w when it
    exists, else the per-workload run (e6)."""
    mix = load(root / "e6mix" / f"hedices_{m}_MIX.json")
    if mix and w in mix.get("per_workload", {}):
        return {**mix["per_workload"][w], "summary": mix["per_workload"][w],
                "records": [r for r in mix.get("records", []) if r.get("workflow") == w]}
    return load(root / "e6" / f"hedices_{m}_{w}.json")


def fig6_7(root: Path, rep: Path, out: dict) -> None:
    lat = {m: [mean_ms(e6_result(root, m, w)) for w in WL] for m in MODES}
    if not any(v for vs in lat.values() for v in vs):
        out["fig6"] = "not run"
        if (root / "e7").exists():  # a Fig 7 run alone (plan e7)
            fig7_share(root, rep, out)
        return
    inc = {m: [pct(lat[m][i], lat["full"][i]) for i in range(4)] for m in MODES if m != "full"}
    cold = e6cold(root)
    if cold:  # alpha=0 over cache formation, against full over the same phase
        inc["alpha0"] = cold["increase_pct"]
    out["fig6"] = {"mean_latency_ms": lat, "increase_vs_full_pct": inc, "alpha0_cache_formation": cold,
                   "paper_increase_pct": PAPER["fig6_increase_pct"]}
    err = {m: [ci95_ms(e6_result(root, m, w)) for w in WL] for m in MODES}
    ymax = max(v + (e or 0) for m in MODES for v, e in zip(lat[m], err[m]) if v)
    with plt.rc_context(SERIF):  # the paper's Fig 6: bars = mean, whiskers = 95 % CI, label = increase over full
        fig, ax = plt.subplots(figsize=(3.3, 1.95))
        pos = grouped(ax, [WL_TICK[w] for w in WL], MODES, lat, MODE_COLORS, [MODE_LABEL[m] for m in MODES], err)
        for m in MODES[1:]:
            for i, x in enumerate(pos[m]):
                if inc[m][i] is not None and lat[m][i]:
                    ax.text(x, lat[m][i] + (err[m][i] or 0) + ymax * 0.01, f"{inc[m][i]:+.1f}%", rotation=90,
                            ha="center", va="bottom", fontsize=6.7, color=INK)
        ax.set_ylim(0, ymax * 1.36); ax.set_ylabel("End-to-end latency (ms)")
        h, l = ax.get_legend_handles_labels()
        top_legend(ax, h, l, ncol=5, fontsize=7.5, y=1.01)
        finish(fig, rep / "fig6_ablation.png")

    fig7_share(root, rep, out)


def fig7_share(root: Path, rep: Path, out: dict) -> None:
    """Fig 7 from its own run (plan e7, configs "e7") where there is one, else from the Fig 6 run."""
    share = {}
    for m in MODES:
        for w in WL:
            res = load(root / "e7" / f"hedices_{m}_{w}.json") or e6_result(root, m, w)
            if not res:
                continue
            # stages with a placement choice across worker classes (edge intent); cloud-intent
            # stages can only run on the cloud class and are not placement decisions
            edge = {sp.split(".")[-1] for sp in EDGE_STAGES.get(w, ())}
            c = Counter(r["device_class"] for r in res["records"] if r["status"] == "ok"
                        and (not edge or r["stage"].split(".")[-1] in edge))
            tot = sum(c.values()) or 1
            share[f"{m}/{w}"] = {k: 100.0 * c.get(k, 0) / tot for k in CLASSES}
    out["fig7_dispatch_share_pct"] = share
    # full / alpha0 pooled over the load sweep (<run>/c*/e7, the Fig 7 run's own concurrency levels), the
    # numbers the summary reports for these two modes
    pooled = {}
    sweep = sorted(p for p in root.glob("c*") if (p / "e7").is_dir())
    for m in ("full", "alpha0"):
        for w in WL:
            edge = {sp.split(".")[-1] for sp in EDGE_STAGES.get(w, ())}
            c = Counter()
            for s in sweep:
                r = load(s / "e7" / f"hedices_{m}_{w}.json")
                for x in (r or {}).get("records", ()):
                    if x["status"] == "ok" and (not edge or x["stage"].split(".")[-1] in edge):
                        c[x["device_class"]] += 1
            if sum(c.values()):
                pooled[f"{m}/{w}"] = {k: 100.0 * c.get(k, 0) / sum(c.values()) for k in CLASSES}
    if pooled:
        out["fig7_dispatch_share_pooled_pct"] = {"sweep": [s.name for s in sweep], "share": pooled}
    show = {**share, **pooled}
    if any(f"{m}/{w}" in show for m in MODES for w in WL):
        # the paper's Fig 7: one heatmap per mode (rows W1-W4, columns NX / Pi5 / AGX), magma colour map 0-100 %,
        # every cell labelled; full and alpha=0 from the load sweep where there is one (the summary's numbers)
        with plt.rc_context(SERIF):
            fig, axes = plt.subplots(1, 5, figsize=(6.6, 1.42))
            for k, (ax, m) in enumerate(zip(axes, MODES)):
                data = [[show.get(f"{m}/{w}", {}).get(c, float("nan")) for c in CLASSES] for w in WL]
                im = ax.imshow(data, cmap="magma_r", vmin=0, vmax=100, aspect="auto")
                ax.grid(False)
                for sp in ax.spines.values():
                    sp.set_visible(False)
                ax.set_xticks(range(3), ["NX", "Pi5", "AGX"], fontsize=8.4)
                ax.set_yticks(range(4), WL if k == 0 else [""] * 4, fontsize=8.4)
                ax.tick_params(axis="x", length=0, pad=2)
                ax.tick_params(axis="y", length=2.5, width=0.6, pad=1.5)
                for i, row in enumerate(data):
                    for j, v in enumerate(row):
                        if v == v:
                            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                                    color="white" if v > 50 else INK)
                ax.set_title(MODE_LABEL[m], fontsize=8.4, pad=2)
            fig.subplots_adjust(left=0.045, right=0.885, top=0.86, bottom=0.17, wspace=0.13)
            cb = fig.colorbar(im, cax=fig.add_axes([0.895, 0.17, 0.008, 0.69]), ticks=[0, 25, 50, 75, 100])
            cb.ax.tick_params(labelsize=8.4, length=2.5, width=0.6)
            cb.set_label("Dispatch share (%)", fontsize=8.4); cb.outline.set_linewidth(0.5)
            finish(fig, rep / "fig7_dispatch.png")


def fig8(root: Path, rep: Path, out: dict) -> None:
    res = {}
    for w in ["MIX", "W1", "W4"]:
        r = load(root / "e8" / f"hedices_full_{w}_cold.json")
        if r:
            recs = sorted((x for x in r["records"] if not x["repair"]), key=lambda x: x["dispatch"])
            res[w] = [(x["sigma"], x["alpha"]) for x in recs]
    if not res:
        out["fig8"] = "not run"
        return
    out["fig8"] = {w: {"dispatches": len(v), "sigma_max": max(s for s, _ in v),
                       "sigma_last50_mean": statistics.fmean(s for s, _ in v[-50:]),
                       "alpha_max": max(a for _, a in v)} for w, v in res.items()}
    # the paper's Fig 8: sigma(t) and alpha(t) as smoothed curves on twin axes over the dispatches, with the three
    # phases shaded and named at the top -- cultivation (alpha near its cap), decay, exploitation (sigma low)
    SIG, ALP = "#003f5c", "#bc5090"
    w, v = next(iter(res.items()))
    sig, alp = [s for s, _ in v], [a for _, a in v]
    n = len(v)
    k = max(9, n // 25) | 1                    # centred moving average over k dispatches
    step = max(1, n // 75)                     # drawn every step dispatches, as the paper's curve
    ss, sa = rolling(sig, k), rolling(alp, k)
    xs = list(range(0, n, step))
    if xs[-1] != n - 1:
        xs.append(n - 1)
    ipk = max(range(n), key=lambda i: sa[i])
    t1 = next((i for i in range(ipk, n) if sa[i] < 0.9 * sa[ipk]), ipk)
    speak = max(ss) or 1.0
    t2 = next((i for i in range(t1, n) if all(x < 0.15 * speak for x in ss[i:])), n)
    out["fig8_phases"] = {"smoothing_window": k, "cultivation_end": t1, "exploitation_start": t2}
    with plt.rc_context(SERIF):
        fig, ax = plt.subplots(figsize=(3.3, 1.45))
        ax2 = ax.twinx()
        for (a0, a1, name, col, tc, al) in ((0, t1, "cultivation", "#a6cee3", "#385e7a", 0.12),
                                            (t1, t2, "decay", "#ffe08a", "#876a25", 0.15),
                                            (t2, n, "exploitation", "#b2df8a", "#3a6f37", 0.15)):
            if a1 > a0:
                ax.axvspan(a0, a1, color=col, alpha=al, zorder=0, lw=0)
                ax.text((a0 + a1) / 2, 0.985, name, transform=ax.get_xaxis_transform(), ha="center", va="top",
                        fontsize=6.1, color=tc)
        ax.plot(xs, [ss[i] for i in xs], color=SIG, lw=1.3, zorder=3, label="σ(t)")
        ax2.plot(xs, [sa[i] for i in xs], color=ALP, lw=1.3, zorder=3, label="α(t)")
        # the paper's axes: σ up to just above its peak (ticks every 0.5), α on 0-0.5 (ticks every 0.1)
        from matplotlib.ticker import MultipleLocator
        ax.set_ylim(0, speak * 1.08); ax.yaxis.set_major_locator(MultipleLocator(0.5))
        ax2.set_ylim(0, 0.5 if max(sa) <= 0.5 else max(sa) * 1.05); ax2.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.set_xlim(0, n); ax.set_xlabel("Dispatch count")
        ax.grid(True, axis="both")
        ax.set_ylabel("σ(t) — cache instability", color=SIG, fontsize=7.8)
        ax2.set_ylabel("α(t) — alignment weight", color=ALP, fontsize=7.8)
        ax.tick_params(axis="y", colors=SIG); ax2.tick_params(axis="y", colors=ALP)
        ax2.grid(False); ax2.spines["right"].set_visible(True); ax2.spines["right"].set_color(ALP)
        ax2.spines["top"].set_visible(False)
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="center right", bbox_to_anchor=(0.985, 0.56), fontsize=7.5, handlelength=2.2)
        finish(fig, rep / "fig8_sigma_alpha.png")


def fig9(root: Path, rep: Path, out: dict) -> None:
    def runs(label, w):
        vals = [mean_ms(load(f)) for f in sorted((root / "e9").glob(f"hedices_{label}_{w}_r*.json"))
                if not f.name.endswith(".net.json")]
        if not vals and (r := load(root / "e9" / f"hedices_{label}_{w}.json")):
            vals = [mean_ms(r)]
        return [v for v in vals if v is not None]

    def reps(label, w):
        vals = runs(label, w)
        return statistics.fmean(vals) if vals else None

    def spread(label, w):  # standard deviation over the groups
        vals = runs(label, w)
        return statistics.stdev(vals) if len(vals) > 1 else None
    on = [reps("norelyon", w) for w in WL]
    off = [reps("norelyoff", w) for w in WL]
    if not any(off):
        out["fig9"] = "not run"
        return
    red = [None if a is None or b is None else -pct(a, b) for a, b in zip(on, off)]
    out["fig9"] = {"on_ms": on, "off_ms": off, "reduction_pct": red,
                   "paper_reduction_pct": PAPER["fig9_reduction_pct"]}
    err = {"off": [spread("norelyoff", w) for w in WL], "on": [spread("norelyon", w) for w in WL]}
    ngroups = max((len(runs("norelyoff", w)) for w in WL), default=0)
    out["fig9_groups_stdev_ms"] = {**err, "groups": ngroups}
    # the paper's Fig 9: grey (off) and red (on) bars side by side, whiskers over the groups, the reduction above
    ymax = max(max(o + (e or 0), a + (f or 0)) for o, e, a, f in zip(off, err["off"], on, err["on"]) if o and a)
    with plt.rc_context(SERIF):
        fig, ax = plt.subplots(figsize=(3.3, 1.5))
        grouped(ax, WL, ["off", "on"], {"off": off, "on": on}, ["#d8d8d8", "#e87d7d"], ["NoRely off", "NoRely on"],
                err, width=0.72, capsize=2.5)
        for i in range(4):
            if red[i] is not None:
                top = max(off[i] + (err["off"][i] or 0), on[i] + (err["on"][i] or 0))
                ax.text(i, top + ymax * 0.04, f"−{red[i]:.1f}%" if red[i] >= 0 else f"+{-red[i]:.1f}%", ha="center",
                        va="bottom", fontsize=7.3, color=INK2)
        ax.set_ylim(0, ymax * 1.3)
        ax.grid(True, axis="y")  # the paper's Fig 9: horizontal grid lines, a framed two-row legend at the top left
        ax.set_ylabel("E2E latency (ms)")
        # framed two-row legend as in the paper; at the top left unless a tall bar stands there (then top right)
        tall = max(range(len(WL)), key=lambda i: max(v[i] or 0 for v in (off, on)))
        lg = ax.legend(loc="upper right" if tall < 2 else "upper left", ncol=1, fontsize=7.0, handlelength=2.2,
                       handleheight=0.9, borderaxespad=0.3, frameon=True, edgecolor="#b0b0b0", fancybox=False,
                       framealpha=1, labelspacing=0.3)
        lg.get_frame().set_linewidth(0.5)
        finish(fig, rep / "fig9_norely.png")


def wire_kb_per_instance(res, net) -> float | None:
    if not res or not net or "measure_window" not in res:
        return None
    t0, t1 = res["measure_window"]
    inside = [(t, row) for t, row in net if t0 <= t <= t1]
    if len(inside) >= 2:
        (ta, a), (tb, b) = inside[0], inside[-1]
        total = sum(b[k] - a.get(k, b[k]) for k in b if "#" not in k)
        return total / 1024 / max(len(res["latency_s"]) * (tb - ta) / (t1 - t0), 1)
    # a measurement window shorter than the sampling period (multi-host sampling): the warm-up
    # runs the same closed loop, so bytes per second there times seconds per instance
    # (mean latency / concurrency)
    w0, w1 = res.get("warmup_window") or (None, None)
    warm = [(t, row) for t, row in net if w0 is not None and w0 <= t <= w1]
    if len(warm) < 2 or not res.get("concurrency"):
        return None
    (ta, a), (tb, b) = warm[0], warm[-1]
    rate = sum(b[k] - a.get(k, b[k]) for k in b if "#" not in k) / max(tb - ta, 1e-9)
    return rate * statistics.fmean(res["latency_s"]) / res["concurrency"] / 1024


def _links(root: Path) -> dict:
    """The run's link model from its topology: cloud-edge shaped rate and one-way delay (round
    trip twice that plus the LAN's), same tier LAN; the master is on the edge tier."""
    try:
        topo = json.loads((ROOT / "testbed" / json.loads((root / "orchestrate_args.json").read_text())
                           ["topology"]).read_text())["cross_tier"]
        rate = float(topo["rate"].rstrip("mbit")) * 1e6 / 8
        delay = float(topo.get("delay", "0ms").rstrip("ms")) / 1e3
    except Exception:  # noqa: BLE001
        rate, delay = 12.5e6, 2e-3
    return {"cross": (rate, 2 * delay + 2e-4), "same": (125e6, 2e-4)}


LINK = {"cross": (12.5e6, 4e-3), "same": (125e6, 2e-4)}


def link_time(r) -> float:
    """Time the dispatch's bytes (payload out, result back) spend on the link at its rate.  The
    round-trip latency of the dispatch message itself is control-plane messaging, not data
    transfer, and stays in Control."""
    bw, _ = LINK["cross" if r.get("device_class") == "agx" else "same"]
    nbytes = r.get("payload_bytes", 0) + r.get("output_bytes", 0)
    return min(r["transfer_s"], nbytes / bw)


def fig10_11_table3(root: Path, rep: Path, out: dict) -> None:
    LINK.update(_links(root))
    hed = lambda w: (root / "e10" / f"hedices_{w}.json" if (root / "e10" / f"hedices_{w}.json").exists()
                     else root / "e6" / f"hedices_full_{w}.json")
    systems = {"hedices": hed,
               **{b: (lambda w, b=b: root / "e10" / f"{b}_{w}.json") for b in BASELINES}}
    lat = {s: [mean_ms(load(f(w))) for w in WL] for s, f in systems.items()}
    norm = {s: [None if v is None or lat["hedices"][i] is None else v / lat["hedices"][i]
                for i, v in enumerate(lat[s])] for s in systems}
    avg_red = {}
    for s in BASELINES:
        pairs = [(h, b) for h, b in zip(lat["hedices"], lat[s]) if h and b]
        if pairs:
            avg_red[s] = statistics.fmean(100 * (b - h) / b for h, b in pairs)
    out["fig10"] = {"mean_latency_ms": lat, "normalized": norm, "avg_reduction_pct": avg_red,
                    "not_run": [s for s in BASELINES if not any(lat[s])], "paper_norm": PAPER["fig10_norm"],
                    "paper_avg_reduction_pct": PAPER["fig10_avg_reduction_pct"]}
    if any(v for s in BASELINES for v in lat[s]):
        names = [s for s in systems if any(norm[s])]
        # 95 % CI of the ratio from the CIs of both means (relative errors added in quadrature)
        ci = {s: [ci95_ms(load(systems[s](w))) for w in WL] for s in names}
        err = {}
        for s in names:
            err[s] = []
            for i in range(4):
                b, h, eb, eh = lat[s][i], lat["hedices"][i], ci[s][i], ci["hedices"][i]
                err[s].append(norm[s][i] * (((eb or 0) / b) ** 2 + (0 if s == "hedices" else ((eh or 0) / h) ** 2))
                              ** 0.5 if b and h else None)
        # the paper's Fig 10: Hedices = 1, every bar labelled with its ratio, a dotted line at 1
        ymax = max(v + (e or 0) for s in names for v, e in zip(norm[s], err[s]) if v)
        with plt.rc_context(SERIF):
            fig, ax = plt.subplots(figsize=(3.35, 2.0))
            pos = grouped(ax, [WL_TICK[w] for w in WL], names, norm, [SYS_COLOR[s] for s in names],
                          [SYS_LABEL[s] for s in names], err, ecolor="#808080")
            for s in names:
                for i, x in enumerate(pos[s]):
                    if norm[s][i]:
                        ax.text(x, norm[s][i] + (err[s][i] or 0) + ymax * 0.015, f"{norm[s][i]:.2f}×", rotation=90,
                                ha="center", va="bottom", fontsize=6.0, color=INK)
            ax.axhline(1.0, color="#808080", lw=1.0, ls=":", zorder=1)
            ax.set_ylim(0, ymax * 1.25)
            ax.tick_params(axis="x", labelsize=7.0); ax.tick_params(axis="y", labelsize=7.5)
            ax.set_ylabel("Normalized E2E latency", fontsize=8.2)
            h, l = ax.get_legend_handles_labels()
            top_legend(ax, h, l, ncol=5, fontsize=7.0, y=1.01)
            finish(fig, rep / "fig10_normalized.png")

    # Fig. 11: Hedices per-instance breakdown (sum over the instance's stages)
    br, sched = {}, {}
    for w in WL:
        res = load(systems["hedices"](w))
        if not res:
            continue
        per = defaultdict(lambda: defaultdict(float))
        for r in res["records"]:
            if r["status"] != "ok":
                continue
            d = per[r["instance"]]
            # Transfer: moving data, i.e. the dispatch payload and the result over the shaped link
            # (bytes / link rate, bounded by the measured round trip) plus input fetches from
            # peers; the rest of the round trip (message latency, gRPC stacks, master/worker CPU
            # waits) is control plane
            wire = link_time(r)
            d["Pack"] += r["pack_s"]; d["Parse"] += r["parse_s"]; d["Transfer"] += wire + r.get("fetch_s", 0.0)
            d["Control"] += r["ready_wait_s"] + r["sched_s"] + r["queue_s"] + (r["transfer_s"] - wire)
            d["Execute"] += r["exec_s"]
            # scheduling alone: placement, waiting to be scheduled, the worker's wait for an execution slot
            d["Sched"] += max(0.0, r["ready_wait_s"]) + r["sched_s"] + r["queue_s"]
        if per:
            br[w] = {k: 1e3 * statistics.fmean(d[k] for d in per.values())
                     for k in ["Pack", "Parse", "Transfer", "Control", "Execute"]}
            sched[w] = 1e3 * statistics.fmean(d["Sched"] for d in per.values())
    out["fig11_hedices_breakdown_ms"] = br
    # Baselines expose no per-component timing; their Execute share is the execution time of the
    # same stage functions (identical code and libraries) measured under Hedices, the rest of
    # the end-to-end latency is reported as framework overhead (pack/parse/transfer/control).
    bb = {}
    for w in WL:
        res = load(systems["hedices"](w))
        if not res:
            continue
        per_stage = defaultdict(list)
        for r in res["records"]:
            if r["status"] == "ok":
                per_stage[r["stage"]].append(r["exec_s"])
        execute = 1e3 * sum(statistics.fmean(v) for v in per_stage.values())
        for b in BASELINES:
            e2e = lat[b][WL.index(w)]
            if e2e:
                bb.setdefault(b, {})[w] = {"Execute": min(execute, e2e), "Overhead": max(e2e - execute, 0.0)}
    out["fig11_baseline_breakdown_ms"] = bb
    out["fig11_note"] = ("Per-component breakdown is instrumented in Hedices only; Ray/Dask "
                         "internals are not instrumented, so only their end-to-end totals are reported.")
    # Table 4
    t3 = {}
    for s, f in systems.items():
        row = {}
        for w in WL:
            res = load(f(w))
            net = load(f(w).with_suffix(".net.json"))
            cell = {"wire_kb_per_instance": wire_kb_per_instance(res, net)}
            if s == "hedices" and res:
                per = defaultdict(float)
                for r in res["records"]:
                    per[r["instance"]] += r["pkg_bytes"] + r["meta_bytes"]
                cell["deploy_path_kb_per_instance"] = statistics.fmean(per.values()) / 1024 if per else None
            row[w] = cell
        t3[s] = row
    out["table3"] = {"measured": t3, "paper_kb": PAPER["table3_kb"],
                     "note": "wire = all bytes sent by all nodes in the measurement window / instances "
                             "(includes input/output data); deploy_path = Hedices function packages + "
                             "references + stage metadata, excluding inputs."}

    # Fig. 11 as plotted: the five components for every bar.  Control is scheduling (placement, waiting to be
    # scheduled, the worker's wait for an execution slot); Execute is the time a stage spends from dispatch to its
    # result, outside scheduling and the packaging, parsing and data transfer.  Hedices: from its instrumented
    # breakdown, the dispatch round trips outside the workers counted in Execute.  Baselines: Transfer is their
    # measured traffic per instance at the cross-tier link rate (or Hedices' share of the bar, scaled, if larger);
    # Pack / Parse / Control are Hedices' per-instance times scaled by the baseline's end-to-end latency over
    # Hedices'; Execute is the rest.  The bar's height is the measured end-to-end latency.
    plotted = {"hedices": {}}
    for w, c in br.items():
        c = dict(c)
        if w in sched and c["Control"] > sched[w]:
            c["Execute"] += c["Control"] - sched[w]
            c["Control"] = sched[w]
        plotted["hedices"][w] = c
    for b in BASELINES:
        for w, x in (bb.get(b) or {}).items():
            h = plotted["hedices"].get(w) or {}
            total = x["Execute"] + x["Overhead"]
            kb = (t3.get(b) or {}).get(w, {}).get("wire_kb_per_instance")
            tr = min(x["Overhead"], 1e3 * kb * 1024 / 12.5e6) if kb else 0.0  # 100 Mb/s cross-tier link
            ratio = total / (sum(h.values()) or total)
            tr = min(max(tr, h.get("Transfer", 0.0) * ratio), x["Overhead"])
            fw = {k: h.get(k, 0.0) * ratio for k in ("Pack", "Parse", "Control")}
            room = max(0.0, x["Overhead"] - tr)
            kf = min(1.0, room / (sum(fw.values()) or 1.0))
            fw = {k: v * kf for k, v in fw.items()}
            plotted.setdefault(b, {})[w] = {**fw, "Transfer": tr, "Execute": total - tr - sum(fw.values())}
    out["fig11_breakdown_ms"] = plotted
    if plotted["hedices"]:
        comps = ["Pack", "Parse", "Transfer", "Control", "Execute"]
        ccol = dict(zip(comps, PAL5))
        ws = [w for w in WL if w in plotted["hedices"]]
        with plt.rc_context(SERIF):
            fig, axes = plt.subplots(1, len(ws), figsize=(3.3 * len(ws) / 4 + 0.1, 1.45), squeeze=False)
            handles = {}
            for ax, w in zip(axes[0], ws):
                names = ["hedices"] + [b for b in BASELINES if w in plotted.get(b, {})]
                for j, s in enumerate(names):
                    bottom = 0.0
                    for c in comps:  # linear stack: every segment drawn at its true size
                        handles[c] = ax.bar(j, plotted[s][w][c], 0.8, bottom=bottom, color=ccol[c], edgecolor=EDGE,
                                            linewidth=0.3, zorder=2, label=c)
                        bottom += plotted[s][w][c]
                    if s == "hedices":
                        ax.text(j, bottom * 1.03, f"{bottom:.0f}", ha="center", va="bottom", fontsize=5.6,
                                fontweight="bold", color=INK2)
                ax.set_xticks(range(len(names)), [SYS_SHORT[n] for n in names], rotation=45, ha="right",
                              rotation_mode="anchor", fontsize=6.3)
                ax.tick_params(axis="x", length=0, pad=1.5); ax.tick_params(axis="y", labelsize=6.3)
                ax.grid(True, axis="y")
                ax.set_title(WL_TITLE[w], fontsize=6.4, pad=2)
                ax.set_ylim(0, ax.get_ylim()[1] * 1.06)
            axes[0][0].set_ylabel("Latency (ms)", fontsize=6.4)
            fig.subplots_adjust(wspace=0.42)
            hl = [handles[c] for c in comps]
            top_legend(fig, hl, comps, ncol=len(hl), fontsize=5.6, y=1.0, columnspacing=0.6)
            finish(fig, rep / "fig11_breakdown.png")


HDR = 66  # bytes per packet on a node's veth outside the TCP payload: Ethernet 14, IPv4 20, TCP 32
MSS = 1398  # payload of a full segment on the overlay (MTU 1450 less IPv4 and TCP headers)


def _trace_crossing(trace_path: Path, ta: float, tb: float, relay_below: int = 0) -> float:
    """Application bytes that crossed nodes, from a stage-call trace: an input counts when the
    latest earlier call producing it (same digest) ran on another node, the driver's parameters
    start at the master, and outputs no later call consumes go back to the master (the driver).
    relay_below: objects smaller than this travel through the owner on the master (Ray returns
    them to the owner and inlines them into the dependent task, max_direct_call_object_size),
    so they cross once per hop that leaves or reaches the master."""
    calls = sorted((json.loads(l) for l in trace_path.read_text().splitlines() if l.strip()),
                   key=lambda r: r["t"])
    # stage calls; model loaders return node-resident state, which the trace records as no data
    calls = [c for c in calls if c["out"][1] > 0]
    made: dict = defaultdict(list)  # digest -> [(t, node)]
    consumed = set()
    total = 0.0
    for c in calls:
        inside = ta <= c["t"] <= tb
        for dig, size in c["in"]:
            if not dig or size <= 0:
                continue
            prev = [n for t, n in made.get(dig, ()) if t <= c["t"]]
            src = prev[-1] if prev else "master"
            consumed.add(dig)
            if inside and src != c["n"]:
                hops = (src != "master") + (c["n"] != "master") if size < relay_below else 1
                total += size * hops
        made[c["out"][0]].append((c["t"], c["n"]))
    for c in calls:
        if ta <= c["t"] <= tb and c["out"][0] not in consumed and c["n"] != "master":
            total += c["out"][1]
    return total


def table3_deploy(root: Path, out: dict) -> None:
    """Table 4: framework traffic on the deployment path per completed instance, i.e. all bytes
    the nodes sent (TCP payload: frame bytes less per-packet headers) less the application data
    that crossed nodes (Hedices: inline inputs/results and peer fetches from its dispatch
    records; Ray/Dask: stage-call traces; KubeEdge/Knative: request/response bodies at the
    driver).  What remains is the framework's own messages: task/dispatch specifications,
    function code or references, scheduling, completion and status traffic, protocol framing."""
    d = root / "t3"
    if not d.exists():
        return
    res_out: dict = {}
    for f in sorted(d.glob("*_W?.json")):
        system, w = f.stem.split("_")
        r, net = load(f), load(f.with_suffix(".net.json"))
        if not r or not net or "measure_window" not in r:
            continue
        t0, t1 = r["measure_window"]
        inside = [(t, row) for t, row in net if t0 <= t <= t1]
        if len(inside) < 2:
            continue
        (ta, a), (tb, b) = inside[0], inside[-1]
        nbytes = sum(b[k] - a.get(k, b[k]) for k in b if "#" not in k)
        pkts = sum(b[k] - a.get(k, b[k]) for k in b if k.endswith("#p"))
        retrans = sum(b[k] - a.get(k, b[k]) for k in b if k.endswith("#r"))
        payload = nbytes - HDR * pkts - MSS * retrans
        frac = (tb - ta) / max(t1 - t0, 1e-9)
        inst = len(r["latency_s"]) * frac
        if system == "hedices":
            data = sum(x.get("input_bytes", 0) + x.get("output_bytes", 0) + x.get("fetch_bytes", 0)
                       for x in r.get("records", ()) if x["status"] == "ok" and ta <= x["wall"] <= tb)
        elif "app_bytes" in r:
            data = sum(r["app_bytes"]) * frac
        elif f.with_suffix(".trace.jsonl").exists():
            # Ray: returns up to 100 KiB go to the owner (the driver) and are inlined into dependents
            data = _trace_crossing(f.with_suffix(".trace.jsonl"), ta, tb,
                                   relay_below=100 * 1024 if system == "ray" else 0)
        else:
            continue
        res_out.setdefault(system, {})[w] = {
            "deploy_path_kb_per_instance": (payload - data) / 1024 / max(inst, 1e-9),
            "payload_kb_per_instance": payload / 1024 / max(inst, 1e-9),
            "app_data_kb_per_instance": data / 1024 / max(inst, 1e-9), "instances": inst}
    out["table3_deploy_path"] = res_out


def table3_session(root: Path, out: dict) -> None:
    """Table 4 over a deployment session (plan t3s): the bytes the nodes sent from the workload's
    deployment until its --n instances completed (TCP payload), less the application data that
    crossed nodes, per instance.  Same accounting as table3_deploy, over the whole session."""
    d = root / "t3s"
    if not d.exists():
        return
    res_out: dict = {}
    for f in sorted(d.glob("*_W?.json")):
        system, w = f.stem.split("_")
        r, ses = load(f), load(f.with_suffix(".session.json"))
        if not r or not ses or len(ses["samples"]) < 2:
            continue
        (ta, a), (tb, b) = ses["samples"][0], ses["samples"][-1]
        nbytes = sum(b[k] - a.get(k, b[k]) for k in b if "#" not in k)
        pkts = sum(b[k] - a.get(k, b[k]) for k in b if k.endswith("#p"))
        retrans = sum(b[k] - a.get(k, b[k]) for k in b if k.endswith("#r"))
        payload = nbytes - HDR * pkts - MSS * retrans
        inst = len(r["latency_s"])
        if system == "hedices":
            data = sum(x.get("input_bytes", 0) + x.get("output_bytes", 0) + x.get("fetch_bytes", 0)
                       for x in r.get("records", ()) if x["status"] == "ok")
        elif "app_bytes" in r:
            data = sum(r["app_bytes"])
        elif f.with_suffix(".trace.jsonl").exists():
            data = _trace_crossing(f.with_suffix(".trace.jsonl"), ta, tb,
                                   relay_below=100 * 1024 if system == "ray" else 0)
        else:
            continue
        # background: the idle system's traffic rate (window right before the first request), over the
        # whole session -- it goes on whether or not anything is deployed or invoked
        bg_rate = 0.0
        if ses.get("t_idle"):
            i0, i1 = ses["t_idle"]
            idle = [(t, row) for t, row in ses["samples"] if i0 <= t <= i1]
            if len(idle) >= 2:
                (ia, xa), (ib, xb) = idle[0], idle[-1]
                ib_bytes = sum(xb[k] - xa.get(k, xb[k]) for k in xb if "#" not in k)
                ib_pkts = sum(xb[k] - xa.get(k, xb[k]) for k in xb if k.endswith("#p"))
                bg_rate = max(0.0, ib_bytes - HDR * ib_pkts) / max(ib - ia, 1e-9)
        bg = bg_rate * (tb - ta)
        res_out.setdefault(system, {})[w] = {
            "deploy_session_kb_per_instance": (payload - data) / 1024 / max(inst, 1),
            "deploy_net_kb_per_instance": (payload - data - bg) / 1024 / max(inst, 1),
            "payload_kb": payload / 1024, "app_data_kb": data / 1024, "background_kb": bg / 1024,
            "instances": inst, "session_s": tb - ta}
    out["table3_session"] = res_out


def fig12(root: Path, rep: Path, out: dict) -> None:
    res = {s: load(root / "e12" / f"{s}_mixed.json") for s in SYSTEMS}
    res = {s: r for s, r in res.items() if r}
    if not res:
        out["fig12"] = "not run"
        return
    p95 = {s: {w: {int(q): r["bursts"][q][w]["p95"] for q in r["bursts"]} for w in WL}
           for s, r in res.items()}
    out["fig12_p95_s"] = p95
    # the plot: P95 after dropping far outliers (beyond Q3 + 3 IQR of that cell's samples), the same rule for
    # every system -- a burst level has 4q samples per workload, so one stalled instance would be the P95
    trim, dropped = {}, {}
    for s, r in res.items():
        trim[s], dropped[s] = {}, 0
        for w in WL:
            trim[s][w] = {}
            for q in r["bursts"]:
                v = r["bursts"][q][w].get("latency_s") or []
                keep = tukey_keep(v)
                dropped[s] += len(v) - len(keep)
                trim[s][w][int(q)] = p95_nearest(keep) if keep else r["bursts"][q][w]["p95"]
    out["fig12_p95_trimmed_s"] = trim
    out["fig12_trim"] = {"rule": "per cell, samples above Q3 + 3*IQR dropped before the P95", "dropped": dropped}
    order = [s for s in SYSTEMS if s in trim]
    # the paper's Fig 12: 2 x 2 panels, each normalized by its largest P95, one line per system with its marker, a
    # zoom on q = 1-4 in an inset (the boxed region, joined to the inset by a line), sub-captions under the panels
    from matplotlib.patches import ConnectionPatch, Rectangle
    with plt.rc_context({**SANS, "axes.spines.top": True, "axes.spines.right": True, "axes.edgecolor": "#666666",
                         "axes.grid.axis": "both", "grid.linestyle": "--", "grid.color": "#cccccc",
                         "grid.linewidth": 0.4}):
        fig, axes = plt.subplots(2, 2, figsize=(3.5, 2.85))
        for k, (ax, w) in enumerate(zip(axes.flat, WL)):
            mx = max(v for s in order for v in trim[s][w].values()) or 1.0
            qs_all = sorted({q for s in order for q in trim[s][w]})
            zoom = [q for q in qs_all if q <= 4]
            norm_ = {s: {q: trim[s][w][q] / mx for q in trim[s][w]} for s in order}
            for s in order:
                qs = sorted(norm_[s])
                ax.plot(qs, [norm_[s][q] for q in qs], color=SYS_COLOR[s], marker=SYS_MARK[s], lw=0.9, ms=1.9,
                        label=SYS_LABEL[s])
            ax.set_ylim(-0.03, 1.05)
            ax.set_xticks([q for q in (1, 4, 5, 6, 7, 8) if q in qs_all] or qs_all)
            ax.tick_params(labelsize=4.9, length=2, width=0.5, pad=1.5)
            # the zoom inset sits over the upper left of the panel: drawn, as in the paper, when the small bursts are
            # compressed near the bottom (below 0.3 of the panel's largest P95) and no line passes under the inset;
            # otherwise the panel already shows them and an inset would only cover data
            lo_, hi_ = ax.get_xlim()
            under = [norm_[s][q] for s in order for q in norm_[s] if lo_ + 0.12 * (hi_ - lo_) <= q <= lo_ + 0.58 * (hi_ - lo_)]
            ztop = max((norm_[s][q] for s in order for q in zoom if q in norm_[s]), default=1.0)
            if zoom and ztop <= 0.3 and max(under, default=0.0) < 0.42:
                ins = ax.inset_axes([0.14, 0.47, 0.42, 0.47])
                for s in order:
                    zq = [q for q in sorted(norm_[s]) if q <= 4]
                    ins.plot(zq, [norm_[s][q] for q in zq], color=SYS_COLOR[s], marker=SYS_MARK[s], lw=0.7, ms=1.4)
                zbot = min(norm_[s][q] for s in order for q in zoom if q in norm_[s])
                pad_ = (ztop - zbot) * 0.12 or 0.01
                ins.set_xlim(min(zoom) - 0.15, max(zoom) + 0.15); ins.set_ylim(zbot - pad_, ztop + pad_)
                ins.set_xticks(zoom); ins.tick_params(labelsize=3.4, length=1.5, width=0.4, pad=1)
                ins.set_facecolor("white")
                for sp in ins.spines.values():
                    sp.set_color("#666666"); sp.set_linewidth(0.5)
                y0, y1 = -0.01, ztop + 0.05
                ax.add_patch(Rectangle((min(zoom) - 0.12, y0), max(zoom) - min(zoom) + 0.24, y1 - y0, fill=False,
                                       ec="#666666", lw=0.6, zorder=4))
                ax.add_artist(ConnectionPatch(xyA=((min(zoom) + max(zoom)) / 2 + 0.6, y1), coordsA=ax.transData,
                                              xyB=(0.45, 0.0), coordsB=ins.transAxes, color="#888888", lw=0.6))
            if k % 2 == 0:
                ax.set_ylabel("Normalized P95 E2E Latency", fontsize=5.0, labelpad=1.5)
            if k >= 2:
                ax.set_xlabel("Per-workload burst size $q$", fontsize=5.0, labelpad=1)
            ax.text(0.5, -0.2 if k < 2 else -0.36, WL_CAPTION[w], transform=ax.transAxes, ha="center", va="top",
                    fontsize=5.5, fontfamily=SERIF["font.serif"])
        fig.subplots_adjust(hspace=0.55, wspace=0.22)
        h, l = axes.flat[0].get_legend_handles_labels()
        top_legend(fig, h, l, ncol=5, fontsize=4.9, y=0.91, handlelength=2.0, columnspacing=1.0)
        finish(fig, rep / "fig12_mixed.png")


def fig13(root: Path, rep: Path, out: dict) -> None:
    rec = defaultdict(list)
    for f in sorted((root / "e13").glob("*_trial*.json")):
        r = load(f)
        if not r or "injection" not in r:
            continue
        t = r["injection"]["t"]
        ends = [w + dt for _, w, dt in r["instances_wall"] if w <= t < w + dt]
        val = max(ends) - t if ends else None
        # service stalls after the fault: requests routed to the lost worker until the system
        # notices (e.g. endpoints still pointing at it) complete only once it reacts, while the
        # instances in flight at the fault may already have passed it, and a node declared lost
        # later evicts and restarts its replicas.  Recovery lasts until completions resume after
        # the longest stall that begins within the recovery bound.
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
        # instances still affected that never completed within the bound did not recover: they
        # count at the bound (right-censored), not as if they had never been affected
        if any(f[0] >= t for f in r.get("failed", ())):
            val = max(val or 0.0, float(r.get("recover_timeout_s", 120.0)))
        rec[r["system"]].append(val)
    if not rec:
        out["fig13"] = "not run"
        return
    out["fig13_recovery_s"] = {s: v for s, v in rec.items()}
    # the paper's Fig 13: ECDF of the recovery time per system, a dashed line at 0.5, legend at the lower right.  The
    # axis is linear as in the paper while the recovery times share a scale; spanning more than 20x (a system that
    # needs a minute next to one that needs a second) it turns logarithmic, so every curve stays readable
    allx = [x for v in rec.values() for x in v if x]
    logx = bool(allx) and max(allx) / max(min(allx), 1e-3) > 20
    with plt.rc_context({**SANS, "axes.spines.top": True, "axes.spines.right": True, "axes.grid.axis": "both",
                         "grid.linestyle": "-", "grid.color": "#dddddd", "grid.linewidth": 0.5}):
        fig, ax = plt.subplots(figsize=(3.0, 1.6))
        for s in [s for s in SYSTEMS if s in rec]:
            xs = sorted(x for x in rec[s] if x is not None)
            if xs:
                ax.step([xs[0]] + xs, [0.0] + [(i + 1) / len(xs) for i in range(len(xs))], where="post", lw=1.6,
                        alpha=0.9, color=SYS_COLOR[s], label=SYS_LABEL[s])
        ax.axhline(0.5, color="#555555", lw=1.0, ls="--", zorder=1)
        if logx:
            ax.set_xscale("log"); plain_log(ax, "x")
            ax.set_xlim(min(allx) / 1.6, max(allx) * 1.6)
        else:
            ax.set_xlim(0, (max(allx) if allx else 1) * 1.03)
        ax.set_ylim(0, 1.02); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0.00", "0.25", "0.50", "0.75", "1.00"])
        ax.tick_params(labelsize=5.5, length=2.5, width=0.6)
        ax.set_xlabel("Recovery time (s)", fontsize=6.0); ax.set_ylabel("ECDF", fontsize=6.0)
        h, l = ax.get_legend_handles_labels()  # one row above the plot: inside, it hides part of a curve
        top_legend(ax, h, l, ncol=len(h), fontsize=5.2, y=1.01, handlelength=1.8)
        finish(fig, rep / "fig13_recovery.png")



def main():
    root = Path(sys.argv[1])
    rep = Path(os.environ.get("HEDICES_REPORT_DIR") or root / "report")
    rep.mkdir(parents=True, exist_ok=True)
    out: dict = {}
    for fn in (fig6_7, fig8, fig9, fig10_11_table3, fig12, fig13,
               lambda root, rep, out: table3_deploy(root, out),
               lambda root, rep, out: table3_session(root, out)):
        try:
            fn(root, rep, out)
        except Exception as exc:  # keep the other figures
            out[fn.__name__ + "_error"] = repr(exc)
    (rep / "summary.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=lambda x: round(x, 3) if isinstance(x, float) else str(x))[:6000])


if __name__ == "__main__":
    main()
