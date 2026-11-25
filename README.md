# README for SenSys AE submission — Hedices

## Project Overview

Hedices is a runtime for repeatedly triggered cloud–edge sensing pipelines. A pipeline is written once with the
Stem–Branch programming model (`@Stem` stages, `@NoRely` independence). **Stem** (the master) places every stage
invocation with StemSched, which sees the function cache of every worker; **Branch** (on every worker) keeps
content-addressed function packages cached and runs independent invocations in parallel.

This artifact contains the runtime, the four applications of the paper (W1 recognition, W2 tracking, W3 detection,
W4 multi-modal fusion) written for Hedices and for the four baselines (Ray, Dask, KubeEdge, Knative), and one script
per experiment that reruns the figures and tables of Section 5 (Evaluation) on our six-device testbed and prints the measured
values next to the paper's.

We apply for the **Artifacts Available**, **Artifacts Evaluated – Functional / Reusable** and **Results Validated –
Reproduced** badges. Reviewers get remote access to the testbed the results were measured on; nothing has to be
built or bought. Questions are welcome through HotCRP at any time.

## TODO List

- [x] Getting started (Fig 8)
- [x] Experiment 1: scheduler ablation (Fig 6) and dispatch distribution (Fig 7)
- [x] Experiment 2: cache dynamics (Fig 8)
- [x] Experiment 3: local parallelism with @NoRely (Fig 9)
- [x] Experiment 4: comparison with Ray / Dask / KubeEdge / Knative (Fig 10, Fig 11, Table 4, Fig 12, Fig 13)
- [x] Pre-collected results of every experiment

## Repository Layout

```
AE-process/            one folder per experiment, each with run.sh; run_all.sh; queue.sh; results/ (pre-collected)
src/hedices/           the runtime: api.py (Stem-Branch interface), stemsched.py (StemSched, Alg. 1), master.py (Stem),
                       branch.py (Branch: function cache, @NoRely), package.py (content-addressed packages)
src/apps/              W1-W4 for every framework: common/ (shared models and processing), hedices/, ray/, dask/,
                       kubeedge/, knative/
src/experiments/       experiment plans, driver, analysis; configs/phys6r.json holds every workload setting
src/testbed/           network namespaces, cgroups and link shaping across the six devices; Kubernetes clusters
src/repro/             run_all.sh (all steps), queue.sh (job queue), summarize.py (paper vs measured), monitor.py
```

## Testbed Access and the Job Queue

| Host | Device | Role | Cores |
|---|---|---|---|
| orin0 | Jetson Orin | master + edge worker nx-3 | 4 (slice) |
| agx | Jetson AGX Orin | cloud worker agx-0 | 12 |
| nx6 / nx7 / nx9 | Jetson Orin NX ×3 | edge workers nx-0 / nx-1 / nx-2 (nx-0 on a 100 Mb/s port) | 8 each |
| pi | Raspberry Pi 5 | edge worker pi5-0 | 4 |

**Login details are provided through HotCRP.** Everything runs on `orin0`, where the code of `src/` is installed at
`/opt/hedices` (with the experiment scripts in `/opt/hedices/AE-process`).

The experiments share one testbed, so runs never overlap: every `run.sh` **queues** its run and returns at once,
printing the job id, the estimated start time and the result directory. Please submit with your own name:

```bash
cd /opt/hedices/AE-process
bash 1-Ablation/run.sh alice          # queue Experiment 1 for "alice"
bash queue.sh status                  # running job (step, estimated end), queued jobs (estimated start), finished jobs
bash queue.sh watch 007               # live view of a running job (Ctrl-C leaves it running); a finished one: its summary
bash queue.sh cancel 007              # remove a queued job, or stop your own running job
```

Results go to `/opt/hedices/results/ae/<id>-<name>/` (`summary.txt`, `run.log`, plots in `<step>/report/`).
Jobs run one after another; a full evaluation takes about 5.5 hours, so please queue early.

## Hardware and Software Requirements

On the testbed (already installed): five NVIDIA Jetson boards and a Raspberry Pi 5 on one LAN; Python 3.12 with
PyTorch 2.14.1, ONNX Runtime 1.31.0, Ray 2.59.0, Dask 2026.8.0, gRPC 1.84.0 (`src/repro/requirements.txt`); k3s
1.32.13, KubeEdge 1.23.1, Knative 1.20.3; the application assets (models and inputs, 650 MB). On the reviewer's
side: an SSH client. Setting up another testbed is described in `src/repro/README.md`.

## Experimental Validation Steps and Expected Results

Each command below is run in `/opt/hedices/AE-process`. Times are for one round (`ROUNDS=1`, the default);
`ROUNDS=3 bash …` takes the medians of three rounds. Every job ends by printing the paper-vs-measured table
(`summary.txt`).

| Folder | Paper items | Time |
|---|---|---|
| `0-GettingStarted` | Fig 8 | 5 min |
| `1-Ablation` | Fig 6, Fig 7 | 1.5 h |
| `2-CacheDynamics` | Fig 8 | 5 min |
| `3-NoRely` | Fig 9 | 1 h |
| `4-EndToEnd` | Fig 10, Fig 11, Table 4, Fig 12, Fig 13 | 3.2 h |
| `run_all.sh` | all of the above | 5.5 h (`ROUNDS=3`: 12 h) |

### Getting Started

- **Steps**: `bash 0-GettingStarted/run.sh <your-name>`
- **Expected Results**: the job's `run.log` shows `SYNC_OK` (code identical on all six hosts), and `summary.txt`
  shows the Fig 8 row of Experiment 2 below.

### Experiment 1: Scheduler Ablation (Fig 6) and Dispatch Distribution (Fig 7)

- **Steps**: `bash 1-Ablation/run.sh <your-name>`
- **Expected Results**: full is the fastest configuration for every workload. Latency increase over full, %
  (ranges over our runs; paper in brackets):

| | α=0 | static | RR | nocache |
|---|---|---|---|---|
| W1 | +4 – +7 (+4) | +9 – +14 (+10) | +20 – +25 (+24) | +53 – +64 (+47) |
| W2 | +12 – +17 (+7) | +17 – +22 (+18) | +25 – +30 (+26) | +50 – +54 (+47) |
| W3 | +5 – +8 (+6) | +13 – +19 (+12) | +17 – +21 (+14) | +23 – +29 (+26) |
| W4 | +6 – +11 (+6) | +14 – +17 (+12) | +24 – +28 (+19) | +28 – +32 (+32) |

  Fig 7, share of edge-stage dispatches on the AGX, % (full and α=0 pooled over a closed-loop load sweep):

| | W1 | W2 | W3 | W4 | paper |
|---|---|---|---|---|---|
| full | 6.0 – 6.4 | 7.9 – 9.0 | 4.8 – 5.4 | 5.1 – 6.0 | 6.3 / 8.8 / 9.3 / 6.3 |
| α=0 | 14 – 20 | 22 – 26 | 10 – 13 | 9.7 – 11.3 | 19.1 / 11.6 / 11.8 / 11.4 |
| static | 62.5 | 70.0 | 75.0 | 55.0 | 61.3 / 60.2 / 72.2 / 55.2 |
| RR | 16.5 – 17 | 16.5 | 16.5 – 17 | 16.5 – 17 | 20.9 / 21.8 / 19.7 / 22.4 |
| nocache | 62.5 | 70.0 | 75.0 | 55.0 | 66.9 / 64.0 / 76.6 / 64.7 |

### Experiment 2: Cache Dynamics (Fig 8)

- **Steps**: `bash 2-CacheDynamics/run.sh <your-name>`
- **Expected Results**: σ(t) rises while the caches form (maximum about 2) and decays to 0; α(t) follows (maximum 0.5).

### Experiment 3: Local Parallelism with @NoRely (Fig 9)

- **Steps**: `bash 3-NoRely/run.sh <your-name>` (runs on a three-worker layout of the testbed)
- **Expected Results**: latency reduction with @NoRely W1 17–25 % (paper 16), W2 15–18 % (17), W3 19–24 % (18),
  W4 13–22 % (13).

### Experiment 4: Comparison with the Baselines (Fig 10, Fig 11, Table 4, Fig 12, Fig 13)

- **Steps**: `bash 4-EndToEnd/run.sh <your-name>`
- **Expected Results**:
  - **Fig 10**: Hedices is the fastest system on every workload. Latency relative to Hedices (median of three
    rounds; paper in brackets): Ray 1.86 / 1.74 / 1.66 / 1.55 (1.55 / 1.47 / 1.41 / 1.44), Dask 1.94 / 2.10 / 1.92 /
    1.87 (1.77 / 1.69 / 1.58 / 1.62), KubeEdge 1.71 / 1.85 / 1.70 / 1.58 (1.71 / 1.65 / 1.53 / 1.57), Knative 1.77 /
    1.97 / 1.66 / 1.58 (1.88 / 1.79 / 1.67 / 1.71).
  - **Fig 11**: Pack / Parse / Transfer stay small next to Control and Execute (Transfer 1–2 % of an instance).
  - **Table 4**: Hedices transfers the least per instance on every workload; the others 1.5–4× of it (paper 1.6–3.2×).
  - **Fig 12**: Hedices has the lowest p95 at every burst size, for every workload.
  - **Fig 13**: Hedices recovers within 0.03–1.6 s; Dask about 30 s, KubeEdge 6–56 s, Knative 0.1 s or 55–64 s
    (the latter when the failed node held the only replica of a stage), Ray about 7 s or up to about 2 min.

## Pre-collected Results

`AE-process/results/` holds the output of the full evaluation on the final code (`ROUNDS=3`), so the expected output
can be compared without rerunning: `summary.txt` (all items against the paper), `<step>/report/*.png` (plots) and
`<step>/report/summary.json`; `one_round/summary_run{1,2,3}.txt` are three independent one-round runs
(`run_all.sh`). Fig 6 in `fig6_r{1,2,3}/`, Fig 7 in `fig7/`, Fig 8 in `fig8/`, Fig 9 in `fig9/`,
Fig 10 in `fig10_r{1,2,3}/`, Fig 11–13 and Table 4 in `bexp_r{1,2,3}/` (the Kubernetes rows of Fig 13 in
`fig13_k8s_r{1,2,3}/`).

## Known Deviations

- **Fig 6**: W2 α=0 and W1 nocache lie above the paper (favourable to Hedices).
- **Fig 7**: α=0 W2 lies above the paper: W2's last edge stage runs after the cloud-tier `aggregate`, and α=0 sends it
  with that dispatch to the AGX. RR is 17 % (one AGX of six devices; the paper's testbed has two of ten).
- **Fig 10**: Dask W2 / W3 are relatively slower than in the paper; Knative W4 is lower (1.58 against 1.71). Hedices
  W1 settles in one of two layouts (about 22 or 25 ms), so the W1 ratios move by about 0.1 between runs.
- **Fig 11**: Transfer is 1–2 % of an instance (paper about 5 %).

## Variance

Over three complete runs and three one-round runs on the final code, every Fig 6 cell keeps its sign (no tie),
Fig 7 agrees within about 1 point for full / static / RR / nocache, and Fig 9 within about 5 points. A single
round's spread sits mostly on the baseline side (e.g. Ray W1 1.8–2.5×); the summary takes medians over the rounds.

## License

Apache License 2.0, see `LICENSE`.
