# Reproducing the evaluation

One script runs every figure and table of the evaluation on the six-device testbed and prints the
measured values next to the paper's:

```bash
cd /opt/hedices            # this repository, at the same path on every host
bash repro/run_all.sh      # or: bash repro/run_all.sh fig6 fig10 summary
```

One round of everything (`ROUNDS=1 bash repro/run_all.sh sync hexp bexp summary`) takes about 5.5 hours, three
rounds about 12 hours. On a shared testbed, queue runs with `repro/queue.sh submit <name> <step>...` (or the
per-experiment scripts in `AE-process/`): jobs run one at a time, and a run started by hand waits for none --
it refuses to start while a queued job holds the testbed.
Each step skips result files that already exist, so an interrupted run is resumed by starting it again.
Results go to `results/repro/` (`REPRO_OUT` changes it); the paper-vs-measured table is written to
`results/repro/summary.txt`.

## Testbed

Six workers on five Jetson boards and a Raspberry Pi 5 (`testbed/topology_phys6r_nd.json`):

| Host | Address | Role | Cores | Link |
|---|---|---|---|---|
| Orin (orin0) | 192.168.3.4 | master + worker nx-3 | 4 (slice) | 1 Gb/s |
| AGX Orin | 192.168.3.5 | agx-0 (cloud tier) | 12 | 1 Gb/s |
| Orin NX | 192.168.3.6 | nx-0 | 8 | 100 Mb/s |
| Orin NX | 192.168.3.7 | nx-1 | 8 | 1 Gb/s |
| Orin NX | 192.168.3.9 | nx-2 | 8 | 1 Gb/s |
| Raspberry Pi 5 | 192.168.3.116 | pi5-0 | 4 | 1 Gb/s |

The testbed (`testbed/tb.py`) joins the hosts' bridges with a VXLAN overlay, runs every node in its own
network namespace and cgroup slice, and shapes the cross-tier links to 100 Mb/s (Fig 6/7 topology `_nd`;
the other figures add nothing on top).  Fig 9 runs on `topology_phys6r_f9b.json` (agx-0, nx-1, nx-2).
Different addresses or users: edit the topology files and `repro/hosts`.

## Setup (once)

On every host, as the user that owns `/opt/hedices`:

1. Clone this repository to `/opt/hedices`.
2. Python 3.12 virtualenv: `python3.12 -m venv /opt/hedices/.venv && /opt/hedices/.venv/bin/pip install -r repro/requirements.txt`.
3. Native execution channel: `bash hedices/native/build.sh` (builds `_hedx.*.so` for the host's Python).
4. Application assets in `/opt/hedices/assets/` (not in git, 650 MB): `script_model_cpu.pt` (W3 detector),
   `yoloxna-320.onnx` (W2 / W4 detector), `pics40.npz` (frames), `w1/` (`model_weights`, `CurrentDSet`,
   `TotalCurrent.npy`: the W1 recognizer and its signal data).  Copy them from the original testbed.
5. Passwordless `sudo` (the testbed creates namespaces, cgroups, qdiscs) and, from the master, key-based
   ssh to every worker (`REPRO_SSH_KEY`, default `~/.ssh/id_hedices`).

On the master only:

6. `cp repro/hosts.example repro/hosts` and adjust it.
7. Kubernetes baselines: `bash repro/k8s_fetch.sh` (k3s 1.32.13, KubeEdge 1.23.1, EdgeMesh 1.17.0,
   Knative 1.20.3 + Kourier); Docker for the local image registry (`testbed/k8s.py` builds and pushes the
   application images itself each time it brings a cluster up).
8. `bash repro/run_all.sh sync` pushes the code to the workers and compares checksums (`SYNC_OK`).
   Run it again after any change to the code.

## What each step runs

All workload settings come from `experiments/configs/phys6r.json`; the steps only fix the run length.

| Step | Command (`python -m experiments.orchestrate --out ...`) | Figure-specific settings in `phys6r.json` |
|---|---|---|
| `fig6` | `--plan e6 --workloads W --modes full,alpha0,static,rr,nocache --warmup 30 --n 200 --cold-cache --nocache-state package --pack-on-dispatch 0`, `ROUNDS` rounds | `e6`: per-workload concurrency and function-cache entries; W4 2-frame instances |
| `fig8` | `--plan e8 --e8-instances 150 --concurrency 4` | |
| `fig12` | `--plan e12 --systems hedices,ray,dask --warmup 60 --rounds 4 --bursts 1,...,8` | |
| `fig13` | `--plan e13 --systems hedices,ray,dask --trials 5 --e13-workload W4` | |
| `fig10` | `--plan e10 --systems S --workloads W --warmup 60 --n 40`, every system, `ROUNDS` rounds (Kubernetes rebuilt each round) | `e10`: concurrency 8 / 4 / 5 / 3; `dask_env` |
| `table3` | `--plan t3s --systems S --warmup 0 --n 200 --concurrency 4` | |
| `fig9` | `--plan e9 --warmup 60 --n 40 --trials 5` on `topology_phys6r_f9b.json` | `e9`: concurrency, input segments, @NoRely batching, slot policy |

Every run goes through `run_guarded` (`repro/guard.sh`): it waits for a stable master uplink
(`LINK_IF`, default `eno1`) and redoes a run up to three times when it failed, rebuilding the testbed.

## Expected results and variance

The values measured with this code (`results/` of 2026-11-12) and the paper's:

| | Measured | Paper |
|---|---|---|
| Fig 6 alpha0 | W1 +4.7 / +10.2, W2 +6.3 / +10.5, W3 +6.1 / +6.3, W4 +4.7 / +2.8 % | +4 / +7 / +6 / +6 % |
| Fig 6 static | +14-16, +23-24, +16, +6.5-8.7 % | +10 / +18 / +12 / +12 % |
| Fig 6 RR | +23, +37, +18-21, +19-20 % | +24 / +26 / +14 / +19 % |
| Fig 6 nocache | +39, +46-47, +19-21, +11-15 % | +47 / +47 / +26 / +32 % |
| Fig 7 AGX share | static and nocache 81 / 85 / 50 / 70 %, full 0 / 0 / 23 / 39-44 % | static 55-72, nocache 64-77, full 6-9 % |
| Fig 9 @NoRely reduction | about 20 / 15 / 20 / 15 % | 16 / 17 / 18 / 13 % |
| Fig 10 Ray / Dask / KubeEdge / Knative vs Hedices | W1 1.91 / 2.24 / 1.93 / 2.03, W2 1.88 / 2.06 / 1.88 / 2.05, W3 1.63 / 1.99 / 1.73 / 1.79, W4 1.47 / 1.83 / 1.69 / 1.56 | W1 1.55 / 1.77 / 1.71 / 1.88, W2 1.47 / 1.69 / 1.65 / 1.79, W3 1.41 / 1.58 / 1.53 / 1.67, W4 1.44 / 1.62 / 1.57 / 1.71 |
| Fig 12 | Hedices has the lowest p95 at every burst size | same |
| Fig 13 | Hedices 0.04-1.4 s, Ray 6.4-7.1 s, Dask 30 s | |

Known deviations: Fig 6 W4 nocache (+11-15 %, the paper +32: seven stages, and the stages chained in one dispatch
parse their shared module once), Fig 7 full on W3 / W4 (the AGX runs detection faster here), Table 4 (Hedices lowest in every column, but 2.5-4.4 KB
per instance against the paper's 12-24, and Ray / Knative 9-41x Hedices against about 2x), Fig 11
(Transfer 1-2 % of an instance, the paper about 5 %).

Single rounds vary: one batch of five Fig 9 trials for W1 gave 17.2 %, another 22.5 %; a Ray round now
and then lands on a bad actor layout (2x the latency).  Compare medians over at least two rounds
(`ROUNDS=3` for a tighter Fig 6 / Fig 10).
