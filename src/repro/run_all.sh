#!/bin/bash
# repro/run_all.sh [STEP ...] -- reproduce the evaluation on the six-device testbed (see repro/README.md).
#
# Run on the master host (orin0), from the repository installed at the same path on every host
# (/opt/hedices).  "all" = sync hexp bexp summary (the whole evaluation); every figure also runs on its own:
#   sync    push hedices/ experiments/ testbed/ apps/ tools/ to every worker host and check md5
#   fig6    scheduler ablation: W1-W4 x full/alpha0/static/rr/nocache, ROUNDS rounds
#   fig7    dispatch distribution per device class on its own setting (configs "e7"), one round
#   fig8    alignment dynamics under co-deployment (cold start)
#   fig9    @NoRely on/off on its own topology (topology_phys6r_f9b.json), FIG9_TRIALS trials
#   fig10   end-to-end comparison (+ Fig 11 breakdown): 5 systems x W1-W4 in steady state, ROUNDS rounds
#   table3  network traffic per instance: steady state + deployment session, 5 systems x W1-W4
#   fig12   bursts 1-8, p95: 5 systems x W1-W4
#   fig13   recovery after a worker failure: 5 systems, W4, 5 trials
#   summary paper-vs-measured table of everything above (repro/summarize.py)
#   hexp    fig6 fig7 fig8 fig9;  bexp  fig10-13 and table3 as one campaign (plan eB), ROUNDS rounds
# fig10 / table3 / fig12 / fig13 are the phases of the bexp campaign run on their own (plan eB --eb-phases).
# Every workload / figure setting is read from experiments/configs/phys6r.json; nothing is set here
# beyond what each step's command line shows.  Results go to $REPRO_OUT (default results/repro);
# a step whose result files exist is skipped, so an interrupted run resumes where it stopped.
set -u
cd "$(dirname "$0")/.." && export ROOT=$PWD
. repro/guard.sh
PY=$ROOT/.venv/bin/python
O="sudo -E $PY -m experiments.orchestrate"
R=${REPRO_OUT:-results/repro}; mkdir -p "$R/logs"
ROUNDS=${ROUNDS:-2}
FIG9_TRIALS=${FIG9_TRIALS:-5}
WL="W1 W2 W3 W4"
SYSTEMS="hedices ray dask kubeedge knative"
# long runs and cold starts trip the stalled-sample check (>= 2 samples above 5x the median) without a
# stall; every final measurement was taken with it off
export NO_STALLCHK=${NO_STALLCHK:-1}
# Fig 13's failures per system: FIG13_TRIALS overrides configs e13 "trials" for this run
[ -n "${FIG13_TRIALS:-}" ] && export HEDICES_E13_TRIALS=$FIG13_TRIALS
# scan overrides must not leak into a reproduction
unset HEDICES_E6_CACHE HEDICES_E6_SEGMENT HEDICES_E6_CONC HEDICES_INPUT_IMU HEDICES_INPUT_SEGMENT HEDICES_CACHE_ENTRIES

own() { sudo chown -R "$(id -un)" "$@" 2>/dev/null; }
analyze() { $PY -m experiments.analyze "$1" > "$1/analyze.log" 2>&1 || echo "   analyze $1 failed, see $1/analyze.log"; }
# the testbed reads its topology at start-up: tear the cluster down under the old one before switching
topo() {
  local want=topology_phys6r_$1.json
  if [ "${HEDICES_TOPOLOGY:-}" != "$want" ]; then
    [ -n "${HEDICES_TOPOLOGY:-}" ] && sudo -E $PY testbed/tb.py down-cluster > /dev/null 2>&1
    export HEDICES_TOPOLOGY=$want
    sudo -E $PY testbed/tb.py down-cluster > /dev/null 2>&1
  fi
}
log() { echo "$(date +%H:%M:%S) $*"; }

step_sync() { bash repro/sync.sh; }


step_fig6() {
  # nocache redeploys a stage at every dispatch with no cache anywhere: the master packages the function anew too
  # (--pack-on-dispatch 1, Sec 4.4: a deployment introspects and serializes the function), ships and parses it; the
  # cached modes reuse the master's content-addressed package on a miss (--pack-on-dispatch 0)
  topo nd
  for r in $(seq 1 "$ROUNDS"); do for w in $WL; do
    log "fig6 round $r $w"
    run_guarded "$R/logs/fig6_r${r}_$w" $O --out "$R/fig6_r$r" --plan e6 --workloads $w \
      --modes full,alpha0,static,rr --warmup 30 --n 200 --cold-cache --nocache-state package --pack-on-dispatch 0
    run_guarded "$R/logs/fig6_r${r}_${w}_nocache" $O --out "$R/fig6_r$r" --plan e6 --workloads $w \
      --modes nocache --warmup 30 --n 200 --cold-cache --nocache-state package --pack-on-dispatch 1
  done; own "$R/fig6_r$r"; analyze "$R/fig6_r$r"; done
}

step_fig7() {
  # Fig 7 on its own setting (configs "e7": the Fig 6 run where edge caches / load do not force a stage off its tier)
  topo nd; log "fig7"
  run_guarded "$R/logs/fig7" $O --out "$R/fig7" --plan e7 --workloads W1,W2,W3,W4 \
    --modes full,alpha0,static,rr,nocache --warmup 30 --n 200 --cold-cache --nocache-state package --pack-on-dispatch 0
  # the cost-based modes (full, alpha0) also over a load sweep: on six workers a stage's placement is discrete (all
  # of its dispatches on the AGX or none), so Fig 7 pools their dispatches over closed-loop concurrency FIG7_CONC
  # (each workload at the levels of its configs "e7" "load_sweep", default 2-5)
  # (each workload at the levels of its configs "e7" "load_sweep", default 2-5)
  for c in ${FIG7_CONC:-2 3 4 5 6}; do
    HEDICES_E7_SWEEP=1 HEDICES_E6_CONC=$c run_guarded "$R/logs/fig7_c$c" $O --out "$R/fig7/c$c" --plan e7 --workloads W1,W2,W3,W4 \
      --modes full,alpha0 --warmup 30 --n 200 --cold-cache --nocache-state package --pack-on-dispatch 0
  done
  own "$R/fig7"; analyze "$R/fig7"
}

step_fig8() {
  topo nd; log "fig8"
  run_guarded "$R/logs/fig8" $O --out "$R/fig8" --plan e8 --e8-instances 150 --concurrency 4
  own "$R/fig8"; analyze "$R/fig8"
}

# one or more phases of the baseline comparison (plan eB, exactly as in bexp) for all five systems: eb_part DIR PHASES TAG
eb_part() {
  local d=$1 ph=$2 tag=$3 s
  topo nd
  for s in $SYSTEMS; do
    case $s in kubeedge|knative)  # a Kubernetes baseline on a freshly built cluster
      sudo -E $PY testbed/k8s.py down > "$R/logs/${tag}_${s}_down.log" 2>&1
      bash repro/registry_gc.sh || exit 1 ;; esac
    log "$tag $s"
    run_guarded "$R/logs/${tag}_$s" $O --out "$d" --plan eB --systems $s --eb-phases "$ph" --warmup 60 --n ${FIG10_N:-200} \
      --eb-session-n 200 --eb-burst-systems hedices,ray,dask,kubeedge,knative --rounds 4 --bursts 1,2,3,4,5,6,7,8 \
      --trials 5 --e13-workload W4
  done
  own "$d"; analyze "$d"
  sudo -E $PY testbed/k8s.py down > /dev/null 2>&1
}

step_fig12() { eb_part "$R/fig12" burst fig12; }

step_fig13() { eb_part "$R/fig13" failure fig13; }

step_fig10() {
  topo nd
  for r in $(seq 1 "$ROUNDS"); do for s in $SYSTEMS; do
    case $s in kubeedge|knative)  # every round of a Kubernetes baseline on a freshly built cluster
      sudo -E $PY testbed/k8s.py down > "$R/logs/fig10_r${r}_${s}_down.log" 2>&1
      bash repro/registry_gc.sh || exit 1 ;; esac
    for w in $WL; do
      log "fig10 round $r $s $w"
      run_guarded "$R/logs/fig10_r${r}_${s}_$w" $O --out "$R/fig10_r$r" --plan e10 --systems $s --workloads $w --warmup 60 --n ${FIG10_N:-200}
    done
  done; own "$R/fig10_r$r"; analyze "$R/fig10_r$r"; done
  sudo -E $PY testbed/k8s.py down > /dev/null 2>&1
}

# Table 4 reads the steady state (all traffic per instance) and the deployment session (Dask's stage boundaries)
step_table3() { eb_part "$R/table3" steady,session table3; }
step_table4() { step_table3; }  # the paper's number for this table (results stay in table3/)

step_fig9() {
  topo f9b; log "fig9"
  run_guarded "$R/logs/fig9" $O --out "$R/fig9" --plan e9 --warmup 60 --n 40 --trials "$FIG9_TRIALS"
  own "$R/fig9"; analyze "$R/fig9"
}

# the two campaigns of the evaluation as one step each:
#   hexp  Hedices alone (Fig 6 / 7 / 8 / 9): the main topology for Fig 6, Fig 7 and
#         Fig 8, then Fig 9 on its three-worker topology (on six workers W1 / W2 do not queue: -2 / +2 %)
#   bexp  the comparison with the baselines (Table 4, Fig 10 / 11 / 12 / 13) as plan eB, ROUNDS rounds: per system,
#         steady state then the deployment session for each workload, then bursts and failures (all five systems)
step_hexp() { step_fig6; step_fig7; step_fig8; step_fig9; }

step_bexp() {
  topo nd
  for r in $(seq 1 "$ROUNDS"); do for s in $SYSTEMS; do
    case $s in kubeedge|knative)
      sudo -E $PY testbed/k8s.py down > "$R/logs/bexp_r${r}_${s}_down.log" 2>&1
      bash repro/registry_gc.sh || exit 1 ;; esac
    log "bexp round $r $s"
    run_guarded "$R/logs/bexp_r${r}_$s" $O --out "$R/bexp_r$r" --plan eB --systems $s --warmup 60 --n ${FIG10_N:-200} \
      --eb-session-n 200 --eb-burst-systems hedices,ray,dask,kubeedge,knative --rounds 4 --bursts 1,2,3,4,5,6,7,8 \
      --trials 5 --e13-workload W4
  done; own "$R/bexp_r$r"; analyze "$R/bexp_r$r"; done
  sudo -E $PY testbed/k8s.py down > /dev/null 2>&1
}

step_summary() { $PY repro/summarize.py "$R" | tee "$R/summary.txt"; }

STEPS=("$@"); [ ${#STEPS[@]} -eq 0 ] && STEPS=(all)
[ "${STEPS[0]}" = all ] && STEPS=(sync hexp bexp summary)
for s in "${STEPS[@]}"; do
  declare -F "step_$s" > /dev/null || { echo "unknown step: $s"; exit 2; }
done

# one run on the testbed at a time: a run started by hand takes the lock the queue (repro/queue.sh) runs its jobs under
if [ -z "${AE_QUEUE_JOB:-}" ] && [ -z "${REPRO_CHILD:-}" ]; then
  mkdir -p "$ROOT/queue" 2>/dev/null; exec 9>>"$ROOT/queue/testbed.lock"
  flock -n 9 || { echo "The testbed is running another job (bash repro/queue.sh status). Queue this run instead:"
                  echo "  ROUNDS=$ROUNDS bash repro/queue.sh submit <your-name> ${STEPS[*]}"; exit 3; }
fi

# On a terminal the steps run in the background (their output in $R/run.log) and repro/monitor.py redraws a live
# view in place: the step, elapsed / remaining time, CPU / memory / temperature of every host, the latest log lines.
# NO_MONITOR=1 prints the log to the terminal instead.
if [ -t 1 ] && [ -z "${NO_MONITOR:-}" ] && [ -z "${REPRO_CHILD:-}" ]; then
  REPRO_CHILD=1 bash "$ROOT/repro/run_all.sh" "${STEPS[@]}" > "$R/run.log" 2>&1 &
  child=$!
  $PY repro/monitor.py --out "$R" --steps "${STEPS[*]}" --rounds "$ROUNDS" --pid "$child"
  if kill -0 "$child" 2>/dev/null; then
    echo "The run goes on in the background (log: $R/run.log); watch it again with:"
    echo "  $PY repro/monitor.py --out $R --steps \"${STEPS[*]}\" --rounds $ROUNDS --pid $child"
    exit 0
  fi
  wait "$child"; rc=$?
  tail -n 3 "$R/run.log"
  [ -f "$R/summary.txt" ] && echo "summary: $R/summary.txt"
  exit $rc
fi

: > "$R/.progress"
for s in "${STEPS[@]}"; do
  echo "$(date +%s) start $s" >> "$R/.progress"
  "step_$s"
  echo "$(date +%s) end $s $?" >> "$R/.progress"
done
log "done: ${STEPS[*]} -> $R"
