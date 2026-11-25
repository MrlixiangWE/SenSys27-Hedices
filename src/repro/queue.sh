#!/bin/bash
# repro/queue.sh -- one testbed, several users: runs of repro/run_all.sh wait in a FIFO queue and run one at a time.
#
#   queue.sh submit NAME STEP...   queue a run (ROUNDS from the environment, default 1); prints its id and position
#   queue.sh status                the running job (step, elapsed, estimated end), the queue with estimated starts,
#                                  and the last finished jobs with their summary files
#   queue.sh watch [ID]            live view of the running job (Ctrl-C leaves it running); a finished job: its summary
#   queue.sh cancel ID             remove a queued job; a running job is stopped (only by the user who queued it)
#   queue.sh runner                the worker loop (installed as the systemd service hedices-ae-queue)
#
# Every job writes to its own directory, results/ae/<id>-<NAME>/ (run.log, summary.txt, reports).
# repro/run_all.sh started by hand waits for the testbed lock as well, so it never overlaps a queued job.
set -u
cd "$(dirname "$0")/.." && ROOT=$PWD
Q=${AE_QUEUE:-$ROOT/queue}
PY=$ROOT/.venv/bin/python
mkdir -p "$Q/pending" "$Q/running" "$Q/done" 2>/dev/null

# expected minutes per step (repro/monitor.py holds the same table)
est() {  # est ROUNDS STEP... -> minutes
  local r=$1 m=0 s; shift
  for s in "$@"; do case $s in
    sync|fig8|summary) m=$((m + 1)) ;; fig6) m=$((m + 20 * r)) ;; fig7) m=$((m + 60)) ;; fig9) m=$((m + 55)) ;;
    fig12) m=$((m + 50)) ;; fig13) m=$((m + 85)) ;; fig10) m=$((m + 60 * r)) ;; table3|table4) m=$((m + 85)) ;;
    hexp) m=$((m + 20 * r + 60 + 2 + 55)) ;; bexp) m=$((m + 190 * r)) ;; all) m=$((m + 2 + 20 * r + 60 + 2 + 55 + 190 * r)) ;;
    *) m=$((m + 5)) ;; esac; done
  echo $m
}
field() { sed -n "s/^$2=//p" "$1" | head -1; }
hm() { printf '%d:%02d' $(($1 / 60)) $(($1 % 60)); }
clock() { date -d "@$1" +'%a %H:%M'; }

# minutes left of the running job: its estimate minus the time since it started, at least 5
left_running() {
  local j; j=$(ls "$Q/running"/*.job 2>/dev/null | head -1); [ -n "$j" ] || { echo 0; return; }
  local t0 e; t0=$(field "$j" STARTED); t0=${t0:-$(date +%s)}; e=$(est "$(field "$j" ROUNDS)" $(field "$j" STEPS))
  local l=$((e - ($(date +%s) - t0) / 60)); [ $l -lt 5 ] && l=5; echo $l
}

submit() {
  local name=${1:-}; shift || true
  [[ $name =~ ^[A-Za-z0-9_-]{1,32}$ ]] && [ $# -gt 0 ] || { echo "usage: queue.sh submit NAME STEP...   (NAME: letters, digits, - or _)"; exit 2; }
  local s; for s in "$@"; do grep -q "^step_$s()" repro/run_all.sh || [ "$s" = all ] || { echo "unknown step: $s"; exit 2; }; done
  local id; id=$(exec 8>>"$Q/.seq.lock"; flock 8; n=$(( $(cat "$Q/.seq" 2>/dev/null || echo 0) + 1 )); echo $n > "$Q/.seq"; printf '%03d' $n)
  local r=${ROUNDS:-1}
  printf 'NAME=%s\nUSER=%s\nSTEPS=%s\nROUNDS=%s\nOUT=results/ae/%s-%s\nSUBMITTED=%s\n' \
    "$name" "$(id -un)" "$*" "$r" "$id" "$name" "$(date +%s)" > "$Q/pending/$id.job.tmp"
  mv "$Q/pending/$id.job.tmp" "$Q/pending/$id.job"; chmod a+r "$Q/pending/$id.job"
  local ahead; ahead=$(( $(ls "$Q/pending"/*.job | wc -l) - 1 ))
  local wait=$(left_running) j
  for j in $(ls "$Q/pending"/*.job | sort); do [ "$j" = "$Q/pending/$id.job" ] && break
    wait=$((wait + $(est "$(field "$j" ROUNDS)" $(field "$j" STEPS)))); done
  local e; e=$(est "$r" "$@")
  echo "queued job $id ($name: $*, ROUNDS=$r) -- $ahead job(s) ahead of it"
  echo "  starts about $(clock $(( $(date +%s) + wait * 60 ))), runs about $(hm "$e") h, results in results/ae/$id-$name/"
  echo "  follow it with: bash $ROOT/AE-process/queue.sh status   /   bash $ROOT/AE-process/queue.sh watch $id"
}

status() {
  local now j; now=$(date +%s)
  echo "== running"
  j=$(ls "$Q/running"/*.job 2>/dev/null | head -1)
  if [ -n "$j" ]; then
    local out t0; out=$(field "$j" OUT); t0=$(field "$j" STARTED)
    local step; step=$(awk '$2=="start"{s=$3} END{print s}' "$ROOT/$out/.progress" 2>/dev/null)
    if [ -z "$t0" ]; then  # taken from the queue, waiting for the testbed lock (a run started by hand holds it)
      echo "  $(basename "$j" .job)  $(field "$j" NAME)  [$(field "$j" STEPS), ROUNDS=$(field "$j" ROUNDS)]  waiting for the testbed"
    else
      echo "  $(basename "$j" .job)  $(field "$j" NAME)  [$(field "$j" STEPS), ROUNDS=$(field "$j" ROUNDS)]  step ${step:-starting}," \
           "running $(hm $(( (now - t0) / 60 ))) h, ends about $(clock $(( now + $(left_running) * 60 )))"
    fi
  else echo "  (none)"; fi
  echo "== queued"
  local wait; wait=$(left_running); local any=0
  for j in $(ls "$Q/pending"/*.job 2>/dev/null | sort); do any=1
    local e; e=$(est "$(field "$j" ROUNDS)" $(field "$j" STEPS))
    echo "  $(basename "$j" .job)  $(field "$j" NAME)  [$(field "$j" STEPS), ROUNDS=$(field "$j" ROUNDS)]  starts about $(clock $(( now + wait * 60 ))), runs about $(hm "$e") h"
    wait=$((wait + e)); done
  [ $any = 1 ] || echo "  (none)"
  echo "== finished (latest 5)"
  any=0
  for j in $(ls "$Q/done"/*.job 2>/dev/null | sort | tail -5); do any=1
    local out rc; out=$(field "$j" OUT); rc=$(field "$j" RC)
    echo "  $(basename "$j" .job)  $(field "$j" NAME)  [$(field "$j" STEPS)]  $( [ "$rc" = 0 ] && echo ok || echo "exit $rc" ), ended $(clock "$(field "$j" ENDED)")  ->  $out/summary.txt"
  done
  [ $any = 1 ] || echo "  (none)"
}

watch() {
  local id=${1:-} j
  [ -n "$id" ] && j=$(ls "$Q"/*/"$id".job 2>/dev/null | head -1) || j=$(ls "$Q/running"/*.job 2>/dev/null | head -1)
  [ -n "$j" ] || { echo "no such job (queue.sh status lists them)"; exit 1; }
  local out; out=$(field "$j" OUT)
  case $j in
    */running/*) exec "$PY" repro/monitor.py --out "$out" --steps "$(field "$j" STEPS)" --rounds "$(field "$j" ROUNDS)" --pid "$(cat "$Q/running/.pid" 2>/dev/null || echo 0)" ;;
    */pending/*) echo "job $(basename "$j" .job) is still queued:"; status ;;
    *) [ -f "$out/summary.txt" ] && cat "$out/summary.txt" || tail -n 20 "$out/run.log" ;;
  esac
}

cancel() {
  local id=${1:-}; [ -n "$id" ] || { echo "usage: queue.sh cancel ID"; exit 2; }
  if [ -f "$Q/pending/$id.job" ]; then rm -f "$Q/pending/$id.job" && echo "removed queued job $id"; return; fi
  local j=$Q/running/$id.job
  [ -f "$j" ] || { echo "job $id is neither queued nor running"; exit 1; }
  [ "$(field "$j" USER)" = "$(id -un)" ] || { echo "job $id was queued by $(field "$j" USER); only they can stop it"; exit 1; }
  touch "$Q/running/.cancel" && echo "stopping job $id (the testbed is reset before the next job starts)"
}

reset_testbed() {  # what a failed run leaves behind: k8s clusters, network namespaces, CPU frequency caps
  sudo -E "$PY" testbed/k8s.py down > /dev/null 2>&1
  for t in nd f9b; do HEDICES_TOPOLOGY=topology_phys6r_$t.json sudo -E "$PY" testbed/tb.py down-cluster > /dev/null 2>&1; done
  bash testbed/cpufreq.sh check > /dev/null 2>&1
}

runner() {
  exec 9>"$Q/testbed.lock"
  while :; do
    local j; j=$(ls "$Q/pending"/*.job 2>/dev/null | sort | head -1)
    [ -n "$j" ] || { sleep 10; continue; }
    local id; id=$(basename "$j" .job)
    mv "$j" "$Q/running/" || continue; j=$Q/running/$id.job
    flock 9
    echo "STARTED=$(date +%s)" >> "$j"
    local out; out=$(field "$j" OUT); mkdir -p "$out"; chmod a+rx "$out"
    rm -f "$Q/running/.cancel"
    AE_QUEUE_JOB=$id REPRO_OUT=$out ROUNDS=$(field "$j" ROUNDS) NO_MONITOR=1 \
      setsid bash repro/run_all.sh $(field "$j" STEPS) > "$out/run.log" 2>&1 9>&- &
    local pid=$!; echo $pid > "$Q/running/.pid"
    while kill -0 $pid 2>/dev/null; do
      if [ -f "$Q/running/.cancel" ]; then kill -TERM -- -$pid 2>/dev/null; sleep 5; kill -KILL -- -$pid 2>/dev/null; echo "cancelled" >> "$out/run.log"; fi
      sleep 5
    done
    wait $pid; local rc=$?
    reset_testbed
    chmod -R a+rX "$out" 2>/dev/null
    printf 'ENDED=%s\nRC=%s\n' "$(date +%s)" "$rc" >> "$j"
    mv "$j" "$Q/done/"; rm -f "$Q/running/.pid" "$Q/running/.cancel"
    flock -u 9
  done
}

case ${1:-} in
  submit) shift; submit "$@" ;;
  status) status ;;
  watch) shift; watch "${1:-}" ;;
  cancel) shift; cancel "${1:-}" ;;
  runner) runner ;;
  *) sed -n '2,12p' "$0"; exit 2 ;;
esac
