# repro/guard.sh (sourced): run_guarded OUT CMD... -- runs CMD only on a link that has been stable for 2 min,
# and redoes it (up to 3 tries) when the master's uplink went down while it ran, a run failed (rc!=0:
# the testbed is rebuilt) or (unless NO_STALLCHK is set) samples stalled.  CMD's output goes to OUT.log;
# the log of every failed try is kept as OUT.tryN.log.  A failed KubeEdge / Knative run also takes the
# k8s cluster down: down-cluster removes the netns under it while k8s/state still names the flavor.
# LINK_IF: the master's LAN interface (default eno1); without it the link check is skipped.
ROOT=${ROOT:-/opt/hedices}
LINK_IF=${LINK_IF:-eno1}
flaps_since() {
  [ -e /sys/class/net/$LINK_IF ] || { echo 0; return; }
  sudo -n journalctl -k --no-pager --since "$1" | grep -c "$LINK_IF: link down"
}
stable() {
  [ -e /sys/class/net/$LINK_IF ] || return 0
  [ "$(cat /sys/class/net/$LINK_IF/speed 2>/dev/null)" = 1000 ] && \
    [ "$(sudo -n journalctl -k --no-pager --since -2min | grep -c "$LINK_IF: link")" = 0 ]
}
run_guarded() { local o=$1; shift; local try
  mkdir -p "$(dirname "$o")"
  for try in 1 2 3; do
    local w=0; until stable; do sleep 15; w=$((w+15)); [ $w -ge 900 ] && { echo "ABORT: $LINK_IF not stable at 1000 Mb/s for 15 min"; exit 1; }; done
    local t0; t0=$(date '+%F %T'); "$@" > "$o.log" 2>&1
    if grep -qE "rc=[1-9]" "$o.log"; then
      echo "   $(basename "$o"): a run failed (rc!=0), rebuild the testbed and redo (try $try)"; cp "$o.log" "$o.try$try.log"
      case " $* " in *" kubeedge "*|*" knative "*)
        sudo -E "$ROOT/.venv/bin/python" "$ROOT/testbed/k8s.py" down > /dev/null 2>&1 ;; esac
      sudo -E "$ROOT/.venv/bin/python" "$ROOT/testbed/tb.py" down-cluster > /dev/null 2>&1; sleep 20
    elif [ "$(flaps_since "$t0")" != 0 ]; then
      echo "   $(basename "$o"): $LINK_IF flapped during the run, redo (try $try)"; cp "$o.log" "$o.try$try.log"
    elif [ -z "${NO_STALLCHK:-}" ] && ! "$ROOT/.venv/bin/python" "$ROOT/repro/stallchk.py" "$o" > /dev/null; then
      echo "   $(basename "$o"): stalled samples, redo (try $try)"; cp "$o.log" "$o.try$try.log"
    else return 0; fi
  done; echo "   $(basename "$o"): gave up after 3 tries"; return 1; }
