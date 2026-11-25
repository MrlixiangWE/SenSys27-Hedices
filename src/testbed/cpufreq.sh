#!/bin/bash
# testbed/cpufreq.sh set <class> <kHz> | restore <class> | check
# Cap (or restore) the CPU frequency of one device class's hosts: a per-figure compute setting (configs
# "cpu_max_khz", applied by experiments/orchestrate.py around a workload's runs). agx: the AGX host; pi5: the Raspberry Pi; nx: the three NX
# boards that host nothing else (the master's own board, nx-3's host, is left alone). Writes scaling_max_freq on
# every CPU of the host; restore puts back cpuinfo_max_freq.
# Safeguard: every cap is recorded in $MARK and removed on restore; "check" (run by orchestrate at start-up) restores
# every class that is marked or whose hosts run below their top frequency, so no later experiment inherits a cap.
set -u
op=$1; cls=${2:-}; khz=${3:-}
MARK=${HEDICES_CPUFREQ_MARK:-/opt/hedices/results/.cpufreq_capped}
hosts_of() {
  case $1 in
    agx) echo "lixiang@192.168.3.5" ;;
    nx)  echo "orin@192.168.3.6 orin@192.168.3.7 orin@192.168.3.9" ;;
    pi5) echo "leexiangdanaer@192.168.3.116" ;;
    *) echo "cpufreq: unknown class $1" >&2; return 2 ;;
  esac
}
as=${SUDO_USER:-$(id -un)}
KEY=$(eval echo "~$as")/.ssh/id_hedices  # the testbed key (repro/sync.sh, qsync.sh): authorized on every worker host
rsh() {  # rsh <user@host> <command>: as the testbed account when run under sudo
  local k=(); [ -f "$KEY" ] && k=(-i "$KEY")
  if [ "$(id -u)" = 0 ] && [ "$as" != root ]; then
    sudo -u "$as" ssh -n "${k[@]}" -o ConnectTimeout=8 -o BatchMode=yes "$1" "$2"
  else
    ssh -n "${k[@]}" -o ConnectTimeout=8 -o BatchMode=yes "$1" "$2"
  fi
}
SET='for f in /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_max_freq; do echo KHZ | sudo -n tee $f > /dev/null; done'
RESTORE='for d in /sys/devices/system/cpu/cpu[0-9]*/cpufreq; do cat $d/cpuinfo_max_freq | sudo -n tee $d/scaling_max_freq > /dev/null; done'
CAPPED='for d in /sys/devices/system/cpu/cpu[0-9]*/cpufreq; do [ "$(cat $d/scaling_max_freq)" -lt "$(cat $d/cpuinfo_max_freq)" ] && echo capped && break; done'
restore() {
  for h in $(hosts_of "$1"); do rsh "$h" "$RESTORE"; echo "cpufreq restore $1 $h: $(rsh "$h" 'cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq')"; done
  [ -f "$MARK" ] && { grep -vx "$1" "$MARK" > "$MARK.tmp" || true; mv "$MARK.tmp" "$MARK"; [ -s "$MARK" ] || rm -f "$MARK"; }
  return 0
}
case $op in
  set)
    for h in $(hosts_of "$cls"); do rsh "$h" "${SET//KHZ/$khz}"; echo "cpufreq set $cls $h: $(rsh "$h" 'cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq')"; done
    grep -qx "$cls" "$MARK" 2>/dev/null || echo "$cls" >> "$MARK" ;;
  restore) restore "$cls" ;;
  check)
    for c in agx nx pi5; do
      need=0; grep -qx "$c" "$MARK" 2>/dev/null && need=1
      for h in $(hosts_of "$c"); do [ -n "$(rsh "$h" "$CAPPED" 2>/dev/null)" ] && need=1; done
      [ $need = 1 ] && { echo "cpufreq check: $c was left capped, restoring"; restore "$c"; }
    done
    true ;;
  *) echo "usage: cpufreq.sh set <agx|nx|pi5> <kHz> | restore <agx|nx|pi5> | check"; exit 2 ;;
esac
