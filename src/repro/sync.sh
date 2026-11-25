#!/bin/bash
# repro/sync.sh: push the master's copy of the code to every worker host and compare a checksum of it.
# Hosts: one user@address per line in repro/hosts (default repro/hosts.example); key: REPRO_SSH_KEY.
cd "$(dirname "$0")/.." || exit 1
HOSTS=${REPRO_HOSTS:-repro/hosts}; [ -f "$HOSTS" ] || HOSTS=repro/hosts.example
K=${REPRO_SSH_KEY:-$HOME/.ssh/id_hedices}
DIRS="hedices experiments testbed apps tools"
SUMDIRS="hedices experiments testbed apps"  # tools/ may hold host-local helpers
SSH=(ssh -n -i "$K" -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new)
sum() {  # checksum of the source files of $SUMDIRS under directory $1
  echo "cd $1 && find $SUMDIRS \\( -name '*.py' -o -name '*.json' -o -name '*.yaml' -o -name '*.cpp' -o -name '*.sh' \\) -not -path '*/__pycache__/*' | LC_ALL=C sort | xargs md5sum | md5sum | cut -c1-8"
}
here=$(bash -c "$(sum "$PWD")")
echo "master $here"
bad=0
while read -r h; do
  [ -z "$h" ] || [ "${h#\#}" != "$h" ] && continue
  for d in $DIRS; do
    rsync -a --exclude=__pycache__ --exclude='*.pyc' --exclude='*.so' \
      -e "ssh -i $K -o BatchMode=yes -o StrictHostKeyChecking=accept-new" "$d/" "$h:$PWD/$d/" < /dev/null || bad=1
  done
  there=$("${SSH[@]}" "$h" "$(sum "$PWD")")
  echo "$h $there"; [ "$there" = "$here" ] || bad=1
done < "$HOSTS"
[ $bad = 0 ] && echo "SYNC_OK" || { echo "SYNC_MISMATCH"; exit 1; }
