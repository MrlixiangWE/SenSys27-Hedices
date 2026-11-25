#!/bin/bash
# repro/registry_gc.sh: keep the master's root disk clear of kubelet's image-filesystem eviction line
# (imagefs.available < 15 %).  Every Kubernetes start rebuilds the application images
# (hedices-bench/*:latest) and pushes them to the testbed registry; the registry keeps every earlier
# revision, about 1 GB a build.  This drops the earlier revisions of hedices-bench/* only (the mirrored
# upstream images are referenced by digest and stay), garbage-collects their blobs, restarts the
# registry (its blob-descriptor cache), and prunes the docker build cache.  Run with the clusters down.
cd "$(dirname "$0")/.." || exit 1
R=${REGISTRY_DIR:-$PWD/k8s-assets/registry}/docker/registry/v2/repositories/hedices-bench
if sudo test -d "$R"; then
  for repo in $(sudo ls "$R"); do
    cur=$(sudo cat "$R/$repo/_manifests/tags/latest/current/link" 2>/dev/null | cut -d: -f2)
    [ -n "$cur" ] || continue
    for sub in revisions/sha256 tags/latest/index/sha256; do
      for d in $(sudo ls "$R/$repo/_manifests/$sub"); do
        [ "$d" != "$cur" ] && sudo rm -rf "$R/$repo/_manifests/$sub/$d"
      done
    done
  done
  sudo docker exec hed-registry registry garbage-collect /etc/docker/registry/config.yml > /dev/null 2>&1
  sudo docker restart hed-registry > /dev/null
fi
sudo docker builder prune -f --keep-storage 2GB > /dev/null 2>&1
used=$(df --output=pcent / | tail -1 | tr -dc 0-9)
echo "root disk ${used}% used"
[ "$used" -lt 85 ] || { echo "DISK_HIGH: kubelet evicts pods below 15 % free; free space before running Kubernetes"; exit 1; }
