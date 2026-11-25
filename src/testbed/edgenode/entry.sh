#!/bin/sh
# cgroup v2 nesting: leave the root group to the pods and delegate every controller
mkdir -p /sys/fs/cgroup/init
for p in $(cat /sys/fs/cgroup/cgroup.procs); do echo "$p" > /sys/fs/cgroup/init/cgroup.procs 2>/dev/null || true; done
sed -e 's/ / +/g' -e 's/^/+/' < /sys/fs/cgroup/cgroup.controllers > /sys/fs/cgroup/cgroup.subtree_control
mkdir -p /etc/cni/net.d
cat > /etc/cni/net.d/10-bridge.conflist <<CNI
{"cniVersion": "1.0.0", "name": "edge", "plugins": [
  {"type": "bridge", "bridge": "cni0", "isGateway": true, "ipMasq": true, "hairpinMode": true,
   "ipam": {"type": "host-local", "ranges": [[{"subnet": "$POD_CIDR"}]], "routes": [{"dst": "0.0.0.0/0"}]}},
  {"type": "portmap", "capabilities": {"portMappings": true}}]}
CNI
containerd --config /etc/containerd/config.toml > /var/log/containerd.log 2>&1 &
until [ -S /run/containerd/containerd.sock ]; do sleep 0.5; done
# the edgecore configuration (with the join token) appears once CloudCore is up
until [ -s /etc/kubeedge/config/edgecore.yaml ]; do sleep 1; done
exec edgecore --config /etc/kubeedge/config/edgecore.yaml
