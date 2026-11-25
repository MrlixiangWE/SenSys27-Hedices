#!/bin/bash
# repro/k8s_fetch.sh [DIR]: download the Kubernetes baselines' binaries and manifests (arm64) into DIR
# (default /opt/hedices/k8s-assets), the pinned versions the evaluation used.
set -x
D=${1:-/opt/hedices/k8s-assets}; mkdir -p "$D" && cd "$D" || exit 1
G=https://github.com
get() { [ -s "$2" ] || curl -fL --retry 3 -o "$2" "$1"; }
get $G/k3s-io/k3s/releases/download/v1.32.13%2Bk3s1/k3s-arm64 k3s && chmod +x k3s
get $G/k3s-io/k3s/releases/download/v1.32.13%2Bk3s1/k3s-airgap-images-arm64.tar.zst k3s-airgap-images-arm64.tar.zst
get $G/knative/serving/releases/download/knative-v1.20.3/serving-crds.yaml serving-crds.yaml
get $G/knative/serving/releases/download/knative-v1.20.3/serving-core.yaml serving-core.yaml
get $G/knative-extensions/net-kourier/releases/download/knative-v1.20.1/kourier.yaml kourier.yaml
get $G/kubeedge/kubeedge/releases/download/v1.23.1/kubeedge-v1.23.1-linux-arm64.tar.gz kubeedge.tgz && tar xzf kubeedge.tgz
get $G/kubeedge/kubeedge/releases/download/v1.23.1/keadm-v1.23.1-linux-arm64.tar.gz keadm.tgz && tar xzf keadm.tgz
mkdir -p edgemesh
for f in 01-configmap.yaml 02-serviceaccount.yaml 03-clusterrole.yaml 04-clusterrolebinding.yaml 05-daemonset.yaml; do
  get https://raw.githubusercontent.com/kubeedge/edgemesh/v1.17.0/build/agent/resources/$f edgemesh/$f
done
ls -la . edgemesh
echo FETCH_DONE
