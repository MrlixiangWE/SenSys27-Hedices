# KubeEdge edge node of the emulated testbed: containerd + CNI bridge + edgecore.
FROM ubuntu:24.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      containerd runc containernetworking-plugins iptables iproute2 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --set iptables /usr/sbin/iptables-legacy \
    && update-alternatives --set ip6tables /usr/sbin/ip6tables-legacy
COPY edgecore /usr/local/bin/edgecore
COPY entry.sh /entry.sh
COPY containerd.toml /etc/containerd/config.toml
COPY hosts.toml /etc/containerd/certs.d/docker.io/hosts.toml
RUN chmod +x /entry.sh /usr/local/bin/edgecore
