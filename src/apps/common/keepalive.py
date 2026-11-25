"""A light HTTP client for the stage calls of the KubeEdge / Knative pipeline drivers.

One keep-alive http.client connection per thread and host instead of a requests.Session shared by
every driver thread.  At the Fig 10 load levels the requests overhead per call queued the driver's
calls (tools/k8s_hop_probe.py, results/qk4: a 0-ms stage at concurrency 8 took 11.7 ms with
requests, 4.8 ms with this).  ``post`` mirrors the requests call the pipelines make (url, data=,
timeout=) and returns an object with .content, .headers and raise_for_status(); connection errors
are raised as requests.ConnectionError, so the pipelines' retry on another replica is unchanged.
The Host header is the service name, as a Knative ingress needs to route the call.
"""
import http.client
import threading
from urllib.parse import urlsplit

import requests
import urllib3
from requests.structures import CaseInsensitiveDict


class Response:
    def __init__(self, url, status, content, headers):
        self.url, self.status_code, self.content, self.headers = url, status, content, headers

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} for {self.url}", response=self)


class KeepAlive:
    def __init__(self):
        self._local = threading.local()

    def _conn(self, host, port, timeout):
        conns = self._local.__dict__.setdefault("conns", {})
        c = conns.get((host, port))
        if c is None:
            c = conns[(host, port)] = http.client.HTTPConnection(host, port, timeout=timeout)
        return c

    def post(self, url, data=None, timeout=None):
        u = urlsplit(url)
        host, port = u.hostname, u.port or 80
        path = u.path or "/"
        for attempt in (0, 1):  # a kept-alive connection the server has closed: reconnect once
            c = self._conn(host, port, timeout)
            try:
                c.request("POST", path, body=data or b"",
                          headers={"Content-Type": "application/octet-stream"})
                r = c.getresponse()
                return Response(url, r.status, r.read(), CaseInsensitiveDict(r.getheaders()))
            except (OSError, http.client.HTTPException) as e:
                c.close()
                self._local.conns.pop((host, port), None)
                if attempt:
                    raise requests.ConnectionError(str(e)) from e


class Pooled:
    """The same post() over one urllib3 PoolManager shared by the driver threads: the request layer
    under requests without its session machinery.  At concurrency 8 a 0-ms stage call took 11.8 ms
    with requests, 7.3 ms with this and 4.7 ms with KeepAlive (results/qk7, KubeEdge); W1, whose
    five calls per instance at concurrency 8 queue on the driver, uses it."""

    def __init__(self, maxsize=16):
        self._pm = urllib3.PoolManager(maxsize=maxsize)

    def post(self, url, data=None, timeout=None):
        try:
            r = self._pm.request("POST", url, body=data or b"", timeout=timeout, retries=False,
                                 headers={"Content-Type": "application/octet-stream"})
        except urllib3.exceptions.HTTPError as e:
            raise requests.ConnectionError(str(e)) from e
        return Response(url, r.status, r.data, CaseInsensitiveDict(r.headers))
