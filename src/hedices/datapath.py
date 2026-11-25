"""Branch data plane: stage outputs a Branch holds are fetched by peers over a plain TCP stream.

A peer fetch moves one stored object (hundreds of KB for frames) and needs none of the RPC
machinery: it is a request for an object id answered by its bytes.  Carried as a gRPC message it
pays HTTP/2 framing and several copies of the payload; on a persistent TCP connection it is a
length-prefixed write the kernel sends from the store's buffer.  Control traffic (dispatches,
heartbeats) stays on gRPC.

  request   u16 length + object id (utf-8)
  reply     i64 length (-1: not held) + bytes
"""
from __future__ import annotations

import socket
import struct
import threading

_REQ = struct.Struct("!H")
_REP = struct.Struct("!q")


def _recv_exact(sock: socket.socket, n: int) -> bytearray:
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        k = sock.recv_into(view[got:], n - got)
        if k == 0:
            raise ConnectionError("peer closed the data connection")
        got += k
    return buf


class DataServer:
    """Serves objects of ``get(oid) -> bytes | None`` on ``host:port`` (port 0: any free)."""

    def __init__(self, get, host: str = "0.0.0.0", port: int = 0):
        self._get = get
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((host, port))
        self._sock.listen(64)
        self.port = self._sock.getsockname()[1]
        self._stop = threading.Event()
        threading.Thread(target=self._accept, daemon=True, name="data-accept").start()

    def _accept(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            threading.Thread(target=self._serve, args=(conn,), daemon=True, name="data-conn").start()

    def _serve(self, conn: socket.socket) -> None:
        with conn:
            try:
                while True:
                    (n,) = _REQ.unpack(_recv_exact(conn, _REQ.size))
                    data = self._get(bytes(_recv_exact(conn, n)).decode())
                    if data is None:
                        conn.sendall(_REP.pack(-1))
                    else:
                        conn.sendall(_REP.pack(len(data)))
                        conn.sendall(data)
            except (OSError, ConnectionError, struct.error):
                return

    def stop(self) -> None:
        self._stop.set()
        try:
            self._sock.close()
        except OSError:
            pass


class DataClient:
    """Persistent connections to the peers' data servers, one idle pool per peer."""

    def __init__(self):
        self._idle: dict[tuple, list] = {}
        self._lock = threading.Lock()

    def fetch(self, host: str, port: int, oid: str, timeout: float) -> bytearray | None:
        key = (host, port)
        with self._lock:
            pool = self._idle.setdefault(key, [])
            sock = pool.pop() if pool else None
        if sock is None:
            sock = socket.create_connection(key, timeout=timeout)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.settimeout(timeout)
        try:
            req = oid.encode()
            sock.sendall(_REQ.pack(len(req)) + req)
            (n,) = _REP.unpack(_recv_exact(sock, _REP.size))
            data = None if n < 0 else _recv_exact(sock, n)  # bytes-like, no extra copy
        except BaseException:
            sock.close()
            raise
        with self._lock:
            self._idle.setdefault(key, []).append(sock)
        return data

    def close(self) -> None:
        with self._lock:
            for pool in self._idle.values():
                for s in pool:
                    s.close()
            self._idle.clear()
