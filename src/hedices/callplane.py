"""Stem-to-Branch call plane: dispatch messages on one persistent TCP stream per worker.

A dispatch is a pickled message answered by a pickled reply; on gRPC each one also pays call
object setup, HTTP/2 framing and hand-offs between gRPC's completion threads and the
interpreter (about 0.6 ms of a 1 ms loopback round trip on these CPUs).  Here calls are
multiplexed by id on one stream: the Stem writes a frame, a Branch thread pool runs the handler,
and a reader thread on the Stem completes the call's future.  Registration and heartbeats stay
on gRPC; a worker without a call port is called over gRPC as before.

  frame   u32 length + u64 call id + bytes        (both directions)
"""
from __future__ import annotations

import concurrent.futures as cf
import itertools
import queue
import socket
import struct
import threading
import time

import grpc

_H = struct.Struct("!IQ")


def _recv_exact(sock: socket.socket, n: int) -> bytearray:
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        k = sock.recv_into(view[got:], n - got)
        if k == 0:
            raise ConnectionError("call stream closed")
        got += k
    return buf


UNACKED_S = 5.0  # a peer that acknowledges nothing for this long is gone


def _bound_unacked(sock: socket.socket) -> None:
    """A write to a vanished peer (link down, no RST) would block forever once the send buffer
    fills, and the Stem writes under its scheduling lock: bound unacknowledged data and blocked
    sends, so the stream fails and the worker is handled as lost."""
    if hasattr(socket, "TCP_USER_TIMEOUT"):
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_USER_TIMEOUT, int(UNACKED_S * 1000))
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDTIMEO, struct.pack("ll", int(UNACKED_S), 0))


class Unavailable(grpc.RpcError):
    """The worker's call stream broke: handled like gRPC UNAVAILABLE (the worker is gone)."""

    def code(self):
        return grpc.StatusCode.UNAVAILABLE


class DeadlineExceeded(grpc.RpcError):
    def code(self):
        return grpc.StatusCode.DEADLINE_EXCEEDED


class CallServer:
    """Runs ``handler(request: bytes, None) -> bytes`` for frames arriving on host:port."""

    def __init__(self, handler, host: str = "0.0.0.0", port: int = 0, max_workers: int = 64):
        self._handler = handler
        self._pool = cf.ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="call")
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((host, port))
        self._sock.listen(16)
        self.port = self._sock.getsockname()[1]
        self._conns: set = set()
        threading.Thread(target=self._accept, daemon=True, name="call-accept").start()

    def _accept(self) -> None:
        while True:
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            _bound_unacked(conn)
            self._conns.add(conn)
            threading.Thread(target=self._serve, args=(conn,), daemon=True, name="call-conn").start()

    def _serve(self, conn: socket.socket) -> None:
        lock = threading.Lock()

        def run(cid: int, req: bytes) -> None:
            rep = self._handler(req, None)
            try:
                with lock:
                    conn.sendall(_H.pack(len(rep), cid) + rep if len(rep) < 65536 else _H.pack(len(rep), cid))
                    if len(rep) >= 65536:
                        conn.sendall(rep)
            except OSError:
                pass

        with conn:
            try:
                while True:
                    n, cid = _H.unpack(_recv_exact(conn, _H.size))
                    self._pool.submit(run, cid, bytes(_recv_exact(conn, n)))
            except (OSError, ConnectionError, struct.error):
                return

    def stop(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass
        for conn in list(self._conns):  # the Stem sees the stream break, as for a dead worker
            try:
                conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        self._pool.shutdown(wait=False, cancel_futures=True)


class CallClient:
    """The Stem's side of one worker's call stream, used where a gRPC method object was:
    ``client(payload, timeout)`` blocks, ``client.future(payload, timeout)`` returns a future."""

    def __init__(self, host: str, port: int, connect_timeout: float = 5.0):
        self._sock = socket.create_connection((host, port), timeout=connect_timeout)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        _bound_unacked(self._sock)
        self._sock.settimeout(None)
        self._lock = threading.Lock()
        self._pending: dict[int, tuple[cf.Future, float]] = {}
        self._ids = itertools.count(1)
        self._broken: BaseException | None = None
        # frames go out from a writer thread: the Stem dispatches under its scheduling lock and
        # must never block on a peer that stopped acknowledging
        self._outq: queue.SimpleQueue = queue.SimpleQueue()
        threading.Thread(target=self._reader, daemon=True, name="call-reader").start()
        threading.Thread(target=self._writer, daemon=True, name="call-writer").start()
        threading.Thread(target=self._deadlines, daemon=True, name="call-deadlines").start()

    def future(self, payload: bytes, timeout: float | None = None) -> cf.Future:
        fut: cf.Future = cf.Future()
        cid = next(self._ids)
        deadline = time.monotonic() + timeout if timeout else float("inf")
        with self._lock:
            if self._broken is not None:
                fut.set_exception(Unavailable(str(self._broken)))
                return fut
            self._pending[cid] = (fut, deadline)
        self._outq.put((cid, payload))
        return fut

    def _writer(self) -> None:
        while True:
            item = self._outq.get()
            if item is None or self._broken is not None:
                return
            cid, payload = item
            try:
                if len(payload) < 65536:
                    self._sock.sendall(_H.pack(len(payload), cid) + payload)
                else:
                    self._sock.sendall(_H.pack(len(payload), cid))
                    self._sock.sendall(payload)
            except OSError as exc:
                self._fail(exc)
                return

    def __call__(self, payload: bytes, timeout: float | None = None) -> bytes:
        return self.future(payload, timeout).result()

    def _reader(self) -> None:
        try:
            while True:
                n, cid = _H.unpack(_recv_exact(self._sock, _H.size))
                data = bytes(_recv_exact(self._sock, n))
                with self._lock:
                    ent = self._pending.pop(cid, None)
                if ent is not None and ent[0].set_running_or_notify_cancel():
                    ent[0].set_result(data)
        except (OSError, ConnectionError, struct.error) as exc:
            self._fail(exc)

    def _deadlines(self) -> None:
        while self._broken is None:
            time.sleep(0.5)
            now = time.monotonic()
            with self._lock:
                late = [cid for cid, (_, dl) in self._pending.items() if dl <= now]
                ents = [self._pending.pop(cid) for cid in late]
            for fut, _ in ents:
                if fut.set_running_or_notify_cancel():
                    fut.set_exception(DeadlineExceeded("deadline exceeded"))

    def _fail(self, exc: BaseException) -> None:
        with self._lock:
            if self._broken is None:
                self._broken = exc
            ents, self._pending = list(self._pending.values()), {}
        for fut, _ in ents:
            if fut.set_running_or_notify_cancel():
                fut.set_exception(Unavailable(str(exc)))

    def close(self) -> None:
        self._outq.put(None)
        try:
            self._sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._sock.close()
