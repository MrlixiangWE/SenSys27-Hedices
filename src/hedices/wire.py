"""Wire format and gRPC plumbing: persistent channels carrying pickled messages."""
from __future__ import annotations

import pickle
from concurrent import futures

import grpc

MAX_MSG = 512 * 1024 * 1024
OPTIONS = [
    ("grpc.max_send_message_length", MAX_MSG),
    ("grpc.max_receive_message_length", MAX_MSG),
    ("grpc.keepalive_time_ms", 20000),
]
# servers accept the clients' keepalive pings: with gRPC's default (one ping per 5 min) a server
# answers them with GOAWAY "too_many_pings" and the channel reconnects mid-run
SERVER_OPTIONS = OPTIONS + [
    ("grpc.keepalive_permit_without_calls", 1),
    ("grpc.http2.min_recv_ping_interval_without_data_ms", 10000),
    ("grpc.http2.max_ping_strikes", 0),
]


def dumps(obj) -> bytes:
    return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)


def loads(data: bytes):
    return pickle.loads(data)


def serve(service: str, handlers: dict, address: str, max_workers: int = 32):
    """Start a gRPC server whose methods take and return raw bytes."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers), options=SERVER_OPTIONS)
    rpc = {name: grpc.unary_unary_rpc_method_handler(fn) for name, fn in handlers.items()}
    server.add_generic_rpc_handlers((grpc.method_handlers_generic_handler(service, rpc),))
    port = server.add_insecure_port(address)
    if port == 0:
        raise OSError(f"cannot bind {address}")
    server.start()
    return server, port


def channel(address: str) -> grpc.Channel:
    return grpc.insecure_channel(address, options=OPTIONS)


def method(ch: grpc.Channel, service: str, name: str):
    return ch.unary_unary(f"/{service}/{name}")


class ObjRef:
    """Reference to a stage output held in a Branch object store (paper Sec. 4.4: input references)."""

    __slots__ = ("oid", "holder")

    def __init__(self, oid: str, holder: str):
        self.oid = oid
        self.holder = holder

    def __reduce__(self):
        return (ObjRef, (self.oid, self.holder))

    def __repr__(self):
        return f"ObjRef({self.oid}@{self.holder})"


def map_refs(obj, fn):
    """Replace every ObjRef nested in tuples, lists and dicts with fn(ref)."""
    if isinstance(obj, ObjRef):
        return fn(obj)
    if isinstance(obj, tuple):
        return tuple(map_refs(x, fn) for x in obj)
    if isinstance(obj, list):  # a list subclass keeps its type (apps.common.assets.Segment)
        out = [map_refs(x, fn) for x in obj]
        return out if type(obj) is list else type(obj)(out)
    if isinstance(obj, dict):
        return {k: map_refs(v, fn) for k, v in obj.items()}
    return obj
