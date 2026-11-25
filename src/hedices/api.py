"""Stem-Branch programming interface (paper Sec. 3.4, Table 2).

Pipeline stages are ordinary Python functions annotated with ``@Stem`` and,
optionally, ``@NoRely``.  A workflow is a plain function that calls stages;
``Workflow`` canonicalizes it into a stage DAG by tracing those calls
(Sec. 4.2, step 1).
"""
from __future__ import annotations

import contextvars
import functools
import inspect
import re
from dataclasses import dataclass, field
from typing import Any, Callable

TARGETS = ("cloud", "edge", "any")

_TRACER: contextvars.ContextVar = contextvars.ContextVar("hedices_tracer", default=None)


def parse_mem_mb(mem) -> float:
    """'512MB' / '1GB' / 256 -> megabytes."""
    if isinstance(mem, (int, float)):
        return float(mem)
    m = re.fullmatch(r"\s*([\d.]+)\s*([KMGT]?)(?:i?B)?\s*", str(mem), re.IGNORECASE)
    if not m:
        raise ValueError(f"bad memory spec: {mem!r}")
    scale = {"": 1.0, "K": 1 / 1024, "M": 1.0, "G": 1024.0, "T": 1024.0 * 1024}
    return float(m.group(1)) * scale[m.group(2).upper()]


@dataclass(eq=False)
class StageSpec:
    """Placement intent A_v and resource demand R_v of one stage."""

    fn: Callable
    target: str = "any"
    cpus: float = 1.0
    mem_mb: float = 512.0
    norely_flag: str | None = None
    signature: inspect.Signature = field(init=False, repr=False)

    def __post_init__(self):
        self.signature = inspect.signature(self.fn)

    @property
    def name(self) -> str:
        return f"{self.fn.__module__}.{self.fn.__qualname__}"

    @property
    def short_name(self) -> str:
        return self.fn.__qualname__


def Stem(target: str = "any", cpus: float = 1, mem="512MB"):
    """Declare a workflow stage, its placement intent and its resource demand."""
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}, got {target!r}")

    def deco(fn):
        spec = StageSpec(fn=fn, target=target, cpus=float(cpus), mem_mb=parse_mem_mb(mem),
                         norely_flag=getattr(fn, "__hedices_norely__", None))

        @functools.wraps(fn)
        def stage(*args, **kwargs):
            tracer = _TRACER.get()
            if tracer is None:
                return fn(*args, **kwargs)
            return tracer.record(spec, args, kwargs)

        stage.__hedices_stage__ = spec
        return stage

    return deco


def NoRely(flag: str = "req_id"):
    """Declare that executions over different workflow instances are independent.

    ``flag`` names the argument whose value identifies the workflow instance.
    Works above or below ``@Stem``.
    """

    def deco(fn):
        spec = getattr(fn, "__hedices_stage__", None)
        target = spec.fn if spec is not None else fn
        if flag not in inspect.signature(target).parameters:
            raise ValueError(f"@NoRely flag {flag!r} is not a parameter of {target.__qualname__}")
        if spec is not None:
            spec.norely_flag = flag
        else:
            fn.__hedices_norely__ = flag
        return fn

    return deco


@dataclass(frozen=True)
class Ref:
    """Placeholder for the output of a traced stage call."""

    node: int


@dataclass(frozen=True)
class Param:
    """Placeholder for a workflow parameter during tracing."""

    name: str


@dataclass
class Node:
    id: int
    spec: StageSpec
    args: tuple
    kwargs: dict
    deps: frozenset


def walk(obj, fn):
    """Map ``fn`` over every Ref/Param nested in tuples, lists and dicts."""
    if isinstance(obj, (Ref, Param)):
        return fn(obj)
    if isinstance(obj, tuple):
        return tuple(walk(x, fn) for x in obj)
    if isinstance(obj, list):  # a list subclass keeps its type (apps.common.assets.Segment)
        out = [walk(x, fn) for x in obj]
        return out if type(obj) is list else type(obj)(out)
    if isinstance(obj, dict):
        return {k: walk(v, fn) for k, v in obj.items()}
    return obj


class _Tracer:
    def __init__(self):
        self.nodes: list[Node] = []

    def record(self, spec: StageSpec, args, kwargs) -> Ref:
        deps: set[int] = set()

        def collect(x):
            if isinstance(x, Ref):
                deps.add(x.node)
            return x

        walk((args, kwargs), collect)
        node = Node(len(self.nodes), spec, tuple(args), dict(kwargs), frozenset(deps))
        self.nodes.append(node)
        return Ref(node.id)


class Workflow:
    """Runtime DAG G_j = (V_j, E_j) of a workflow function."""

    def __init__(self, fn: Callable, name: str | None = None):
        self.fn = fn
        self.name = name or f"{fn.__module__}.{fn.__qualname__}"
        self.signature = inspect.signature(fn)
        params = [Param(p) for p in self.signature.parameters]
        tracer = _Tracer()
        token = _TRACER.set(tracer)
        try:
            self.output = fn(*params)
        finally:
            _TRACER.reset(token)
        self.nodes = tracer.nodes
        if not self.nodes:
            raise ValueError(f"workflow {self.name} calls no @Stem stages")
        self.children: dict[int, list[int]] = {n.id: [] for n in self.nodes}
        for n in self.nodes:
            for d in n.deps:
                self.children[d].append(n.id)
        self.roots = [n.id for n in self.nodes if not n.deps]
        outs: set[int] = set()
        walk(self.output, lambda x: outs.add(x.node) if isinstance(x, Ref) else x)
        self.output_nodes = frozenset(outs)

    def stages(self) -> list[StageSpec]:
        seen: dict[int, StageSpec] = {}
        for n in self.nodes:
            seen.setdefault(id(n.spec), n.spec)
        return list(seen.values())

    def bind(self, args, kwargs) -> dict[str, Any]:
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        return dict(bound.arguments)

    @staticmethod
    def resolve(obj, params: dict, results: dict):
        return walk(obj, lambda x: params[x.name] if isinstance(x, Param) else results[x.node])

    def __repr__(self):
        edges = sum(len(n.deps) for n in self.nodes)
        return f"Workflow({self.name}, stages={len(self.nodes)}, edges={edges})"
