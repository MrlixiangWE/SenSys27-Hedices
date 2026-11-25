"""Function-granular packaging (paper Sec. 3.2 Eq. 2, Sec. 4.4).

A package holds the stage function and the user-level code it reaches -- its own module and
every application module it imports, transitively -- serialized by value with cloudpickle,
plus a dependency manifest.  Library dependencies (installed distributions, the standard
library) are listed in the manifest instead of being shipped.  Node-local asset modules
(HEDICES_LOCAL_MODULES, default ``apps.common.assets``: model weights and data that every
node holds) stay by reference, and the asset handles they create (lazily loaded models) are
node-resident: a worker resolves them to its own loaded instance, so re-receiving a package
re-ships and re-parses code but does not reload model weights.  The content identifier is
h_g = H(code(g), deps(g), version(g)).
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import importlib.util
import inspect
import io
import marshal
import os
import pickle
import sys
import sysconfig
import textwrap
import threading
import time
import types
from dataclasses import dataclass

import cloudpickle

from .api import StageSpec

_STDLIB = set(sys.builtin_module_names) | set(getattr(sys, "stdlib_module_names", ()))
_LIB_DIRS = tuple({os.path.realpath(sysconfig.get_paths()[k]) for k in ("stdlib", "platstdlib", "purelib", "platlib")})
LOCAL_MODULES = set(filter(None, os.environ.get("HEDICES_LOCAL_MODULES", "apps.common.assets").split(",")))
# HEDICES_SPLIT_DEPS=1: the application modules a stage imports become dependency packages of
# their own (one per module, shared by every stage that imports it); the stage's function package
# then refers to them by name, so U_v = {function package, its module packages}
SPLIT_DEPS = os.environ.get("HEDICES_SPLIT_DEPS", "0") == "1"
# HEDICES_PKG_FORMAT=source: a stage is still one package (one cache entry, re-shipped whole on a
# miss), but its application modules travel as source text that the worker compiles, instead of
# cloudpickled function objects; packing is then text assembly rather than per-function reduction
PKG_FORMAT = os.environ.get("HEDICES_PKG_FORMAT", "source")
# HEDICES_PKG_MINIFY (default on): the module sources a package carries are minified -- docstrings
# dropped, comments and layout gone (ast.unparse) -- so a package is smaller to ship and quicker to
# compile on the worker; the code is the same.  0 ships the source text as written.
PKG_MINIFY = os.environ.get("HEDICES_PKG_MINIFY", "1") != "0"
_MINIFIED: dict = {}


def _minify(src: str) -> str:
    import ast

    tree = ast.parse(src)
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and body
                and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            node.body = body[1:] or [ast.Pass()]
    return ast.unparse(tree) + "\n"


def _shipped_source(m: types.ModuleType) -> str:
    """The source text of an application module as packages carry it."""
    src = inspect.getsource(m)
    if not PKG_MINIFY:
        return src
    out = _MINIFIED.get((m.__name__, src))
    if out is None:
        out = _MINIFIED[(m.__name__, src)] = _minify(src)
    return out


_USER: dict = {}


def is_user_module(m: types.ModuleType) -> bool:
    """Application code: a module with a source file outside the interpreter's library dirs."""
    name = m.__name__
    if name in LOCAL_MODULES or name.split(".")[0] in _STDLIB or name.startswith("hedices"):
        return False
    f = getattr(m, "__file__", None)
    key = (name, f)
    if key not in _USER:  # a module's file does not move while the process runs
        _USER[key] = bool(f) and not os.path.realpath(f).startswith(_LIB_DIRS)
    return _USER[key]


def _user_modules(home: types.ModuleType) -> list[types.ModuleType]:
    """The stage's module and the application modules it imports, transitively."""
    seen, stack = {home.__name__: home}, [home]
    while stack:
        m = stack.pop()
        for obj in list(vars(m).values()):
            mod = obj if isinstance(obj, types.ModuleType) else sys.modules.get(getattr(obj, "__module__", "") or "")
            if isinstance(mod, types.ModuleType) and mod.__name__ not in seen and is_user_module(mod):
                seen[mod.__name__] = mod
                stack.append(mod)
    return list(seen.values())


def _is_asset_handle(obj) -> bool:
    """Objects created by a node-local asset module that cannot be imported by name (e.g. the
    lazy loaders ``assets.once`` returns)."""
    if type(obj) is not types.FunctionType or obj.__module__ not in LOCAL_MODULES:
        return False
    mod = sys.modules.get(obj.__module__)
    return getattr(mod, obj.__name__, None) is not obj


class _Pickler(cloudpickle.Pickler):
    # Asset handles are recognized in reducer_override, which the C pickler consults only for
    # objects outside its built-in types (a persistent_id hook would be called back from C for
    # every object: half of the packaging time went into those calls)
    def reducer_override(self, obj):
        if type(obj) is types.FunctionType and _is_asset_handle(obj):
            buf = io.BytesIO()
            _Pickler(buf, protocol=5).dump(_HandleBody(obj))
            return _asset_handle, (_handle_key(obj), buf.getvalue())
        return super().reducer_override(obj)


def _handle_key(fn) -> str:
    h = hashlib.sha256(fn.__module__.encode() + marshal.dumps(fn.__code__))
    for c in fn.__closure__ or ():
        v = c.cell_contents
        if v is fn:
            continue
        h.update(f"{getattr(v, '__module__', '')}.{getattr(v, '__qualname__', type(v).__name__)}".encode())
        h.update(_source(v).encode() if callable(v) else repr(v).encode())
    return h.hexdigest()[:32]


class _HandleBody:
    """An asset handle is rebuilt on the worker inside its node-local module's namespace (shared
    locks and registries); only its closure (the user's builder) travels by value."""

    def __init__(self, fn):
        self.fn = fn

    def __reduce__(self):
        fn = self.fn
        cells = tuple(_SELF if c.cell_contents is fn else c.cell_contents for c in fn.__closure__ or ())
        return (_rebuild_handle, (fn.__module__, marshal.dumps(fn.__code__), fn.__name__, fn.__defaults__, cells))


_SELF = "__hedices_self__"  # a closure cell that refers to the handle itself


def _rebuild_handle(modname, code, name, defaults, cells):
    mod = importlib.import_module(modname)
    closure = tuple(types.CellType(None if isinstance(c, str) and c == _SELF else c) for c in cells)
    fn = types.FunctionType(marshal.loads(code), mod.__dict__, name, defaults, closure)
    for cell, c in zip(closure, cells):
        if isinstance(c, str) and c == _SELF:
            cell.cell_contents = fn
    return fn


_HANDLES: dict[str, object] = {}
_HANDLES_LOCK = threading.RLock()  # handle bodies may hold handles


_LOADING = threading.local()  # handle table of the package being loaded on this thread


# runtime hook applied to what asset handles return (the Branch installs its model optimizer)
VALUE_HOOK = None


def _hooked(handle, hook):
    def get():
        return hook(handle())
    return get


def _asset_handle(key, body):
    """Unpickling an asset handle: one instance per handle table (node-wide, or per load)."""
    handles = getattr(_LOADING, "handles", None)
    table = _HANDLES if handles is None else handles
    with _HANDLES_LOCK:
        if key not in table:
            fn = _load(body, handles)
            table[key] = _hooked(fn, VALUE_HOOK) if VALUE_HOOK is not None else fn
        return table[key]


def _load(blob: bytes, handles: dict | None = None):
    prev = getattr(_LOADING, "handles", None)
    _LOADING.handles = handles
    try:
        if not PKG_SLICE:  # whole-module bundles of a module name are identical: no lock needed
            return pickle.loads(blob)
        with _BUNDLE_LOCK:  # (a lock here serializes every package load of the Branch)
            return pickle.loads(blob)
    finally:
        _LOADING.handles = prev


def load_package(blob: bytes, cold: bool = False):
    """Worker side: rebuild a stage function from its package.  Asset handles stay node-resident
    across packages unless ``cold`` (no function cache): then the function and the state it
    initializes (loaded models) are rebuilt for this invocation alone."""
    return _load(blob, {} if cold else None)


@dataclass(frozen=True)
class FunctionPackage:
    name: str
    hash: str
    blob: bytes
    manifest: tuple
    version: str
    build_s: float

    @property
    def size(self) -> int:
        return len(self.blob)


def _code_names(code: types.CodeType) -> set[str]:
    names = set(code.co_names)
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            names |= _code_names(const)
    return names


def _closure(fn) -> tuple[dict[str, object], set[types.ModuleType]]:
    """The stage function plus same-module helpers it references, transitively."""
    home = fn.__module__
    items: dict[str, object] = {}
    modules: set[types.ModuleType] = set()
    stack = [fn]
    while stack:
        f = stack.pop()
        key = f"{f.__module__}.{f.__qualname__}"
        if key in items:
            continue
        items[key] = f
        if not inspect.isfunction(f):
            continue
        g = f.__globals__
        for name in _code_names(f.__code__):
            obj = g.get(name)
            if obj is None:
                continue
            if isinstance(obj, types.ModuleType):
                modules.add(obj)
                continue
            spec = getattr(obj, "__hedices_stage__", None)
            if spec is not None:
                obj = spec.fn
            owner = getattr(obj, "__module__", None)
            if (inspect.isfunction(obj) or inspect.isclass(obj)) and owner == home:
                stack.append(obj)
            elif hasattr(obj, "content_digest"):  # by-value data, e.g. shipped model weights
                items[f"{home}.{name}"] = obj
            elif owner and owner in sys.modules:
                modules.add(sys.modules[owner])
    return items, modules


_SOURCES: dict = {}


def _source(obj) -> str:
    if hasattr(obj, "content_digest"):
        return obj.content_digest
    key = getattr(obj, "__code__", None)
    if key is not None and key in _SOURCES:  # source text of a loaded code object is fixed
        return _SOURCES[key]
    try:
        src = textwrap.dedent(inspect.getsource(obj))
    except (OSError, TypeError):
        code = getattr(obj, "__code__", None)
        src = code.co_code.hex() if code is not None else repr(obj)
    if key is not None:
        _SOURCES[key] = src
    return src


def _manifest(modules, home: str) -> tuple:
    tops = set()
    for m in modules:
        top = m.__name__.split(".")[0]
        if top in _STDLIB or m.__name__ == home or top.startswith("_"):
            continue
        tops.add(top)
    return tuple((top, _dist_version(top)) for top in sorted(tops))


_VERSIONS: dict[str, str] = {}


def _dist_version(top: str) -> str:
    """Installed version of a library (fixed for the life of the process: looked up once)."""
    if top not in _VERSIONS:
        try:
            _VERSIONS[top] = importlib.metadata.version(top)
        except importlib.metadata.PackageNotFoundError:
            _VERSIONS[top] = str(getattr(sys.modules.get(top), "__version__", ""))
    return _VERSIONS[top]


def _install_submodule_index() -> None:
    """cloudpickle finds the submodules a function's globals reach by scanning all of
    sys.modules once per function; index them by package instead, rebuilt whenever a module is
    imported (same result, without the scan that dominates packaging a stage)."""
    cp = sys.modules.get("cloudpickle.cloudpickle")
    if cp is None or getattr(cp, "_hedices_indexed", False):
        return
    state = {"n": -1, "by_pkg": {}}

    def submodules(prefix):
        if state["n"] != len(sys.modules):
            by_pkg: dict[str, list[str]] = {}
            for name in list(sys.modules):
                if name is None:
                    continue
                parts = name.split(".")
                for i in range(1, len(parts)):
                    by_pkg.setdefault(".".join(parts[:i]) + ".", []).append(name)
            state["n"], state["by_pkg"] = len(sys.modules), by_pkg
        return state["by_pkg"].get(prefix, ())

    memo: dict = {}

    def _find_imported_submodules(code, top_level_dependencies):
        deps = list(top_level_dependencies)
        key = (len(sys.modules), code, tuple(id(x) for x in deps))
        hit = memo.get(key)
        if hit is not None:
            return list(hit[1])
        subimports = []
        names = set(code.co_names)
        for x in deps:
            if isinstance(x, types.ModuleType) and getattr(x, "__package__", None):
                prefix = x.__name__ + "."
                for name in submodules(prefix):
                    if name in sys.modules and not set(name[len(prefix):].split(".")) - names:
                        subimports.append(sys.modules[name])
        memo[key] = (deps, subimports)  # holding deps keeps the ids in the key unique
        return list(subimports)

    cp._find_imported_submodules = _find_imported_submodules
    cp._hedices_indexed = True


_install_submodule_index()


_PICKLE_LOCK = threading.Lock()


_MODULES_BY_SOURCE: dict[str, types.ModuleType] = {}
_MODULES_LOCK = threading.Lock()


# Branch side: the module sources a package ships are compiled by helper interpreters next to the
# Branch (set_compilers), so that loading a package does not hold this interpreter's lock for the
# compile -- calls already executing on the worker keep running while a package is parsed.  The
# helper returns the code object marshalled (as a .pyc holds it); loading it here is a fraction of
# the compile.  Same code; a broken helper falls back to compiling here.
_COMPILER_SRC = r"""
import marshal, struct, sys
rd, wr = sys.stdin.buffer, sys.stdout.buffer
def readn(n):
    b = b""
    while len(b) < n:
        c = rd.read(n - len(b))
        if not c:
            raise SystemExit
        b += c
    return b
while True:
    n1, n2 = struct.unpack("<II", readn(8))
    name, src = readn(n1).decode(), readn(n2).decode()
    try:
        out = b"\0" + marshal.dumps(compile(src, name, "exec"))
    except BaseException as e:
        out = b"\1" + repr(e).encode()
    wr.write(struct.pack("<I", len(out)) + out)
    wr.flush()
"""


class _Compilers:
    def __init__(self, n: int):
        import queue
        import subprocess
        self.free: "queue.Queue" = queue.Queue()
        for _ in range(n):
            self.free.put(subprocess.Popen([sys.executable, "-I", "-S", "-c", _COMPILER_SRC],
                                           stdin=subprocess.PIPE, stdout=subprocess.PIPE))

    @staticmethod
    def _readn(f, n: int) -> bytes:
        b = b""
        while len(b) < n:
            c = f.read(n - len(b))
            if not c:
                raise EOFError
            b += c
        return b

    def compile(self, source: str, filename: str):
        import struct
        p = self.free.get()
        try:
            a, s = filename.encode(), source.encode()
            p.stdin.write(struct.pack("<II", len(a), len(s)) + a + s)
            p.stdin.flush()
            out = self._readn(p.stdout, struct.unpack("<I", self._readn(p.stdout, 4))[0])
        except Exception:  # a dead helper: compile here, and do not hand the helper out again
            p = None
            return compile(source, filename, "exec")
        finally:
            if p is not None:
                self.free.put(p)
        if out[:1] != b"\0":
            return compile(source, filename, "exec")  # raises the same error here
        return marshal.loads(out[1:])


_COMPILERS: _Compilers | None = None
# set by the Branch: True while other calls run there, i.e. while compiling here would hold the lock
# they need; a load with nothing else running compiles here (no round trip to a helper)
COMPILE_CONTENDED = None


def set_compilers(n: int) -> None:
    """Branch start-up: ``n`` helper interpreters compile the module sources of received packages
    (0: compile in this interpreter)."""
    global _COMPILERS
    _COMPILERS = _Compilers(n) if n > 0 else None
    if _COMPILERS is not None:
        _COMPILERS.compile("pass", "<warm>")


def _install_module(name: str, source: str, h: str):
    """Worker side of a module package: the module is executed once per content hash and kept
    in sys.modules (a re-received package is parsed again but keeps the module's state, e.g. the
    asset handles it created)."""
    # one dispatch message (a stage and the successors chained on it) parses a module source it carries
    # several times over once: the stages of a chain ship the same application module
    seen = getattr(_LOADING, "message", None)
    if seen is not None and h in seen:
        with _MODULES_LOCK:
            mod = _MODULES_BY_SOURCE.get(h)
            if mod is not None:
                sys.modules[name] = mod
                return mod
    fname = f"<hedices:{name}>"
    offload = _COMPILERS is not None and (COMPILE_CONTENDED is None or COMPILE_CONTENDED())
    code = _COMPILERS.compile(source, fname) if offload else compile(source, fname, "exec")
    with _MODULES_LOCK:
        mod = _MODULES_BY_SOURCE.get(h)
        if mod is None:
            mod = types.ModuleType(name)
            mod.__package__ = name.rpartition(".")[0]
            mod.__hedices_hash__ = h
            sys.modules[name] = mod
            try:
                exec(code, mod.__dict__)
            except BaseException:
                sys.modules.pop(name, None)
                raise
            if VALUE_HOOK is not None:  # the asset handles the module created, as for by-value ones
                for k, v in list(vars(mod).items()):
                    if _is_asset_handle(v):
                        setattr(mod, k, _hooked(v, VALUE_HOOK))
            _MODULES_BY_SOURCE[h] = mod
        sys.modules[name] = mod
    if seen is not None:
        seen.add(h)
    return mod


def _module_order(user: list) -> list:
    """Dependency packages of a stage in import order (a module after the user modules it uses)."""
    names = {m.__name__ for m in user}
    order, seen = [], set()

    def visit(m):
        if m.__name__ in seen:
            return
        seen.add(m.__name__)
        for x in vars(m).values():
            dep = x if isinstance(x, types.ModuleType) else sys.modules.get(getattr(x, "__module__", "") or "")
            if isinstance(dep, types.ModuleType) and dep.__name__ in names and dep is not m:
                visit(dep)
        order.append(m)
    for m in sorted(user, key=lambda m: m.__name__):
        visit(m)
    return order


_MODULE_PKGS: dict = {}


def module_package(m: types.ModuleType, version: str = "1") -> FunctionPackage:
    src = _shipped_source(m)
    key = (m.__name__, src, version)
    pkg = _MODULE_PKGS.get(key)
    if pkg is None:
        t0 = time.perf_counter()
        mods = {sys.modules[x.__module__] for x in vars(m).values()
                if getattr(x, "__module__", None) in sys.modules and not is_user_module(sys.modules[x.__module__])}
        mods |= {x for x in vars(m).values() if isinstance(x, types.ModuleType) and not is_user_module(x)}
        manifest = _manifest(mods, m.__name__)
        h = hashlib.sha256(f"module:{m.__name__}".encode() + src.encode() + repr(manifest).encode()
                           + version.encode()).hexdigest()[:32]
        buf = io.BytesIO()
        pickle.dump(_ModuleBody(m.__name__, src, h), buf, protocol=5)
        pkg = _MODULE_PKGS[key] = FunctionPackage(m.__name__, h, buf.getvalue(), manifest, version,
                                                  time.perf_counter() - t0)
    return pkg


class _ModuleBody:
    def __init__(self, name, source, h):
        self.name, self.source, self.h = name, source, h

    def __reduce__(self):
        return (_install_module, (self.name, self.source, self.h))


def build_packages(spec: StageSpec, version: str = "1") -> list[FunctionPackage]:
    """The packages U_v of a stage: its function package first, then (HEDICES_SPLIT_DEPS) the
    application modules it imports, in the order a worker installs them."""
    fn = spec.fn
    if not SPLIT_DEPS and PKG_FORMAT != "source":
        return [build_package(spec, version)]
    module = sys.modules.get(fn.__module__)
    user = _user_modules(module) if module is not None and fn.__module__ != "__main__" else []
    deps = [m for m in _module_order(user) if m.__name__ != fn.__module__]
    if not SPLIT_DEPS:
        return [source_package(spec, deps, version)]
    return [build_package(spec, version, by_ref=deps)] + [module_package(m, version) for m in deps]


# HEDICES_PKG_SLICE=1 (default off): function-granular packages.  Of each application module a stage
# imports, its package carries only the top-level statements the stage function reaches --
# transitively: the helpers it calls, the constants, imports and decorators they use, and what
# those reach in the other carried modules -- instead of the whole module (which holds the code of
# every stage of the application).  Statements that bind no name are always kept.  0 ships whole
# modules (the default: sliced packages made every miss so cheap that the RR / alpha0 / nocache gaps
# fell below the reference, v2t 2026-11-13).
PKG_SLICE = os.environ.get("HEDICES_PKG_SLICE", "0") == "1"


def _bound_names(st) -> set | None:
    """Top-level names a statement binds (None: binds none, kept in every slice)."""
    import ast
    if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {st.name}
    if isinstance(st, ast.Import):
        return {a.asname or a.name.split(".")[0] for a in st.names}
    if isinstance(st, ast.ImportFrom):
        return None if st.module == "__future__" else {a.asname or a.name for a in st.names}
    targets = (st.targets if isinstance(st, ast.Assign) else
               [st.target] if isinstance(st, (ast.AnnAssign, ast.AugAssign)) else None)
    if targets is None:
        return None
    out = set()
    for t in targets:
        for n in ast.walk(t):
            if isinstance(n, ast.Name):
                out.add(n.id)
    return out or None


def _slices(deps: list, need: dict) -> dict:
    """{module name: source} -- each module of ``deps`` cut to what ``need`` ({module name: names}
    used from it) reaches."""
    import ast
    dep = {m.__name__: m for m in deps}
    mods = {}
    for name, m in dep.items():
        body = ast.parse(_shipped_source(m)).body
        defs: dict[str, list[int]] = {}
        always = []
        for i, st in enumerate(body):
            b = _bound_names(st)
            if b is None:
                always.append(i)
            else:
                for n in b:
                    defs.setdefault(n, []).append(i)
        # names of this module bound to another carried module (import aliases), or to an object
        # imported from one (``from x import y``)
        alias = {k: v.__name__ for k, v in vars(m).items() if isinstance(v, types.ModuleType) and v.__name__ in dep}
        imported = {k: (v.__module__, getattr(v, "__name__", k)) for k, v in vars(m).items()
                    if not isinstance(v, types.ModuleType) and getattr(v, "__module__", None) in dep
                    and v.__module__ != name}
        mods[name] = (body, defs, always, alias, imported)
    keep = {name: set() for name in dep}
    work = [(mod, n) for mod, ns in need.items() if mod in dep for n in ns]
    whole = set()

    def take(mod, i):
        if i in keep[mod]:
            return
        keep[mod].add(i)
        body, defs, _, alias, imported = mods[mod]
        for node in ast.walk(body[i]):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in alias:
                work.append((alias[node.value.id], node.attr))
            if isinstance(node, ast.Name):
                if node.id in defs:
                    work.append((mod, node.id))
                if node.id in imported:
                    work.append(imported[node.id])
                if node.id in alias and not any(isinstance(p, ast.Attribute) and p.value is node
                                                for p in ast.walk(body[i])):
                    whole.add(alias[node.id])  # the module object itself is used: keep all of it

    for mod in dep:
        for i in mods[mod][2]:
            take(mod, i)
    while work or whole:
        while work:
            mod, n = work.pop()
            for i in mods[mod][1].get(n, ()):
                take(mod, i)
        if whole:
            mod = whole.pop()
            for i in range(len(mods[mod][0])):
                take(mod, i)
    out = {}
    for mod in dep:
        body = mods[mod][0]
        out[mod] = ast.unparse(ast.Module(body=[body[i] for i in sorted(keep[mod])], type_ignores=[])) + "\n"
    return out


def _stage_needs(fn, deps: list) -> dict:
    """{carried module name: names the stage function (and its same-module helpers) use from it}:
    every name its code mentions, for a module it references (an over-approximation: attribute
    names that are not top-level names of the module select nothing)."""
    items, _ = _closure(fn)
    dep = {m.__name__ for m in deps}
    need = {n: set() for n in dep}
    for f in items.values():
        if not inspect.isfunction(f):
            continue
        names = _code_names(f.__code__)
        for n in names:
            v = f.__globals__.get(n)
            if isinstance(v, types.ModuleType) and v.__name__ in dep:
                need[v.__name__] |= names
            elif getattr(v, "__module__", None) in dep:
                need[v.__module__].add(getattr(v, "__name__", n))
    return need


def source_package(spec: StageSpec, deps: list, version: str = "1") -> FunctionPackage:
    """One package holding the sources of the stage's application modules (installed first, in
    import order; PKG_SLICE: only what the stage reaches of each) and the stage function, which
    refers to them by name."""
    t0 = time.perf_counter()
    entry = build_package(spec, version, by_ref=deps)
    mods = [module_package(m, version) for m in deps]  # hash and manifest per module source
    if PKG_SLICE:
        cut = _slices(deps, _stage_needs(spec.fn, deps))
        srcs = [cut[m.__name__] for m in deps]
        hashes = [hashlib.sha256(f"slice:{m.__name__}".encode() + s.encode() + repr(p.manifest).encode()
                                 + version.encode()).hexdigest()[:32] for m, s, p in zip(deps, srcs, mods)]
    else:
        srcs, hashes = [_shipped_source(m) for m in deps], [p.hash for p in mods]
    buf = io.BytesIO()
    pickle.dump(_Bundle([_ModuleBody(m.__name__, s, h) for m, s, h in zip(deps, srcs, hashes)],
                        entry.blob), buf, protocol=5)
    h = hashlib.sha256("".join([entry.hash] + hashes).encode()).hexdigest()[:32]
    manifest = tuple(sorted(set(entry.manifest).union(*(p.manifest for p in mods))))
    return FunctionPackage(spec.name, h, buf.getvalue(), manifest, version, time.perf_counter() - t0)


class _Bundle:
    def __init__(self, modules, entry):
        self.modules, self.entry = modules, entry

    def __reduce__(self):  # the module bodies unpickle (install) before the entry is loaded
        return (_load_entry, (self.modules, self.entry))


# a bundle's modules are installed and its entry bound to them under one lock: with sliced
# packages two stages carry different bodies of the same module name, and an entry must find the
# bodies of its own bundle in sys.modules
_BUNDLE_LOCK = threading.RLock()


def _load_entry(modules, entry):
    for mod in modules:
        sys.modules[mod.__name__] = mod
    return pickle.loads(entry)


def build_package(spec: StageSpec, version: str = "1", by_ref=()) -> FunctionPackage:
    t0 = time.perf_counter()
    fn = spec.fn
    items, modules = _closure(fn)
    module = sys.modules.get(fn.__module__)
    user = _user_modules(module) if module is not None and fn.__module__ != "__main__" else []
    if by_ref:  # shipped as dependency packages, imported by name on the worker
        refs = {m.__name__ for m in by_ref}
        user = [m for m in user if m.__name__ not in refs]
    for m in user:
        modules |= {sys.modules[x.__module__] for x in vars(m).values()
                    if getattr(x, "__module__", None) in sys.modules and not is_user_module(sys.modules[x.__module__])}
        modules |= {x for x in vars(m).values() if isinstance(x, types.ModuleType)}
    with _PICKLE_LOCK:  # by-value registration is process-global in cloudpickle
        for m in user:
            cloudpickle.register_pickle_by_value(m)
        try:
            buf = io.BytesIO()
            _Pickler(buf, protocol=5).dump(fn)
            blob = buf.getvalue()
        finally:
            for m in user:
                cloudpickle.unregister_pickle_by_value(m)
    manifest = _manifest({m for m in modules if not is_user_module(m)}, fn.__module__)
    h = hashlib.sha256()
    for key in sorted(items):
        h.update(key.encode())
        h.update(_source(items[key]).encode())
    for m in sorted(user, key=lambda m: m.__name__):
        if m.__name__ != fn.__module__:
            h.update(m.__name__.encode())
            h.update(inspect.getsource(m).encode())
    h.update(repr(manifest).encode())
    h.update(version.encode())
    return FunctionPackage(spec.name, h.hexdigest()[:32], blob, manifest, version,
                           time.perf_counter() - t0)


def check_manifest(manifest) -> None:
    """Worker-side check that every library dependency is installed locally."""
    missing = [name for name, _ in manifest if importlib.util.find_spec(name) is None]
    if missing:
        raise ImportError(f"package needs modules missing on this worker: {missing}")
