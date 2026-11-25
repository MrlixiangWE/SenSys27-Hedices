"""Node-local model/data assets shared by the four applications, loaded once per process."""
import os
import threading
from pathlib import Path

import numpy as np

ASSETS = Path(os.environ.get("HEDICES_ASSETS", Path(__file__).resolve().parents[2] / "assets"))
_lock = threading.RLock()


class Segment(list):
    """The k consecutive samples (frames, IMU windows) one instance carries under a per-figure
    input setting HEDICES_INPUT_SEGMENT=k (configs/<name>.json); see ``segmentwise``."""


def segment() -> int:
    return max(1, int(os.environ.get("HEDICES_INPUT_SEGMENT", "1")))


def segmentwise(fn):
    """A stage function over one sample that also takes a Segment: it then runs on each sample in
    turn (the other arguments, e.g. a model handle, go to every call) and returns the Segment of
    the results.  Without a Segment argument the call is unchanged."""
    import functools

    @functools.wraps(fn)
    def run(*args, **kw):
        n = next((len(a) for a in args if isinstance(a, Segment)), None)
        if n is None:
            return fn(*args, **kw)
        cols = [a if isinstance(a, Segment) else [a] * n for a in args]
        return Segment(fn(*xs, **kw) for xs in zip(*cols))
    return run


def once(build):
    """Build an asset at most once per worker process, even under concurrent callers."""
    def get():
        if not hasattr(get, "value"):
            with _lock:
                if not hasattr(get, "value"):
                    get.value = build()
        return get.value
    return get


@once
def torch():
    import torch as t

    if os.environ.get("OMP_NUM_THREADS"):
        t.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    return t


def seeded(arch):
    """Instantiate a network with fixed initial weights (no trained checkpoint is shipped)."""
    torch().manual_seed(0)
    return arch().eval()


@once
def frames():
    z = np.load(ASSETS / "pics40.npz", allow_pickle=True)
    return [np.ascontiguousarray(f) for f in z["img"]]


def frame(idx, size):
    from PIL import Image

    f = frames()[idx % len(frames())]
    return np.asarray(Image.fromarray(f).resize(size, Image.BICUBIC))


def jpeg(img, quality=90):
    """Frames travel between stages JPEG-encoded, as a camera delivers them."""
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def unjpeg(data):
    import io

    from PIL import Image

    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
