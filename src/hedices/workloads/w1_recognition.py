"""W1 Recognition: appliance-state recognition from a current trace (paper Listing 1)."""
import os

import numpy as np

from hedices import NoRely, Stem
from hedices.workloads import models as M

USE_LEGACY_MLP = os.environ.get("HEDICES_W1_MODEL", "lstm") == "legacy_mlp"
RECOGNIZER = M.app_model("lstm")


def decode_power_trace(signal):
    return np.frombuffer(signal, dtype=np.int32).astype(np.float32)


def normalize_window(trace):
    x = (trace - trace.mean()) / (trace.std() + 1e-6)
    return np.convolve(x, np.full(5, 0.2, np.float32), mode="same").astype(np.float32)


def lstm_infer(features):
    t = M.torch()
    x = t.from_numpy(np.ascontiguousarray(features)).view(1, -1)
    with t.no_grad():
        if USE_LEGACY_MLP:
            return np.array([int(m(x).argmax(axis=1)[0]) for m in M.legacy_mlps()])
        return RECOGNIZER(x).argmax(-1).numpy()[0]


def majority_vote(label):
    return sorted(M.APPLIANCES[i] for i, on in enumerate(label) if on)


@Stem("edge", cpus=1, mem="1GB")
def load_signal(req_id, signal):
    return decode_power_trace(signal)


@Stem("edge", cpus=2, mem="1GB")
@NoRely(flag="req_id")
def preproc(req_id, trace):
    return normalize_window(trace)


@Stem("edge", cpus=2, mem="2GB")
@NoRely(flag="req_id")
def recognize(req_id, features):
    return lstm_infer(features)


@Stem("cloud", cpus=1, mem="512MB")
def aggregate(req_id, label):
    return majority_vote(label)


def workflow(req_id, signal):
    trace = load_signal(req_id, signal)
    features = preproc(req_id, trace)
    label = recognize(req_id, features)
    return aggregate(req_id, label)


def make_input(i: int):
    w = M.current_windows()
    return (f"w1-{i}", np.asarray(w[i % len(w)], dtype=np.int32).tobytes()), {}
