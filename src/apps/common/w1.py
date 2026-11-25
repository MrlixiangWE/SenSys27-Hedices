"""W1 Recognition: appliance-state recognition from a household current trace (UK-DALE).

An instance carries one 320-sample window, or with HEDICES_INPUT_SEGMENT=k (a per-figure input
setting, configs/<name>.json) a recording segment of k consecutive windows: every stage handles
the k windows (per-window normalization, the LSTM over the k windows as one batch, per-window
states) and the aggregate is the majority over the windows.  k = 1 is the single-window pipeline.
"""
import numpy as np

from apps.common import assets as A

WINDOW = 320  # samples per window: 20 steps x 16 features of the LSTM

APPLIANCES = ["Lamp1", "Lamp2", "Humidifier", "Television", "AirCleaner", "Roomba", "MicrowaveOven",
              "Refrigerator", "Monitor", "Touchpad1", "Charger", "Laptop", "Display", "Touchpad2", "PC"]


@A.once
def current_windows():
    return np.load(A.ASSETS / "w1" / "TotalCurrent.npy", mmap_mode="r")


def _recognizer_arch():
    nn = A.torch().nn

    class Recognizer(nn.Module):
        """2-layer LSTM over a 320-sample window, on/off logits for 15 appliances."""

        def __init__(self, n_apps=15, steps=20, feat=16, hidden=160):
            super().__init__()
            self.steps, self.feat = steps, feat
            self.lstm = nn.LSTM(feat, hidden, num_layers=2, batch_first=True)
            self.head = nn.Linear(hidden, n_apps * 2)

        def forward(self, x):
            h, _ = self.lstm(x.view(x.shape[0], self.steps, self.feat))
            return self.head(h[:, -1]).view(x.shape[0], -1, 2)

    return Recognizer()


@A.once
def load_recognizer():
    return A.seeded(_recognizer_arch)


def decode_power_trace(signal):
    return np.frombuffer(signal, dtype=np.int32).astype(np.float32)


def normalize_window(trace):
    if trace.size > WINDOW:  # a segment: every window on its own
        return np.concatenate([normalize_window(w) for w in trace.reshape(-1, WINDOW)])
    x = (trace - trace.mean()) / (trace.std() + 1e-6)
    return np.convolve(x, np.full(5, 0.2, np.float32), mode="same").astype(np.float32)


# the same computation as normalize_window / lstm_infer, as the separate steps of the pipeline
def standardize(trace):
    return ((trace - trace.mean()) / (trace.std() + 1e-6)).astype(np.float32)


def smooth(x):
    return np.convolve(x, np.full(5, 0.2, np.float32), mode="same").astype(np.float32)


def lstm_logits(features, model=None):
    """Per-window on/off logits (15, 2); for a segment of k windows (k, 15, 2), one batch."""
    t = A.torch()
    x = t.from_numpy(np.ascontiguousarray(features)).view(-1, WINDOW)
    with t.no_grad():
        out = (model or load_recognizer())(x).numpy()
    return out[0] if len(out) == 1 else out


def decode_states(logits):
    return logits.argmax(-1)


def lstm_infer(features, model=None):
    t = A.torch()
    x = t.from_numpy(np.ascontiguousarray(features)).view(1, -1)
    with t.no_grad():
        return (model or load_recognizer())(x).argmax(-1).numpy()[0]


def majority_vote(label):
    label = np.asarray(label)
    if label.ndim == 2:  # a segment: an appliance is on when it is on in most of its windows
        label = 2 * label.sum(0) > len(label)
    return sorted(APPLIANCES[i] for i, on in enumerate(label) if on)


def inputs(n=8):
    w, k = current_windows(), A.segment()
    return [(f"w1-{i}", np.asarray(w[np.arange(i * k, (i + 1) * k) % len(w)], dtype=np.int32).tobytes())
            for i in range(n)]
