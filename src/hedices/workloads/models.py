"""Node-local model and data store for the workloads.

Models and datasets are provisioned on every worker ahead of time (like
node-local images).  They are loaded lazily, once per process, and are not
part of any function package.  This module is installed with the runtime, so
stage packages reference it through their dependency manifest.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np

ASSETS = Path(os.environ.get("HEDICES_ASSETS", Path(__file__).resolve().parents[2] / "assets"))
APPLIANCES = ["Lamp1", "Lamp2", "Humidifier", "Television", "AirCleaner", "Roomba", "MicrowaveOven",
              "Refrigerator", "Monitor", "Touchpad1", "Charger", "Laptop", "Display", "Touchpad2", "PC"]

_lock = threading.Lock()
_cache: dict = {}


def _get(key, build):
    value = _cache.get(key)
    if value is None:
        with _lock:
            value = _cache.get(key)
            if value is None:
                value = _cache[key] = build()
    return value


def torch():
    import torch as _torch

    def build():
        if os.environ.get("OMP_NUM_THREADS"):
            _torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
        return _torch

    return _get("torch", build)


def prepare_assets() -> None:
    """One-time conversion so workers can memory-map the W1 dataset."""
    npy = ASSETS / "w1" / "TotalCurrent.npy"
    if not npy.exists():
        d = np.load(ASSETS / "w1" / "CurrentDSet" / "FinalCurrentData.npz", allow_pickle=True)
        np.save(npy, d["TotalCurrent"])


def current_windows():
    return _get("current", lambda: np.load(ASSETS / "w1" / "TotalCurrent.npy", mmap_mode="r"))


def _lstm_arch():
    t = torch()

    class Recognizer(t.nn.Module):
        def __init__(self, n_apps=15, steps=20, feat=16, hidden=64):
            super().__init__()
            self.steps, self.feat = steps, feat
            self.lstm = t.nn.LSTM(feat, hidden, num_layers=2, batch_first=True)
            self.head = t.nn.Linear(hidden, n_apps * 2)

        def forward(self, x):
            h, _ = self.lstm(x.view(x.shape[0], self.steps, self.feat))
            return self.head(h[:, -1]).view(x.shape[0], -1, 2)

    return Recognizer()


def _seeded(arch):
    torch().manual_seed(0)
    return arch().eval()


def lstm():
    """W1 recognizer: 2-layer LSTM over a 320-sample current window, 15 appliances x on/off."""
    return _get("lstm", lambda: _seeded(_lstm_arch))


def legacy_mlps():
    """The 15 per-appliance checkpoints shipped with the 2025 experiments (MLPs)."""

    def build():
        t = torch()
        ck = ASSETS / "w1" / "model_weights" / "checkpoint"
        return [t.load(ck / f"model{i}.ckpt", map_location="cpu", weights_only=False).eval()
                for i in range(15)]

    return _get("legacy_mlps", build)


def yolox():
    """YOLOX-nano, 320x320 ONNX model from the 2025 experiments.

    One session per native-thread count: Branch grants t_v per call, and with no
    grant the library default (OMP_NUM_THREADS if set, else all cores) is used.
    """
    from hedices.runtime import native_threads

    t = native_threads()
    if t is None:
        t = int(os.environ.get("OMP_NUM_THREADS", "0"))

    def build():
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = t
        so.inter_op_num_threads = 1
        return ort.InferenceSession(str(ASSETS / "yoloxna-320.onnx"), so,
                                    providers=["CPUExecutionProvider"])

    return _get(("yolox", t), build)


W3_MODEL = os.environ.get("HEDICES_W3_MODEL", "mobilenet")
W3_INPUT = 600 if W3_MODEL == "resnet50" else 320


def frcnn_mobilenet():
    """Scaled-down W3 detector: Faster R-CNN MobileNetV3-Large 320 FPN (21 VOC classes).

    Box filtering and NMS are left to the cloud-side post-processing stage.
    """

    def build():
        import torchvision

        t = torch()
        t.manual_seed(0)
        m = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(
            weights=None, weights_backbone=None, num_classes=21,
            box_score_thresh=0.0, box_nms_thresh=1.0, box_detections_per_img=100)
        return m.eval()

    return _get("frcnn_mobilenet", build)


def frcnn():
    """Faster R-CNN (ResNet-50, VOC) TorchScript model from the 2025 experiments."""

    def build():
        import torchvision  # noqa: F401  registers torchvision::roi_pool used by the scripted model

        t = torch()
        return t.jit.load(str(ASSETS / "script_model_cpu.pt"), map_location="cpu").eval()

    return _get("frcnn", build)


def _fusion_arch():
    nn = torch().nn
    return nn.Sequential(
        nn.Conv2d(24, 64, 3, padding=1), nn.ReLU(),
        nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
        nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(),
        nn.Conv2d(256, 256, 3, stride=2, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 10))


def fusion_cnn():
    """W4 fusion CNN, ~1.0M parameters (~4 MB fp32)."""
    return _get("fusion", lambda: _seeded(_fusion_arch))


_ARCH = {"lstm": (_lstm_arch, lstm), "fusion": (_fusion_arch, fusion_cnn)}
APP_MODELS = os.environ.get("HEDICES_APP_MODELS", "local")


class AppModel:
    """Application-trained weights shipped by value inside the stage's function package.

    Pickling carries the serialized state dict; unpickling (package load on the
    worker) rebuilds the network, so the loaded model lives as long as the
    cached function and a cache miss pays transfer + load.
    """

    def __init__(self, name: str):
        import hashlib
        import io

        buf = io.BytesIO()
        torch().save(_ARCH[name][1]().state_dict(), buf)
        self.name, self.state = name, buf.getvalue()
        self.content_digest = hashlib.sha256(self.state).hexdigest()
        self._net = None

    def __getstate__(self):
        return {"name": self.name, "state": self.state, "content_digest": self.content_digest}

    def __setstate__(self, d):
        self.__dict__.update(d)
        self._net = None
        self.net()

    def net(self):
        if self._net is None:
            import io

            net = _ARCH[self.name][0]()
            net.load_state_dict(torch().load(io.BytesIO(self.state)))
            self._net = net.eval()
        return self._net

    def __call__(self, *a):
        return self.net()(*a)


class LocalModel:
    """Default: the model is a node-local asset loaded once per worker process."""

    def __init__(self, name: str):
        self.name = name

    def __call__(self, *a):
        return _ARCH[self.name][1]()(*a)


def app_model(name: str):
    return AppModel(name) if APP_MODELS == "package" else LocalModel(name)


def frames():
    def build():
        z = np.load(ASSETS / "pics40.npz", allow_pickle=True)
        return [np.ascontiguousarray(f) for f in z["img"]]

    return _get("frames", build)


def frame(idx: int, size: tuple[int, int] | None = None) -> np.ndarray:
    from PIL import Image

    f = frames()[idx % len(frames())]
    if size is None:
        return f
    return np.asarray(Image.fromarray(f).resize(size, Image.BICUBIC))


def imu_window(idx: int, n: int = 200) -> np.ndarray:
    rng = np.random.default_rng(idx)
    t = np.arange(n, dtype=np.float32)[:, None]
    base = np.sin(t / 15.0 + rng.uniform(0, 6.28, (1, 6))).astype(np.float32)
    return base + 0.3 * rng.standard_normal((n, 6)).astype(np.float32)
