"""W4 Fusion: branching camera + IMU pipeline with a cloud-side 4 MB fusion CNN (Fig. 1d)."""
import os

import numpy as np

from apps.common import assets as A
from apps.common import yolox as Y

INPUT = 320
GRID = 64
ACTIVITIES = ["idle", "walking", "running", "cycling", "driving", "stairs", "falling", "sitting",
              "standing", "lying"]


def new_detector(threads=None):
    return Y.yolox_session("yoloxna-320.onnx", threads)


load_detector = A.once(new_detector)


def _fusion_arch():
    nn = A.torch().nn
    return nn.Sequential(
        nn.Conv2d(24, 64, 3, padding=1), nn.ReLU(),
        nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
        nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(),
        nn.Conv2d(256, 256, 3, stride=2, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, len(ACTIVITIES)))


@A.once
def load_fusion():
    return A.seeded(_fusion_arch)


@A.segmentwise
def preprocess_frame(frame):
    canvas, ratio = Y.letterbox(A.unjpeg(frame), INPUT)
    return A.jpeg(canvas), ratio


def detect(tensor, sess=None, score_thr=0.25):
    x, ratio = tensor
    sess = sess or load_detector()
    pred = Y.yolox_decode(sess.run(None, {sess.get_inputs()[0].name: Y.normalize(A.unjpeg(x))})[0], INPUT)[0]
    scores = pred[:, 4:5] * pred[:, 5:]
    cls = scores.argmax(1)
    conf = scores[np.arange(len(cls)), cls]
    m = conf > score_thr
    b = pred[m, :4]
    xyxy = np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2,
                     b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1) / ratio
    return np.concatenate([xyxy, conf[m, None], cls[m, None]], 1).astype(np.float32)


def filter_and_normalize(imu_sample):
    k = np.full(9, 1 / 9, np.float32)
    smooth = np.stack([np.convolve(imu_sample[:, c], k, mode="same") for c in range(6)], 1)
    z = (smooth - smooth.mean(0)) / (smooth.std(0) + 1e-6)
    energy = (np.abs(np.fft.rfft(z, axis=0)[1:9]) ** 2).mean(0)
    return np.concatenate([smooth.mean(0), smooth.std(0), energy[:4]]).astype(np.float32)


# the same computation as detect / filter_and_normalize, as the separate steps of the pipeline
@A.segmentwise
def detect_candidates(tensor, sess=None, score_thr=0.25):
    """Run the detector; the candidates above the score threshold as (cx, cy, w, h, conf, cls)."""
    x, ratio = tensor
    sess = sess or load_detector()
    pred = Y.yolox_decode(sess.run(None, {sess.get_inputs()[0].name: Y.normalize(A.unjpeg(x))})[0], INPUT)[0]
    scores = pred[:, 4:5] * pred[:, 5:]
    cls = scores.argmax(1)
    conf = scores[np.arange(len(cls)), cls]
    m = conf > score_thr
    return np.concatenate([pred[m, :4], conf[m, None], cls[m, None]], 1).astype(np.float32), ratio


@A.segmentwise
def to_boxes(candidates):
    """Centre/size candidates to corner boxes in the coordinates of the original frame."""
    c, ratio = candidates
    b = c[:, :4]
    xyxy = np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2,
                     b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1) / ratio
    return np.concatenate([xyxy, c[:, 4:]], 1).astype(np.float32)


@A.segmentwise
def smooth_imu(imu_sample):
    k = np.full(9, 1 / 9, np.float32)
    return np.stack([np.convolve(imu_sample[:, c], k, mode="same") for c in range(6)], 1)


@A.segmentwise
def imu_features(smooth):
    z = (smooth - smooth.mean(0)) / (smooth.std(0) + 1e-6)
    energy = (np.abs(np.fft.rfft(z, axis=0)[1:9]) ** 2).mean(0)
    return np.concatenate([smooth.mean(0), smooth.std(0), energy[:4]]).astype(np.float32)


@A.segmentwise
def fusion_model(objects, imu_state, model=None):
    """Rasterize detections into per-class occupancy maps, stack IMU features, return logits."""
    t = A.torch()
    x = np.zeros((24, GRID, GRID), np.float32)
    for x1, y1, x2, y2, conf, cls in objects:
        c = int(cls) % 8
        a, b = int(max(x1, 0) / INPUT * GRID), int(max(y1, 0) / INPUT * GRID)
        e, f = int(min(x2, INPUT - 1) / INPUT * GRID) + 1, int(min(y2, INPUT - 1) / INPUT * GRID) + 1
        x[c, b:f, a:e] = np.maximum(x[c, b:f, a:e], conf)
    x[8:24] = imu_state[:, None, None]
    with t.no_grad():
        return (model or load_fusion())(t.from_numpy(x)[None])[0].numpy()


@A.segmentwise
def aggregate_decision(logits):
    p = np.exp(logits - logits.max())
    p /= p.sum()
    k = int(p.argmax())
    return {"activity": ACTIVITIES[k], "confidence": round(float(p[k]), 4)}


def imu_window(idx, n=200):
    rng = np.random.default_rng(idx)
    t = np.arange(n, dtype=np.float32)[:, None]
    base = np.sin(t / 15.0 + rng.uniform(0, 6.28, (1, 6))).astype(np.float32)
    return base + 0.3 * rng.standard_normal((n, 6)).astype(np.float32)


def imu_length():
    """Samples of an instance's IMU window: 200, times HEDICES_INPUT_IMU (a per-figure input setting,
    configs/<name>.json: a longer IMU recording per camera frame)."""
    return 200 * max(1, int(os.environ.get("HEDICES_INPUT_IMU", "1")))


def inputs(n=8):
    k, m = A.segment(), imu_length()
    mix = [int(x) for x in os.environ.get("HEDICES_INPUT_SEGMENT_MIX", "").split(",") if x.strip()]
    if mix:
        # a per-figure input setting: instance i carries mix[i % len(mix)] consecutive frames (a camera
        # that delivers a varying number of frames per trigger); stages run per frame as for a segment
        out, j = [], 0
        for i in range(n):
            kk = max(1, mix[i % len(mix)])
            out.append((f"w4-{i}", A.Segment(A.jpeg(A.frame(f, (INPUT, INPUT))) for f in range(j, j + kk)),
                        A.Segment(imu_window(f, m) for f in range(j, j + kk))))
            j += kk
        return out
    if k == 1:
        return [(f"w4-{i}", A.jpeg(A.frame(i, (INPUT, INPUT))), imu_window(i, m)) for i in range(n)]
    # a segment: k consecutive camera frames with the IMU window of each (stages run per frame)
    return [(f"w4-{i}", A.Segment(A.jpeg(A.frame(j, (INPUT, INPUT))) for j in range(i * k, (i + 1) * k)),
             A.Segment(imu_window(j, m) for j in range(i * k, (i + 1) * k))) for i in range(n)]
