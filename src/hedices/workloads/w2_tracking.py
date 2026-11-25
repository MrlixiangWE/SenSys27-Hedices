"""W2 Tracking: YOLOX detect-then-track on 224x224 frames with periodic re-detection.

Cloud-side frame I/O, edge-side inference.  Frames are letterboxed into the
320x320 input of the YOLOX-nano model used in the 2025 experiments.
"""
import numpy as np

from hedices import NoRely, Stem
from hedices.workloads import models as M

REDETECT_EVERY = 3
INPUT = 320


def letterbox(img, size=INPUT):
    r = min(size / img.shape[0], size / img.shape[1])
    from PIL import Image

    h, w = int(img.shape[0] * r), int(img.shape[1] * r)
    out = np.full((size, size, 3), 114, dtype=np.uint8)
    out[:h, :w] = np.asarray(Image.fromarray(img).resize((w, h), Image.BILINEAR))
    x = out[:, :, ::-1].astype(np.float32) / 255.0
    x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None]), r


def decode(outputs, size=INPUT):
    grids, strides = [], []
    for stride in (8, 16, 32):
        n = size // stride
        xv, yv = np.meshgrid(np.arange(n), np.arange(n))
        grids.append(np.stack((xv, yv), 2).reshape(1, -1, 2))
        strides.append(np.full((1, n * n, 1), stride))
    grids, strides = np.concatenate(grids, 1), np.concatenate(strides, 1)
    outputs[..., :2] = (outputs[..., :2] + grids) * strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * strides
    return outputs


def nms(boxes, scores, thr):
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order, keep = scores.argsort()[::-1], []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(ovr <= thr)[0] + 1]
    return keep


def yolo_infer(frame, score_thr=0.25, nms_thr=0.45):
    x, ratio = letterbox(frame)
    sess = M.yolox()
    pred = decode(sess.run(None, {sess.get_inputs()[0].name: x})[0])[0]
    boxes = pred[:, :4]
    scores = pred[:, 4:5] * pred[:, 5:]
    cls = scores.argmax(1)
    conf = scores[np.arange(len(cls)), cls]
    m = conf > score_thr
    if not m.any():
        return np.zeros((0, 6), np.float32)
    b = boxes[m]
    xyxy = np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2,
                     b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1) / ratio
    keep = nms(xyxy, conf[m], nms_thr)
    return np.concatenate([xyxy[keep], conf[m][keep, None], cls[m][keep, None]], 1).astype(np.float32)


def update_tracks(frame, dets, frame_idx):
    """Appearance descriptors (colour histograms) for each tracked box."""
    if dets is None or len(dets) == 0:
        h, w = frame.shape[:2]
        dets = np.array([[0, 0, w, h, 0.0, -1]], np.float32)
    tracks = []
    for k, (x1, y1, x2, y2, conf, cls) in enumerate(dets):
        patch = frame[max(int(y1), 0):max(int(y2), 1), max(int(x1), 0):max(int(x2), 1)]
        if patch.size == 0:
            continue
        hist = np.concatenate([np.histogram(patch[..., c], bins=16, range=(0, 255))[0] for c in range(3)])
        tracks.append((k, int(cls), float(conf), hist.astype(np.float32) / max(patch.size, 1)))
    return {"frame": frame_idx, "redetect": frame_idx % REDETECT_EVERY == 0, "tracks": tracks}


@Stem("cloud", cpus=1, mem="1GB")
def read_frame(req_id, frame_idx):
    return M.frame(frame_idx, (224, 224))


@Stem("edge", cpus=2, mem="1GB")
@NoRely(flag="req_id")
def detect(req_id, frame, frame_idx):
    if frame_idx % REDETECT_EVERY:
        return None
    return yolo_infer(frame)


@Stem("edge", cpus=1, mem="512MB")
@NoRely(flag="req_id")
def track(req_id, frame, dets, frame_idx):
    return update_tracks(frame, dets, frame_idx)


@Stem("cloud", cpus=1, mem="512MB")
def aggregate(req_id, tracks):
    return {"frame": tracks["frame"], "n": len(tracks["tracks"]),
            "classes": sorted({c for _, c, _, _ in tracks["tracks"] if c >= 0})}


def workflow(req_id, frame_idx):
    frame = read_frame(req_id, frame_idx)
    dets = detect(req_id, frame, frame_idx)
    tracks = track(req_id, frame, dets, frame_idx)
    return aggregate(req_id, tracks)


def make_input(i: int):
    return (f"w2-{i}", i), {}
