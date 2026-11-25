"""W4 Fusion: branching camera + IMU pipeline with a cloud-side fusion CNN (paper Listing 2)."""
import numpy as np

from hedices import NoRely, Stem
from hedices.workloads import models as M

INPUT = 320
FUSION = M.app_model("fusion")


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


def yolo_infer(frame, score_thr=0.25):
    x, ratio = letterbox(frame)
    sess = M.yolox()
    pred = decode(sess.run(None, {sess.get_inputs()[0].name: x})[0])[0]
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


def fusion_model(objects, imu_state, grid=64):
    t = M.torch()
    x = np.zeros((24, grid, grid), np.float32)
    for x1, y1, x2, y2, conf, cls in objects:
        c = int(cls) % 8
        a, b = int(max(x1, 0) / INPUT * grid), int(max(y1, 0) / INPUT * grid)
        e, f = int(min(x2, INPUT - 1) / INPUT * grid) + 1, int(min(y2, INPUT - 1) / INPUT * grid) + 1
        x[c, b:f, a:e] = np.maximum(x[c, b:f, a:e], conf)
    x[8:24] = imu_state[:, None, None]
    with t.no_grad():
        logits = FUSION(t.from_numpy(x)[None])
    return int(logits.argmax())


@Stem("edge", cpus=2, mem="2GB")
@NoRely(flag="req_id")
def detect_objects(req_id, frame):
    return yolo_infer(frame)


@Stem("edge", cpus=1, mem="512MB")
@NoRely(flag="req_id")
def preproc_imu(req_id, imu_sample):
    return filter_and_normalize(imu_sample)


@Stem("cloud", cpus=4, mem="4GB")
def fuse(req_id, objects, imu_state):
    return fusion_model(objects, imu_state)


def workflow(req_id, frame, imu_sample):
    objects = detect_objects(req_id, frame)
    imu_state = preproc_imu(req_id, imu_sample)
    return fuse(req_id, objects, imu_state)


def make_input(i: int):
    return (f"w4-{i}", M.frame(i, (INPUT, INPUT)), M.imu_window(i)), {}
