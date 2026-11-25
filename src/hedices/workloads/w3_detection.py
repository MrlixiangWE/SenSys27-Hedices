"""W3 Detection: Faster R-CNN with cloud-side I/O and post-processing, edge-side inference.

Default (scaled down for the 4-core testbed): MobileNetV3-320 Faster R-CNN on
320x320 frames.  HEDICES_W3_MODEL=resnet50 selects the 2025 ResNet-50 model
on 600x600 frames, decoded as in the 2025 code (DecodeBox + per-class NMS).
"""
import numpy as np

from hedices import NoRely, Stem
from hedices.workloads import models as M

NUM_CLASSES = 20
STD = (0.1, 0.1, 0.2, 0.2)


def preprocess_input(image):
    x = image.astype(np.float32) / 255.0
    return np.ascontiguousarray(np.transpose(x, (2, 0, 1))[None])


def loc2bbox(t, src, loc):
    w = src[:, 2:3] - src[:, 0:1]
    h = src[:, 3:4] - src[:, 1:2]
    cx = src[:, 0:1] + 0.5 * w
    cy = src[:, 1:2] + 0.5 * h
    dx, dy, dw, dh = loc[:, 0::4], loc[:, 1::4], loc[:, 2::4], loc[:, 3::4]
    ncx, ncy = dx * w + cx, dy * h + cy
    nw, nh = t.exp(dw) * w, t.exp(dh) * h
    out = t.zeros_like(loc)
    out[:, 0::4], out[:, 1::4] = ncx - 0.5 * nw, ncy - 0.5 * nh
    out[:, 2::4], out[:, 3::4] = ncx + 0.5 * nw, ncy + 0.5 * nh
    return out


def decode_boxes(raw, shape, confidence=0.5, nms_iou=0.3):
    t = M.torch()
    from torchvision.ops import nms

    cls_locs, scores, rois = (t.from_numpy(a) for a in raw)
    n = NUM_CLASSES + 1
    loc = (cls_locs[0] * t.tensor(STD).repeat(n)).view(-1, n, 4)
    roi = rois.view(-1, 1, 4).expand_as(loc)
    boxes = loc2bbox(t, roi.reshape(-1, 4), loc.reshape(-1, 4)).view(-1, n, 4)
    prob = t.softmax(scores[0], dim=-1)
    out = []
    for c in range(1, n):
        m = prob[:, c] > confidence
        if m.any():
            b, s = boxes[m, c], prob[m, c]
            keep = nms(b, s, nms_iou)
            for k in keep.tolist():
                out.append([*b[k].tolist(), float(s[k]), c - 1])
    return np.array(out, np.float32).reshape(-1, 6)


def filter_candidates(raw, confidence=0.02, nms_iou=0.3):
    t = M.torch()
    from torchvision.ops import batched_nms

    boxes, scores, labels = (t.from_numpy(a) for a in raw)
    m = scores > confidence
    boxes, scores, labels = boxes[m], scores[m], labels[m]
    keep = batched_nms(boxes, scores, labels, nms_iou)
    return t.cat([boxes[keep], scores[keep, None], (labels[keep, None] - 1).float()], 1).numpy()


@Stem("cloud", cpus=1, mem="1GB")
def load_image(req_id, frame_idx):
    return M.frame(frame_idx, (M.W3_INPUT, M.W3_INPUT))


@Stem("edge", cpus=2, mem="2GB")
@NoRely(flag="req_id")
def infer(req_id, image):
    t = M.torch()
    with t.no_grad():
        if M.W3_MODEL == "resnet50":
            cls_locs, scores, rois, _ = M.frcnn()(t.from_numpy(preprocess_input(image)))
            return "resnet50", (cls_locs.numpy(), scores.numpy(), rois.numpy())
        out = M.frcnn_mobilenet()([t.from_numpy(preprocess_input(image))[0]])[0]
        return "mobilenet", (out["boxes"].numpy(), out["scores"].numpy(), out["labels"].numpy())


@Stem("cloud", cpus=2, mem="1GB")
def postprocess(req_id, raw):
    kind, tensors = raw
    if kind == "resnet50":
        return decode_boxes(tensors, (600, 600))
    return filter_candidates(tensors)


@Stem("cloud", cpus=1, mem="512MB")
def aggregate(req_id, dets):
    classes, counts = np.unique(dets[:, 5].astype(int), return_counts=True) if len(dets) else ([], [])
    return {int(c): int(n) for c, n in zip(classes, counts)}


def workflow(req_id, frame_idx):
    image = load_image(req_id, frame_idx)
    raw = infer(req_id, image)
    dets = postprocess(req_id, raw)
    return aggregate(req_id, dets)


def make_input(i: int):
    return (f"w3-{i}", i), {}
