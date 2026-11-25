from hedices import NoRely, Stem, Workflow
from hedices.master import Master
from apps.common import w4 as L


@Stem("edge", cpus=1, mem="512MB")
@NoRely(flag="req_id")
def preproc_frame(req_id, frame):
    return L.preprocess_frame(frame)


@Stem("edge", cpus=2, mem="2GB")
@NoRely(flag="req_id")
def detect_objects(req_id, tensor):
    return L.detect_candidates(tensor)


@Stem("edge", cpus=1, mem="512MB")
@NoRely(flag="req_id")
def to_boxes(req_id, candidates):
    return L.to_boxes(candidates)


@Stem("edge", cpus=1, mem="512MB")
@NoRely(flag="req_id")
def smooth_imu(req_id, imu_sample):
    return L.smooth_imu(imu_sample)


@Stem("edge", cpus=1, mem="512MB")
@NoRely(flag="req_id")
def imu_features(req_id, smooth):
    return L.imu_features(smooth)


@Stem("cloud", cpus=4, mem="4GB")
def fuse(req_id, objects, imu_state):
    return L.fusion_model(objects, imu_state)


@Stem("cloud", cpus=1, mem="512MB")
def aggregate(req_id, logits):
    return L.aggregate_decision(logits)


def workflow(req_id, frame, imu_sample):
    tensor = preproc_frame(req_id, frame)
    candidates = detect_objects(req_id, tensor)
    objects = to_boxes(req_id, candidates)
    smooth = smooth_imu(req_id, imu_sample)
    imu_state = imu_features(req_id, smooth)
    logits = fuse(req_id, objects, imu_state)
    return aggregate(req_id, logits)


if __name__ == "__main__":
    with Master() as stem:
        wf = Workflow(workflow)
        futures = [stem.submit(wf, *args) for args in L.inputs()]
        for f in futures:
            print(f.result())
