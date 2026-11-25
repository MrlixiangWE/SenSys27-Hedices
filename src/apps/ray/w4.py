import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

import apps
from apps.common import assets as A
from apps.common import w4 as L
from apps.ray.balance import LeastLoaded

EDGE, CLOUD = {"edge": 1}, {"cloud": 1}
RUNTIME_ENV = {"py_modules": [apps], "pip": ["numpy", "pillow", "onnxruntime", "torch"]}  # ship app code and deps


@ray.remote(num_cpus=1, memory=512 << 20, resources=EDGE)
def preproc_frame(frame):
    return L.preprocess_frame(frame)


@ray.remote(memory=2 << 30, resources=EDGE, max_restarts=-1, max_task_retries=-1)
class Detector:
    """Keeps the YOLOX session resident (a task would reload it whenever Ray recycles its worker
    process); restarted with its session on another node after a failure."""

    def __init__(self, threads):
        self.sess = L.new_detector(threads)  # native threads sized to the CPUs this replica holds

    def detect(self, tensor):
        return L.detect_candidates(tensor, self.sess)


# num_cpus=0: a few microseconds of post-processing on the model replica's output.  Ray leases a
# task where its argument lives; the replicas hold every CPU of the small edge nodes, and a task
# asking for a CPU there waits for the raylet's next scheduling round (about 100 ms) before it
# spills to another node.
@ray.remote(num_cpus=0, memory=512 << 20, resources=EDGE)
def to_boxes(candidates):
    return L.to_boxes(candidates)


@ray.remote(num_cpus=1, memory=512 << 20, resources=EDGE)
def smooth_imu(imu_sample):
    return L.smooth_imu(imu_sample)


@ray.remote(num_cpus=1, memory=512 << 20, resources=EDGE)
def imu_features(smooth):
    return L.imu_features(smooth)


@ray.remote(memory=4 << 30, resources=CLOUD, max_restarts=-1, max_task_retries=-1)
class Fusion:
    """Keeps the fusion CNN resident on cloud nodes; restarted with its model after a failure."""

    def __init__(self, threads):
        A.torch().set_num_threads(threads)
        self.model = L.load_fusion()

    def fuse(self, objects, imu_state):
        return L.fusion_model(objects, imu_state, self.model)


@ray.remote(num_cpus=1, memory=512 << 20, resources=CLOUD)
def aggregate(logits):
    return L.aggregate_decision(logits)


def replicas(actor, tier, cpus, share):
    """Per-device replicas: each node of the tier gets as many as its CPU share allows, pinned there."""
    pool = []
    for node in ray.nodes():
        if node["Alive"] and tier in node["Resources"]:
            pin = NodeAffinitySchedulingStrategy(node["NodeID"], soft=True)
            for _ in range(max(1, int(node["Resources"]["CPU"] * share) // cpus)):
                pool.append(actor.options(num_cpus=cpus, scheduling_strategy=pin).remote(cpus))
    return LeastLoaded(pool)


def workflow(detectors, fusions, req_id, frame, imu_sample):
    tensor = preproc_frame.remote(frame)
    candidates = next(detectors).detect.remote(tensor)
    objects = to_boxes.remote(candidates)
    smooth = smooth_imu.remote(imu_sample)
    imu_state = imu_features.remote(smooth)
    logits = next(fusions).fuse.remote(objects, imu_state)
    return aggregate.remote(logits)


if __name__ == "__main__":
    ray.init(runtime_env=RUNTIME_ENV)
    detectors = replicas(Detector, "edge", cpus=2, share=0.5)
    fusions = replicas(Fusion, "cloud", cpus=4, share=0.5)
    refs = [workflow(detectors, fusions, *args) for args in L.inputs()]
    for out in ray.get(refs):
        print(out)
