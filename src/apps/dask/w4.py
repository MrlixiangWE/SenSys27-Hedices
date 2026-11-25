import shutil
from pathlib import Path

from dask.distributed import Client, PipInstall, WorkerPlugin, get_worker

import apps
from apps.common import assets as A
from apps.common import w4 as L

GB = 1 << 30


class Models(WorkerPlugin):
    """Loads YOLOX on edge workers and the fusion CNN on cloud workers before tasks arrive, with
    native threads limited to the CPUs one task reserves (workers run several tasks concurrently)."""

    name = "w4-models"

    def setup(self, worker):
        tiers = worker.state.total_resources
        if "edge" in tiers:
            self.detector = L.new_detector(RESOURCES[detect_objects]["CPU"])
        if "cloud" in tiers:
            A.torch().set_num_threads(RESOURCES[fuse]["CPU"])
            self.fusion = L.load_fusion()


def preproc_frame(frame):
    return L.preprocess_frame(frame)


def detect_objects(tensor):
    return L.detect_candidates(tensor, get_worker().plugins[Models.name].detector)


def to_boxes(candidates):
    return L.to_boxes(candidates)


def smooth_imu(imu_sample):
    return L.smooth_imu(imu_sample)


def imu_features(smooth):
    return L.imu_features(smooth)


def fuse(objects, imu_state):
    return L.fusion_model(objects, imu_state, get_worker().plugins[Models.name].fusion)


def aggregate(logits):
    return L.aggregate_decision(logits)


# Consecutive stages of one tier run as a single Dask task (what dask.optimize's linear-chain fusion
# does to a delayed graph): the scheduler is asked once per tier instead of once per stage, and the
# intermediate values stay in the worker's memory.  The stage functions are the ones above.
def frame_stages(frame):
    return to_boxes(detect_objects(preproc_frame(frame)))


def imu_stages(imu_sample):
    return imu_features(smooth_imu(imu_sample))


def cloud_stages(objects, imu_state):
    return aggregate(fuse(objects, imu_state))


# placement tier and resource demand of each stage (workers advertise their tier, CPU and MEM)
RESOURCES = {preproc_frame: {"edge": 1, "CPU": 1, "MEM": GB // 2},
             detect_objects: {"edge": 1, "CPU": 2, "MEM": 2 * GB},
             to_boxes: {"edge": 1, "CPU": 1, "MEM": GB // 2},
             smooth_imu: {"edge": 1, "CPU": 1, "MEM": GB // 2},
             imu_features: {"edge": 1, "CPU": 1, "MEM": GB // 2},
             fuse: {"cloud": 1, "CPU": 4, "MEM": 4 * GB},
             aggregate: {"cloud": 1, "CPU": 1, "MEM": GB // 2}}
RESOURCES[frame_stages] = {"edge": 1, "CPU": 2, "MEM": 2 * GB}  # the largest demand among its stages
RESOURCES[imu_stages] = {"edge": 1, "CPU": 1, "MEM": GB // 2}
RESOURCES[cloud_stages] = {"cloud": 1, "CPU": 4, "MEM": 4 * GB}


def submit(client, stage, req_id, *args):
    # explicit keys: Dask would otherwise merge repeated triggers that carry identical inputs
    return client.submit(stage, *args, key=f"{stage.__name__}-{req_id}", resources=RESOURCES[stage])


def workflow(client, req_id, frame, imu_sample):
    objects = submit(client, frame_stages, req_id, frame)
    imu_state = submit(client, imu_stages, req_id, imu_sample)
    return submit(client, cloud_stages, req_id, objects, imu_state)


def deploy(client, packages):
    """Install dependencies and ship the application package to every worker."""
    client.register_plugin(PipInstall(packages))
    client.upload_file(shutil.make_archive("apps", "zip", Path(apps.__file__).parent.parent, "apps"))


if __name__ == "__main__":
    client = Client()
    deploy(client, ["numpy", "pillow", "onnxruntime", "torch"])
    client.register_plugin(Models())
    futures = [workflow(client, *args) for args in L.inputs()]
    for out in client.gather(futures):
        print(out)
