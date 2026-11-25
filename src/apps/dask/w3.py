import shutil
from pathlib import Path

from dask.distributed import Client, PipInstall, WorkerPlugin, get_worker

import apps
from apps.common import assets as A
from apps.common import w3 as L

GB = 1 << 30


class Models(WorkerPlugin):
    """Loads Faster R-CNN once on every edge worker before tasks arrive, with native threads
    limited to the CPUs one infer task reserves (workers run several tasks concurrently)."""

    name = "w3-models"

    def setup(self, worker):
        if "edge" in worker.state.total_resources:
            A.torch().set_num_threads(RESOURCES[infer]["CPU"])
            self.detector = L.load_detector()


def load_image(frame_idx):
    return L.load_image(frame_idx)


def preprocess(image):
    return L.preprocess_input(image)


def infer(tensor):
    return L.frcnn_infer(tensor, get_worker().plugins[Models.name].detector)


def postprocess(raw):
    return L.postprocess(raw)


def aggregate(dets):
    return L.count_classes(dets)


# Consecutive stages of one tier run as a single Dask task (what dask.optimize's linear-chain fusion
# does to a delayed graph): the scheduler is asked once per tier instead of once per stage, and the
# intermediate values stay in the worker's memory.  The stage functions are the ones above.
def edge_stages(frame_idx):
    return infer(preprocess(load_image(frame_idx)))


def cloud_stages(raw):
    return aggregate(postprocess(raw))


# placement tier and resource demand of each stage (workers advertise their tier, CPU and MEM)
RESOURCES = {load_image: {"edge": 1, "CPU": 1, "MEM": GB},
             preprocess: {"edge": 1, "CPU": 1, "MEM": GB},
             infer: {"edge": 1, "CPU": 2, "MEM": 2 * GB},
             postprocess: {"cloud": 1, "CPU": 2, "MEM": GB},
             aggregate: {"cloud": 1, "CPU": 1, "MEM": GB // 2}}
RESOURCES[edge_stages] = {"edge": 1, "CPU": 2, "MEM": 2 * GB}  # the largest demand among its stages
RESOURCES[cloud_stages] = {"cloud": 1, "CPU": 2, "MEM": GB}


# Operator tuning from per-stage profiling, as tools/gen_k8s.py AVOID for KubeEdge / Knative: the edge
# task holds infer, which executes about 5x slower on pi5 than on NX (results/qk3); keep it off pi5.
AVOID = {edge_stages: ("pi5",)}
_workers: dict = {}


def allowed(client, stage):
    """Addresses of the edge workers a stage may run on (None: any worker with its resources)."""
    if stage not in AVOID:
        return None
    if stage not in _workers:
        info = client.scheduler_info()["workers"]
        _workers[stage] = [a for a, d in info.items() if "edge" in d.get("resources", {})
                           and not str(d.get("name", "")).startswith(AVOID[stage])] or None
    return _workers[stage]


def submit(client, stage, req_id, *args):
    # explicit keys: Dask would otherwise merge repeated triggers that carry identical inputs
    return client.submit(stage, *args, key=f"{stage.__name__}-{req_id}", resources=RESOURCES[stage],
                         workers=allowed(client, stage))


def workflow(client, req_id, frame_idx):
    raw = submit(client, edge_stages, req_id, frame_idx)
    return submit(client, cloud_stages, req_id, raw)


def deploy(client, packages):
    """Install dependencies and ship the application package to every worker."""
    client.register_plugin(PipInstall(packages))
    client.upload_file(shutil.make_archive("apps", "zip", Path(apps.__file__).parent.parent, "apps"))


if __name__ == "__main__":
    client = Client()
    deploy(client, ["numpy", "pillow", "torch", "torchvision"])
    client.register_plugin(Models())
    futures = [workflow(client, *args) for args in L.inputs()]
    for out in client.gather(futures):
        print(out)
