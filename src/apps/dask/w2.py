import shutil
from pathlib import Path

from dask.distributed import Client, PipInstall, WorkerPlugin, get_worker

import apps
from apps.common import w2 as L

GB = 1 << 30


class Models(WorkerPlugin):
    """Loads the YOLOX session once on every edge worker before tasks arrive, with native threads
    limited to the CPUs one retrack task reserves (workers run several tasks concurrently)."""

    name = "w2-models"

    def setup(self, worker):
        if "edge" in worker.state.total_resources:
            self.detector = L.new_detector(RESOURCES[retrack]["CPU"])


def read_frame(frame_idx):
    return L.load_frame(frame_idx)


def preprocess(frame):
    return L.preprocess(frame)


def retrack(tensor, frame_idx):
    return L.redetect(tensor, frame_idx, get_worker().plugins[Models.name].detector)


def track(frame, dets, frame_idx):
    return L.update_tracks(frame, dets, frame_idx)


def aggregate(tracks):
    return L.summarize_tracks(tracks)


def track_update(summary):
    return L.track_update(summary)


# Consecutive stages of one tier run as a single Dask task (what dask.optimize's linear-chain fusion
# does to a delayed graph): the scheduler is asked once per tier instead of once per stage, and the
# intermediate values stay in the worker's memory.  The stage functions are the ones above.
def edge_stages(frame_idx):
    frame = read_frame(frame_idx)
    return track(frame, retrack(preprocess(frame), frame_idx), frame_idx)


# placement tier and resource demand of each stage (workers advertise their tier, CPU and MEM)
RESOURCES = {read_frame: {"edge": 1, "CPU": 1, "MEM": GB},
             preprocess: {"edge": 1, "CPU": 1, "MEM": GB // 2},
             retrack: {"edge": 1, "CPU": 2, "MEM": GB},
             track: {"edge": 1, "CPU": 1, "MEM": GB // 2},
             aggregate: {"cloud": 1, "CPU": 1, "MEM": GB // 2},
             track_update: {"edge": 1, "CPU": 1, "MEM": GB // 4}}
RESOURCES[edge_stages] = {"edge": 1, "CPU": 2, "MEM": GB}  # the largest demand among its stages


def submit(client, stage, req_id, *args):
    # explicit keys: Dask would otherwise merge repeated triggers that carry identical inputs
    return client.submit(stage, *args, key=f"{stage.__name__}-{req_id}", resources=RESOURCES[stage])


def workflow(client, req_id, frame_idx):
    tracks = submit(client, edge_stages, req_id, frame_idx)
    summary = submit(client, aggregate, req_id, tracks)
    return submit(client, track_update, req_id, summary)


def deploy(client, packages):
    """Install dependencies and ship the application package to every worker."""
    client.register_plugin(PipInstall(packages))
    client.upload_file(shutil.make_archive("apps", "zip", Path(apps.__file__).parent.parent, "apps"))


if __name__ == "__main__":
    client = Client()
    deploy(client, ["numpy", "pillow", "onnxruntime"])
    client.register_plugin(Models())
    futures = [workflow(client, *args) for args in L.inputs()]
    for out in client.gather(futures):
        print(out)
