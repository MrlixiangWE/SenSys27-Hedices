import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor

import requests

from apps.common import w3 as L

URLS = {s: os.environ.get(f"{s.upper()}_URL", f"http://w3-{s.replace('_', '-')}.default.svc.cluster.local")
        for s in ("load_image", "preprocess", "infer", "postprocess", "aggregate")}
# per-call bound: a call to a replica on a failed node is abandoned after this and retried on another
# replica (the Service picks the endpoint per connection) instead of waiting for the node to be declared lost
STAGE_TIMEOUT = float(os.environ.get("HEDICES_K8S_STAGE_TIMEOUT", "5"))
session = requests.Session()


RETRY_S = float(os.environ.get("HEDICES_K8S_RETRY_S", "120"))


def call(stage, *args):
    """Invoke one stage service; a call lost with a failed pod is retried on another replica for up to RETRY_S.
    The bound is in time, not attempts: a refused connection or a 503 from a route whose only replica is gone
    returns at once, a call lost with its pod after STAGE_TIMEOUT -- either way the retries cover the time
    Kubernetes takes to replace the replica."""
    deadline = time.monotonic() + RETRY_S
    while True:
        try:
            r = session.post(f"{URLS[stage]}/{stage}", data=pickle.dumps(args), timeout=STAGE_TIMEOUT)
            r.raise_for_status()
            return pickle.loads(r.content)
        except requests.RequestException:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.5)  # a refused connection (route not ready, pod gone) fails at once: let it settle


def workflow(req_id, frame_idx):
    image = call("load_image", frame_idx)
    tensor = call("preprocess", image)
    raw = call("infer", tensor)
    dets = call("postprocess", raw)
    return call("aggregate", dets)


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=8) as pool:
        for out in pool.map(lambda args: workflow(*args), L.inputs()):
            print(out)
