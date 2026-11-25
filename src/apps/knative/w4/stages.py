import os
import pickle
import time

from flask import Flask, request

from apps.common import w4 as L

STAGES = {"preproc_frame": L.preprocess_frame,
          "detect_objects": L.detect_candidates,
          "to_boxes": L.to_boxes,
          "smooth_imu": L.smooth_imu,
          "imu_features": L.imu_features,
          "fuse": L.fusion_model,
          "aggregate": L.aggregate_decision}
MODELS = {"detect_objects": L.load_detector,
          "fuse": L.load_fusion}
STAGE = os.environ["STAGE"]  # each Deployment/Service runs one stage
POD = os.environ.get("HOSTNAME", "")  # the pod name; with the execution time, in each response (stage tracing)
app = Flask(__name__)


@app.post(f"/{STAGE}")
def run():
    args = pickle.loads(request.data)
    t = time.perf_counter()
    out = STAGES[STAGE](*args)
    ms = 1e3 * (time.perf_counter() - t)
    return pickle.dumps(out), 200, {"X-Pod": POD, "X-Exec-Ms": f"{ms:.3f}"}


if __name__ == "__main__":
    if STAGE in MODELS:
        MODELS[STAGE]()  # load before the port opens, so the queue-proxy probe gates traffic
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8080")), threaded=True)
