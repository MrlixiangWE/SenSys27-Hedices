"""Worker preload for baselines: import the workload dependencies at worker start (warm process)."""
from hedices.workloads import models as M  # noqa: F401

M.torch()
import onnxruntime  # noqa: F401,E402
import torchvision  # noqa: F401,E402
