"""Name -> (workflow function, input generator) for the four workloads."""
from hedices.workloads import w1_recognition, w2_tracking, w3_detection, w4_fusion

WORKLOADS = {
    "W1": w1_recognition,
    "W2": w2_tracking,
    "W3": w3_detection,
    "W4": w4_fusion,
}
