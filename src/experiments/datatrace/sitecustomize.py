"""Measurement-only hook (Table 4): with HEDICES_DATATRACE=<dir> on PYTHONPATH's front, every
process that imports a shared stage module (apps.common.w1-w4) records, per stage call, the node
it ran on and the digest and serialized size of each input and of the output.  Matching digests
between producer and consumer calls tells which application data crossed nodes.  The benchmark
applications are not modified; the hook wraps the shared functions as they are imported."""
import os

if os.environ.get("HEDICES_DATATRACE"):
    from experiments.datatrace import trace

    trace.install()
