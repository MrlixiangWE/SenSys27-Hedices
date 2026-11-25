#!/bin/bash
# Experiment 4: comparison with Ray / Dask / KubeEdge / Knative -- Fig 10, Fig 11, Table 4, Fig 12, Fig 13 (about 3.2 hours a round)
# usage: bash run.sh <your-name>        (ROUNDS=<n> for more rounds; default 1)
# The testbed runs one job at a time: this queues the run and prints when it starts. Results: results/ae/<id>-<name>/
exec bash "${HEDICES_ROOT:-/opt/hedices}/repro/queue.sh" submit "${1:?usage: bash run.sh <your-name>}" sync bexp summary
