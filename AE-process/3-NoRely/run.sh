#!/bin/bash
# Experiment 3: local parallelism with @NoRely, Fig 9 (about 1 hour)
# usage: bash run.sh <your-name>        (ROUNDS=<n> for more rounds; default 1)
# The testbed runs one job at a time: this queues the run and prints when it starts. Results: results/ae/<id>-<name>/
exec bash "${HEDICES_ROOT:-/opt/hedices}/repro/queue.sh" submit "${1:?usage: bash run.sh <your-name>}" sync fig9 summary
