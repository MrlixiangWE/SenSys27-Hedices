#!/bin/bash
# Experiment 2: cache and alignment dynamics, Fig 8 (about 5 minutes)
# usage: bash run.sh <your-name>        (ROUNDS=<n> for more rounds; default 1)
# The testbed runs one job at a time: this queues the run and prints when it starts. Results: results/ae/<id>-<name>/
exec bash "${HEDICES_ROOT:-/opt/hedices}/repro/queue.sh" submit "${1:?usage: bash run.sh <your-name>}" sync fig8 summary
