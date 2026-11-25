#!/bin/bash
# Experiment 1: scheduler ablation (Fig 6) and dispatch distribution per device class (Fig 7) (about 1.5 hours)
# usage: bash run.sh <your-name>        (ROUNDS=<n> for more rounds; default 1)
# The testbed runs one job at a time: this queues the run and prints when it starts. Results: results/ae/<id>-<name>/
exec bash "${HEDICES_ROOT:-/opt/hedices}/repro/queue.sh" submit "${1:?usage: bash run.sh <your-name>}" sync fig6 fig7 summary
