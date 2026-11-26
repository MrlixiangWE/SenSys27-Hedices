#!/bin/bash
# Every experiment in one job: Fig 6, 7, 8, 9, 10, 11, Table 4, Fig 12, 13 (about 15 hours with ROUNDS=1)
# usage: bash run_all.sh <your-name>     (ROUNDS=3 for the medians of three rounds, about 42 hours)
exec bash "${HEDICES_ROOT:-/opt/hedices}/repro/queue.sh" submit "${1:?usage: bash run_all.sh <your-name>}" sync hexp bexp summary
