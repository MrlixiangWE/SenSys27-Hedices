#!/bin/bash
# The job queue of the testbed:  bash queue.sh status | watch [ID] | cancel ID
exec bash "${HEDICES_ROOT:-/opt/hedices}/repro/queue.sh" "$@"
