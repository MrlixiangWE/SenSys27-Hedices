#!/bin/bash
# Build the native execution channel for this host's Python (run on each host).
set -e
cd "$(dirname "$0")"
PY=${PY:-../../.venv/bin/python}
INC=$($PY -c "import sysconfig; print(sysconfig.get_paths()['include'])")
EXT=$($PY -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
g++ -O2 -std=c++17 -shared -fPIC -Wall -I"$INC" _hedx.cpp -o "_hedx$EXT" -lrt
echo "built _hedx$EXT"
