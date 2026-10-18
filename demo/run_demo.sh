#!/usr/bin/env bash
# ALEI demo: keyword rules vs classifier on 3 redacted awards. One screen.
#   ./run_demo.sh                 uses python3 on PATH
#   PYTHON=/path/to/python ./run_demo.sh
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec "$PY" demo.py 2>&1
