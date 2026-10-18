#!/usr/bin/env bash
# ALEI demo: keyword rules vs classifier on 3 redacted awards. One screen.
#
# Interpreter, first one found:
#   1. $PYTHON                      e.g.  PYTHON=/path/to/python ./run_demo.sh
#   2. demo/.venv/bin/python
#   3. <repo>/.venv/bin/python      make it with:  python3.12 -m venv .venv && .venv/bin/pip install -r requirements-s390x.txt
#   4. python3 on PATH
# The classifier needs the versions in VERSIONS.txt. demo.py checks them before loading the model and,
# on a mismatch, explains how to fix it (and how to run the rules baseline) instead of failing.
set -euo pipefail
cd "$(dirname "$0")"
if [ -n "${PYTHON:-}" ]; then PY="$PYTHON"
elif [ -x ./.venv/bin/python ]; then PY=./.venv/bin/python
elif [ -x ../.venv/bin/python ]; then PY=../.venv/bin/python
else PY=python3
fi
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec "$PY" demo.py 2>&1
