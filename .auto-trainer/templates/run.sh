#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=4
PY="/Users/hook/miniforge3/bin/python"
"$PY" preflight.py
"$PY" train.py
"$PY" eval.py
