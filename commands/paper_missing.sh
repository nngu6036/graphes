#!/usr/bin/env bash
# Complete commands for the missing-items report. Default is preview only.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$SCRIPT_DIR/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON_BIN:-python}" scripts/run_grapher_paper_experiments.py "$@"
