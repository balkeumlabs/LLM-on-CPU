#!/usr/bin/env bash
# Convenience wrapper (Linux/macOS). Forwards all args to the master script.
# Usage: ./run.sh [--report-only] [--model NAME] [--cpu-only]
set -euo pipefail
cd "$(dirname "$0")"
# Prefer the project venv if it exists, else system python.
if [ -x ".venv/bin/python" ]; then
  PY=".venv/bin/python"
else
  PY="$(command -v python3 || command -v python)"
fi
exec "$PY" run_experiment.py "$@"
