#!/usr/bin/env bash
set -euo pipefail
python -m src.evaluate
python -m src.run_experiments
# --- IGNORE ---