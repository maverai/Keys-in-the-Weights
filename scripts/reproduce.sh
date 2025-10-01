#!/usr/bin/env bash
set -euo pipefail
# Clean+install and run the full experiment end-to-end.
pip install -r requirements.txt
python -m src.evaluate
python -m src.run_experiments
# --- IGNORE ---