#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate
python data/download.py
python preprocess.py
python features.py
python train.py
python evaluate.py
python explain.py
echo "Done. Serve with: uvicorn app.backend.main:app --host 127.0.0.1 --port 8000"
