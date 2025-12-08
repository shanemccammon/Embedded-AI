#!/usr/bin/env bash
set -euo pipefail

echo "[1/4] Creating virtual environment (.venv) if missing..."
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi

echo "[2/4] Activating virtual environment..."
# shellcheck disable=SC1091
source .venv/bin/activate

echo "[3/4] Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "[4/4] Running model pipeline..."
# Prevent legacy keras import issues if user has TF_USE_LEGACY_KERAS exported globally
unset TF_USE_LEGACY_KERAS 2>/dev/null || true
#also force the env for this process to not include legacy flag
env -u TF_USE_LEGACY_KERAS python model.py

echo "Done. Artifacts in ./models"