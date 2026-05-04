#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
MODEL_DIR="${MODEL_DIR:-./models/emotion_mlp}"
python3 ./predict-emotion.py --model-dir "$MODEL_DIR" "$@"
