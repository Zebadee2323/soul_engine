#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
python3 ./predict-emotion.py --model-dir ./models/emotion_mlp "$@"
