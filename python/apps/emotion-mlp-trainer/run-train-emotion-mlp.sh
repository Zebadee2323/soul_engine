#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
rm -rf ./models/emotion_mlp/
python3 ./train-emotion-mlp.py --train-data-dir ./train_data/ --model-dir ./models/emotion_mlp "$@"
