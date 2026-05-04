#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
python3 ./sources/uldisvalainis_audio_emotions.py --max-wavs-per-emotion 3000
