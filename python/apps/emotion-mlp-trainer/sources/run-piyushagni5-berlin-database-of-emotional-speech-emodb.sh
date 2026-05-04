#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
python3 ./sources/piyushagni5_berlin_database_of_emotional_speech_emodb.py
