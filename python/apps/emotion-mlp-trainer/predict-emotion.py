#!/usr/bin/env python3
"""Predict emotion probabilities for a WAV file using a trained emotion MLP."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from emotion_mlp_common import (
    MODEL_FILENAME,
    TRAINING_METADATA_FILENAME,
    extract_feature_vector,
    validate_feature_order,
)


def build_parser() -> argparse.ArgumentParser:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Classify emotion in a WAV file.")
    parser.add_argument("wav_path", type=Path, help="WAV file to classify.")
    parser.add_argument("--feature-config", type=Path, default=script_dir / "feature-extractors.yaml")
    parser.add_argument("--model-dir", type=Path, default=script_dir / "models" / "emotion_mlp")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON only.")
    return parser


def predict(args: argparse.Namespace) -> dict[str, object]:
    if not args.wav_path.is_file():
        raise FileNotFoundError(f"WAV file not found: {args.wav_path}")
    if not args.feature_config.is_file():
        raise FileNotFoundError(f"Feature config not found: {args.feature_config}")

    import pyafex
    import tensorflow as tf

    metadata_path = args.model_dir / TRAINING_METADATA_FILENAME
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Training metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    feature_names, feature_values = extract_feature_vector(pyafex, args.feature_config, args.wav_path)
    validate_feature_order(feature_names, list(metadata["feature_names"]))

    features = np.asarray([feature_values], dtype=np.float32)

    model = tf.keras.models.load_model(args.model_dir / MODEL_FILENAME)
    probabilities = model.predict(features, verbose=0)[0]
    label_order = list(metadata["label_order"])
    ranked = sorted(
        (
            {"label": label, "probability": float(probabilities[index])}
            for index, label in enumerate(label_order)
        ),
        key=lambda item: item["probability"],
        reverse=True,
    )
    return {
        "wav_path": str(args.wav_path),
        "prediction": ranked[0]["label"],
        "confidence": ranked[0]["probability"],
        "probabilities": ranked,
        "features": {name: float(value) for name, value in zip(feature_names, feature_values, strict=True)},
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = predict(args)
    except Exception as error:  # noqa: BLE001 - CLI should report failures without traceback by default.
        print(f"predict-emotion: {error}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Prediction: {result['prediction']} ({result['confidence']:.1%})")
        print("Probabilities:")
        for item in result["probabilities"]:
            print(f"  {item['label']:<10} {item['probability']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
