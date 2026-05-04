#!/usr/bin/env python3
"""Predict emotion probabilities for WAV files using a trained emotion MLP."""

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
    parser = argparse.ArgumentParser(description="Classify emotion in one WAV file or a folder of WAV files.")
    parser.add_argument("wav_path", type=Path, help="WAV file or directory of .wav files to classify.")
    parser.add_argument("--feature-config", type=Path, default=script_dir / "feature-extractors.yaml")
    parser.add_argument("--model-dir", type=Path, default=script_dir / "models" / "emotion_mlp")
    parser.add_argument("--recursive", action="store_true", help="Search directories recursively for .wav files.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of ranked emotions to show in compact output.")
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        help="Keep processing remaining WAV files if one file fails.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON only.")
    return parser


def find_wav_paths(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".wav":
            raise ValueError(f"Input file is not a .wav: {input_path}")
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"WAV file or directory not found: {input_path}")

    iterator = input_path.rglob("*") if recursive else input_path.iterdir()
    wav_paths = sorted(path for path in iterator if path.is_file() and path.suffix.lower() == ".wav")
    if not wav_paths:
        search_kind = "recursive " if recursive else ""
        raise FileNotFoundError(f"No .wav files found in {search_kind}directory: {input_path}")
    return wav_paths


def ranked_probabilities(probabilities: np.ndarray, label_order: list[str]) -> list[dict[str, object]]:
    return sorted(
        (
            {"label": label, "probability": float(probabilities[index])}
            for index, label in enumerate(label_order)
        ),
        key=lambda item: item["probability"],
        reverse=True,
    )


def load_predictor(args: argparse.Namespace):
    if not args.feature_config.is_file():
        raise FileNotFoundError(f"Feature config not found: {args.feature_config}")

    import pyafex
    import tensorflow as tf

    metadata_path = args.model_dir / TRAINING_METADATA_FILENAME
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Training metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model = tf.keras.models.load_model(args.model_dir / MODEL_FILENAME)
    return pyafex, model, metadata


def predict_wav(
    *,
    pyafex,
    model,
    metadata: dict[str, object],
    feature_config: Path,
    wav_path: Path,
) -> dict[str, object]:
    feature_names, feature_values = extract_feature_vector(pyafex, feature_config, wav_path)
    validate_feature_order(feature_names, list(metadata["feature_names"]))

    features = np.asarray([feature_values], dtype=np.float32)

    probabilities = model.predict(features, verbose=0)[0]
    label_order = list(metadata["label_order"])
    ranked = ranked_probabilities(probabilities, label_order)
    return {
        "wav_path": str(wav_path),
        "prediction": ranked[0]["label"],
        "confidence": ranked[0]["probability"],
        "probabilities": ranked,
        "features": {name: float(value) for name, value in zip(feature_names, feature_values, strict=True)},
    }


def predict(args: argparse.Namespace) -> list[dict[str, object]]:
    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1.")

    wav_paths = find_wav_paths(args.wav_path, args.recursive)
    pyafex, model, metadata = load_predictor(args)

    results: list[dict[str, object]] = []
    for wav_path in wav_paths:
        try:
            results.append(
                predict_wav(
                    pyafex=pyafex,
                    model=model,
                    metadata=metadata,
                    feature_config=args.feature_config,
                    wav_path=wav_path,
                )
            )
        except Exception as error:  # noqa: BLE001 - batch mode can optionally keep going.
            if not args.skip_errors:
                raise
            results.append({"wav_path": str(wav_path), "error": str(error)})
    return results


def format_probability(item: dict[str, object]) -> str:
    return f"{item['label']} {float(item['probability']):.1%}"


def print_compact_results(results: list[dict[str, object]], input_path: Path, top_k: int) -> None:
    if len(results) == 1 and "error" not in results[0] and input_path.is_file():
        result = results[0]
        print(f"Prediction: {result['prediction']} ({float(result['confidence']):.1%})")
        print("Probabilities:")
        for item in result["probabilities"]:
            print(f"  {item['label']:<10} {float(item['probability']):.4f}")
        return

    rows: list[tuple[str, str, str, str]] = []
    for result in results:
        wav_path = Path(str(result["wav_path"]))
        display_path = str(wav_path)
        try:
            display_path = str(wav_path.relative_to(input_path if input_path.is_dir() else input_path.parent))
        except ValueError:
            pass

        if "error" in result:
            rows.append((display_path, "ERROR", "-", str(result["error"])))
            continue

        top_probabilities = ", ".join(
            format_probability(item) for item in result["probabilities"][:top_k]
        )
        rows.append(
            (
                display_path,
                str(result["prediction"]),
                f"{float(result['confidence']):.1%}",
                top_probabilities,
            )
        )

    headers = ("file", "prediction", "conf", f"top {top_k}")
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    print(f"{headers[0]:<{widths[0]}}  {headers[1]:<{widths[1]}}  {headers[2]:>{widths[2]}}  {headers[3]}")
    print(f"{'-' * widths[0]}  {'-' * widths[1]}  {'-' * widths[2]}  {'-' * widths[3]}")
    for row in rows:
        print(f"{row[0]:<{widths[0]}}  {row[1]:<{widths[1]}}  {row[2]:>{widths[2]}}  {row[3]}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        results = predict(args)
    except Exception as error:  # noqa: BLE001 - CLI should report failures without traceback by default.
        print(f"predict-emotion: {error}", file=sys.stderr)
        return 1

    if args.json:
        payload: dict[str, object] | list[dict[str, object]]
        payload = results[0] if len(results) == 1 and args.wav_path.is_file() else results
        print(json.dumps(payload, indent=2))
    else:
        print_compact_results(results, args.wav_path, args.top_k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
