#!/usr/bin/env python3
"""Build TensorFlow training data from the Kaggle audio-emotions dataset.

The output label for each WAV is a one-hot probability vector ordered as:

    angry, happy, sad, neutral, fearful, disgusted, surprised

Example:

    ./emotion-dataset-transform.py --max-wavs-per-emotion 10 --output-dir ./data/emotions
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DATASET_REF = "uldisvalainis/audio-emotions"
LABELS = ("angry", "happy", "sad", "neutral", "fearful", "disgusted", "surprised")
LABEL_ALIASES = {
    "angry": "angry",
    "anger": "angry",
    "happy": "happy",
    "happiness": "happy",
    "sad": "sad",
    "sadness": "sad",
    "neutral": "neutral",
    "fearful": "fearful",
    "fear": "fearful",
    "disgusted": "disgusted",
    "disgust": "disgusted",
    "surprised": "surprised",
    "surprise": "surprised",
    # The Kaggle dataset's surprise folder is commonly referenced with this typo.
    "suprised": "surprised",
}


@dataclass(frozen=True)
class Example:
    wav_path: Path
    label: str


def build_parser() -> argparse.ArgumentParser:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description=(
            "Download uldisvalainis/audio-emotions, extract pyafex features from "
            "WAV files, and save TensorFlow-ready training data."
        )
    )
    parser.add_argument(
        "--feature-config",
        type=Path,
        default=script_dir / "feature-extractors.yaml",
        help="pyafex YAML extractor config. Defaults to this app's feature-extractors.yaml.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "data" / "audio-emotions",
        help="Directory to write the TensorFlow dataset and metadata into.",
    )
    parser.add_argument(
        "--max-wavs-per-emotion",
        type=int,
        default=None,
        help="Maximum WAV count to include from each emotion folder. Omit to include all WAVs.",
    )
    parser.add_argument(
        "--dataset-ref",
        default=DATASET_REF,
        help=f"KaggleHub dataset ref to download. Defaults to {DATASET_REF}.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Seed used when shuffling each emotion's WAV files before applying the max limit.",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Do not shuffle WAV files before applying --max-wavs-per-emotion.",
    )
    parser.add_argument(
        "--skip-failures",
        action="store_true",
        help="Skip WAV files that fail feature extraction instead of stopping.",
    )
    return parser


def canonical_label_for_path(path: Path, dataset_root: Path) -> str | None:
    """Return the canonical emotion label represented by one of the path folders."""
    try:
        relative_parts = path.relative_to(dataset_root).parts[:-1]
    except ValueError:
        relative_parts = path.parts[:-1]

    for part in reversed(relative_parts):
        normalized = part.strip().lower().replace("-", "_").replace(" ", "_")
        if normalized in LABEL_ALIASES:
            return LABEL_ALIASES[normalized]
    return None


def collect_examples(
    dataset_root: Path,
    max_wavs_per_emotion: int | None,
    shuffle: bool,
    seed: int,
) -> list[Example]:
    """Find WAV files grouped by emotion and apply the per-emotion limit locally."""
    grouped: dict[str, list[Path]] = defaultdict(list)
    for wav_path in sorted(dataset_root.rglob("*.wav")):
        label = canonical_label_for_path(wav_path, dataset_root)
        if label is not None:
            grouped[label].append(wav_path)

    missing_labels = [label for label in LABELS if not grouped[label]]
    if missing_labels:
        raise RuntimeError(
            "No WAV files were found for emotion folder(s): "
            + ", ".join(missing_labels)
            + f"\nSearched under: {dataset_root}"
        )

    rng = random.Random(seed)
    examples: list[Example] = []
    for label in LABELS:
        wav_paths = list(grouped[label])
        if shuffle:
            rng.shuffle(wav_paths)
        if max_wavs_per_emotion is not None:
            wav_paths = wav_paths[:max_wavs_per_emotion]
        examples.extend(Example(wav_path=wav_path, label=label) for wav_path in wav_paths)

    return examples


def extract_feature_vector(pyafex: Any, config_path: Path, wav_path: Path) -> tuple[list[str], list[float]]:
    """Run pyafex and return scalar feature names and values in pyafex result order."""
    analysis = pyafex.analyze_audio_file_with_yaml(str(config_path), str(wav_path))

    feature_names: list[str] = []
    feature_values: list[float] = []
    for feature in analysis["features"]:
        name = str(feature["name"])
        status = feature.get("status")

        if status != "complete":
            raise RuntimeError(f"Feature {name!r} returned status {status!r}.")
        if "value" not in feature:
            raise RuntimeError(f"Feature {name!r} did not include a 'value' field.")
        if feature["value"] is None:
            raise RuntimeError(f"Feature {name!r} produced a null 'value'.")

        feature_names.append(name)
        feature_values.append(float(feature["value"]))

    return feature_names, feature_values


def one_hot(label: str) -> list[float]:
    return [1.0 if candidate == label else 0.0 for candidate in LABELS]


def save_training_data(
    output_dir: Path,
    feature_rows: list[list[float]],
    label_rows: list[list[float]],
    feature_names: list[str],
    examples: list[Example],
    skipped: list[dict[str, str]],
) -> None:
    """Save arrays, a tf.data.Dataset, and metadata for reproducible training."""
    import numpy as np
    import tensorflow as tf

    output_dir.mkdir(parents=True, exist_ok=True)

    features = np.asarray(feature_rows, dtype=np.float32)
    labels = np.asarray(label_rows, dtype=np.float32)

    np.save(output_dir / "features.npy", features)
    np.save(output_dir / "labels.npy", labels)

    dataset_dir = output_dir / "tf_dataset"
    tf.data.Dataset.from_tensor_slices((features, labels)).save(str(dataset_dir))

    metadata = {
        "example_count": int(features.shape[0]),
        "feature_count": int(features.shape[1]) if features.ndim == 2 else 0,
        "label_order": list(LABELS),
        "feature_names": feature_names,
        "tf_dataset": str(dataset_dir),
        "features_npy": str(output_dir / "features.npy"),
        "labels_npy": str(output_dir / "labels.npy"),
        "examples": [
            {"wav_path": str(example.wav_path), "label": example.label}
            for example in examples
        ],
        "skipped": skipped,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def transform_dataset(args: argparse.Namespace) -> None:
    if args.max_wavs_per_emotion is not None and args.max_wavs_per_emotion < 1:
        raise ValueError("--max-wavs-per-emotion must be at least 1 when provided.")
    if not args.feature_config.is_file():
        raise FileNotFoundError(f"Feature config not found: {args.feature_config}")

    import kagglehub
    import pyafex

    print(f"Downloading Kaggle dataset: {args.dataset_ref}", flush=True)
    dataset_root = Path(kagglehub.dataset_download(args.dataset_ref))
    print(f"Dataset available at: {dataset_root}", flush=True)

    examples = collect_examples(
        dataset_root=dataset_root,
        max_wavs_per_emotion=args.max_wavs_per_emotion,
        shuffle=not args.no_shuffle,
        seed=args.seed,
    )
    print(f"Extracting features from {len(examples)} WAV files...", flush=True)

    expected_feature_names: list[str] | None = None
    processed_examples: list[Example] = []
    feature_rows: list[list[float]] = []
    label_rows: list[list[float]] = []
    skipped: list[dict[str, str]] = []

    for index, example in enumerate(examples, start=1):
        try:
            feature_names, feature_values = extract_feature_vector(
                pyafex=pyafex,
                config_path=args.feature_config,
                wav_path=example.wav_path,
            )
            if expected_feature_names is None:
                expected_feature_names = feature_names
            elif feature_names != expected_feature_names:
                raise RuntimeError(
                    f"Feature order changed for {example.wav_path}: "
                    f"expected {expected_feature_names}, got {feature_names}"
                )

            processed_examples.append(example)
            feature_rows.append(feature_values)
            label_rows.append(one_hot(example.label))
        except Exception as error:  # noqa: BLE001 - CLI converts extractor failures to readable output.
            if not args.skip_failures:
                raise
            skipped.append({"wav_path": str(example.wav_path), "error": str(error)})

        if index % 25 == 0 or index == len(examples):
            print(f"Processed {index}/{len(examples)} WAV files", flush=True)

    if expected_feature_names is None or not processed_examples:
        raise RuntimeError("No training examples were created.")

    save_training_data(
        output_dir=args.output_dir,
        feature_rows=feature_rows,
        label_rows=label_rows,
        feature_names=expected_feature_names,
        examples=processed_examples,
        skipped=skipped,
    )
    print(f"Wrote TensorFlow training data to: {args.output_dir}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        transform_dataset(args)
    except Exception as error:  # noqa: BLE001 - CLI should report failures without a traceback by default.
        print(f"emotion-dataset-transform: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
