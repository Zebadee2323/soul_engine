#!/usr/bin/env python3
"""Build TensorFlow training data from the Kaggle audio-emotions dataset.

The output label for each WAV is a one-hot probability vector ordered as:

    angry, happy, sad, neutral, fearful, disgusted, surprised

Example:

    ./emotion-dataset-transform.py --max-wavs-per-emotion 10 --output-dir ./data/emotions
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
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


@dataclass(frozen=True)
class FeatureExtractionResult:
    index: int
    example: Example
    feature_names: list[str] | None = None
    feature_values: list[float] | None = None
    error: str | None = None


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
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help=(
            "Number of worker processes to use for CPU-bound audio feature extraction. "
            "Defaults to the available CPU count."
        ),
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


def extract_example_features(
    index: int,
    example: Example,
    config_path: Path,
) -> FeatureExtractionResult:
    """Extract features for one example in a worker process."""
    try:
        import pyafex

        feature_names, feature_values = extract_feature_vector(
            pyafex=pyafex,
            config_path=config_path,
            wav_path=example.wav_path,
        )
    except Exception as error:  # noqa: BLE001 - caller decides whether failures are fatal.
        return FeatureExtractionResult(index=index, example=example, error=str(error))

    return FeatureExtractionResult(
        index=index,
        example=example,
        feature_names=feature_names,
        feature_values=feature_values,
    )


def extract_feature_rows(
    examples: list[Example],
    config_path: Path,
    workers: int,
    skip_failures: bool,
) -> tuple[list[list[float]], list[list[float]], list[str], list[Example], list[dict[str, str]]]:
    """Extract features concurrently while preserving the input example order."""
    if workers < 1:
        raise ValueError("--workers must be at least 1.")

    worker_count = min(workers, len(examples))
    if worker_count == 1:
        print("Using 1 worker process for feature extraction.", flush=True)
        results = []
        for index, example in enumerate(examples):
            results.append(extract_example_features(index, example, config_path))
            completed = index + 1
            if completed % 25 == 0 or completed == len(examples):
                print(f"Processed {completed}/{len(examples)} WAV files", flush=True)
    else:
        print(f"Using {worker_count} worker processes for feature extraction.", flush=True)
        results_by_index: dict[int, FeatureExtractionResult] = {}
        completed = 0
        with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(extract_example_features, index, example, config_path)
                for index, example in enumerate(examples)
            ]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results_by_index[result.index] = result
                completed += 1

                if result.error is not None and not skip_failures:
                    for pending in futures:
                        pending.cancel()
                    raise RuntimeError(f"{result.example.wav_path}: {result.error}")

                if completed % 25 == 0 or completed == len(examples):
                    print(f"Processed {completed}/{len(examples)} WAV files", flush=True)

        results = [results_by_index[index] for index in range(len(examples))]

    expected_feature_names: list[str] | None = None
    processed_examples: list[Example] = []
    feature_rows: list[list[float]] = []
    label_rows: list[list[float]] = []
    skipped: list[dict[str, str]] = []

    for result in results:
        if result.error is not None:
            if not skip_failures:
                raise RuntimeError(f"{result.example.wav_path}: {result.error}")
            skipped.append({"wav_path": str(result.example.wav_path), "error": result.error})
            continue

        if result.feature_names is None or result.feature_values is None:
            raise RuntimeError(f"{result.example.wav_path}: feature extraction returned no data.")

        if expected_feature_names is None:
            expected_feature_names = result.feature_names
        elif result.feature_names != expected_feature_names:
            raise RuntimeError(
                f"Feature order changed for {result.example.wav_path}: "
                f"expected {expected_feature_names}, got {result.feature_names}"
            )

        processed_examples.append(result.example)
        feature_rows.append(result.feature_values)
        label_rows.append(one_hot(result.example.label))

    if expected_feature_names is None or not processed_examples:
        raise RuntimeError("No training examples were created.")

    return feature_rows, label_rows, expected_feature_names, processed_examples, skipped


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
    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")
    if not args.feature_config.is_file():
        raise FileNotFoundError(f"Feature config not found: {args.feature_config}")

    import kagglehub

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

    (
        feature_rows,
        label_rows,
        expected_feature_names,
        processed_examples,
        skipped,
    ) = extract_feature_rows(
        examples=examples,
        config_path=args.feature_config.resolve(),
        workers=args.workers,
        skip_failures=args.skip_failures,
    )

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
