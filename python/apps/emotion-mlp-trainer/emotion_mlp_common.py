#!/usr/bin/env python3
"""Shared feature, normalisation, and metadata helpers for emotion MLP scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

LABELS = ("Anger", "Disgust", "Fear", "Happy", "Neutral", "Sad")
MODEL_FILENAME = "emotion_mlp.keras"
TRAINING_METADATA_FILENAME = "training_metadata.json"
MAX_FRAME_LENGTH_SECONDS = 2.0


def extract_feature_vector(
    pyafex: Any,
    config_path: Path,
    wav_path: Path,
    *,
    trim_silence: bool = False,
    max_frame_length: float = 0.0,
) -> tuple[list[str], list[float]]:
    """Run pyafex and return scalar feature names and values in pyafex result order."""
    analysis = pyafex.analyze_audio_file_with_yaml(
        str(config_path),
        str(wav_path),
        trim_silence=trim_silence,
        max_frame_length=max_frame_length,
    )

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


def load_dataset_metadata(train_data_dir: Path) -> dict[str, Any]:
    metadata_path = train_data_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Dataset metadata not found: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def resolve_data_file(train_data_dir: Path, metadata_value: str | None, fallback_name: str) -> Path:
    """Resolve dataset array paths written with either absolute or cwd-relative metadata."""
    candidates: list[Path] = []
    if metadata_value:
        recorded = Path(metadata_value)
        candidates.append(recorded)
        if not recorded.is_absolute():
            candidates.append(train_data_dir / recorded)
            candidates.append(train_data_dir / recorded.name)
    candidates.append(train_data_dir / fallback_name)

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not resolve {fallback_name}. Tried: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def load_training_arrays(train_data_dir: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    metadata = load_dataset_metadata(train_data_dir)
    features_path = resolve_data_file(train_data_dir, metadata.get("features_npy"), "features.npy")
    labels_path = resolve_data_file(train_data_dir, metadata.get("labels_npy"), "labels.npy")

    features = np.load(features_path).astype(np.float32)
    labels = np.load(labels_path).astype(np.float32)
    if features.ndim != 2:
        raise ValueError(f"Expected features.npy to be 2D, got shape {features.shape}.")
    if labels.ndim != 2:
        raise ValueError(f"Expected labels.npy to be 2D, got shape {labels.shape}.")
    if features.shape[0] != labels.shape[0]:
        raise ValueError(f"Feature/label row mismatch: {features.shape[0]} vs {labels.shape[0]}.")
    return features, labels, metadata


def validate_feature_order(actual: list[str], expected: list[str]) -> None:
    if actual != expected:
        raise ValueError(f"Feature order mismatch. Expected {expected}, got {actual}.")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
