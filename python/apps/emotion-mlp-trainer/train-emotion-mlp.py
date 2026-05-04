#!/usr/bin/env python3
"""Train a TensorFlow MLP emotion classifier from dataset-transform.py output."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from emotion_mlp_common import (
    LABELS,
    MODEL_FILENAME,
    TRAINING_METADATA_FILENAME,
    load_training_arrays,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Train an emotion MLP from train_data/features.npy and labels.npy."
    )
    parser.add_argument("--train-data-dir", type=Path, default=script_dir / "train_data")
    parser.add_argument("--model-dir", type=Path, default=script_dir / "models" / "emotion_mlp")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    return parser


def build_model(input_dim: int, class_count: int, learning_rate: float):
    import tensorflow as tf

    regularizer = tf.keras.regularizers.l2(1.0e-4)
    inputs = tf.keras.Input(shape=(input_dim,), name="audio_features")
    x = tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=regularizer)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    x = tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=regularizer)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Dense(32, activation="relu", kernel_regularizer=regularizer)(x)
    x = tf.keras.layers.Dropout(0.15)(x)
    outputs = tf.keras.layers.Dense(class_count, activation="softmax", name="emotion_probabilities")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="emotion_feature_mlp")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top_3_accuracy")],
    )
    return model


def stratified_split_indices(labels: np.ndarray, validation_split: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Create a deterministic per-class validation split for one-hot labels."""
    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    validation_indices: list[int] = []

    class_ids = labels.argmax(axis=1)
    for class_id in range(labels.shape[1]):
        indices = np.flatnonzero(class_ids == class_id)
        if indices.size == 0:
            continue
        rng.shuffle(indices)
        validation_count = int(round(indices.size * validation_split))
        validation_count = max(1, validation_count) if indices.size > 1 else 0
        validation_count = min(validation_count, indices.size - 1)
        validation_indices.extend(indices[:validation_count].tolist())
        train_indices.extend(indices[validation_count:].tolist())

    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)
    if not train_indices or not validation_indices:
        raise ValueError("Not enough labelled examples to create a train/validation split.")
    return np.asarray(train_indices, dtype=np.int64), np.asarray(validation_indices, dtype=np.int64)


def class_weight_for(labels: np.ndarray) -> dict[int, float]:
    class_counts = labels.sum(axis=0)
    total = float(labels.shape[0])
    class_count = float(labels.shape[1])
    return {
        index: total / (class_count * float(count))
        for index, count in enumerate(class_counts)
        if count > 0.0
    }


def train(args: argparse.Namespace) -> None:
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if not 0.0 < args.validation_split < 0.5:
        raise ValueError("--validation-split must be greater than 0 and less than 0.5.")

    import tensorflow as tf

    tf.keras.utils.set_random_seed(args.seed)

    features, labels, dataset_metadata = load_training_arrays(args.train_data_dir)
    label_order = list(dataset_metadata.get("label_order", list(LABELS)))
    if labels.shape[1] != len(label_order):
        raise ValueError(
            f"Label array width {labels.shape[1]} does not match metadata label_order length {len(label_order)}."
        )

    model = build_model(
        input_dim=features.shape[1],
        class_count=labels.shape[1],
        learning_rate=args.learning_rate,
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=25, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=8, min_lr=1.0e-5, verbose=1
        ),
    ]

    train_indices, validation_indices = stratified_split_indices(
        labels=labels,
        validation_split=args.validation_split,
        seed=args.seed,
    )
    train_features = features[train_indices]
    train_labels = labels[train_indices]
    validation_data = (features[validation_indices], labels[validation_indices])

    history = model.fit(
        train_features,
        train_labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=validation_data,
        shuffle=True,
        class_weight=class_weight_for(train_labels),
        callbacks=callbacks,
        verbose=2,
    )

    args.model_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.model_dir / MODEL_FILENAME
    model.save(model_path)

    write_json(
        args.model_dir / TRAINING_METADATA_FILENAME,
        {
            "model_path": str(model_path),
            "input_feature_count": int(features.shape[1]),
            "input_normalization": "pre_normalized_by_dataset_transform",
            "label_order": label_order,
            "feature_names": dataset_metadata["feature_names"],
            "train_data_dir": str(args.train_data_dir),
            "epochs_requested": args.epochs,
            "epochs_trained": len(history.history["loss"]),
            "batch_size": args.batch_size,
            "validation_split": args.validation_split,
            "training_example_count": int(train_labels.shape[0]),
            "validation_example_count": int(validation_data[1].shape[0]),
            "history": {key: [float(value) for value in values] for key, values in history.history.items()},
        },
    )
    print(f"Saved model to: {model_path}", flush=True)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        train(args)
    except Exception as error:  # noqa: BLE001 - CLI should report failures without traceback by default.
        print(f"train-emotion-mlp: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
