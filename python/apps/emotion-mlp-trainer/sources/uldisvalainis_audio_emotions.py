#!/usr/bin/env python3
"""Create a source_data manifest CSV from uldisvalainis/audio-emotions."""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path


DATASET_REF = "uldisvalainis/audio-emotions"
LABELS = ("Anger", "Disgust", "Fear", "Happy", "Neutral", "Sad")
LABEL_ALIASES = {
    "angry": "Anger",
    "anger": "Anger",
    "disgusted": "Disgust",
    "disgust": "Disgust",
    "fearful": "Fear",
    "fear": "Fear",
    "happy": "Happy",
    "happiness": "Happy",
    "neutral": "Neutral",
    "sad": "Sad",
    "sadness": "Sad",
}


def build_parser() -> argparse.ArgumentParser:
    script_dir = Path(__file__).resolve().parent
    app_dir = script_dir.parent
    parser = argparse.ArgumentParser(
        description=(
            "Download uldisvalainis/audio-emotions and write a source_data CSV "
            "manifest for dataset-transform.py."
        )
    )
    parser.add_argument(
        "--dataset-ref",
        default=DATASET_REF,
        help=f"KaggleHub dataset ref to download. Defaults to {DATASET_REF}.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=app_dir / "source_data" / "uldisvalainis_audio_emotions.csv",
        help="CSV manifest path to write.",
    )
    parser.add_argument(
        "--max-wavs-per-emotion",
        type=int,
        default=None,
        help="Maximum WAV count to include from each emotion folder. Omit to include all WAVs.",
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
    return parser


def canonical_label_for_path(path: Path, dataset_root: Path) -> str | None:
    """Return the manifest emotion label represented by one of the path folders."""
    try:
        relative_parts = path.relative_to(dataset_root).parts[:-1]
    except ValueError:
        relative_parts = path.parts[:-1]

    for part in reversed(relative_parts):
        normalized = part.strip().lower().replace("-", "_").replace(" ", "_")
        if normalized in LABEL_ALIASES:
            return LABEL_ALIASES[normalized]
    return None


def collect_wav_paths(
    dataset_root: Path,
    max_wavs_per_emotion: int | None,
    shuffle: bool,
    seed: int,
) -> list[tuple[Path, str]]:
    """Find WAV files grouped by supported emotion and apply the per-emotion limit locally."""
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
    rows: list[tuple[Path, str]] = []
    for label in LABELS:
        wav_paths = list(grouped[label])
        if shuffle:
            rng.shuffle(wav_paths)
        if max_wavs_per_emotion is not None:
            wav_paths = wav_paths[:max_wavs_per_emotion]
        rows.extend((wav_path.resolve(), label) for wav_path in wav_paths)

    return rows


def label_vector(label: str) -> list[float]:
    return [1.0 if candidate == label else 0.0 for candidate in LABELS]


def write_manifest(output_csv: Path, rows: list[tuple[Path, str]]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["absolute_wav_path", *LABELS])
        for wav_path, label in rows:
            writer.writerow([str(wav_path), *label_vector(label)])


def create_manifest(args: argparse.Namespace) -> None:
    if args.max_wavs_per_emotion is not None and args.max_wavs_per_emotion < 1:
        raise ValueError("--max-wavs-per-emotion must be at least 1 when provided.")

    import kagglehub

    print(f"Downloading Kaggle dataset: {args.dataset_ref}", flush=True)
    dataset_root = Path(kagglehub.dataset_download(args.dataset_ref))
    print(f"Dataset available at: {dataset_root}", flush=True)

    rows = collect_wav_paths(
        dataset_root=dataset_root,
        max_wavs_per_emotion=args.max_wavs_per_emotion,
        shuffle=not args.no_shuffle,
        seed=args.seed,
    )
    write_manifest(args.output_csv, rows)
    print(f"Wrote {len(rows)} manifest rows to: {args.output_csv}", flush=True)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        create_manifest(args)
    except Exception as error:  # noqa: BLE001 - CLI should report failures without traceback by default.
        print(f"uldisvalainis-audio-emotions-source: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
