#!/usr/bin/env python3
"""Create a source_data manifest CSV from ejlok1/cremad."""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path


DATASET_REF = "ejlok1/cremad"
AUDIO_DIR_NAME = "AudioWAV"
LABELS = ("Anger", "Disgust", "Fear", "Happy", "Neutral", "Sad")
FILENAME_LABELS = {
    "ANG": "Anger",
    "DIS": "Disgust",
    "FEA": "Fear",
    "HAP": "Happy",
    "NEU": "Neutral",
    "SAD": "Sad",
}


def build_parser() -> argparse.ArgumentParser:
    script_dir = Path(__file__).resolve().parent
    app_dir = script_dir.parent
    parser = argparse.ArgumentParser(
        description=(
            "Download ejlok1/cremad and write a source_data CSV manifest "
            "for dataset-transform.py."
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
        default=app_dir / "source_data" / "ejlok1_cremad.csv",
        help="CSV manifest path to write.",
    )
    parser.add_argument(
        "--max-wavs-per-emotion",
        type=int,
        default=None,
        help="Maximum WAV count to include for each emotion. Omit to include all WAVs.",
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


def audio_wav_root(dataset_root: Path) -> Path:
    """Return the CREMA-D AudioWAV directory, tolerating one wrapper directory."""
    direct_audio_root = dataset_root / AUDIO_DIR_NAME
    if direct_audio_root.is_dir():
        return direct_audio_root

    matching_roots = [
        candidate
        for candidate in dataset_root.rglob(AUDIO_DIR_NAME)
        if candidate.is_dir()
    ]
    if len(matching_roots) == 1:
        return matching_roots[0]

    if not matching_roots:
        raise RuntimeError(
            f"Could not find the {AUDIO_DIR_NAME}/ directory under: {dataset_root}"
        )

    raise RuntimeError(
        f"Found multiple {AUDIO_DIR_NAME}/ directories under {dataset_root}: "
        + ", ".join(str(path) for path in matching_roots)
    )


def canonical_label_for_filename(path: Path) -> str | None:
    """Return the manifest emotion label encoded in a CREMA-D WAV filename."""
    for token in path.stem.upper().split("_"):
        label = FILENAME_LABELS.get(token)
        if label is not None:
            return label
    return None


def collect_wav_paths(
    dataset_root: Path,
    max_wavs_per_emotion: int | None,
    shuffle: bool,
    seed: int,
) -> list[tuple[Path, str]]:
    """Find AudioWAV WAV files grouped by emotion and apply the per-emotion limit locally."""
    grouped: dict[str, list[Path]] = defaultdict(list)
    wav_root = audio_wav_root(dataset_root)
    for wav_path in sorted(wav_root.rglob("*.wav")):
        label = canonical_label_for_filename(wav_path)
        if label is not None:
            grouped[label].append(wav_path)

    missing_labels = [label for label in LABELS if not grouped[label]]
    if missing_labels:
        raise RuntimeError(
            "No WAV files were found for filename emotion code(s): "
            + ", ".join(missing_labels)
            + f"\nSearched under: {wav_root}"
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
        print(f"ejlok1-cremad-source: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
