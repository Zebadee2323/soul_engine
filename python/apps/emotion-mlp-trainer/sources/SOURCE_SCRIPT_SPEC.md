# Source Script Spec

Source scripts convert one external emotion-audio dataset into one manifest CSV that
`../dataset-transform.py` can consume. They should hide every dataset-specific detail
behind a small CLI: download or locate the dataset, find supported WAV files, map each
file to the shared emotion labels, and write the manifest.

## Required manifest format

Write a CSV with exactly these columns, in this order:

```csv
absolute_wav_path,Anger,Disgust,Fear,Happy,Neutral,Sad
```

Rules:

- `absolute_wav_path` must be an absolute filesystem path to a `.wav` file.
- The six emotion columns must be numeric values.
- Most source scripts should emit one-hot labels using `1.0` for the matching emotion
  and `0.0` for the others.
- Soft labels are allowed only when the source dataset genuinely provides them.
- Every row must have at least one emotion value greater than `0.0`.
- Use UTF-8 and normal CSV quoting via Python's `csv` module.

The shared label order is:

```python
LABELS = ("Anger", "Disgust", "Fear", "Happy", "Neutral", "Sad")
```

## Script location and naming

Place source scripts in this directory:

```text
python/apps/emotion-mlp-trainer/sources/
```

Use a stable dataset-ref-style filename with punctuation converted to underscores, for
example:

```text
uldisvalainis_audio_emotions.py
ejlok1_cremad.py
```

The default output CSV should live in:

```text
python/apps/emotion-mlp-trainer/source_data/<script_name>.csv
```

## Required CLI behaviour

Each source script must be executable as a standalone CLI and return a non-zero exit
code on failure.

Include these options unless there is a clear dataset-specific reason not to:

- `--dataset-ref`: dataset identifier used by the downloader. For Kaggle datasets this
  should be the KaggleHub dataset ref, for example `ejlok1/cremad`.
- `--output-csv`: output manifest path. Default to `../source_data/<script_name>.csv`.
- `--max-wavs-per-emotion`: optional per-emotion cap. Omit to include every supported
  WAV.
- `--seed`: seed used when shuffling before applying the cap. Default to `1337`.
- `--no-shuffle`: preserve deterministic sorted order instead of shuffling.

Source scripts should print:

- which dataset is being downloaded or read;
- where the dataset is available locally;
- how many manifest rows were written and to which CSV path.

## Dataset handling

A source script is responsible for all dataset-specific decisions, including:

- finding the audio root, even if the downloaded archive contains one wrapper folder;
- recursively discovering `.wav` files when appropriate;
- mapping folder names, filename tokens, metadata rows, or annotations to the shared
  labels;
- ignoring emotions outside the shared six-label set unless intentionally mapped;
- raising a clear error when any required shared emotion has no matching WAV files.

Keep those decisions local to the source script. Do not require callers or
`dataset-transform.py` to know how a particular dataset names folders, files, or
metadata columns.

## Row ordering and balancing

Rows must be deterministic for a given set of inputs and seed.

Recommended collection flow:

1. Group WAV paths by canonical label.
2. Sort paths before any randomization.
3. Shuffle each label group with `random.Random(seed)` unless `--no-shuffle` is set.
4. Apply `--max-wavs-per-emotion` to each group after shuffling.
5. Emit rows in shared `LABELS` order.

This keeps dataset balancing local to the source script and makes rebuilds
reproducible.

## Runner script

When useful, add a small shell runner next to the Python script:

```text
run-<script-name-with-hyphens>.sh
```

Runner scripts should follow this pattern:

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
python3 ./sources/<script_name>.py
```

Only add default limits or flags when they are intentional for that dataset.

## Implementation expectations

- Use `pathlib.Path` for filesystem paths.
- Keep constants near the top: dataset ref, labels, and dataset-specific mappings.
- Keep manifest writing generic and boring: `write_manifest(output_csv, rows)`.
- Validate CLI arguments before starting expensive work.
- Catch top-level exceptions in `main()` and print a concise error message to stderr.
- Do not extract audio features, train models, or write TensorFlow data here; that is
  `dataset-transform.py`'s job.

A good source script presents a small, consistent interface while hiding the messy
implementation details of one external dataset.
