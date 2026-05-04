#!/usr/bin/env python3

import argparse
import json
import sys

import pyafex


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract afex audio features from an audio file.")
    parser.add_argument("audio_file", nargs="?", help="Path to the WAV audio file to analyze.")
    parser.add_argument("config", nargs="?", help="Path to the YAML extractor config.")
    parser.add_argument("--list-extractors", action="store_true", help="List built-in extractor names and exit.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print the JSON result.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_extractors:
        print("\n".join(pyafex.builtin_feature_names()))
        return 0

    if args.audio_file is None or args.config is None:
        parser.error("audio_file and config are required unless --list-extractors is used.")

    try:
        result = pyafex.analyze_audio_file_with_yaml(args.config, args.audio_file)
    except Exception as error:  # noqa: BLE001 - CLI should turn native errors into readable stderr.
        print(f"pyafex-cli: {error}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2 if args.pretty else None, sort_keys=args.pretty))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
