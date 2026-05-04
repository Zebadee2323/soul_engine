#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../../.." && pwd)"

cli="${PYAFEX_CLI:-}"
if [[ -z "${cli}" ]]; then
    for candidate in \
        "${repo_root}/build/python/apps/pyafex-cli/pyafex-cli"; do
        if [[ -x "${candidate}" ]]; then
            cli="${candidate}"
            break
        fi
    done
fi

if [[ -z "${cli}" || ! -x "${cli}" ]]; then
    cat >&2 <<ERROR
Could not find a built pyafex-cli executable.

Build it first, for example:
  cmake --build <your-build-dir> --target pyafex_cli

Or point this script at it explicitly:
  PYAFEX_CLI=/path/to/pyafex-cli ${0}
ERROR
    exit 1
fi

exec "${cli}" "${script_dir}/test.wav" "${script_dir}/extractors.yaml" --pretty
