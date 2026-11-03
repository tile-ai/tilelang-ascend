#!/usr/bin/env bash
# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
set -euo pipefail

tilelang_examples_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
source "$tilelang_examples_root/set_env.sh"

if [[ -n "${TILELANG_TEST_PYTHON:-}" ]]; then
    tilelang_examples_python="$TILELANG_TEST_PYTHON"
elif [[ -x "$tilelang_examples_root/.venv/bin/python" ]]; then
    tilelang_examples_python="$tilelang_examples_root/.venv/bin/python"
else
    tilelang_examples_python="python3"
fi

exec "$tilelang_examples_python" \
    "$tilelang_examples_root/scripts/testing/run_ascend_examples.py" "$@"
