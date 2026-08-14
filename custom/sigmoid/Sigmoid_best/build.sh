#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

rm -rf build dist cann_bench.egg-info

python setup.py bdist_wheel

echo "=== Build complete ==="
ls -la dist/
