#!/bin/bash
# build_perf_rope.sh — Wrapper to build perf_rope_ascendc via CMake.
#
# Requires $ASCEND_HOME_PATH set (run 'source set_env.sh' first).
# The rotary_position_embedding op package must be installed to CANN first.
#
# Usage:
#   bash build_perf_rope.sh
#   bash build_perf_rope.sh --clean    # rm -rf build/ and rebuild

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
BINARY="$BUILD_DIR/perf_rope_ascendc"

if [[ -z "${ASCEND_HOME_PATH:-}" && -z "${ASCEND_TOOLKIT_HOME:-}" ]]; then
    echo "Error: ASCEND_HOME_PATH not set. Run 'source set_env.sh' first." >&2
    exit 1
fi

if [[ "${1:-}" == "--clean" ]]; then
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"

echo "Running cmake..."
cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR"

echo ""
echo "Running make..."
make -C "$BUILD_DIR" -j"$(nproc)"

echo ""
if [[ -x "$BINARY" ]]; then
    echo "Build OK: $BINARY"
    echo "Verify:   $BINARY --help"
else
    echo "Error: build failed, $BINARY not produced" >&2
    exit 1
fi
