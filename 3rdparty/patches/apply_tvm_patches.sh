#!/bin/bash
# Apply local patches to the pinned 3rdparty/tvm submodule.
#
# Shared by install_ascend.sh, build_wheel_ascend.sh, and setup.py so that every
# build path (bash install, wheel build, and `USE_ASCEND=true pip install -e .`)
# picks up the fixes kept under 3rdparty/patches/. Without this, only
# install_ascend.sh would apply the patches and the other two paths would build
# an unpatched TVM.
#
# Behaviour:
#   - idempotent: an already-applied patch is detected (reverse --check) and
#     skipped, so re-running install / incremental builds is safe;
#   - FATAL on failure: if a patch cannot apply (e.g. the pinned tvm was bumped
#     and the context no longer matches) we exit non-zero instead of silently
#     building an unpatched TVM.
#
# Usage: bash 3rdparty/patches/apply_tvm_patches.sh [repo_root]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${1:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
TVM_DIR="$REPO_ROOT/3rdparty/tvm"
PATCH_DIR="$REPO_ROOT/3rdparty/patches"

if ! git -C "$TVM_DIR" rev-parse --git-dir >/dev/null 2>&1; then
    echo "[patch] ERROR: tvm submodule not found at $TVM_DIR" >&2
    echo "          run 'git submodule update --init --recursive' first" >&2
    exit 1
fi

if [ ! -d "$PATCH_DIR" ]; then
    echo "[patch] no patch directory at $PATCH_DIR, nothing to do"
    exit 0
fi

for patch in "$PATCH_DIR"/tvm_*.patch; do
    [ -e "$patch" ] || continue
    patch_name="$(basename "$patch")"
    if git -C "$TVM_DIR" apply --reverse --check "$patch" >/dev/null 2>&1; then
        echo "  [patch] $patch_name already applied, skipping"
    elif git -C "$TVM_DIR" apply --check "$patch" >/dev/null 2>&1; then
        git -C "$TVM_DIR" apply "$patch"
        echo "  [patch] $patch_name applied"
    else
        echo "  [patch] ERROR: cannot apply $patch_name to 3rdparty/tvm" >&2
        echo "          (the pinned tvm submodule may have changed; regenerate the patch)" >&2
        exit 1
    fi
done
