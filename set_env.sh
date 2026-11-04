#!/bin/bash

if [ -n "${ZSH_VERSION:-}" ]; then
    tilelang_env_script_path="${(%):-%N}"
else
    tilelang_env_script_path="${BASH_SOURCE[0]}"
fi

TL_ROOT=$(dirname "$(readlink -f "$tilelang_env_script_path")")
export TL_ROOT

# Read-only health check: the adapted tl_templates/ascend/common.h requires
# catlass >= v2.0 headers; a stale submodule (e.g. after `git pull` without
# updating submodules) breaks every JIT compile with confusing errors. We
# only print a fix hint here — environment activation never mutates the
# working tree. Deployments without a 3rdparty tree simply skip the check.
if [ -d "$TL_ROOT/3rdparty" ] && [ ! -f "$TL_ROOT/3rdparty/catlass/include/catlass/gemm/tile/atlasa2/copy_gm_to_l1.hpp" ]; then
    echo "[tilelang] WARNING: catlass submodule is stale (common.h requires catlass >= v2.0)." >&2
    echo "[tilelang] fix with: git submodule sync --recursive && git submodule update --init --recursive" >&2
fi

case "${PYTHONPATH:-}" in
    "$TL_ROOT"|"$TL_ROOT":*) ;;
    *) PYTHONPATH="$TL_ROOT${PYTHONPATH:+:$PYTHONPATH}" ;;
esac
export PYTHONPATH

# disable the import of tvm when using torch_npu
export ACL_OP_INIT_MODE=1

unset tilelang_env_script_path
