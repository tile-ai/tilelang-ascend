#!/bin/bash

if [ -n "${ZSH_VERSION:-}" ]; then
    tilelang_env_script_path="${(%):-%N}"
else
    tilelang_env_script_path="${BASH_SOURCE[0]}"
fi

TL_ROOT=$(dirname "$(readlink -f "$tilelang_env_script_path")")
export TL_ROOT
case "${PYTHONPATH:-}" in
    "$TL_ROOT"|"$TL_ROOT":*) ;;
    *) PYTHONPATH="$TL_ROOT${PYTHONPATH:+:$PYTHONPATH}" ;;
esac
export PYTHONPATH

# disable the import of tvm when using torch_npu
export ACL_OP_INIT_MODE=1

unset tilelang_env_script_path
