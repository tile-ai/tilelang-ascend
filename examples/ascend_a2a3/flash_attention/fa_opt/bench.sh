#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$script_dir"

python run.py \
  --iter-mode zip \
  --B 2 2 1 \
  --S 131072 65536 32768 \
  --q-heads 12 12 12 \
  --kv-heads 1 1 1 \
  --D 128 128 128 \
  --log ./log \
  --tl ./auto_pipeline/h16_d128.py \
  --ascendc ./reference/ascendc.py
