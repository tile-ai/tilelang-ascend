#!/bin/bash

# Get the script name from argument, default to auto_sync
SCRIPT="${1:-flash_attn_bhsd_auto_pipeline_h16_d128.py}"

# Remove .py extension if provided
SCRIPT="${SCRIPT%.py}"

python run.py \
  --iter-mode zip \
  --B 2 \
  --S 65536 \
  --q-heads 12 \
  --kv-heads 1 \
  --D 128 \
  --log ./log \
  --tl "./${SCRIPT}.py" \
  --ascendc ./flash_attn_bhsd_ascendc.py
