python run.py \
  --iter-mode zip \
  --B 2 2 1 \
  --S 131072 65536 32768 \
  --q-heads 12 12 12 \
  --kv-heads 1 1 1 \
  --D 128 128 128 \
  --log ./log \
  --tl ./flash_attn_bhsd_auto_pipeline_h16_d128.py \
  --ascendc ./flash_attn_bhsd_ascendc.py
