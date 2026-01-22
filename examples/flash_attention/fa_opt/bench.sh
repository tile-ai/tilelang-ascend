python run.py \
    --iter-mode zip \
    --B 2 4 8 16 \
    --S 8192 4096 2048 1024 \
    --H 32 32 32 32 \
    --D 512 512 512 512 \
    --log ./log \
    --tl ./flash_attn_bhsd_auto_pipeline_h32_d512.py \
    --ascendc ./flash_attn_bhsd_ascendc.py