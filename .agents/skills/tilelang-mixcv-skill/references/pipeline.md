# MixCV Pipeline Pattern

## Producer-consumer flow

1. Cube stage writes intermediate workspace
2. sync_block_set marks stage completion
3. Vector stage waits with sync_block_wait
4. Vector stage consumes intermediate data

## Good practice

- keep sync id consistent between set and wait
- minimize workspace footprint by tiling
- isolate stage-specific memory buffers
