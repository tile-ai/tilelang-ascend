# Flash Attention Pattern (Expert-leaning)

## Stage A: score computation in Cube scope

- tile Q and K
- compute score tiles with gemm
- emit intermediates to workspace
- signal readiness via sync_block_set

## Stage B: softmax and accumulation in Vector scope

- wait on sync_block_wait
- cast, scale, exp, reduce using v-prefix APIs
- update running max and running sum

## Stage C: value accumulation and output

- consume V tiles
- accumulate weighted outputs
- normalize and store final output
