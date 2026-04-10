# Flash Attention Pattern (Developer mode)

## MixCV feature definition (Developer mode)

- A kernel is treated as MixCV when it has:
  - Cube-side T.gemm compute, and
  - Vector-side at least one v-prefix op (for example T.vmul, T.vadd, T.vexp, T.vcast, T.vbrc).

## Typical style

- allocate with alloc_shared and alloc_fragment
- use v-prefix vector APIs for softmax math
- use Pipelined for staged loops when possible

## Migration tip

Start from Developer style for correctness, then migrate hot paths to Expert style blocks.
