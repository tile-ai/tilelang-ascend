# Cube API Guide

## Memory

- T.alloc_shared(shape, dtype) (Developer mode)
- T.alloc_L1(shape, dtype) (Expert mode only)
- T.alloc_L0C(shape, accum_dtype) (Expert mode only)

## Data movement

- T.copy(src, dst)
- T.load_nd2nz(src, dst, size) (Expert mode only)

## Compute

- T.gemm(A, B, C, initC=True or False, b_transpose=True or False, size=[M, K, N])

## Store path

- T.copy(C_buf, C_out)
- T.store_fixpipe(C_buf, C_out, size=[M, N], enable_nz2nd=True) (Expert mode only)

## Scope recommendation

Use explicit T.Scope("Cube") for cube sections in expert mode kernels.
Don't use explicit T.Scope for cube sections in developer mode kernels.

## Mode guidance

- Expert mode: ND -> NZ (load_nd2nz), cube compute, NZ -> ND (store_fixpipe).
- Developer mode: Keep ND tensors and use T.copy without explicit NZ conversion.
