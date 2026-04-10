# Matmul Pattern

## Standard K-loop accumulation

- for each k tile
- load A and B into L1
- gemm into L0C with initC=(k==0)
- store once final tile is complete

## Practical notes

- Keep K tile size aligned with target constraints
- Validate transpose configuration for B path
- Validate numerical tolerance with torch reference

## Example files in examples/gemm

- example_gemm.py: fp16 input gemm with fp32 destination/accumulation
- example_gemm_int82int32.py: int8 input gemm with int32 destination/accumulation
- matmul.py: general matmul pattern
- matmul_dynamic_shape.py: dynamic-shape matmul pattern

## Data type safety note

- For fp16 input gemm, destination/accumulation should use fp32.
- Setting destination to fp16 for fp16 input gemm may cause runtime hang.
