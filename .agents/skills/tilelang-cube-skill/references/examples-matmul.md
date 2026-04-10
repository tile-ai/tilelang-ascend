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
