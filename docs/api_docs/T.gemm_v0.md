# T.gemm_v0

## 1. Description

Performs a fractal matrix multiply-accumulate operation: `C += op(A) × op(B)`, where `op` is an optional transpose. When `init=True`, the accumulator C is cleared before computation, equivalent to `C = op(A) × op(B)`.

## 2. Function Prototype

### 2.1 Function Definition

```python
def gemm_v0(
    A: Buffer | BufferRegion,
    B: Buffer | BufferRegion,
    C: Buffer | BufferRegion,
    transpose_A: bool = False,
    transpose_B: bool = False,
    init: bool = False,
    kL0Size: int = 128,
    n_actual: int = None,
)
```

### 2.2 Parameters

| Parameter | Input/Output | Description | Type | Required/Optional |
|-----------|-------------|-------------|------|-------------------|
| A | Input | Left matrix, last two dimensions are matrix dimensions, must be in L1 | tensor | Required |
| B | Input | Right matrix, last two dimensions are matrix dimensions, must be in L1 | tensor | Required |
| C | Input/Output | Output matrix (accumulator), last two dimensions are matrix dimensions, must be in L0C | tensor | Required |
| transpose_A | Input | Whether to transpose matrix A | bool | Optional (default `False`) |
| transpose_B | Input | Whether to transpose matrix B | bool | Optional (default `False`) |
| init | Input | Whether to clear accumulator C before computation (`True`: clear then compute C=A×B; `False`: accumulate on existing C, C+=A×B) | bool | Optional (default `False`) |
| kL0Size | Input | K-axis tile size controlling L1→L0A/L0B data movement granularity, must be a multiple of 16 and ≤ 4095 | int | Optional (default `128`) |
| n_actual | Input | Runtime output column count (≤ N and multiple of 16), only effective on transpose_B path | int | Optional (default `None`) |

> **Type Description**:
> - **tensor**: A buffer allocated via `T.alloc_L1`, `T.alloc_L0C`, etc., or its slice (BufferRegion)
>
> **Memory Hierarchy**:
> - A, B must be in L1 (allocated via `T.alloc_L1`). gemm_v0 internally transfers data from L1 to L0A/L0B for matrix multiplication
> - C must be in L0C (allocated via `T.alloc_L0C`). The Mmad hardware instruction output is written to L0C
> - Data flow: `GM → L1 (user transfer) → L0A/L0B (gemm_v0 auto transfer) → Mmad compute → L0C (result) → GM (user transfer)`

### 2.3 Specifications

#### 2.3.1 DataType Support

| Platform | A | B | C |
|----------|:---:|:---:|:---:|
| Ascend A2 / A3 | float16, bfloat16, float32, int8 | float16, bfloat16, float32, int8 | float32 (when A/B are floating-point), int32 (when A/B are int8) |

> float32 input: ascendc backend not supported yet (compile error), pto backend available. See [issue #1016](https://github.com/tile-ai/tilelang-ascend/issues/1016)

#### 2.3.2 Shape Support

- A, B, C: ≥2D, last two dimensions are matrix dimensions (M×K, K×N, M×N)

#### 2.3.3 A/B/C dtype Combination Constraints

A and B must have the same dtype. C dtype is determined by A/B dtype:

| A/B dtype | C dtype |
|-----------|---------|
| float16 | float32 |
| bfloat16 | float32 |
| float32 | float32 (pto only) |
| int8 | int32 |

#### 2.3.4 Fractal Dimensions

The hardware uses fractal matrices as the minimum compute unit. Fractal dimensions per dtype:

| A/B dtype | A fractal (L0A) | B fractal (L0B) | C fractal (L0C) |
|-----------|-----------------|-----------------|-----------------|
| float16 / bfloat16 (2B) | 16×16 | 16×16 | 16×16 |
| float32 (4B) | 16×8 | 8×16 | 16×16 |
| int8 (1B) | 16×32 | 32×16 | 16×16 |

### 2.4 Constraints

1. A, B, C must be ≥2D tensors. For >2D, all leading dimensions must be 1
2. K dimension of A and B must match: if `transpose_A=False`, K = `A.shape[-1]`; otherwise K = `A.shape[-2]`. Same for B. Both K values must be equal
3. A and B must have the same dtype
4. A, B must be in L1, C must be in L0C (hardware constraint). gemm_v0 internally transfers A/B from L1 to L0A/L0B; users do not need to manually transfer to L0A/L0B
5. A starting address must be 512-byte aligned (hardware constraint)
6. B starting address must be 512-byte aligned (hardware constraint)
7. C starting address must be 256-element aligned (hardware constraint)
8. Mmad single-call m/n/k range is `[0, 4095]`. When any of m/n/k is 0, the instruction is not executed (hardware constraint). gemm_v0 automatically splits via tiling; overall matrix M/N/K is not limited by this range
9. When `M=1`, hardware enables GEMV (General Matrix-Vector Multiplication) by default. In this case, A must be in ND format instead of ZZ format (hardware constraint)
10. `kL0Size` must be a multiple of 16 and ≤ 4095

## 3. Examples

**Example 1: Basic usage (fp16 input, fp32 accumulation, init=True)**

```python
block_M, block_K, block_N = 128, 128, 128
A_L1 = T.alloc_L1((block_M, block_K), "float16")
B_L1 = T.alloc_L1((block_K, block_N), "float16")
C_L0 = T.alloc_L0C((block_M, block_N), "float")
T.gemm_v0(A_L1, B_L1, C_L0, init=True)
```

**Example 2: Accumulation mode (init=False, accumulate on existing C)**

```python
A_L1 = T.alloc_L1((block_M, block_K), "float16")
B_L1 = T.alloc_L1((block_K, block_N), "float16")
C_L0 = T.alloc_L0C((block_M, block_N), "float")
T.gemm_v0(A_L1, B_L1, C_L0, init=False)  # C_L0 += A_L1 × B_L1
```

**Example 3: Transpose matrix B**

```python
A_L1 = T.alloc_L1((block_M, block_K), "float16")
B_L1 = T.alloc_L1((block_N, block_K), "float16")  # B shape is (N, K)
C_L0 = T.alloc_L0C((block_M, block_N), "float")
T.gemm_v0(A_L1, B_L1, C_L0, transpose_B=True, init=True)
```
