# T.Parallel on TileLang-Ascend

This document describes how **`T.Parallel`** works in TileLang's programming model. It covers design goals, user guide and supported semantics. (especially for **Ascend** targets).


## 1. Background and Goals

### 1.1 Background

In TileLang, `T.Parallel` is the **primitive** for expressing **intra-tile element-wise parallel computation**.  
At the **IR level**, it abstracts parallel loops that represent data-parallel semantics while hiding hardware details, making kernel development simpler and more portable.

In Ascend kernels, the typical compute flow looks like this:

1. Split large tensors into **tiles**.
2. Load each tile into on-chip **UB (Unified Buffer)** memory.
3. Perform **Load → Compute → Store**.
4. Within the **Compute** stage, operate over all elements of a tile using vectorized instructions.

`T.Parallel` captures this “compute-stage” vectorized semantics at the IR level.

### 1.2 Design Objectives

The main purpose is to provide a **unified IR abstraction** for expressing vector operations within tiles.

#### 1.2.1 Alignment with TileLang IR Operators

The use of symbolic mathematical APIs (e.g., `T.exp`, `T.log`, `T.max`, etc.) is encouraged within `T.Parallel` instead of explicitly referencing low-level vector ops. This ensures:
 
- Compatibility with upstream IR  
- Backend portability (e.g., CPU, GPU, Ascend)

#### 1.2.2 Coordination with AscendC Vector Capabilities

TileLang-Ascend also integrates AscendC-specific features:

- Vector primitives are wrapped as `T.tile.xxx` under `ascend_tile.py`.
- Users can flexibly choose between **symbolic APIs** (e.g., `+`, `*`, `T.max`, …) with `T.Parallel` or **explicit vector intrinsics** (e.g., `T.tile.add`, etc.).



## 2. User guide

### 2.1 Basic Syntax

`T.Parallel` expresses element-wise parallel iteration.

**1D Example:**

```python
for j in T.Parallel(block_N // VEC_NUM):
    c_ub[j] = a_ub[j] + b_ub[j]
```

**2D Example:**

```python
for (i, j) in T.Parallel(block_M // VEC_NUM, block_N):
    c_ub[i, j] = a_ub[i, j] + b_ub[i, j]
```

Each iteration of `(i, j)` executes independently, representing a parallelizable region.


## 3. Supported Operations

### 3.1 Binary Operations

Binary operations patterns supported by `T.Parallel` include:


|    Category    |     Formula     | TileLang Expression |
| -------------- | --------------- | ------------------- |
| Addition       | `c = a + b`     | `a + b`             |
| Subtraction    | `c = a - b`     | `a - b`             |
| Multiplication | `c = a * b`     | `a * b`             |
| Division       | `c = a / b`     | `a / b`             |
| Min            | `c = min(a, b)` | `T.min(a, b)`       |
| Max            | `c = max(a, b)` | `T.max(a, b)`       |



**Integer Bitwise Operations**

| Category |   Formula    | TileLang Expression |
| -------- | ------------ | ------------------- |
| AND      | `c = a & b`  | `a & b`             |
| OR       | `c = a \| b` | `a \| b`            |



### 3.2 Unary Operations

**Floating-Point Unary Operations**

| Category |     Formula     | TileLang Expression |
| -------- | --------------- | ------------------- |
| Abs      | `y = \|x\|`     | `T.abs(a)`          |
| Exp      | `y = e^x`       | `T.exp(a)`          |
| Log      | `y = log(x)`    | `T.log(a)`          |
| Sqrt     | `y = sqrt(x)`   | `T.sqrt(a)`         |
| Rsqrt    | `y = 1/sqrt(x)` | `T.rsqrt(a)`        |
| ReLU     | `y = max(x, 0)` | `T.max(a, 0)`       |

**Integer Unary Operations**

|  Category   |   Formula    | TileLang Expression |
| ----------- | ------------ | ------------------- |
| Bitwise NOT | `y = ~x`     | `~a`                |
| Left Shift  | `y = x << s` | `a << scalar_val`   |
| Right Shift | `y = x >> s` | `a >> scalar_val`   |




### 3.3 Vector–Scalar Operations and Broadcasting

`T.Parallel` natively supports binary operations between vectors and scalars, as well as broadcasting along rows.

**Vector–Scalar Example**

```python
for j in T.Parallel(block_N):
    c_ub[j] = a_ub[j] + 1
```


**Row-Wise Broadcast Example**

```python
for (i, j) in T.Parallel(block_M // VEC_NUM, block_N):
    c_ub[i, j] = a_ub[i, j] * b_ub[i]
```

- `a_ub.shape = (block_M // VEC_NUM, block_N)`
- `b_ub.shape = (block_M // VEC_NUM,)`


### 3.4 Row-Split Pattern

`T.Parallel` can flexibly combine **sequential (row)** and **parallel (column)** dimensions.

```python
for i in range(block_M // VEC_NUM):  # Row sequential
    for j in T.Parallel(block_N):    # Column parallel
        c_ub[i, j] = a_ub[i, j] * b_ub[i, j]
```

This enables partial parallelization when full tiling is unnecessary.


## 4. Conclusion
For vector operations on Ascend C, both programming paradigms are supported.
### 4.1 Use `T.Parallel` with Symbolic APIs


High-level expression focused on clarity and portability:

```python
@T.prim_func
def main(A: T.Buffer((M, N), "float16"), B: T.Buffer((M, N), "float16")):
    with T.Scope("V"):
        a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), "float16")
        b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), "float16")

        T.copy(A, a_ub)

        for (i, j) in T.Parallel(block_M // VEC_NUM, block_N):
            b_ub[i, j] = T.exp(a_ub[i, j])

        T.copy(b_ub, B)
```

This approach performs data parallel computation over the tile region using symbolic APIs.

### 4.2 Use Ascend C vector instructions on tile-level data.


```python
@T.prim_func
def main(A: T.Buffer((M, N), "float16"), B: T.Buffer((M, N), "float16")):
    with T.Scope("V"):
        T.copy(A, a_ub)
        T.tile.exp(b_ub, a_ub)
        T.copy(b_ub, B)
```

In this approach, tile-level vector instructions are directly invoked.

## Future Support Scenarios

- Vertical slicing
- Non-linear index access
- Complex nested expressions