# Tilelang.language.Kernel

## 1. OP概述

简介：`tilelang.language.Kernel` 用于定义内核启动域的上下文构造接口。

```
T.Kernel(blocks, threads, is_cpu, prelude, is_npu, pipeline)
```

## 2. OP规格

### 2.1 参数说明

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `*blocks` | 常量 \| `tir.PrimExpr` \| `List[tir.PrimExpr]` | 是 | 网格各维度的 extent，1～3 维，表示 `gridDim.x`、`gridDim.y`、`gridDim.z`。可为常量或符号表达式（如 `T.ceildiv(N, block_N)`）。 |
| `threads` | `int` \| `List[int]` \| `Tuple` \| `None` | 否（GPU 有默认值） | 块内线程数。单整数表示 `blockDim.x`，列表/元组表示 `blockDim.(x,y,z)`，不足 3 维时未给出维度默认为 1。取 `-1` 表示跳过 `threadIdx.x` 绑定。非 CPU 且未传时默认为 `128`。 |
| `is_cpu` | `bool` | 否 | 为 `True` 表示 CPU kernel，不绑定 `threadIdx.x/y/z` 和 `blockIdx.x/y/z`，索引为 for 循环变量。默认 `False`。 |
| `prelude` | `str` \| `None` | 否 | 在生成的内核代码前注入的 C 代码（通过 `pragma_import_c` 等机制）。默认 `None`。 |
| `is_npu` | `bool` | 否 | 为 `True` 表示 NPU kernel，使用 NPU 的 block 语义（仅 1 维 block）。默认 `False`。 |
| `pipeline` | `bool` | 否 | 流水线相关开关，当前在 Python 层仅传入 attrs，具体语义由后端使用。默认 `False`。 |

### 2.2 支持规格

#### 2.2.1 DataType支持

**T.Kernel 本身不约束张量或标量的数据类型**。

- 只定义「网格/块/线程」的维度和索引变量（`tir.Var`），数据类型由内核体内的 `T.alloc_shared`、`T.alloc_ub`、缓冲区声明以及具体计算决定。
- 索引变量的 dtype 由 TIR 的迭代/变量约定决定（一般为整型 `int32`）。

#### 2.2.2 Shape支持

**blocks（网格维度）**

- 维度数：**1～3**
- 每维为 `tir.PrimExpr`（常量或符号，如 `T.ceildiv(M, block_M)`）
- 语义：第 1 维对应 `blockIdx.x`，第 2 维对应 `blockIdx.y`，第 3 维对应 `blockIdx.z`；未显式给出的维度视为 1（由 `KernelLaunch` 内扩展）。

### 2.3 特殊限制说明

**NPU 模式（`is_npu=True`）**

- **必须有且仅有 1 个 block 维度**：`len(blocks) == 1`，否则会触发 `AssertionError: "NPU kernel must have exactly one block dimension"`。
- 进入上下文时返回的是前 2 个 iter_var 的变量（`frames[0].iter_var.var`, `frames[1].iter_var.var`），用于 NPU 的 cube/vector 索引（`cid, vid`）。

### 2.4 使用方法

#### 1：GEMM

`examples/gemm/example_gemm.py`：用「总 block 数」启动，在 kernel 里把 `cid` 拆成 (by, bx)。

```
@T.prim_func
def gemm(
    A: T.Tensor((M, K), dtype),
    B: T.Tensor((K, N), dtype),
    C: T.Tensor((M, N), dtype),
):
    # 一维 block 数 = (M/block_M) * (N/block_N)
    with T.Kernel(T.ceildiv(N, block_N) * T.ceildiv(M, block_M), is_npu=True) as (cid, _):
        by = cid // T.ceildiv(N, block_N)
        bx = cid % T.ceildiv(N, block_N)

        A_shared = T.alloc_shared((block_M, block_K), dtype)
        B_shared = T.alloc_shared((block_K, block_N), dtype)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

        for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
            T.copy(A[by * block_M, k * block_K], A_shared)
            T.copy(B[k * block_K, bx * block_N], B_shared)
            T.gemm(A_shared, B_shared, C_local, initC=(k == 0))

        T.copy(C_local, C[by * block_M, bx * block_N])
```

#### 2：Flash Attention

`examples/flash_attn_npuir_dev.py`：block 数 = 序列方向上的块数，用 cid 算当前块的偏移。

```
with T.Kernel(T.ceildiv(seq_len, block_m), is_npu=True) as (cid, _):
    offset = cid * block_m
    Q_shared = T.alloc_shared([block_m, dim], dtype)
    T.copy(Q[offset, 0], Q_shared, size=[block_m, dim])
    # ... 后续用 offset、block_m 做 L1/Fragment 与 Cube 计算
```

#### 3：固定 NPU 并行度

`examples/elementwise/vec_add_2d.py`：block 数固定为 `BLOCK_SIZE`（如 20），每个 block 用 `cid` 区分，在循环里算真实 `block_id`，再拆成 (block_id_m, block_id_n)。

```
BLOCK_SIZE = 20
m_num = M // block_M
n_num = N // block_N

with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
    A_VEC = T.alloc_ub((block_M, block_N), dtype)
    B_VEC = T.alloc_ub((block_M, block_N), dtype)
    C_VEC = T.alloc_ub((block_M, block_N), dtype)
    for i in T.serial(T.ceildiv(m_num * n_num, BLOCK_SIZE)):
        block_id = i * BLOCK_SIZE + cid
        if block_id < m_num * n_num:
            block_id_m = block_id // n_num
            block_id_n = block_id % n_num
            bx = block_id_m * block_M
            by = block_id_n * block_N
            T.copy(A[bx, by], A_VEC)
            T.copy(B[bx, by], B_VEC)
            T.npuir_add(A_VEC, B_VEC, C_VEC)
            T.copy(C_VEC, C[bx, by])
```