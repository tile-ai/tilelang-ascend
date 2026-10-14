# T.alloc_fragment

## 1. 功能说明

分配 fragment 层级存储，在 Ascend 平台由 `InferAllocScope` pass 根据在 GEMM 中的位置自动推断为 L0A（`wmma.matrix_a`）/ L0B（`wmma.matrix_b`）/ L0C（`wmma.accumulator`）；未在 GEMM 中使用时默认映射到 L0C。

## 2. 函数原型

### 2.1 函数定义

```python
def alloc_fragment(
    shape: tuple,
    dtype: str,
    scope: str = "local.fragment",
) -> Buffer
```

### 2.2 参数说明

| 参数名 | 输入/输出 | 描述 | 类型 | 必填/可选 |
|--------|----------|------|------|----------|
| shape | 输入 | buffer 的形状 | 整数元组 | 必填 |
| dtype | 输入 | buffer 的数据类型（如 `"float16"`、`"float32"`、`"int32"`） | 字符串 | 必填 |
| scope | 输入 | 内存作用域，通常使用默认值由编译器自动推断 | 字符串 | 可选（默认 `"local.fragment"`） |

> **返回值说明**：
> - 返回 `T.Buffer` 对象，可用于 `T.copy`、计算原语等操作

### 2.3 参数规格

#### 2.3.1 DataType 支持

| 平台 | dtype |
|------|:-----:|
| Ascend A2 / A3 | float16, bfloat16, float32, int32 |

> alloc_fragment 通常用作 GEMM 累加器。

#### 2.3.2 Shape 支持

- 支持 1D 和 2D（GEMM 场景下须为 2D 矩阵分形）

#### 2.3.3 Ascend 平台内存层级映射

`InferAllocScope` pass 在编译期根据 buffer 在 GEMM 中的位置推断最终硬件 scope：

| GEMM 位置 | 推断 scope | Ascend 硬件映射 |
|----------|-----------|----------------|
| position 0（左矩阵 A） | `wmma.matrix_a` | L0A |
| position 1（右矩阵 B） | `wmma.matrix_b` | L0B |
| position 2（累加器 C） | `wmma.accumulator` | L0C |
| 未在 GEMM 中使用 | `wmma.accumulator` | L0C |

### 2.4 约束条件

1. 必须在 `T.Kernel` 作用域内调用
2. GEMM 场景下 shape 须为 2D 矩阵分形

## 3. 示例代码

**示例 1：分配 GEMM 累加器**

```python
block_M, block_N = 128, 128
accum_dtype = "float32"

C_L0 = T.alloc_fragment((block_M, block_N), accum_dtype)
```

**示例 2：GEMM 完整分配**

```python
block_M, block_K, block_N = 128, 128, 128
dtype = "float16"
accum_dtype = "float32"

A_L1 = T.alloc_shared((block_M, block_K), dtype)
B_L1 = T.alloc_shared((block_K, block_N), dtype)
C_L0 = T.alloc_fragment((block_M, block_N), accum_dtype)
```
