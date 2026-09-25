# T.alloc_shared

## 1. 功能说明

分配 shared 层级存储，在 Ascend 平台由 `InferAllocScope` pass 根据使用场景自动推断为 L1 Buffer（`shared.l1`）或 Unified Buffer（`shared.ub`）。

## 2. 函数原型

### 2.1 函数定义

```python
def alloc_shared(
    shape: tuple,
    dtype: str,
    scope: str = "shared",
) -> Buffer
```

### 2.2 参数说明

| 参数名 | 输入/输出 | 描述 | 类型 | 必填/可选 |
|--------|----------|------|------|----------|
| shape | 输入 | buffer 的形状 | 整数元组 | 必填 |
| dtype | 输入 | buffer 的数据类型（如 `"float16"`、`"float32"`、`"int32"`） | 字符串 | 必填 |
| scope | 输入 | 内存作用域，通常使用默认值由编译器自动推断 | 字符串 | 可选（默认 `"shared"`） |

> **返回值说明**：
> - 返回 `T.Buffer` 对象，可用于 `T.copy`、计算原语等操作

### 2.3 参数规格

#### 2.3.1 DataType 支持

| 平台 | dtype |
|------|:-----:|
| Ascend A2 / A3 | float16, float32, bfloat16, int8, int16, int32, uint8, uint16, uint32 |

> 仅 ascendc 支持：int64, uint64

#### 2.3.2 Shape 支持

- 支持 1D 和 2D

#### 2.3.3 Ascend 平台内存层级映射

`InferAllocScope` pass 在编译期根据 buffer 的实际使用场景推断最终硬件 scope：

| 使用场景 | 推断 scope | Ascend 硬件映射 |
|---------|-----------|----------------|
| 仅 Cube 计算路径 | `shared.l1` | L1 Buffer |
| 仅 Vector 计算路径 | `shared.ub` | Unified Buffer（UB） |
| 混合使用（Cube + Vector） | `shared.l1` | L1 Buffer |

### 2.4 约束条件

1. 必须在 `T.Kernel` 作用域内调用

## 3. 示例代码

**示例 1：GEMM 中分配 L1 缓冲**

```python
block_M, block_K, block_N = 128, 128, 128
dtype = "float16"

A_L1 = T.alloc_shared((block_M, block_K), dtype)
B_L1 = T.alloc_shared((block_K, block_N), dtype)
```

**示例 2：Vector 计算中分配 UB 缓冲**

```python
block_M, block_N = 128, 128
dtype = "float16"

a_ub = T.alloc_shared((block_M, block_N), dtype)
b_ub = T.alloc_shared((block_M, block_N), dtype)
```
