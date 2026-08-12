# T.tile.atomic_add

## 1. 功能说明

将本地 tensor（UB/L0C/L1）的数据原子累加到 GM 目标 tensor：`dst[GM][i] += src[local][i]`

## 2. 函数原型

### 2.1 函数定义

```python
def atomic_add(
    dst: Buffer | BufferRegion | BufferLoad,
    src: Buffer | BufferRegion | BufferLoad,
)
```

### 2.2 参数说明

| 参数名 | 输入/输出 | 描述 | 类型 | 必填/可选 |
|--------|----------|------|------|----------|
| dst | 输出 | GM 上的目标区域，累加结果写回此处 | 张量（tensor，scope 必须为 global） | 必填 |
| src | 输入 | 本地 tensor，其数据将被原子累加到 dst | 张量（tensor，scope 必须为 local） | 必填 |

> **类型说明**：
> - **tensor**：通过 `T.alloc_ub`、`T.alloc_shared`、`T.alloc_L0C`、`T.alloc_L1` 等分配的缓冲区（Buffer），或其切片（BufferRegion）
> - **global scope**：通过 `T.alloc_gm` 或外部传入的 GM buffer
> - **local scope**：UB、L0C、L1 等非 GM 内存区域

### 2.3 参数规格

#### 2.3.1 DataType 支持

| 平台 | dst | src |
|------|:---:|:---:|
| Ascend A2 / A3 | float16, float32, int16, int32, bfloat16 | float16, float32, int16, int32, bfloat16 |

> int8 仅 pto 后端支持。

#### 2.3.2 Shape 支持

- 支持 1D 和 2D

### 2.4 约束条件

1. dst 的 buffer scope 必须为 `global`（GM），否则抛出 `ValueError`
2. src 的 buffer scope 必须为 local（UB/L0C/L1 等），不能为 `global`，否则抛出 `ValueError`
3. dst 与 src 的 dtype 必须相同（所有路径）
4. 仅支持 local（VECOUT/L0C/L1）→ GM 方向的 DMA 搬运附带原子累加（硬件约束）
5. 累加前 GM 需清零：DMA 原子累加不会自动清零，开发者需在调用前确保 dst GM 已清零
6. 完成后自动关闭原子操作：tilelang 在原子累加完成后自动调用 `disable_dma_atomic`，开发者无需手动关闭

## 3. 示例代码

**示例 1：UB → GM 原子累加**

```python
C_gm = T.alloc_gm((256,), "float32")
src_ub = T.alloc_ub((256,), "float32")
T.tile.fill(src_ub, 1.0)
T.tile.atomic_add(C_gm, src_ub)
```

**示例 2：L0C → GM 原子累加（GEMM split-K 场景）**

```python
C_gm = T.alloc_gm((block_M, block_N), "float32")
C_L0 = T.alloc_L0C((block_M, block_N), "float32")
T.gemm_v0(A_L1, B_L1, C_L0, init=True)
T.tile.atomic_add(C_gm[by * block_M, bx * block_N], C_L0)
```
