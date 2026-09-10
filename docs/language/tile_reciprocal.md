# T.tile.reciprocal

## 1. 功能说明

对源操作数逐元素执行求倒数运算：`dst[i] = 1/src[i]`

## 2. 函数原型

### 2.1 函数定义

```python
def reciprocal(
    dst: Buffer | BufferRegion,
    src0: Buffer | BufferRegion,
)
```

### 2.2 参数说明

| 参数名 | 输入/输出 | 描述 | 类型 | 必填/可选 |
|--------|----------|------|------|----------|
| dst | 输出 | 存放求倒数运算结果 | 张量（tensor） | 必填 |
| src0 | 输入 | 源操作数 | 张量（tensor） | 必填 |

> **类型说明**：
> - **tensor**：通过 `T.alloc_ub`、`T.alloc_shared` 等分配的缓冲区（Buffer），或其切片（BufferRegion）

### 2.3 参数规格

#### 2.3.1 DataType 支持

| 平台 | dst | src0 |
|------|:---:|:----:|
| Ascend A2 / A3 | float16, float32 | float16, float32 |
| Ascend A5 | float16, float32, int64, uint64 | float16, float32, int64, uint64 |

> **后端差异说明**：A5 平台上 int64/uint64 仅 Ascend C 后端支持。

#### 2.3.2 Shape 支持

- 支持 1D 和 2D

### 2.4 约束条件

1. dst 与 src0 的元素总数必须相同
2. 操作数地址需 32 字节对齐（硬件约束）
3. src0 输入不应为 0，否则结果未定义
4. INTRINSIC 模式精度有限：float16 不满足双千分之一，float32 不满足双万分之一；需要高精度时应考虑用 `T.tile.div` 替代

## 3. 示例代码

**示例 1：逐元素求倒数运算**

```python
src0 = T.alloc_ub((256,), "float16")
dst = T.alloc_ub((256,), "float16")
T.tile.reciprocal(dst, src0)  # dst[i] = 1 / src0[i]
```
