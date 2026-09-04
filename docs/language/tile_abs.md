# T.tile.abs

## 1. 功能说明

对源操作数逐元素执行绝对值运算：`dst[i] = |src[i]|`

## 2. 函数原型

### 2.1 函数定义

```python
def abs(
    dst: Buffer | BufferRegion,
    src0: Buffer | BufferRegion,
)
```

### 2.2 参数说明

| 参数名 | 输入/输出 | 描述 | 类型 | 必填/可选 |
|--------|----------|------|------|----------|
| dst | 输出 | 存放绝对值运算结果 | 张量（tensor） | 必填 |
| src0 | 输入 | 源操作数 | 张量（tensor） | 必填 |

> **类型说明**：
> - **tensor**：通过 `T.alloc_ub`、`T.alloc_shared` 等分配的缓冲区（Buffer），或其切片（BufferRegion）

### 2.3 参数规格

#### 2.3.1 DataType 支持

| 平台 | dst | src0 |
|------|:---:|:----:|
| Ascend A2 / A3 | float16, float32 | float16, float32 |
| Ascend A5 | uint8, int8, int16, float16, int32, float32, int64 | uint8, int8, int16, float16, int32, float32, int64 |

#### 2.3.2 Shape 支持

- 支持 1D 和 2D

### 2.4 约束条件

1. dst 与 src0 的元素总数必须相同
2. 操作数地址需 32 字节对齐（硬件约束）

## 3. 示例代码

**示例 1：逐元素绝对值运算（float16）**

```python
src0 = T.alloc_ub((256,), "float16")
dst = T.alloc_ub((256,), "float16")
T.tile.abs(dst, src0)  # dst[i] = |src0[i]|
```

**示例 2：逐元素绝对值运算（int32，仅 A5）**

```python
src0 = T.alloc_ub((256,), "int32")
dst = T.alloc_ub((256,), "int32")
T.tile.abs(dst, src0)  # dst[i] = |src0[i]|，A5 平台支持 int32
```
