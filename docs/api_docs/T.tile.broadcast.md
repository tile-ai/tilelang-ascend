# T.tile.broadcast

## 1. 功能说明

将输入张量沿指定维度广播到目标 shape：`dst[i, j] = src[i, 0]`（axis=1）或 `dst[i, j] = src[0, j]`（axis=0）

## 2. 函数原型

### 2.1 函数定义

```python
def broadcast(
    dst: Buffer | BufferRegion,
    src: Buffer | BufferRegion,
    axis: int | None = None,
    *,
    tmp: Buffer | BufferRegion | None = None,
)
```

### 2.2 参数说明

| 参数名 | 输入/输出 | 描述 | 类型 | 必填/可选 |
|--------|----------|------|------|----------|
| dst | 输出 | 存放广播结果 | 张量（tensor，scope 必须为 UB） | 必填 |
| src | 输入 | 待广播的源张量 | 张量（tensor，scope 必须为 UB） | 必填 |
| axis | 输入 | 广播维度：0 表示沿行方向广播，1 表示沿列方向广播；None 时根据 shape 自动推断 | 整数（0 或 1）/ None | 可选（默认 `None`） |
| tmp | 输入 | 可选的临时缓冲区，供后端内部使用 | 张量（tensor，scope 为 `shared.ub`） | 可选（默认 `None`） |

> **类型说明**：
> - **tensor**：通过 `T.alloc_ub`、`T.alloc_shared` 等分配的缓冲区（Buffer），或其切片（BufferRegion）

### 2.3 参数规格

#### 2.3.1 DataType 支持

| 平台 | dst | src |
|------|:---:|:---:|
| Ascend A2 / A3 | int8, uint8, int16, uint16, float16, bfloat16, float32, int32, uint32 | 同 dst |

#### 2.3.2 Shape 支持

- 支持 1D 和 2D

> **已知限制**：
> - 2D axis=1（src=(M,1)→dst=(M,N)）当 M≥2 时，ascendc、pto 均不支持
> - 1D axis=0 在 pto 后端不支持

### 2.4 约束条件

1. dst 与 src 的 dtype 必须一致
2. axis 仅支持 0 或 1；axis=None 时根据 src/dst 的 shape 自动推断广播轴
3. 广播维度上 src 的 size 必须为 1，非广播维度上 src 和 dst 的 size 必须相同
4. 1D src 广播到 2D dst 时，自动推断广播轴；无法推断时抛出 `ValueError`
5. 不支持 src 与 dst 地址重叠（硬件约束）
6. 仅支持 ND 格式（硬件约束）
7. dim=2 且 axis=0 时，srcShape[1] 必须 32 字节对齐（即元素个数 × 元素大小 % 32 == 0）（硬件约束，非对齐时静默出错而非报错）

## 3. 示例代码

**示例 1：1D src 广播到 2D dst（axis 自动推断）**

```python
src = T.alloc_ub((4, 1), "float16")
dst = T.alloc_ub((4, 16), "float16")
T.tile.broadcast(dst, src)  # axis=1 自动推断：src[4,1] → dst[4,16]
```

**示例 2：axis=0，沿列方向广播**

```python
src = T.alloc_ub((1, 16), "float32")
dst = T.alloc_ub((8, 16), "float32")
T.tile.broadcast(dst, src, axis=0)  # src[1,16] → dst[8,16]
```

**示例 3：1D src 广播到 1D dst（需要 src 为 [1]）**

```python
src = T.alloc_ub((1,), "float16")
dst = T.alloc_ub((256,), "float16")
T.tile.broadcast(dst, src, axis=0)
```
