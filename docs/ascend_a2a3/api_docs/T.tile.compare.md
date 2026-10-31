# T.tile.compare

## 1. 功能说明

对两个操作数逐元素比较，将结果以 bit-packed 形式写入 dst：`dst.bit[i] = (src0[i] <mode> src1[i]) ? 1 : 0`

## 2. 函数原型

### 2.1 函数定义

```python
def compare(
    dst: Buffer | BufferRegion,
    src0: Buffer | BufferRegion,
    src1: Buffer | BufferRegion | BufferLoad | PrimExpr,
    mode: str,
)
```

### 2.2 参数说明

| 参数名 | 输入/输出 | 描述 | 类型 | 必填/可选 |
|--------|----------|------|------|----------|
| dst | 输出 | 存放比较结果（bit-packed，uint8） | 张量（tensor） | 必填 |
| src0 | 输入 | 第一个源操作数 | 张量（tensor） | 必填 |
| src1 | 输入 | 第二个源操作数，支持 tensor 或 scalar | 张量（tensor）/ 标量（scalar） | 必填 |
| mode | 输入 | 比较模式，决定比较运算符 | 字符串，取值见 [2.3.3 mode 说明](#233-mode-说明) | 必填 |

> **类型说明**：
> - **tensor**：通过 `T.alloc_ub`、`T.alloc_shared` 等分配的缓冲区（Buffer），或其切片（BufferRegion）
> - **scalar**：单个元素值，可以是 buffer 元素访问（BufferLoad）或 Python 标量/表达式（PrimExpr/float）

### 2.3 参数规格

#### 2.3.1 DataType 支持

| 平台 | dst | src0 | src1 |
|------|:---:|:----:|:----:|
| Ascend A2 / A3 | int8, uint8 | float16, float32, int32 | float16, float32, int32 |

- 当 src1 为 scalar 时，其数据类型须与 src0 一致
- A2/A3 平台 int32 仅支持 `mode = "EQ"`

#### 2.3.2 Shape 支持

- 支持 1D 和 2D

#### 2.3.3 mode 说明

mode 决定比较运算符，共 6 种取值：

| mode | 含义 | 运算语义 |
|------|------|---------|
| `"EQ"` | 等于（equal to） | `src0[i] == src1[i]` |
| `"NE"` | 不等于（not equal to） | `src0[i] != src1[i]` |
| `"GT"` | 大于（greater than） | `src0[i] > src1[i]` |
| `"GE"` | 大于等于（greater than or equal to） | `src0[i] >= src1[i]` |
| `"LT"` | 小于（less than） | `src0[i] < src1[i]` |
| `"LE"` | 小于等于（less than or equal to） | `src0[i] <= src1[i]` |

### 2.4 约束条件

1. src0 和 src1 的 dtype 必须相同（src1 为 scalar 时，数据类型须与 src0 一致）
2. dst 的 dtype 须为 `uint8`（bit-packed 比较结果）
3. src1 为 tensor 时，shape 须与 src0 一致
4. 操作数地址需 32 字节对齐（硬件约束）
5. src0 的元素数量所占空间需 256 字节对齐（硬件约束）
6. A2/A3 平台 int32 仅支持 `"EQ"` 模式（硬件约束）
7. src1 为 `PrimExpr`/`float`（scalar）时，内部 dispatch 到 `tl.ascend_compare_scalar`；src1 为 `Buffer`/`BufferRegion`（tensor）时，dispatch 到 `tl.ascend_compare`

## 3. 示例代码

**示例 1：tensor-tensor 比较**

```python
src0 = T.alloc_ub((256,), "float16")
src1 = T.alloc_ub((256,), "float16")
dst  = T.alloc_ub((32,),  "uint8")  # 256 elements ÷ 8 bits/byte = 32 bytes
T.tile.compare(dst, src0, src1, "LT")  # dst bit = 1 if src0 < src1
```

**示例 2：tensor-scalar 比较**

```python
src0 = T.alloc_ub((256,), "float16")
dst  = T.alloc_ub((32,),  "uint8")
T.tile.compare(dst, src0, 0.0, "GT")  # dst bit = 1 if src0 > 0.0
```

**示例 3：配合 T.tile.select 使用**

```python
src0 = T.alloc_ub((256,), "float16")
src1 = T.alloc_ub((256,), "float16")
cmp_mask = T.alloc_ub((32,), "uint8")
dst = T.alloc_ub((256,), "float16")

T.tile.compare(cmp_mask, src0, src1, "GT")
T.tile.select(dst, cmp_mask, src0, src1, "VSEL_TENSOR_TENSOR_MODE")  # 等价于 max(src0, src1)
```
