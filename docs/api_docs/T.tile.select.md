# T.tile.select

## 1. 功能说明

根据 selMask 的比特位，从 src0 或 src1 中选择元素写入 dst：`dst[i] = selMask.bit[i] ? src0[i] : src1[i]`

## 2. 函数原型

### 2.1 函数定义

```python
def select(
    dst: Buffer | BufferRegion,
    selMask: Buffer,
    src0: Buffer | BufferRegion,
    src1: Buffer | BufferLoad | PrimExpr,
    selMode: str,
)
```

### 2.2 参数说明

| 参数名 | 输入/输出 | 描述 | 类型 | 必填/可选 |
|--------|----------|------|------|----------|
| dst | 输出 | 存放选择结果 | 张量（tensor） | 必填 |
| selMask | 输入 | 选择掩码，每个 bit 控制一个元素的选择来源（bit=1 选 src0，bit=0 选 src1） | 张量（tensor，bit-packed，dtype 为 uint8） | 必填 |
| src0 | 输入 | bit=1 时选择的来源 | 张量（tensor） | 必填 |
| src1 | 输入 | bit=0 时选择的来源，支持 tensor 或 scalar | 张量（tensor）/ 标量（scalar） | 必填 |
| selMode | 输入 | 选择模式，决定 selMask 的解读方式和 src1 的类型 | 字符串，取值见 [2.3.3 selMode 说明](#233-selmode-说明) | 必填 |

> **类型说明**：
> - **tensor**：通过 `T.alloc_ub`、`T.alloc_shared` 等分配的缓冲区（Buffer），或其切片（BufferRegion）
> - **scalar**：单个元素值，可以是 buffer 元素访问（BufferLoad）或 Python 标量/表达式（PrimExpr）

### 2.3 参数规格

#### 2.3.1 DataType 支持

| 平台 | dst | src0 | src1 | selMask |
|------|:---:|:----:|:----:|:-------:|
| Ascend A2 / A3 | float16, float32 | float16, float32 | float16, float32 | uint8 |

- 当 src1 为 scalar 时，其数据类型须与 src0 一致

#### 2.3.2 Shape 支持

- 支持 1D 和 2D
- 更高维 buffer 需通过切片降维为 1D/2D 的 BufferRegion 传入
#### 2.3.3 selMode 说明

selMode 决定 selMask 的解读方式，共 3 种模式：

| selMode | 含义 | 适用场景 |
|---------|------|---------|
| `"VSEL_CMPMASK_SPR"` | bit-packed mask，每轮迭代复用同一份 mask（最多 `256 / sizeof(T)` 个 element） | 配合 `T.tile.compare` 的结果使用，mask 来自比较输出 |
| `"VSEL_TENSOR_SCALAR_MODE"` | mask 连续存放，多轮迭代连续消耗；src1 为标量 | src0 为 tensor，src1 为常量值 |
| `"VSEL_TENSOR_TENSOR_MODE"` | mask 连续存放，多轮迭代连续消耗；src1 为 tensor | src0 和 src1 均为 tensor |

**src1 类型与 selMode 的对应关系**：
- src1 为 `PrimExpr` / `float`（scalar）→ 必须用 `"VSEL_TENSOR_SCALAR_MODE"`
- src1 为 `BufferLoad`（单个元素）→ 必须用 `"VSEL_CMPMASK_SPR"` 或 `"VSEL_TENSOR_TENSOR_MODE"`
- src1 为 `Buffer`（tensor）→ 必须用 `"VSEL_CMPMASK_SPR"` 或 `"VSEL_TENSOR_TENSOR_MODE"`

### 2.4 约束条件

1. dst 与 src0 的 shape 必须一致
2. src1 为 tensor 时，shape 须与 src0 一致
3. selMask 的 dtype 须为 `uint8`（bit-packed 掩码）
4. 操作数地址需 32 字节对齐（硬件约束）
5. src1 仅支持 tensor（Buffer/BufferRegion）或 scalar（PrimExpr/float）；BufferLoad 单元素访问形式当前不可用
6. `"VSEL_CMPMASK_SPR"` 模式下 selMask 有效元素数上限为 `256 / sizeof(T)`（256 for float16，128 for float32）（硬件约束）
7. `"VSEL_TENSOR_SCALAR_MODE"` 和 `"VSEL_TENSOR_TENSOR_MODE"` 模式在 Ascend A2/A3 上需预留 8KB Unified Buffer 临时空间（硬件约束）

## 3. 示例代码

**示例 1：tensor-tensor 模式（selMode = "VSEL_TENSOR_TENSOR_MODE"）**

```python
src0 = T.alloc_ub((256,), "float16")
src1 = T.alloc_ub((256,), "float16")
mask = T.alloc_ub((32,),  "uint8")   # 256 elements ÷ 8 bits/byte = 32 bytes
dst  = T.alloc_ub((256,), "float16")
T.tile.select(dst, mask, src0, src1, "VSEL_TENSOR_TENSOR_MODE")
```

**示例 2：tensor-scalar 模式（selMode = "VSEL_TENSOR_SCALAR_MODE"）**

```python
src0 = T.alloc_ub((256,), "float16")
mask = T.alloc_ub((32,),  "uint8")
dst  = T.alloc_ub((256,), "float16")
T.tile.select(dst, mask, src0, 0.0, "VSEL_TENSOR_SCALAR_MODE")  # src1 = 0.0
```

**示例 3：配合 T.tile.compare 使用（selMode = "VSEL_CMPMASK_SPR"）**

```python
src0 = T.alloc_ub((256,), "float16")
src1 = T.alloc_ub((256,), "float16")
cmp_mask = T.alloc_ub((32,), "uint8")  # T.tile.compare 的 bit-packed 结果
dst = T.alloc_ub((256,), "float16")

T.tile.compare(cmp_mask, src0, src1, "GT")                       # src0 > src1 的位置 bit=1
T.tile.select(dst, cmp_mask, src0, src1, "VSEL_CMPMASK_SPR")     # 选较大值 → 等价于 max(src0, src1)
```
