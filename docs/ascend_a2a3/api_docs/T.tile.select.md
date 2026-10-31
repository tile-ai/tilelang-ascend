# T.tile.select

## 1. 功能说明

根据 `selMask` 的比特位从 `src0` 或 `src1` 中选择元素，结果写入 `dst`：`dst[i] = selMask.bit[i] ? src0[i] : src1[i]`

## 2. 函数原型

### 2.1 函数定义

```python
def select(
    dst: Buffer | BufferRegion,
    selMask: Buffer,
    src0: Buffer | BufferRegion,
    src1: Buffer | BufferLoad | PrimExpr,
    selMode: str,
    *,
    tmp: Buffer | BufferRegion | None = None,
)
```

### 2.2 参数说明

| 参数名 | 输入/输出 | 描述 | 类型 | 必填/可选 |
|--------|----------|------|------|----------|
| dst | 输出 | 存放选择结果 | 张量（tensor） | 必填 |
| selMask | 输入 | 选择掩码；每个比特控制一个元素的数据来源（bit=1 选 src0，bit=0 选 src1） | 张量（bit-packed，dtype uint8） | 必填 |
| src0 | 输入 | bit=1 时选择的源 | 张量（tensor） | 必填 |
| src1 | 输入 | bit=0 时选择的源；支持张量、BufferLoad 或标量 | 张量（tensor）/ 标量（scalar） | 必填 |
| selMode | 输入 | 选择模式，决定 selMask 的解释方式及 src1 的类型 | 字符串，见 [2.3.3 selMode](#233-selmode) | 必填 |
| tmp | 输入 | 可选 UB 临时存储空间；其标量 dtype 由 lowering 重解释，无语义含义 | 张量（tensor）/ None | 可选（默认 `None`） |

> **类型说明**：
> - **tensor**：通过 `T.alloc_ub`、`T.alloc_shared` 等分配的缓冲区（Buffer），或其切片（BufferRegion）
> - **scalar**：Python 标量或表达式（PrimExpr），如 `1.0`、`0.0`

### 2.3 参数规格

#### 2.3.1 数据类型支持

| 平台 | dst | src0 | src1 | selMask |
|------|:---:|:----:|:----:|:-------:|
| Ascend A2 / A3 | float16, float32 | float16, float32 | float16, float32 | uint8 |

- src1 为标量时，其 dtype 必须与 src0 一致
- 三种 selMode 均支持相同的数据类型

#### 2.3.2 Shape 支持

- 支持 1D 和 2D
- 高维缓冲区需通过切片转为 1D/2D BufferRegion 传入

#### 2.3.3 selMode

selMode 决定 selMask 的解释方式，共 3 种模式：

| selMode | 说明 | 适用场景 |
|---------|------|---------|
| `"VSEL_CMPMASK_SPR"` | bit-packed 掩码，可跨迭代复用 | 与 `T.tile.compare` 配合使用；掩码来自比较输出 |
| `"VSEL_TENSOR_SCALAR_MODE"` | 掩码连续存储，逐迭代消耗；src1 为标量 | src0 为张量，src1 为常量 |
| `"VSEL_TENSOR_TENSOR_MODE"` | 掩码连续存储，逐迭代消耗；src1 为张量 | src0 和 src1 均为张量 |

**src1 类型与 selMode 对应关系**：
- src1 为 `PrimExpr` / `float`（标量）→ 必须使用 `"VSEL_TENSOR_SCALAR_MODE"`
- src1 为 `Buffer` / `BufferRegion`（张量）→ 必须使用 `"VSEL_CMPMASK_SPR"` 或 `"VSEL_TENSOR_TENSOR_MODE"`
- src1 为 `BufferLoad`（单元素访问）→ 必须使用 `"VSEL_CMPMASK_SPR"` 或 `"VSEL_TENSOR_TENSOR_MODE"`

### 2.4 约束条件

1. dst 与 src0 的 shape 必须相同
2. src1 为张量时，其 shape 必须与 src0 一致
3. selMask 为 bit-packed 掩码，dtype 必须为 uint8，元素个数 = 数据元素个数 / 8
4. 操作数地址需 32 字节对齐（硬件约束）
5. src1 支持张量（Buffer/BufferRegion）、BufferLoad（单元素访问）或标量（PrimExpr/float）
6. `"VSEL_CMPMASK_SPR"` 模式复用比较掩码寄存器，每次调用最多处理 `256 / sizeof(T)` 个元素（float16 为 128，float32 为 64）。超出该上限会导致精度错误（硬件约束）
7. `"VSEL_TENSOR_SCALAR_MODE"` 和 `"VSEL_TENSOR_TENSOR_MODE"` 模式需预留 UB 最后 8KB 作为临时空间（硬件约束）

## 3. 示例代码

**示例 1：张量-张量模式（selMode = "VSEL_TENSOR_TENSOR_MODE"）**

```python
src0 = T.alloc_ub((256,), "float16")
src1 = T.alloc_ub((256,), "float16")
mask = T.alloc_ub((32,),  "uint8")   # 256 元素 / 8 比特/字节 = 32 字节
dst  = T.alloc_ub((256,), "float16")
T.tile.select(dst, mask, src0, src1, "VSEL_TENSOR_TENSOR_MODE")
```

**示例 2：张量-标量模式（selMode = "VSEL_TENSOR_SCALAR_MODE"）**

```python
src0 = T.alloc_ub((256,), "float16")
mask = T.alloc_ub((32,),  "uint8")
dst  = T.alloc_ub((256,), "float16")
T.tile.select(dst, mask, src0, 0.0, "VSEL_TENSOR_SCALAR_MODE")  # src1 = 0.0
```

**示例 3：与 T.tile.compare 配合（selMode = "VSEL_CMPMASK_SPR"）**

```python
src0 = T.alloc_ub((256,), "float16")
src1 = T.alloc_ub((256,), "float16")
cmp_mask = T.alloc_ub((32,), "uint8")  # 来自 T.tile.compare 的 bit-packed 结果
dst = T.alloc_ub((256,), "float16")

T.tile.compare(cmp_mask, src0, src1, "GT")                       # bit=1 表示 src0 > src1
T.tile.select(dst, cmp_mask, src0, src1, "VSEL_CMPMASK_SPR")     # 选择较大值 = max(src0, src1)
```

**示例 4：显式指定 tmp 临时空间**

```python
src0 = T.alloc_ub((256,), "float16")
src1 = T.alloc_ub((256,), "float16")
mask = T.alloc_ub((32,),  "uint8")
dst  = T.alloc_ub((256,), "float16")
tmp  = T.alloc_ub((512,), "uint8")  # UB 临时空间，dtype 由 lowering 重解释
T.tile.select(dst, mask, src0, src1, "VSEL_TENSOR_TENSOR_MODE", tmp=tmp)
```

> `tmp` 省略时框架自动申请所需空间；显式传入时需提供足够的 UB 容量，dtype 无语义含义（lowering 按 `src0.dtype` 重解释）