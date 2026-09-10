# T.tile.select

## 1. Description

Selects elements from `src0` or `src1` based on `selMask` bit values and writes the result to `dst`: `dst[i] = selMask.bit[i] ? src0[i] : src1[i]`

## 2. Function Prototype

### 2.1 Function Definition

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

### 2.2 Parameters

| Parameter | Input/Output | Description | Type | Required/Optional |
|-----------|-------------|-------------|------|-------------------|
| dst | Output | Stores the selection result | tensor | Required |
| selMask | Input | Selection mask; each bit controls the source of one element (bit=1 selects from src0, bit=0 selects from src1) | tensor (bit-packed, dtype uint8) | Required |
| src0 | Input | Source selected when bit=1 | tensor | Required |
| src1 | Input | Source selected when bit=0; supports tensor, BufferLoad, or scalar | tensor / scalar | Required |
| selMode | Input | Selection mode; determines how selMask is interpreted and the type of src1 | string, see [2.3.3 selMode](#233-selmode) | Required |
| tmp | Input | Optional complete UB scratch storage; its scalar dtype is reinterpreted by lowering and has no semantic meaning | tensor / None | Optional (default `None`) |

> **Type Notes**:
> - **tensor**: A buffer (Buffer) allocated via `T.alloc_ub`, `T.alloc_shared`, etc., or its slice (BufferRegion)
> - **scalar**: A Python scalar or expression (PrimExpr), e.g. `1.0`, `0.0`

### 2.3 Specifications

#### 2.3.1 DataType Support

| Platform | dst | src0 | src1 | selMask |
|----------|:---:|:----:|:----:|:-------:|
| Ascend A2 / A3 | float16, float32 | float16, float32 | float16, float32 | uint8 |

- When src1 is a scalar, its dtype must match src0
- All three selMode values support the same set of dtypes

#### 2.3.2 Shape Support

- Supports 1D and 2D
- Higher-dimensional buffers must be passed as 1D/2D BufferRegion via slicing

#### 2.3.3 selMode

selMode determines how selMask is interpreted. There are 3 modes:

| selMode | Description | Use Case |
|---------|-------------|----------|
| `"VSEL_CMPMASK_SPR"` | bit-packed mask, reused across iterations | Used with `T.tile.compare` results; mask comes from comparison output |
| `"VSEL_TENSOR_SCALAR_MODE"` | mask stored contiguously, consumed across iterations; src1 is a scalar | src0 is a tensor, src1 is a constant value |
| `"VSEL_TENSOR_TENSOR_MODE"` | mask stored contiguously, consumed across iterations; src1 is a tensor | Both src0 and src1 are tensors |

**src1 type and selMode correspondence**:
- src1 is `PrimExpr` / `float` (scalar) → must use `"VSEL_TENSOR_SCALAR_MODE"`
- src1 is `Buffer` / `BufferRegion` (tensor) → must use `"VSEL_CMPMASK_SPR"` or `"VSEL_TENSOR_TENSOR_MODE"`
- src1 is `BufferLoad` (single element access) → must use `"VSEL_CMPMASK_SPR"` or `"VSEL_TENSOR_TENSOR_MODE"`

### 2.4 Constraints

1. dst and src0 must have the same shape
2. When src1 is a tensor, its shape must match src0
3. selMask is a bit-packed mask; dtype must be uint8, element count = data element count / 8
4. Operand addresses must be 32-byte aligned (hardware constraint)
5. src1 supports tensor (Buffer/BufferRegion), BufferLoad (single element access), or scalar (PrimExpr/float)
6. `"VSEL_CMPMASK_SPR"` mode reuses the compare mask register; the maximum element count per call is `256 / sizeof(T)` (128 for float16, 64 for float32). Exceeding this causes precision errors (hardware constraint)
7. `"VSEL_TENSOR_SCALAR_MODE"` and `"VSEL_TENSOR_TENSOR_MODE"` modes require reserving the last 8KB of Unified Buffer as temporary space (hardware constraint)

## 3. Examples

**Example 1: tensor-tensor mode (selMode = "VSEL_TENSOR_TENSOR_MODE")**

```python
src0 = T.alloc_ub((256,), "float16")
src1 = T.alloc_ub((256,), "float16")
mask = T.alloc_ub((32,),  "uint8")   # 256 elements / 8 bits/byte = 32 bytes
dst  = T.alloc_ub((256,), "float16")
T.tile.select(dst, mask, src0, src1, "VSEL_TENSOR_TENSOR_MODE")
```

**Example 2: tensor-scalar mode (selMode = "VSEL_TENSOR_SCALAR_MODE")**

```python
src0 = T.alloc_ub((256,), "float16")
mask = T.alloc_ub((32,),  "uint8")
dst  = T.alloc_ub((256,), "float16")
T.tile.select(dst, mask, src0, 0.0, "VSEL_TENSOR_SCALAR_MODE")  # src1 = 0.0
```

**Example 3: Used with T.tile.compare (selMode = "VSEL_CMPMASK_SPR")**

```python
src0 = T.alloc_ub((256,), "float16")
src1 = T.alloc_ub((256,), "float16")
cmp_mask = T.alloc_ub((32,), "uint8")  # bit-packed result from T.tile.compare
dst = T.alloc_ub((256,), "float16")

T.tile.compare(cmp_mask, src0, src1, "GT")                       # bit=1 where src0 > src1
T.tile.select(dst, cmp_mask, src0, src1, "VSEL_CMPMASK_SPR")     # select the larger value -> equivalent to max(src0, src1)
```
