# T.tile.gather

## 1. 功能说明

根据字节地址偏移从源 buffer 中按元素收集到目的 buffer：`dst[i] = src[src_base_addr + src_offset[i]]`，其中 src_base_addr 和 src_offset 的单位均为字节（Bytes）。

## 2. 函数原型

### 2.1 函数定义

```python
def gather(
    dst: Buffer | BufferRegion,
    src: Buffer | BufferRegion,
    src_offset: Buffer | BufferRegion,
    src_base_addr: PrimExpr,
    *,
    tmp: Buffer | BufferRegion | None = None,
)
```

### 2.2 参数说明

| 参数名 | 输入/输出 | 描述 | 类型 | 必填/可选 |
|--------|----------|------|------|----------|
| dst | 输出 | 收集结果的存放位置 | 张量（tensor） | 必填 |
| src | 输入 | 源数据表 | 张量（tensor） | 必填 |
| src_offset | 输入 | 每个元素在 src 中对应的字节地址偏移 | 张量（tensor） | 必填 |
| src_base_addr | 输入 | src 的起始基地址偏移，单位为字节（Bytes） | 整数 | 必填 |
| tmp | 输入 | 可选的 UB 临时缓冲区，由 lowering 重解释 dtype，无语义意义 | 张量（tensor） | 可选 |

> **类型说明**：
> - **tensor**：通过 `T.alloc_ub`、`T.alloc_shared` 等分配的缓冲区（Buffer），或其切片（BufferRegion）

### 2.3 参数规格

#### 2.3.1 DataType 支持

| 平台 | dst | src | src_offset |
|------|:---:|:---:|:----------:|
| Ascend A2 / A3 | float16, float32, bfloat16, int16, uint16, int32, uint32 | 同 dst | uint32 |

#### 2.3.2 Shape 支持

- 无限制，支持任意维度
- pto 后端要求处理元素数 ≥ 512

### 2.4 约束条件

1. dst 与 src 的 dtype 必须相同
2. src_offset 的 dtype 必须为 uint32
3. count（处理元素个数）由 `min(dst_size, offset_size)` 自动推导，dst 至少需容纳 count 个元素
4. src_offset 中的偏移值必须保证 src 元素类型位宽对齐（如 float16 需 2 字节对齐，float32 需 4 字节对齐）（硬件约束）
5. 偏移后的地址不能超出 src 的 buffer 范围（硬件约束）
6. 所有操作数地址需 32 字节对齐（硬件约束）
7. A2/A3 平台：src_offset 中的地址偏移不能超出 uint32_t 的范围（硬件约束）

> **pto 后端限制**：`src_base_addr` 参数在 pto 后端被忽略（codegen 假定为 0），非零 `src_base_addr` 仅在 ascendc 后端生效。

## 3. 示例代码

**示例 1：基本 gather（逆序收集）**

```python
x_ub = T.alloc_ub((128,), "float16")
offset_ub = T.alloc_ub((128,), "uint32")
dst_ub = T.alloc_ub((128,), "float16")
T.tile.gather(dst_ub, x_ub, offset_ub, 0)
```

**示例 2：带基地址偏移的 gather**

```python
table_ub = T.alloc_ub((512,), "float16")
offset_ub = T.alloc_ub((128,), "uint32")
dst_ub = T.alloc_ub((128,), "float16")
T.tile.gather(dst_ub, table_ub, offset_ub, 64)
```
