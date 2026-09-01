# T.tile.bitwise_lshift

## 1. 功能说明

将源张量中的每个元素左移指定的标量位数：`dst[i] = src0[i] << scalarValue`

> **移位语义说明**：Ascend A2 / A3（910B3）实测有符号与无符号类型均表现为逻辑左移，即高位丢弃、低位补 0。该结果与 Ascend C 文档对有符号类型的描述存在差异，正在 [Issue #1718](https://github.com/tile-ai/tilelang-ascend/issues/1718) 中跟踪。

## 2. 函数原型

### 2.1 函数定义

```python
def bitwise_lshift(
    dst: Buffer | BufferRegion,
    src0: Buffer | BufferRegion,
    scalarValue: PrimExpr,
)
```

### 2.2 参数说明

| 参数名 | 输入/输出 | 描述 | 类型 | 必填/可选 |
|--------|----------|------|------|----------|
| dst | 输出 | 存放左移运算结果 | 张量（tensor） | 必填 |
| src0 | 输入 | 源操作数 | 张量（tensor） | 必填 |
| scalarValue | 输入 | 位移位数 | 标量（scalar） | 必填 |

> **类型说明**：
> - **tensor**：通过 `T.alloc_ub`、`T.alloc_shared` 等分配的缓冲区（Buffer），或其切片（BufferRegion）
> - **scalar**：Python 标量或 PrimExpr

### 2.3 参数规格

#### 2.3.1 DataType 支持

| 平台 | dst | src0 | scalarValue |
|------|:---:|:----:|:-----------:|
| Ascend A2 / A3 | int16, uint16, int32, uint32 | 同 dst | 标量 |

#### 2.3.2 Shape 支持

- 支持 1D 和 2D

### 2.4 约束条件

1. 输入和输出张量必须位于 UB 内存
2. dst 与 src0 的元素个数必须相同
3. src0 的 dtype 必须与 dst 一致
4. 仅支持 int16、uint16、int32 和 uint32
5. scalarValue 当前仅支持标量；张量位移量暂未开放（参见 [Issue #1719](https://github.com/tile-ai/tilelang-ascend/issues/1719)）
6. scalarValue 的 dtype 无需与 dst 一致
7. 操作数地址需 32 字节对齐（硬件约束）

## 3. 示例代码

**示例 1：标量左移（2D）**

```python
src0 = T.alloc_ub((64, 256), "int16")
dst = T.alloc_ub((64, 256), "int16")
T.tile.bitwise_lshift(dst, src0, 2)
```

**示例 2：1D 标量左移**

```python
src0 = T.alloc_ub((1024,), "int32")
dst = T.alloc_ub((1024,), "int32")
T.tile.bitwise_lshift(dst, src0, 4)
```

**示例 3：张量切片左移**

```python
src0 = T.alloc_ub((64, 256), "int32")
dst = T.alloc_ub((64, 256), "int32")
for i in range(64):
    T.tile.bitwise_lshift(dst[i, :], src0[i, :], 1)
```
