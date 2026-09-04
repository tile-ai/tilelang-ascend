# T.tile.bitwise_rshift

## 1. 功能说明

将源张量中的每个元素右移指定的标量位数：`dst[i] = src0[i] >> scalarValue`

> **移位语义说明**：无符号类型执行逻辑右移，高位补 0；有符号类型执行算术右移，高位复制符号位。

## 2. 函数原型

### 2.1 函数定义

```python
def bitwise_rshift(
    dst: Buffer | BufferRegion,
    src0: Buffer | BufferRegion,
    scalarValue: PrimExpr,
)
```

### 2.2 参数说明

| 参数名 | 输入/输出 | 描述 | 类型 | 必填/可选 |
|--------|----------|------|------|----------|
| dst | 输出 | 存放右移运算结果 | 张量（tensor） | 必填 |
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
5. scalarValue 仅支持标量，其 dtype 无需与 dst 一致
6. 当前不支持 `roundEn` 舍入参数
7. int64 和 uint64 暂不可用，编译会触发异常退出（参见 [Issue #1720](https://github.com/tile-ai/tilelang-ascend/issues/1720)）
8. 操作数地址需 32 字节对齐（硬件约束）

## 3. 示例代码

**示例 1：标量右移（2D）**

```python
src0 = T.alloc_ub((64, 256), "int16")
dst = T.alloc_ub((64, 256), "int16")
T.tile.bitwise_rshift(dst, src0, 3)
```

**示例 2：1D 标量右移**

```python
src0 = T.alloc_ub((1024,), "int32")
dst = T.alloc_ub((1024,), "int32")
T.tile.bitwise_rshift(dst, src0, 4)
```

**示例 3：张量切片右移**

```python
src0 = T.alloc_ub((64, 256), "int32")
dst = T.alloc_ub((64, 256), "int32")
for i in range(64):
    T.tile.bitwise_rshift(dst[i, :], src0[i, :], 1)
```
