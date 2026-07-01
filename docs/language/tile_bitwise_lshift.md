# T.tile.bitwise_lshift

## 1. 功能说明

对 buffer 中每个元素执行标量左移运算：`dst[i] = src[i] << shift_amount`

无符号类型执行逻辑左移（高位丢弃，低位补 0），有符号类型执行算术左移（次高位丢弃，低位补 0）。

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
| scalarValue | 输入 | 位移位数（标量） | 标量（scalar） | 必填 |

> **类型说明**：
> - **tensor**：通过 `T.alloc_ub`、`T.alloc_shared` 等分配的缓冲区（Buffer），或其切片（BufferRegion）
> - **scalar**：Python 标量或表达式（PrimExpr），表示位移位数，不支持 tensor-tensor 位移

### 2.3 参数规格

#### 2.3.1 DataType 支持

| 平台 | dst | src0 | scalarValue |
|------|:---:|:----:|:-----------:|
| Ascend A2 / A3 | int16, uint16, int32, uint32 | int16, uint16, int32, uint32 | 与 dst dtype 一致 |
| Ascend A5 | int8, uint8, int16, uint16, int32, uint32, int64, uint64 | int8, uint8, int16, uint16, int32, uint32, int64, uint64 | 与 dst dtype 一致 |

> **说明**：scalarValue 的数据类型需与 dst 元素类型一致（Ascend C 约束）。

#### 2.3.2 Shape 支持

- 支持 1D 和 2D

### 2.4 约束条件

1. dst 与 src0 的元素总数必须相同
2. src0 的 dtype 必须与 dst 一致
3. 仅支持整数类型，不支持浮点类型
4. scalarValue 必须为标量（PrimExpr），不支持 tensor-tensor 位移
5. scalarValue 的数据类型需与 dst 元素类型一致
6. 操作数地址需 32 字节对齐（硬件约束）

## 3. 示例代码

**示例 1：标量左移**

```python
src0 = T.alloc_ub((256,), "int16")
dst = T.alloc_ub((256,), "int16")
T.tile.bitwise_lshift(dst, src0, 2)  # dst[i] = src0[i] << 2
```
