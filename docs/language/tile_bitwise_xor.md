# T.tile.bitwise_xor

## 1. 功能说明

对两个操作数逐元素执行按位异或运算：`dst[i] = src0[i] ^ src1[i]`

## 2. 函数原型

### 2.1 函数定义

```python
def bitwise_xor(
    dst: Buffer | BufferRegion,
    src0: Buffer | BufferRegion,
    src1: Buffer | BufferRegion,
)
```

### 2.2 参数说明

| 参数名 | 输入/输出 | 描述 | 类型 | 必填/可选 |
|--------|----------|------|------|----------|
| dst | 输出 | 存放按位异或运算结果 | 张量（tensor） | 必填 |
| src0 | 输入 | 第一个源操作数 | 张量（tensor） | 必填 |
| src1 | 输入 | 第二个源操作数（仅支持 tensor，不支持 scalar） | 张量（tensor） | 必填 |

> **类型说明**：
> - **tensor**：通过 `T.alloc_ub`、`T.alloc_shared` 等分配的缓冲区（Buffer），或其切片（BufferRegion）
> - bitwise_xor 的 src1 仅支持 Buffer/BufferRegion，不支持 BufferLoad 或 PrimExpr 标量

### 2.3 参数规格

#### 2.3.1 DataType 支持

| 平台 | dst | src0 | src1 |
|------|:---:|:----:|:----:|
| Ascend A2 / A3 | int16, uint16 | int16, uint16 | int16, uint16 |
| Ascend A5 | int8, uint8, int16, uint16, int32, uint32 | int8, uint8, int16, uint16, int32, uint32 | int8, uint8, int16, uint16, int32, uint32 |

> **说明**：A2/A3 上 bitwise_xor 为复合实现（TOR → TAND → TNOT → TAND），需要临时缓冲区；A5 上有原生 vxor 指令支持。

#### 2.3.2 Shape 支持

- 支持 1D 和 2D

### 2.4 约束条件

1. dst 与 src0 的 shape 必须相同
2. src1 的 shape 也必须与 dst 相同
3. src0 和 src1 的 dtype 必须与 dst 一致
4. 仅支持整数类型，不支持浮点类型
5. src1 不支持标量（scalar），仅支持 tensor
6. 接口内部使用框架自动申请的临时缓冲区（大小与 src 相同），无需用户手动分配
7. A2/A3 上为复合实现，dst/src0/src1/tmp 四个操作数地址不得重叠
8. 操作数地址需 32 字节对齐（硬件约束）

## 3. 示例代码

**示例 1：tensor-tensor 按位异或**

```python
src0 = T.alloc_ub((256,), "int16")
src1 = T.alloc_ub((256,), "int16")
dst = T.alloc_ub((256,), "int16")
T.tile.bitwise_xor(dst, src0, src1)  # dst[i] = src0[i] ^ src1[i]
```
