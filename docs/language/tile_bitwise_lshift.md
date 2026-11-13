# T.tile.bitwise_lshift

## 1. 功能说明

将源张量中的每个元素左移指定的位数（标量或逐元素张量）：`dst[i] = src0[i] << scalarValue[i]`（张量形式）或 `dst[i] = src0[i] << scalarValue`（标量形式）

> **移位语义说明**：对有符号与无符号整数类型，本接口均执行逻辑左移，即高位丢弃、低位补 0。

## 2. 函数原型

### 2.1 函数定义

```python
def bitwise_lshift(
    dst: Buffer | BufferRegion,
    src0: Buffer | BufferRegion,
    scalarValue: Buffer | BufferRegion | PrimExpr,
)
```

### 2.2 参数说明

| 参数名 | 输入/输出 | 描述 | 类型 | 必填/可选 |
|--------|----------|------|------|----------|
| dst | 输出 | 存放左移运算结果 | 张量（tensor） | 必填 |
| src0 | 输入 | 源操作数 | 张量（tensor） | 必填 |
| scalarValue | 输入 | 位移位数（标量或逐元素张量） | 张量（tensor）/ 标量（scalar） | 必填 |

> **类型说明**：
> - **tensor**：通过 `T.alloc_ub`、`T.alloc_shared` 等分配的缓冲区（Buffer），或其**物理连续**的切片（BufferRegion）。二维切片需占满整行（如 `buf[r0:r1, :]`）或仅涉及单行（如 `buf[0:1, a:b]`）；跨多行的列切片（如 `buf[:, a:b]`）不受支持，将产生错误结果
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
5. scalarValue 支持标量移位和逐元素张量移位；张量形式须与 src0 的 dtype 和 shape 一致
6. scalarValue 的 dtype 无需与 dst 一致
7. 操作数地址需 32 字节对齐（硬件约束）
8. 行宽对齐（当前 Ascend C 后端实现的支持范围，非硬件限制）：对完整的二维
   tile，Ascend C 后端当前要求**每一行**的逻辑数据字节数为 32 的倍数。对 int16/uint16（每元素 2 字节），即 tile 内维元素数需为 16 的倍数；
   对 int32/uint32（每元素 4 字节），需为 8 的倍数。
   整个 tile 总字节数为 32 的倍数并不足够：例如 `(64, 100)` 的 int16 tile 总大小为
   `64 × 100 × 2 = 12800` 字节，但每一行为 `100 × 2 = 200` 字节，不满足行宽
   要求。当前实现对此类输入可能编译并运行成功但返回错误结果，且暂无编译期
   拒绝检查。起始地址对齐（第 7 条）与行宽约束是相互独立的两个条件，需分别满足。

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
