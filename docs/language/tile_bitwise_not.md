# T.tile.bitwise_not

## 1. 功能说明

对操作数逐元素执行按位取反运算：`dst[i] = ~src[i]`

## 2. 函数原型

### 2.1 函数定义

```python
def bitwise_not(
    dst: Buffer | BufferRegion,
    src0: Buffer | BufferRegion,
)
```

### 2.2 参数说明

| 参数名 | 输入/输出 | 描述 | 类型 | 必填/可选 |
|--------|----------|------|------|----------|
| dst | 输出 | 存放按位取反运算结果 | 张量（tensor） | 必填 |
| src0 | 输入 | 源操作数 | 张量（tensor） | 必填 |

> **类型说明**：
> - **tensor**：通过 `T.alloc_ub`、`T.alloc_shared` 等分配的缓冲区（Buffer），或其切片（BufferRegion）

### 2.3 参数规格

#### 2.3.1 DataType 支持

| 平台 | dst | src0 |
|------|:---:|:----:|
| Ascend A2 / A3 | int16, uint16 | int16, uint16 |
| Ascend A5 | int8, uint8, int16, uint16, int32, uint32 | int8, uint8, int16, uint16, int32, uint32 |

> **说明**：A2/A3 上 int32/uint32 需通过 ReinterpretCast 转为 int16/uint16 后调用。

#### 2.3.2 Shape 支持

- 支持 1D 和 2D

### 2.4 约束条件

1. dst 与 src0 的 shape 必须相同
2. src0 的 dtype 必须与 dst 一致
3. 仅支持整数类型，不支持浮点类型
4. 操作数地址需 32 字节对齐（硬件约束）

## 3. 示例代码

**示例 1：按位取反**

```python
src0 = T.alloc_ub((256,), "int16")
dst = T.alloc_ub((256,), "int16")
T.tile.bitwise_not(dst, src0)  # dst[i] = ~src0[i]
```
