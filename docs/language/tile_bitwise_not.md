# T.tile.bitwise_not

## 1. 功能说明

对操作数逐元素执行按位取反运算：`dst[i] = ~src0[i]`

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

| 平台 / 后端 | dst | src0 |
|-------------|:---:|:----:|
| Ascend A2 / A3（Ascend C） | int16, uint16 | 同 dst |
| Ascend A2 / A3（PTO） | int8, uint8, int16, uint16 | 同 dst |

#### 2.3.2 Shape 支持

- 支持 2D

### 2.4 约束条件

1. 输入和输出张量必须位于 UB 内存
2. dst 与 src0 的元素个数必须相同
3. src0 的 dtype 必须与 dst 一致
4. 仅支持上述 DataType 表中列出的整数类型
5. 操作数地址需 32 字节对齐（硬件约束）
6. Ascend C 后端要求 tile 字节数为 32 的倍数，否则可能产生错误结果（参见 [Issue #1717](https://github.com/tile-ai/tilelang-ascend/issues/1717)）

## 3. 示例代码

**示例 1：按位取反**

```python
src0 = T.alloc_ub((256,), "int16")
dst = T.alloc_ub((256,), "int16")
T.tile.bitwise_not(dst, src0)
```

**示例 2：张量切片按位取反**

```python
src0 = T.alloc_ub((128, 256), "int16")
dst = T.alloc_ub((128, 256), "int16")
T.tile.bitwise_not(dst[0:128, 0:256], src0[0:128, 0:256])
```

**示例 3：int8 按位取反（仅 PTO 后端）**

```python
src0 = T.alloc_ub((256,), "int8")
dst = T.alloc_ub((256,), "int8")
T.tile.bitwise_not(dst, src0)
```
