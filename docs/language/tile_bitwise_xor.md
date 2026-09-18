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
    *,
    tmp: Buffer | BufferRegion | None = None,
)
```

### 2.2 参数说明

| 参数名 | 输入/输出 | 描述 | 类型 | 必填/可选 |
|--------|----------|------|------|----------|
| dst | 输出 | 存放按位异或运算结果 | 张量（tensor） | 必填 |
| src0 | 输入 | 第一个源操作数 | 张量（tensor） | 必填 |
| src1 | 输入 | 第二个源操作数 | 张量（tensor） | 必填 |
| tmp | 输入 | 临时缓冲区 | 张量（tensor） | 可选 |

> **类型说明**：
> - **tensor**：通过 `T.alloc_ub`、`T.alloc_shared` 等分配的缓冲区（Buffer），或其切片（BufferRegion）
> - **tmp**：可选的一维 UB 临时缓冲区；省略时由框架自动分配

### 2.3 参数规格

#### 2.3.1 DataType 支持

| 平台 / 后端 | dst | src0 | src1 |
|-------------|:---:|:----:|:----:|
| Ascend A2 / A3（Ascend C） | int16, uint16 | 同 dst | 同 dst |
| Ascend A2 / A3（PTO） | int8, int16, uint16 | 同 dst | 同 dst |

#### 2.3.2 Shape 支持

- 支持 1D 和 2D

### 2.4 约束条件

1. 输入和输出张量必须位于 UB 内存
2. dst、src0 和 src1 的元素个数必须相同
3. src0 和 src1 的 dtype 必须与 dst 一致
4. 仅支持上述 DataType 表中列出的整数类型
5. src1 仅支持张量，不支持标量
6. tmp 省略时由框架自动分配；显式传入时必须是一维 UB 张量
7. dst、src0、src1 和 tmp 的地址不得重叠
8. 操作数地址需 32 字节对齐（硬件约束）
9. PTO 后端的 uint8 暂不可用（参见 [Issue #1721](https://github.com/tile-ai/tilelang-ascend/issues/1721)）
10. int32 和 uint32 暂不可用，编译会触发异常退出（参见 [Issue #1722](https://github.com/tile-ai/tilelang-ascend/issues/1722)）

## 3. 示例代码

**示例 1：tensor-tensor 按位异或**

```python
src0 = T.alloc_ub((256,), "int16")
src1 = T.alloc_ub((256,), "int16")
dst = T.alloc_ub((256,), "int16")
T.tile.bitwise_xor(dst, src0, src1)
```

**示例 2：张量切片按位异或**

```python
src0 = T.alloc_ub((128, 256), "int16")
src1 = T.alloc_ub((128, 256), "int16")
dst = T.alloc_ub((128, 256), "int16")
T.tile.bitwise_xor(dst[0:128, 0:256], src0[0:128, 0:256], src1[0:128, 0:256])
```

**示例 3：显式 tmp 缓冲区**

```python
src0 = T.alloc_ub((128, 256), "int16")
src1 = T.alloc_ub((128, 256), "int16")
dst = T.alloc_ub((128, 256), "int16")
tmp = T.alloc_ub((128 * 256 * 2,), "uint8")
T.tile.bitwise_xor(dst, src0, src1, tmp=tmp)
```
