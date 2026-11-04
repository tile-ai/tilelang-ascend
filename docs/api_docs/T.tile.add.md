# T.tile.add

## 1. 功能说明

对两个操作数逐元素执行加法：`dst[i] = src0[i] + src1[i]`

## 2. 函数原型

### 2.1 函数定义

```python
def add(
    dst: Buffer | BufferRegion,
    src0: Buffer | BufferRegion,
    src1: Buffer | BufferRegion | BufferLoad | PrimExpr,
)
```

### 2.2 参数说明

| 参数名 | 输入/输出 | 描述 | 类型 | 必填/可选 |
|--------|----------|------|------|----------|
| dst | 输出 | 存放运算结果 | 张量（tensor） | 必填 |
| src0 | 输入 | 第一个源操作数 | 张量（tensor） | 必填 |
| src1 | 输入 | 第二个源操作数，支持 tensor 或 scalar | 张量（tensor）/ 标量（scalar） | 必填 |

> **类型说明**：
> - **tensor**：通过 `T.alloc_ub`、`T.alloc_shared` 等分配的缓冲区（Buffer），或其切片（BufferRegion）
> - **scalar**：单个元素值，可以是 buffer 元素访问（BufferLoad，仅支持 1D 单索引）或 Python 标量/表达式（PrimExpr）

### 2.3 参数规格

#### 2.3.1 DataType 支持

| 平台 | dst | src0 | src1 |
|------|:---:|:----:|:----:|
| Ascend A2 / A3 | float16, float32, int16, int32 | float16, float32, int16, int32 | 同 dst |

- src1 为 tensor 时，dtype 必须与 dst 一致；src1 为 scalar 时自动转换为 buffer 的 dtype

#### 2.3.2 Shape 支持

- 支持 1D 和 2D
- 支持整行切片（如 `buf[2, :]`）及覆盖完整最后一维的连续多行区域（如 `buf[0:2, :]`）
- 更高维 buffer 需通过切片降维为 1D/2D 的切片传入

### 2.4 约束条件

1. dst 与 src0 的大小必须一致
2. src1 为切片（BufferRegion）时，其大小必须与 dst 一致
3. src1 为 Buffer 时大小不做校验：小于 dst 时产生越界读取，大于 dst 时仅前 dst 大小个元素参与运算（不报错）
4. dst、src0、src1（tensor 形式）的 dtype 必须一致；dtype 不一致会在编译期报错
5. 操作数地址需 32 字节对齐（硬件约束）
6. 仅支持整行/整 buffer 的连续区域；2D 列偏移切片（如 `buf[0, 8:40]`）会产生错误结果或触发 aicore 异常（507015），不支持
7. src1 为 buffer 元素访问时仅支持 1D 单索引（如 `buf[i]`）；多维元素访问只取第一个索引，结果错误（实测行为，两后端一致）

## 3. 示例代码

**示例 1：tensor-tensor 加法**

```python
src0 = T.alloc_ub((256,), "float16")
src1 = T.alloc_ub((256,), "float16")
dst  = T.alloc_ub((256,), "float16")
T.tile.add(dst, src0, src1)
```

**示例 2：tensor-scalar 加法**

```python
src0 = T.alloc_ub((256,), "float16")
dst  = T.alloc_ub((256,), "float16")
T.tile.add(dst, src0, 2.0)
```

**示例 3：tensor-BufferLoad 加法**

```python
src0  = T.alloc_ub((256,), "float16")
scalar_ub = T.alloc_ub((8,), "float16")
dst   = T.alloc_ub((256,), "float16")
T.tile.add(dst, src0, scalar_ub[3])  # dst[i] = src0[i] + scalar_ub[3]
```