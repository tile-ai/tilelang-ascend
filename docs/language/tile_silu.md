# T.tile.silu

## 1. 功能说明

对源操作数逐元素执行 SiLU（Sigmoid Linear Unit，也称 Swish）激活运算：`dst[i] = src[i] * sigmoid(src[i])`，即 `dst[i] = src[i] / (1 + e^(-src[i]))`

> silu 为独立实现，直接映射到 `tl.ascend_silu` intrinsic，非 `unary_op` 通用分发器路径。

## 2. 函数原型

### 2.1 函数定义

```python
def silu(
    dst: Buffer | BufferRegion,
    src: Buffer | BufferRegion,
)
```

### 2.2 参数说明

| 参数名 | 输入/输出 | 描述 | 类型 | 必填/可选 |
|--------|----------|------|------|----------|
| dst | 输出 | 存放 SiLU 运算结果 | 张量（tensor） | 必填 |
| src | 输入 | 源操作数 | 张量（tensor） | 必填 |

> **类型说明**：
> - **tensor**：通过 `T.alloc_ub`、`T.alloc_shared` 等分配的缓冲区（Buffer），或其切片（BufferRegion）

### 2.3 参数规格

#### 2.3.1 DataType 支持

| 平台 | dst | src |
|------|:---:|:---:|
| Ascend A2 / A3 | float16, float32 | float16, float32 |

- 整数 dtype（int16/int32 等）会在编译期报错，不支持

#### 2.3.2 Shape 支持

- 支持 1D 和 2D
- 支持整行切片（如 `buf[0:32, :]`）；仅计算切片区域内元素，区域外内容未定义
- 不支持 2D 列偏移切片（如 `buf[:, 8:40]`）：真机实测两端均触发 aicore 异常（507015）

### 2.4 约束条件

1. dst 与 src 的元素总数应相同（无运行时校验，不匹配时产生未定义结果）
2. dst 与 src 的 dtype 必须一致（Ascend C 约束）
3. 操作数地址需 32 字节对齐（硬件约束）
4. **不支持原地运算**（dst 与 src 为同一 buffer）：真机实测 ascendc/pto 结果均错误，须使用独立缓冲区
5. 特殊值遵循 IEEE 语义：`silu(0)=0`、`silu(-inf)=nan`（-inf × 0 未定义）、`silu(inf)=inf`、`silu(nan)=nan`

## 3. 示例代码

**示例 1：1D SiLU**

```python
src = T.alloc_ub((256,), "float16")
dst = T.alloc_ub((256,), "float16")
T.tile.silu(dst, src)
```

**示例 2：2D SiLU**

```python
src = T.alloc_ub((128, 256), "float16")
dst = T.alloc_ub((128, 256), "float16")
T.tile.silu(dst, src)
```