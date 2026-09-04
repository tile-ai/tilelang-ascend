# T.tile.clamp_min

## 1. 功能说明

将 buffer 中的元素钳位到下界：`dst[i] = max(src[i], min_val)`

低于下界的值替换为下界值，大于或等于下界的值保持不变。

> **注意区分**：`T.max`（标量级运算，用于 `T.Parallel` 循环内的逐元素表达式）与 `T.tile.clamp_min`（buffer 级 intrinsic，操作整个 buffer 区域）名称相似但层级不同。

## 2. 函数原型

### 2.1 函数定义

```python
def clamp_min(
    out: Buffer | BufferRegion,
    buffer: Buffer | BufferRegion,
    min_val: PrimExpr,
    count: PrimExpr,
)
```

### 2.2 参数说明

| 参数名 | 输入/输出 | 描述 | 类型 | 必填/可选 |
|--------|----------|------|------|----------|
| out | 输出 | 存放钳位运算结果 | 张量（tensor） | 必填 |
| buffer | 输入 | 源操作数 | 张量（tensor） | 必填 |
| min_val | 输入 | 下界标量值 | 标量（scalar） | 必填 |
| count | 输入 | 参与计算的元素个数 | 整数（integer） | 必填 |

> **类型说明**：
> - **tensor**：通过 `T.alloc_ub`、`T.alloc_shared` 等分配的缓冲区（Buffer），或其切片（BufferRegion）
> - **scalar**：单个元素值，可以是 Python 标量或表达式（PrimExpr）
> - **integer**：正整数表达式（PrimExpr），指定参与计算的元素个数

### 2.3 参数规格

#### 2.3.1 DataType 支持

| 平台 | out | buffer | min_val |
|------|:---:|:----:|:----:|
| Ascend A2 / A3 | float16, float32, int16, int32 | float16, float32, int16, int32 | float16, float32, int16, int32 |
| Ascend A5 | float16, float32, bfloat16, int8, uint8, int16, uint16, int32, uint32, int64, uint64 | float16, float32, bfloat16, int8, uint8, int16, uint16, int32, uint32, int64, uint64 | float16, float32, bfloat16, int8, uint8, int16, uint16, int32, uint32, int64, uint64 |

> **平台说明**：A5 的 int8/uint8/int16/uint16/int32/uint32/bfloat16 dtype 仅在 Ascend C 后端下支持；PTO 后端仅支持 float16 和 float32。A2/A3 的 int16/int32 由 PTO 后端的 TMINS 指令支持。

> **后端差异说明**：A5 平台上 int64/uint64 仅 Ascend C 后端支持。

#### 2.3.2 Shape 支持

- 支持 1D 和 2D
- out 与 buffer 的元素总数必须相同，count 不超过 out 和 buffer 的元素总数

### 2.4 约束条件

1. count 为必填参数，需显式指定参与计算的元素个数（不同于多数 tile API 自动推断 size）
2. out 与 buffer 的 dtype 必须一致
3. out 与 buffer 的 shape 必须兼容（count 不超过 out 和 buffer 的元素总数）
4. min_val 为标量值，不支持 Tensor 级边界
5. 操作数地址需 32 字节对齐（硬件约束）
6. Ascend C 后端不支持源操作数与目的操作数地址重叠
7. Ascend C 的 ClampMin 为高阶 API，仅 A5（950PR/950DT）支持；A2/A3 使用 PTO 后端的 TMINS 指令实现
8. 接口内部使用框架自动申请的临时缓冲区（大小与 src 相同），无需用户手动分配

## 3. 示例代码

**示例 1：ReLU 激活（下界为 0）**

```python
src = T.alloc_ub((1024,), "float16")
dst = T.alloc_ub((1024,), "float16")
T.tile.clamp_min(dst, src, 0.0, 1024)  # 将所有值限制在不低于 0.0，即 ReLU
```

**示例 2：数值稳定性（防止除零）**

```python
denom = T.alloc_ub((512,), "float32")
safe_denom = T.alloc_ub((512,), "float32")
T.tile.clamp_min(safe_denom, denom, 1e-6, 512)  # 防止除零，将分母下限设为 1e-6
```
