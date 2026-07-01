# T.tile.pow

## 1. 功能说明

对两个 buffer 逐元素做幂运算：`dst[i] = src0[i] ^ src1[i]`

以 src0 为底、src1 为指数的逐元素 power 计算。

## 2. 函数原型

### 2.1 函数定义

```python
def pow(
    dst: Buffer | BufferRegion,
    src0: Buffer | BufferRegion,
    src1: Buffer | BufferRegion,
)
```

### 2.2 参数说明

| 参数名 | 输入/输出 | 描述 | 类型 | 必填/可选 |
|--------|----------|------|------|----------|
| dst | 输出 | 存放幂运算结果 | 张量（tensor） | 必填 |
| src0 | 输入 | 底数 buffer | 张量（tensor） | 必填 |
| src1 | 输入 | 指数 buffer | 张量（tensor） | 必填 |

> **类型说明**：
> - **tensor**：通过 `T.alloc_ub`、`T.alloc_shared` 等分配的缓冲区（Buffer），或其切片（BufferRegion）

### 2.3 参数规格

#### 2.3.1 DataType 支持

| 平台 | dst | src0 | src1 |
|------|:---:|:----:|:----:|
| Ascend A2 / A3 | float16, float32, int8, uint8, int16, uint16, int32, uint32 | float16, float32, int8, uint8, int16, uint16, int32, uint32 | float16, float32, int8, uint8, int16, uint16, int32, uint32 |
| Ascend A5 | float16, float32, bfloat16, int8, uint8, int16, uint16, int32, uint32 | float16, float32, bfloat16, int8, uint8, int16, uint16, int32, uint32 | float16, float32, bfloat16, int8, uint8, int16, uint16, int32, uint32 |

> **平台说明**：A5 的 int8/uint8/int16/uint16/int32/uint32 dtype 仅在 Ascend C 后端下支持；PTO 后端在 A5 上额外支持 bfloat16。A2/A3 的 int8/uint8/int16/uint16/int32/uint32 由 PTO-ISA TPOW 指令支持，Ascend C Power 也支持 int32。

#### 2.3.2 Shape 支持

- 支持 1D 和 2D
- dst、src0、src1 的元素总数必须相同

### 2.4 约束条件

1. dst、src0、src1 的 dtype 必须一致（Ascend C 硬件约束）
2. dst、src0、src1 的 shape 必须兼容（元素总数一致）
3. 不支持源操作数与目的操作数地址重叠
4. 操作数地址需 32 字节对齐（硬件约束）
5. 接口内部使用框架自动申请的临时缓冲区（大小为 `2 × N × sizeof(dtype)` 字节，N 为元素个数），无需用户手动分配
6. 不支持 Scalar 指数参数（src1 必须为 buffer），若需固定指数幂运算，需先分配 buffer 并 fill 为常量值

## 3. 示例代码

**示例 1：逐元素幂运算**

```python
base_ub = T.alloc_ub((128,), "float16")
exp_ub = T.alloc_ub((128,), "float16")
dst_ub = T.alloc_ub((128,), "float16")
T.tile.pow(dst_ub, base_ub, exp_ub)  # dst = base ^ exp
```

**示例 2：平方运算（需先 fill 指数 buffer）**

```python
values = T.alloc_ub((256,), "float32")
squared = T.alloc_ub((256,), "float32")
exp_two = T.alloc_ub((256,), "float32")
T.tile.fill(exp_two, 2.0)  # 指数 buffer 填充为 2.0
T.tile.pow(squared, values, exp_two)  # dst = values ^ 2
```
