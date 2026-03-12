# Tilelang.language.atomic_addx4

## 1. OP概述

简介：`tilelang.language.atomic_addx4` 在 NPU 上执行四宽度操作数的原子加法（atomic add）操作

```markup
T.atomic_addx4(dst, src, size=[])
```

## 2. OP规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| - | - | - |
| `src` | `tensor` | 输入tensor |
| `dst`                             | `tensor` | 输出tensor |
| `size`                            | `list` | 手动指定维度（比如`[32, 32]`），覆盖自动推导的维度，适配非标准形状的数据源 |

### 2.2 支持规格

#### 2.2.1 DataType支持

|        | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool/int1 |
| -------- | ------- | ------ | -------- | ------- | -------- | ------- | -------- | ------- | ------ | ------ | ------ | ----------- |
| Ascend | ×    | ×   | ×     | ×    | ×     | ×    | ×     | ×    | √   | √   | ×   | ×        |

#### 2.2.2 Shape支持

结论：atomic_addx4支持输入tensor与输出tensor shape一致/不一致；

### 2.3 特殊限制说明

无

### 2.4 使用方法

以下示例展示了一个二维张量的原子加法计算：

```
def run_atomic_addx4(M, N, block_M, block_N, dtype="float32"):
    m_num = M // block_M
    n_num = N // block_N
    @T.prim_func
    def atomicAddx4Program(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
            shape_M: T.int32, shape_N: T.int32,
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            blockx = cid // n_num
            blocky = cid % n_num
            A_VEC = T.alloc_ub((1, 4), dtype)

            for i, j in T.Parallel(block_M, block_N // 4):
                bx = blockx * block_M + i
                by = blocky * block_N + j * 4

                T.copy(A[bx, by], A_VEC, [1, 4])
                T.atomic_addx4(B[bx, by], A_VEC, [1, 4])            

    return atomicAddx4Program
​
```

## 3. Tilelang Op到Ascend NPU IR Op的转换

**tilelang::atomic_addx4Op**将被降级转换为hivm::StoreOp
