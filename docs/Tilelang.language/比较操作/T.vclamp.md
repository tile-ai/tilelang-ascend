# Tilelang.language.vclamp

## 1. OP概述

简介：`tilelang.language.vclamp`按向量元素，把输入逐元素限制到区间 `[min_val, max_val]`。

```
T.vclamp(src, dst, min_val, max_val)
```

## 2. OP规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| - | - | - |
| `src` | `tensor` | 输入tensor |
| `dst`  | `tensor` | 输出tensor |
| `min_val`  | `scalar` 或 `tensor` | 下界（或逐元素下界） |
| `max_val`  | `scalar` 或 `tensor` | 上界（或逐元素上界） |

### 2.2 支持规格

#### 2.2.1 DataType支持

|   | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool/int1 |
| - | - | - | - | - | - | - | - | - | - | - | - | - |
| Ascend | × | × | × | × | × | × | × | × | √ | √ | × | × |

#### 2.2.2 Shape支持

- `src` 与 `dst` 必须同 shape（逐元素裁剪）
- 当 `min_val` / `max_val` 是 Buffer 时，需与 `src/dst` 可逐元素对应（通常同 shape）
- 当 `min_val` / `max_val` 是标量时，按标量广播到全部元素

### 2.3 特殊限制说明

1. `vclamp` 前端组合逻辑（max + min）。
2. 语义上要求下界不大于上界；若 `min_val > max_val`，结果由 `max/min` 组合逻辑决定，通常不符合业务预期。

### 2.4 使用方法

参考 `testing/npuir/arith_ops/test_clamp_dev.py`：

标量上下界

```python
@tilelang.jit(target="npuir")
def vclamp_scalar_kernel(M, N, dtype="float16"):
    block_size = 1
    clamp_min = 0.0
    clamp_max = 100.0

    @T.prim_func
    def main(
        src: T.Tensor((M, N), dtype),
        dst: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(block_size, is_npu=True) as (cid, _):
            src_ub = T.alloc_shared((M, N), dtype)
            dst_ub = T.alloc_fragment((M, N), dtype)
            T.copy(src, src_ub)
            T.vclamp(src_ub, dst_ub, clamp_min, clamp_max)
            T.copy(dst_ub, dst)

    return main
```

张量上下界（逐元素 min/max）

参考 `testing/npuir/arith_ops/test_clamp_vec_dev.py`：

```python
@tilelang.jit(target="npuir")
def vclamp_tensor_kernel(M, N, dtype="float16"):
    block_size = 1

    @T.prim_func
    def main(
        src: T.Tensor((M, N), dtype),
        dst: T.Tensor((M, N), dtype),
        min_val: T.Tensor((M, N), dtype),
        max_val: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(block_size, is_npu=True) as (cid, _):
            src_ub = T.alloc_shared((M, N), dtype)
            dst_ub = T.alloc_fragment((M, N), dtype)
            min_ub = T.alloc_shared((M, N), dtype)
            max_ub = T.alloc_shared((M, N), dtype)

            T.copy(src, src_ub)
            T.copy(min_val, min_ub)
            T.copy(max_val, max_ub)
            T.vclamp(src_ub, dst_ub, min_ub, max_ub)
            T.copy(dst_ub, dst)

    return main
```

## 3. Tilelang Op到Ascend NPU IR Op的转换

`T.vclamp` 前端展开为两步：

1. `tl.npuir_max(src, min_val, dst)`
2. `tl.npuir_min(dst, max_val, dst)`

在 MLIR codegen 阶段（`src/target/codegen_npuir_api.cc` / `src/target/codegen_npuir_dev.cc`）对应为：

- `tl.npuir_max` -> `mlir::hivm::VMaxOp`
- `tl.npuir_min` -> `mlir::hivm::VMinOp`
