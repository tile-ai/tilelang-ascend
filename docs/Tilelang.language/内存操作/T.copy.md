# Tilelang.language.copy

## 1. OP概述

简介：`tilelang.language.copy` 该算子用于不同内存区域之间（ub-ub， gm-ub）执行数据复制操作。

```
T.copy(src[0:size], dst[0:size] ) [Developer Op]
T.copy(src[0:size], dst[0:size] ) [Expert Op]
或
T.copy(src, dst, size) [Developer Op]
T.copy(src, dst, size) [Expert Op]
```

## 2. OP规格

### 2.1 参数说明

| 参数名    | 类型         | 说明       |
| ----------- | -------------- | ------------ |
| `src` | `tensor` | 输入tensor |
| `dst` | `tensor` | 输出tensor |
| `size` | `list(int)` | 拷贝数据的size |

### 2.2 支持规格

#### 2.2.1 DataType支持

|        | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| -------- | ------- | ------ | -------- | ------- | -------- | ------- | -------- | ------- | ------ | ------ | ------ | ----------- |
| Ascend | √    | √   | √     | √    | √     | √    | √     | √    | √   |√   | √   | ×        |

#### 2.2.2 Shape支持

结论：输入（input）与输出（output）的shape要一致。

### 2.3 特殊限制说明

无

### 2.4 使用方法

示例1：实现了Expert Mode中将一个二维张量（Tensor） 的copy到A_ub中 (gm -> ub)

```
def atomic_add_2d(M, N, block_M, block_N, dtype="float32"):
    m_blocks = M // block_M
    n_blocks = N // block_N
    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_blocks * n_blocks, is_npu=True) as (cid, _):
            bx = (cid // n_blocks) * block_M
            by = (cid % n_blocks) * block_N
            A_ub = T.alloc_ub((block_M, block_N), dtype)
            tile_M = T.min(block_M, M - bx)
            tile_N = T.min(block_N, N - by)
            T.copy(
                A[bx : bx + tile_M, by : by + tile_N],
                A_ub[0:tile_M, 0:tile_N],
            )
            T.npuir_atomic_add(B[bx, by], A_ub, [tile_M, tile_N])
    return main
​
```

示例2：实现了Developer Mode中将一个二维张量（Tensor） 的copy到A_shared中 (gm -> ub)

```
def atomic_add_2d_dev(M, N, block_M, block_N, dtype="float32"):
    m_blocks = M // block_M
    n_blocks = N // block_N
    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_blocks * n_blocks, is_npu=True) as (cid, _):
            bx = (cid // n_blocks) * block_M
            by = (cid % n_blocks) * block_N
            A_shared = T.alloc_shared((block_M, block_N), dtype)
            tile_M = T.min(block_M, M - bx)
            tile_N = T.min(block_N, N - by)
            T.copy(
                A[bx : bx + tile_M, by : by + tile_N],
                A_shared[0:tile_M, 0:tile_N],
            )
            T.npuir_atomic_add(B[bx, by], A_shared, [tile_M, tile_N])
    return main
​
```

## 3. Tilelang Op到Ascend NPU IR Op的转换

**1. 在expert 模式下：**

当src和dst的shape一致时：**tilelang::copyOp**将被下降为memref::CopyOp

否则：**tilelang::copyOp**将被下降为memref::ExtractStridedMetadataOp、memref::DimOp（动态shape）、arith::ConstantIndexOp（非动态shape）、memref::ReinterpretCastOp、memref::CopyOp

**2. 在developer 模式下：**

**GM -> UB:  ​tilelang::copyOp**将被下降为 memref::SubViewOp、memref::AllocOp、bufferization::ToTensorOp、（tensor::DimOp、tensor::EmptyOp、hivm::VCastOp）[for type cast]、tensor::InsertSliceOp

**UB​​​ -> UB​:  tilelang::copyOp**将被下降为 tensor::ExtractSliceOp、（tensor::DimOp、tensor::EmptyOp、hivm::VCastOp）[for type cast]、tensor::InsertSliceOp

**UB -> GM:  tilelang::copyOp**将被下降为 tensor::ExtractSliceOp、（tensor::DimOp、tensor::EmptyOp、hivm::VCastOp）[for type cast]、bufferization::MaterializeInDestinationOp
