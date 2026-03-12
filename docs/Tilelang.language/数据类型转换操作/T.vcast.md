# Tilelang.language.cast

## 1. OP概述

简介：`tilelang.language.cast`该算子返回输入向量中的元素​**逐元素的数据类型转换**​，并可指定不同的舍入（rounding）模式

```
T.vcast(src, dst, round_mode) [Developer Op]
T.vcast(src, dst, round_mode) [Expert Op]
```

## 2. OP规格

### 2.1 参数说明

| 参数名    | 类型         | 说明       |
| ----------- | -------------- | ------------ |
| `src` | `tensor` | 输入tensor |
| `dst` | `tensor` | 输出tensor |
| ```round_mode``` | ```string``` | 舍入模式：{"round", "rint", "floor", "ceil", "trunc", "odd"}|

### 2.2 支持规格

#### 2.2.1 DataType支持

```
| src  | dst  | roundingmode                                      |
|------|------|---------------------------------------------------|
| f32  | f32  | round, rint, floor, ceil, trunc                   |
| f32  | f16  | round, rint, floor, ceil, trunc, odd              |
| f32  | i64  | round, rint, floor, ceil, trunc                   |
| f32  | i32  | round, rint, floor, ceil, trunc                   |
| f32  | i16  | round, rint, floor, ceil, trunc                   |
| f32  | s64  | round, rint, floor, ceil, trunc                   |
| f32  | bf16 | round, rint, floor, ceil, trunc                   |
| f16  | f32  | rint                                              |
| f16  | i32  | round, rint, floor, ceil, trunc                   |
| f16  | i16  | round, rint, floor, ceil, trunc                   |
| f16  | i8   | round, rint, floor, ceil, trunc                   |
| f16  | ui8  | round, rint, floor, ceil, trunc                   |
| f16  | i4   | round, rint, floor, ceil, trunc                   |
| bf16 | f32  | rint                                              |
| bf16 | i32  | round, rint, floor, ceil, trunc                   |
| ui8  | f16  | rint                                              |
| i8   | f16  | rint                                              |
| i8   | i1   | rint                                              |
| i16  | f16  | round, rint, floor, ceil, trunc                   |
| i16  | f32  | rint                                              |
| i32  | f32  | round, rint, floor, ceil, trunc                   |
| i32  | i64  | rint                                              |
| i32  | i16  | rint                                              |
| i64  | i32  | rint                                              |
| i64  | f32  | round, rint, floor, ceil, trunc                   |
| i4   | f16  | rint                                              |
| i1   | f16  | rint                                              |
| i1   | f32  | rint                                              |
```

#### 2.2.2 Shape支持

结论：输入（input）与输出（output）shape要一致。

### 2.3 特殊限制说明

无

### 2.4 使用方法

示例1：实现了Expert Mode中将一个二维张量（Tensor） 从fp32转换成fp16

```
def vec_cast(M, N, block_M, block_N, round_mode="round"):
    m_num = M // block_M
    n_num = N // block_N
    src_dtype = "float32"
    dst_dtype = "float16"
    BLOCK_SIZE = 20
    @T.prim_func
    def main(
        SRC: T.Tensor((M, N), src_dtype),
        DST: T.Tensor((M, N), dst_dtype),
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            SRC_UB = T.alloc_ub((block_M, block_N), src_dtype)
            DST_UB = T.alloc_ub((block_M, block_N), dst_dtype)
            for i in T.serial(T.ceildiv(m_num * n_num, BLOCK_SIZE)):
                block_id = i * BLOCK_SIZE + cid
                if block_id < m_num * n_num:
                    block_id_m = block_id // n_num
                    block_id_n = block_id % n_num
                    bx = block_id_m * block_M
                    by = block_id_n * block_N
                    T.copy(SRC[bx, by], SRC_UB)
                    # cast with rounding mode
                    T.vcast(
                        SRC_UB,
                        DST_UB,
                        round_mode=round_mode,
                    )
                    T.copy(DST_UB, DST[bx, by])
    return main
```

示例2：实现了Developer Mode中将一个二维张量（Tensor） 从fp32转换成fp16

```
def vec_cast(M, N, block_M, block_N, round_mode="round"):
    m_num = M // block_M
    n_num = N // block_N
    src_dtype = "float32"
    dst_dtype = "int32"
    BLOCK_SIZE = 20
    @T.prim_func
    def main(
        SRC: T.Tensor((M, N), src_dtype),
        DST: T.Tensor((M, N), dst_dtype),
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            SRC_UB = T.alloc_shared((block_M, block_N), src_dtype)
            DST_UB = T.alloc_shared((block_M, block_N), dst_dtype)
            for i in T.serial(T.ceildiv(m_num * n_num, BLOCK_SIZE)):
                block_id = i * BLOCK_SIZE + cid
                if block_id < m_num * n_num:
                    block_id_m = block_id // n_num
                    block_id_n = block_id % n_num
                    bx = block_id_m * block_M
                    by = block_id_n * block_N
                    # GM -> UB
                    T.copy(SRC[bx, by], SRC_UB)
                    # cast with rounding mode
                    T.vcast(
                        SRC_UB,
                        DST_UB,
                        round_mode=round_mode,
                    )
                    # UB -> GM
                    T.copy(DST_UB, DST[bx, by])
    return main
​
```

## 3. Tilelang Op到Ascend NPU IR Op的转换

**tilelang::vcastOp**将被下降为hivm::VCastOp
