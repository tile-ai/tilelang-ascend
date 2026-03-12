# Tilelang.language.load_nd2nz

## 1. OP概述

简介：`tilelang.language.load_nd2nz`用于在`Expert`模式中将`GM`中的数据搬运到`L1`中

```
T.load_nd2nz(src, dst, size)
T.npuir_load_nd2nz(src, dst, size)
```

## 2. 规格

### 2.1 参数说明

| 参数名   | 类型       | 描述   |
|-------|----------|------|
| `src` | `tensor` | 源张量（必须来自GM地址空间）  |
| `dst` | `tensor`,`scalar` | 目的张量（必须来自L1地址空间）|
| `size` | `List[int]` | 搬运的张量大小（可缺省）  |

### 2.2 OP 规格

#### 2.2.1 DataType 支持

|              | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
|:-------------|:----:|:-----:|:-----:|:-----:|:------:|:------:|:------:|:-----:|:----:|:----:|:----:|:----:|:----:|
| Ascend A2/A3 |  ×   |   ×   |   ×   |   ×   |   ×    |   ×    |   ×    |   ×   |  √   |  ×   |  ×   |  ×   |  ×

#### 2.2.2 Shape 支持
1.src支持2-4Dtensor
2.dst仅支持2Dtensor
### 2.3 特殊限制说明
1.src和dst应具有相同的数据类型
2.load_nd2nz适用Expert模式，在Developer模式中使用copy接口
### 2.4 使用方法
以下示例实现了在矩阵乘法之前，使用load_nd2nz将数据从GM搬运到L1中
```python
@tilelang.jit(target="npuir")
def matmul(block_M, block_N, K_L1, dtype="float16", accum_dtype="float32"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype)
    ):
        with T.Kernel(m_num*n_num, is_npu=True) as (cid, _):
            with T.Scope("Cube"):
                bx = cid // n_num * block_M
                by = cid % n_num * block_N
                A_BUF = T.alloc_L1([block_M, K_L1], dtype)
                B_BUF = T.alloc_L1([K_L1, block_N], dtype)
                C_BUF = T.alloc_L0C([block_M, block_N], accum_dtype)

                for i in T.serial(T.ceildiv(K, K_L1)):
                    T.load_nd2nz(A[bx, i * K_L1], A_BUF, [block_M, K_L1])
                    T.load_nd2nz(B[i * K_L1, by], B_BUF, [K_L1, block_N])

                    if i == 0:
                        T.gemm(A_BUF, B_BUF, C_BUF, initC=True, b_transpose=False,
                            size=[block_M, K_L1, block_N])
                    else:
                        T.gemm(A_BUF, B_BUF, C_BUF, initC=False, b_transpose=False,
                            size=[block_M, K_L1, block_N])

                    T.store_fixpipe(C_BUF, C[bx, by],
                        size=[block_M, block_N], enable_nz2nd=True)

    return main
```



## 3. Tilelang Op到Ascend NPU IR Op的转换

**tilelang::load_nd2nzOp**将被转换为`hivm::ND2NZOp`