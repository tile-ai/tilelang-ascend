# tilelang.language.wait_flag

## 1. 概述

简介： `tilelang.language.wait_flag` 用于在同一个Cube/Vector核内的不同通道之间显式设置通道阻塞信号，
与`tilelang.language.set_flag` 所设置的同步信号相对应

## 2. 规格

### 2.1 参数说明

| 参数名 | 类型 | 描述 | 可选值 |
| - | - | - | - |
| `other` | `str` | 目标通道 | 见下方表格 |
| `event_id`                                 | `int` | 标识编号 | 0-15 |

**other 参数可选值详表：**

| 类别 | 可选值 |
| - | - |
| 运算通道 | `PIPE_S`,`PIPE_V`,`PIPE_M`,`PIPE_V2` |
| 数据通道                                                         | `PIPE_MTE1`,`PIPE_MTE2`,`PIPE_MTE3`,`PIPE_MTE4`,`PIPE_MTE5`,`PIPE_FIX` |
| 虚拟通道                                                         | `VIRTUAL_PIPE_MTE2_L1A`,`VIRTUAL_PIPE_MTE2_L1B` |
| 特殊通道                                                         | `PIPE_ALL`,`PIPE_NUM`,`PIPE_UNASSIGNED` |

### 2.2 特殊限制说明

* 仅支持在 `T.Scope()` 上下文管理器中调用该接口

### 2.3 使用方法

以下示例实现了计算两个张量的和，并输出到第三个张量里，
同时在数据搬运和向量计算运算中通过 `set_flag` 和 `wait_flag` 显式定义了核内同步方式：

```
@tilelang.jit(target='npuir')
def vec_add(M, N, K, block_M, block_N, dtype="float16"):
    m_num = M // block_M
    n_num = N // block_N
    @T.prim_func
    def main(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx_ = cid // n_num
            bx = bx_ * block_M
            by_ = cid % n_num
            by = by_ * block_N

            A_VEC = T.alloc_ub((block_M, block_N), dtype)
            B_VEC = T.alloc_ub((block_M, block_N), dtype)
            C_VEC = T.alloc_ub((block_M, block_N), dtype)
            with T.rs("PIPE_MTE2"):
                T.copy(A[bx, by], A_VEC)
                T.copy(B[bx, by], B_VEC)
                T.set_flag("PIPE_V", 0)

            with T.rs("PIPE_V"):
                T.wait_flag("PIPE_MTE2", 0)
                T.npuir_add(A_VEC, B_VEC, C_VEC)
                T.set_flag("PIPE_MTE3", 0)

            with T.rs("PIPE_MTE3"):
                T.wait_flag("PIPE_V", 0)
                T.copy(C_VEC, C[bx, by])

    return main

```

### 3. Tilelang Op到Ascend NPU IR Op的转换

tilelang::wait_flagOp将被下降为hivm::WaitFlagOp