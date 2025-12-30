# T.Pipelined on TileLang-Ascend

## Overview

`T.pipelined` is a high-level abstraction in TileLang-Ascend designed to express and optimize pipelined parallelism on Ascend AI accelerators. It enables fine-grained overlapping of computation, memory access within a single core (intra-core), and synchronization across multiple execution across multiple cores (inter-core).

## Usage
### Interface
```python
for var in T.Pipelined(range: int, num_stages: int):
```
Assuming some operations need to be executed repeatedly for a loop of iterations, this statement enables the pipeline for the operations within the loop. The degree of overlap can be controlled by setting different values for `num_stages`, which is a positive integer less than `range - 1`.
### Intra-core case
```python
for k in T.Pipelined(loop_k, num_stages=2):
    T.copy(A[bx * block_M, k * block_K], A_L1)
    T.copy(B[k * block_K, by * block_N], B_L1)

    T.barrier_all()
    if k == 0:
        T.gemm_v0(A_L1, B_L1, C_L0, init=True)
    else:
        T.gemm_v0(A_L1, B_L1, C_L0)

    T.barrier_all()
```
In case above, there are two memory accesses and one computation in one loop: copy_A, copy_B and gemm. Assume `loop_k = 4`, operations are: 
- loop 0 : copy_A_0  -->  copy_B_0  -->  gemm_0
- loop 1 : copy_A_1  -->  copy_B_1  -->  gemm_1
- loop 2 : copy_A_2  -->  copy_B_2  -->  gemm_2
- loop 3 : copy_A_3  -->  copy_B_3  -->  gemm_3

In one loop, gemm depends on copy_A and copy_B. But there are no data dependencies between loop iterations. Pipelining following prefetch-main body-epilogue enables overlapping of computation and memory access, which can significantly improve performance for memory-intensive operators. 

When `num_stages=2`, the execution sequence of tasks is as follows:

| Time | Copy A       | Copy B       | Compute    |
|------|--------------|--------------|------------|
| t₀   | **copy_A_0** | **copy_B_0** |            |
| t₁   | **copy_A_1** | **copy_B_1** |            |
| t₂   | **copy_A_2** | **copy_B_2** | **gemm_0** |
| t₃   | **copy_A_3** | **copy_B_3** | **gemm_1** |
| t₄   |              |              | **gemm_2** |
| t₅   |              |              | **gemm_3** |

In this case, `num_stages=2`, which means prefetch 2 memory access \
prefetch : `copy_A_0 copy_A_1` and `copy_B_0 copy_B_1`\
main body :  `copy_A_2 copy_B_2 gemm_0` and `copy_A_3 copy_B_3 gemm_1` \
epilogue :  `gemm_2` and `gemm_3`

Computation and memory access are mutually overlapped in main body.
### Inter-core case
```python
for k in T.Pipelined(T.ceildiv(seq_len, block_N), num_stages=2):
    T.copy(K[bz, by, k * block_N:(k + 1) * block_N, :], k_l1)
    T.gemm_v0(q_l1, k_l1, acc_s_l0c, transpose_B=True, init=True)
    T.copy(acc_s_l0c, workspace_1[cid, :, :])

    T.tile.fill(acc_s_ub, 0.0)
    T.copy(m_i, m_i_prev)
    T.copy(
        workspace_1[cid, vid * block_M // 2:vid * block_M // 2 + block_M // 2, :],
        acc_s_ub_)
    T.tile.add(acc_s_ub, acc_s_ub, acc_s_ub_)
    T.tile.mul(acc_s_ub, acc_s_ub, sm_scale)
    ...
```
In the above case, operations `copy_K`, `gemm`, and `copy_l0c_to_wk1` executed on the Cube core are collectively referred to as `write_wk1`. The following operations executed on the Vevtor core contains `copy_wk1_to_ub`can be refferred to as `read_wk1`.
Assume `T.ceildiv(seq_len, block_N)=4`, operations are:

- loop 0 : write_wk1_0  -->  read_wk1_0
- loop 1 : write_wk1_1  -->  read_wk1_1
- loop 2 : write_wk1_2  -->  read_wk1_2
- loop 3 : write_wk1_3  -->  read_wk1_3

In this case, `num_stages=2`, which means Two `write_wk1` tasks are issued at once, the execution sequence of tasks is as follows:
| Time | Write Workspace | Read Workspace  |
|------|-----------------|-----------------|
| t₀   | **write_wk1_0** |                 |
| t₁   | **write_wk1_1** | **read_wk1_0**  |
| t₂   | **write_wk1_2** | **read_wk1_1**  |
| t₃   | **write_wk1_3** | **read_wk1_2**  |
| t₄   |                 | **read_wk1_3**  |

Operations on Cube and Vector are mutually overlapped in t₁~t₃.

## Constraints
- Inter-core pipeline and intra-core pipeline cannot be enabled simultaneously.
- Multiple uses of inter-core pipeline are not supported within a single program.
- When using inter-core pipeline, automatic CV separation and automatic synchronization insertion between CV must be enabled : `"tl.ascend_auto_cv_combine": True,
"tl.ascend_auto_cross_core_sync": True`