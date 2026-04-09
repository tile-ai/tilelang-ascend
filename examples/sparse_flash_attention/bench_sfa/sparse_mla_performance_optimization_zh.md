

# 用TileLang-Ascend编写高性能SparseMLA算子

[English](sparse_mla_performance_optimization.md) | **中文**

## 1. 背景

SparseMLA 是 DeepSeek v3.2 中引入的核心注意力机制，本文讨论Paged Attention版本SparseMLA算子在TileLang-Ascend的实现与优化。


## 2. 测试输入

输入定义在 bench_sfa.py 中，公共参数如下：

- T=1
- B=1
- Q_N=128
- KV_N=1
- D=512
- D_rope=64
- sparse_size=2048
- block_size=128
- act_kv_s=2560

三组 shape 只在 KV_S 上不同：

| 编号 | KV_S | 说明 |
| --- | ---: | --- |
| shape0 | 2560 | 短序列 |\
| shape1 | 6400 | 中序列 |
| shape2 | 48000 | 长序列 |

act_kv_s 固定为 2560，sparse_size 固定为 2048，实际参与计算的有效稀疏窗口规模基本不变。

## 3. 性能结果


|  | shape0 | shape1 | shape2 | 平均耗时(us) | AscendC / TileLang |
| --- | ---: | ---: | ---: | ---: | ---: |
| AscendC  | 98.000 | 99.000 | 100.000 | 99.000 | 1.00x |
| TileLang | 109.760 | 109.100 | 108.980 | 109.280 | 0.91x |

*注： AscendC 参考实现来源：* https://gitcode.com/cann/cann-recipes-infer/blob/master/ops/ascendc/torch_ops_extension/custom_ops/converter/npu_sparse_flash_attention.py




## Fixed Core：launch kernel 固定物理计算核

在 Cube-Vector 融合算子中，AIC (Cube) 与 AIV (Vector) 是相互独立的执行单元，二者之间的数据交互必须通过 Global Memory 上的 Workspace 变量中转。因此，Workspace 的分配方式直接影响显存占用和访存性能。针对这一问题，TileLang 提供了两种 launch kernel 的方式：

**默认模式**——按逻辑任务数 launch：
```python
@T.prim_func
def tl_kernel(
    ...
    # Workspace 按逻辑任务数分配，block_num 可能远大于物理核数
    workspace: T.Tensor([block_num, block_M, block_N], dtype),
):
    # ↓ alloc_buffer / annotate_address 随 block_num 展开，每个逻辑任务各执行一次
    buf = T.alloc_L1([block_M, block_N], dtype)
    T.annotate_layout({buf: ...})

    with T.Kernel(block_num, is_npu=True) as (cid, vid):
        # 每个逻辑任务拥有独立的 workspace 切片
        T.copy(result, workspace[cid, :, :])
```

在此模式下，编译器为每个逻辑任务独立执行 `alloc_buffer`、`annotate_address` 等初始化操作，并分配一份独立的 Workspace。当逻辑任务数远大于物理核数时，会产生三个性能问题：**(1)** 初始化操作随任务数冗余执行；**(2)** Workspace 总量随任务数线性膨胀，显存开销巨大；**(3)** 膨胀后的 Workspace 超出 L2 Cache 容量，Cube-Vector 间频繁交互的中间张量被置换到 HBM，遭遇严重的访存瓶颈。

**Fixed Core 模式**——按物理核数 launch：
```python
@T.prim_func
def tl_kernel(
    ...
    # Workspace 按物理核数分配，固定 24 份，尽可能常驻 L2
    workspace: T.Tensor([core_num, block_M, block_N], dtype),
):
    # ↓ alloc_buffer / annotate_address 仅在 24 个物理核上各执行一次
    buf = T.alloc_L1([block_M, block_N], dtype)
    T.annotate_layout({buf: ...})

    with T.Kernel(core_num, is_npu=True) as (cid, vid):
        # 手动将逻辑任务均摊到物理核，每个核复用同一份 workspace
        single_core_load = T.ceildiv(block_num, core_num)
        for block_idx in T.serial(cid * single_core_load, (cid + 1) * single_core_load):
            ...
            T.copy(result, workspace[cid, :, :])  # workspace[cid] 被复用
```

通过 `T.Kernel(core_num, is_npu=True)` 将计算任务约束在固定数量的物理核心上，解决上述问题：

- **减少冗余初始化**：`alloc_buffer`、`annotate_address` 等操作仅在 24 个物理核上各执行一次，不再随逻辑任务数重复执行，显著降低 kernel launch 开销。
- **CV 中转显存优化**：Cube-Vector 数据交互所需的 Workspace 被严格限制为 `core_num` 份（24 份），而非 `block_num` 份，从根本上遏制了显存膨胀。
- **中间数据常驻 L2**：Workspace 分配使中间张量尽可能常驻于片上 L2 Cache，Cube 核写入的结果可被 Vector 核直接从 L2 读取，避免了向 HBM 置换带来的延迟。


## Tiling：切分策略
Tiling 不仅决定核间负载均衡，还直接影响整体计算效率与任务调度方式。这里主要涉及两个关键维度的切分设计：


### 核间切分

核间切分决定了如何将全局计算任务分配到 24 个（A2、A3）物理 AI Core 上并行执行。SparseMLA 的核间切分沿 **batch × seq_len_q × g_block_num × kv_heads** 四个维度展开，其中 `g_block_num` 是对 head group 维度的进一步分块。

以测试输入参数为例（`q_heads=128, kv_heads=1, m_base_size=16`）：

- **Head group 维度分块**：GQA group size `g = q_heads / kv_heads = 128`。由于 `g > m_base_size`，将 Q head 维度按 `m_base_size=16` 切分，得到 `g_block_num = g / m_base_size = 8` 个子块。每个计算核处理 16 个连续的 Q head 子块。



### 核内切分

每个物理核在执行单个逻辑任务时，沿 **seq_len_kv（topk）** 维度进一步分块迭代，
将 `topk=2048` 按 `n_base_size=256` 切分，得到 `n_block_num = ⌈topk / n_base_size⌉ = 8` 次迭代。每次迭代执行一轮完整的 "Gather KV → MM1(QK^T) → Softmax → MM2(PV) → 累加输出" 流程。

提升 `seq_len_kv` 维度分块大小，增加每轮 Gather kv 和 Copy out 的数据规模，有利于提升访存吞吐；同时，增大基本块大小，也能进一步提升cube侧计算效率。



## Sparse KV: 稀疏访存优化
在稀疏场景下，KV 数据在 Global Memory 中呈离散分布，但后续的 Cube 矩阵运算需要连续的内存输入，这里采取了以下机制设计并优化：

- **双vector核访存指令发射**：基于A2/A3芯片Cube核和Vector核数量1：2的特性，利用vector核发射访存指令，将指令下发效率提升一倍。
- **离散 KV Gather 与连续搬出**：利用统一的稀疏索引先将离散的 KV 从 GM 收集到 UB，在这里gather为连续块之后，再一次性搬出到 Workspace 给后续cube核使用。
- **异步拷贝 (核内手动同步)**：手动设置 `set_flag` / `wait_flag` 进行精确控制，使得离散拷入异步执行，大幅度缩减gather kv整体时间。
- **搬入搬出 Ping-Pong (Double Buffer)**：对离散kv gather操作使用双buffer机制，当前数据通过 MTE3 写出到 Workspace 的同时，针对后一块数据的 MTE2 读取指令已经异步发起，使得MTE3与MTE2进行流水掩盖。

基于上述多项优化机制的协同配合，本次稀疏KV访存优化的整体设计目标与实现思路，如下图所示：
![稀疏访存优化](figures/v0_sparse_kv.png)

本样例最终实际的优化前后流水如下图所示，通过手动设置核内同步，MTE2搬入指令可实现连续下发，大幅提高了MTE2并行度；同时搬入搬出Ping-Pong实现了MTE3和MTE2流水掩盖，进一步减少kv gather耗时。

**优化前流水：**
![稀疏访存优化前](figures/v0_sparse_kv_before.png)

**优化后流水：**
![稀疏访存优化后](figures/v0_sparse_kv_after.png)



## Split-K pipelined GEMM v0 implementation: MTE1 与 Cube 流水重叠

在标准的矩阵乘执行过程中，数据需先由 MTE1 从 L1 搬运到 L0A/L0B，随后才交给 Cube 计算单元执行 MMA（矩阵乘加）指令。若按串行逻辑执行，Cube 在等待 MTE1 搬运下一批数据时将完全闲置。`gemm_v0` 模板通过 **K 轴切块 + Ping-Pong 双缓冲 + 细粒度同步** 实现 MTE1 与 Cube 的流水重叠，具体切分如下：

### Cube 核内切分


- **MM1 (Q×K^T) 的 K 维切分**：矩阵乘的 K 维度为 `dim + rope_dim = 512 + 64 = 576`，按 `k_l0_size=64` 切分为 `⌈576/64⌉ = 9` 个子块。L0A/L0B 各申请双份缓冲（`q_l0a[2, 16, 64]`、`kv_l0b[2, 64, 256]`），通过 Ping-Pong 实现 MTE1 搬运与 Cube 计算的流水重叠。

- **MM2 (P×V) 的 K 维切分**：矩阵乘的 K 维度为 `n_base_size=256`，按 `mm2_k_l0_size=32` 切分为 8 个子块。同样采用双缓冲进行流水化。

基于上述切分，双缓冲通过 `kk % 2` 交替写入 L0 缓冲区，并以 `set_flag` / `wait_flag` 进行精确同步：当 Cube 正在对第 $i$ 份 L0 数据执行 `mma` 运算时，MTE1 同时将第 $i+1$ 份数据从 L1 搬入另一半缓冲，实现流水重叠：

```
时间轴 →
MTE1: [搬入 k0] [搬入 k1] [搬入 k2] [搬入 k3] ...
Cube:           [计算 k0] [计算 k1] [计算 k2] ...
                 ↑ MTE1 与 Cube 交替使用双缓冲，流水重叠
```

优化后流水：
![alt text](figures/pipelined_gemm.png)

通过 `gemm_v0` 模板，实现了 Cube 核内部 MTE1 访存与 Cube 计算的重叠，成功掩盖了大部分 MTE1 访存延迟。


## CV pipeline: Cube 核与 Vector 核操作 Overlap

A2和A3机器采用Cube核/Vector核分离架构，可以实现 CV 操作overlap，通过`T.Pipelined`原语在 `seq_len_kv` 循环上引入双stage，编译器将循环内的 CV 串行任务编排为 CV 流水线执行：
- 下一次迭代的 **Vector MTE 访存**（如 V0 阶段的 kv gather 与写出）被与上一次迭代的**Cube核计算**（C1阶段 QK 矩阵乘）掩盖、下一次迭代的**vector计算**（V1 阶段 Online Softmax）与上一次迭代的**Cube核计算**（C2阶段 PV 矩阵乘）掩盖等。通过不同硬件单元相互掩盖，大幅度减少算子耗时。

基于以上描述，CV pipeline采用`T.Pipelined`原语实现自动化“计算-搬运”重叠以及夸核流水并行执行，整体设计思路和优化目标如下：
![CV pipeline 优化](figures/pipelined_optimize.png)

本样例最终实际的优化前后流水如下图所示，通过`T.Pipelined`原语提供的声明式的流水线编程抽象，最大化硬件利用率，显著提升吞吐量。

**优化前流水：**
![CV pipeline Before](figures/pipelined_optimize_before.png)

**优化后流水：**
![CV pipeline After](figures/pipelined_optimize_after.png)

## Vectorization: Broadcast 与 AXPY
在稀疏注意力的实现中，Vector核围绕 Softmax 的动态更新以及输出结果的逐步累加展开。早期实现采用标量获取索引+逐行计算的方式，这种方式引入了额外的scaler计算，同时向量化程度低，无法充分利用vector核的计算能力。


- **Broadcast 向量化**：引入 `m_i_broadcast`、`m_i_prev_broadcast`、`sumexp_broadcast` 等中间张量，通过 broadcast 操作将按行更新的状态扩展为与输入张量对齐的形状，从而避免逐行操作，提升向量指令的执行效率。

- **AXPY指令**：在输出张量（`acc_o`）的更新过程中，将“历史结果缩放 + 当前结果累加”这一具有迭代依赖的步骤，重构为连续的向量 `mul` 与 `add` 运算流水，并以 AXPY（a·X + Y）指令的形式完成，减少指令下发与中间访存耗时。


## 6. 总结

综合最终评测结果，在该场景下，TileLang 编译生成的算子在相同输入条件下可达到 AscendC 参考实现约 0.90× 的性能水平，本文所总结的优化路径可为后续基于 TileLang 开发高性能算子提供参考。
