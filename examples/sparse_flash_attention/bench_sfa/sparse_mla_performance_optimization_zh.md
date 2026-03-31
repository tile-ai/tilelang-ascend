

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




## Fixed Core：固定物理计算核与 L2 常驻缓存

在 SparseMLA 算子中，负责矩阵运算的 AIC (Cube) 和负责向量处理的 AIV (Vector) 是相互独立的执行单元，它们在计算迭代中需要频繁交互大量的局部状态张量（如 `S`、`P` 及 `O`）。在默认机制下，如果随着庞大的逻辑 `block_num` 动态分配用于 Cube-Vector 数据交互的 Workspace，会造成内存占用急剧膨胀。一旦分配空间超出片上缓存容量，这些频繁访问的中间数据就会被置换到远端的 HBM 中，从而遭遇严重的访存带宽瓶颈。

为了解决这一问题，通过显式引入 `with T.Kernel(core_num, is_npu=True) as (cid, vid)` 语法，将计算任务显式约束在Ascend物理核心：
- **计算任务固定映射到物理核心**：由于 A2、A3 机器在物理上有 24 个 AI Core，不论输入产生多少个逻辑计算块，都通过静态负载均衡的方式将其分配给固定的 24 个物理核执行，避免alloc buffer、annotate_address等操作冗余执行。
- **Workspace 显存优化与常驻 L2**：物理核绑定使得用于跨核心数据交互的 Workspace 总容量被严格限制为 24 份。这种固定且紧凑的内存分配，使得中间数据能够完全常驻于高速的片上 L2 Cache 中。AIC 计算出的结果写入 L2 后，AIV 可以直接从中读取消费，避免了高频中间张量向 HBM 置换带来的开销。


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

## Tiling：切分策略
Tiling 不仅决定核间负载均衡，还直接影响整体计算效率与任务调度方式。这里主要涉及两个关键维度的切分设计：

- **head_num核间切分**：将 `head_num` 维度进行细粒度切分，生成更多可并行执行的任务，从而充分利用物理 AI Core，提升整体并行度和计算吞吐。

- **seq_len_kv核内切分**：提升 `seq_len_kv` 维度分块大小，增加每轮 Gather kv 和 Copy out 的数据规模，有利于提升访存吞吐；同时，增大 cube 的基本块大小也能进一步提升其计算效率。



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

## Split-K pipelined GEMM v0 implementation: MTE1 与 Cube 流水重叠

在标准的矩阵乘执行过程中，数据必须先从 L1 搬运到 L0A/L0B（由 MTE1 完成），随后才交给 Cube计算单元 （MMA 矩阵乘加指令）完成计算。如果按照完全同步的串行逻辑，Cube 计算单元在等待后续数据载入时将处于闲置状态。`gemm_v0`模板采取了以下机制：

- **K 轴数据切块与 Ping-Pong 双缓冲**：在内层循环将 K 维度切分为固定容量的子块（例如 `kL0Size = 128`），并在L0 内存中为子块的 A 和 B 矩阵申请出双倍缓冲区。通过 `pp = (kL0Idx & 1)` 动态交替写入地址。
- **细粒度同步**：手动调度同步信号，使得当 Cube 计算单元正在对第 $i$ 份 L0 数据执行 `mma` 指令运算时，MTE1 搬运单元将 L1 中的第 $i+1$ 份数据拷贝进另一半的 L0 缓冲中。

通过调优的 `gemm_v0` 模板，实现了Cube核内部的 MTE1 访存与 Cube 计算的重叠，成功掩盖了算子的绝大部分MTE1访存延时。

## 6. 总结

综合最终评测结果，在该场景下，TileLang 编译生成的算子在相同输入条件下可达到 AscendC 参考实现约 0.90× 的性能水平，本文所总结的优化路径可为后续基于 TileLang 开发高性能算子提供参考。
