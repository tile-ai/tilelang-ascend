# ForeachNorm 性能优化日志

## 基线（iter1, warmup=5, iters=20）
- avg_speedup = 0.1713（目标 0.6）
- 瓶颈：host 侧 `_finalize()` 用 PyTorch op（.sum/.max/.min/.sqrt/.log/.exp/.to），每 tensor 多 2~5 次 NPU kernel launch（P0 host 反模式）

## Step 2: 算子类型判断
- 纯 AIV（Vector 型）归约算子：abs/mul/cast + reduce_sum/max/min
- 无 Cube 计算，无 AIC

## Step 3: 优化点清单（Part A）

[#1] [P0 Host 侧 finalize 内化]（参考: cann-bench-elementwise-optimization.md §"核心反模式：接口层全量 host 拷贝" + optimization-guide.md §2.12）：**适用** — iter1 的 `_finalize()` 在 host 侧用 PyTorch op（partial.sum().sqrt().to() 等），每个 tensor 额外 2~5 次 NPU kernel launch。改为 1-block finalize kernel 在 NPU 上完成 combine+finalize+cast，消除所有 host PyTorch op。

[#2] [纯 AIV 双 buffer 流水线]（参考: performance-antipatterns.md §"纯 AIV memory bound 算子未做流水/双 buffer"）：**适用** — partial kernel 串行 load→cast→compute→reduce，MTE2 与 V pipeline 无 overlap。加 T.Pipelined(num_stages=2) 预取下一 tile。

[#3] [launch_cores 自适应]（参考: performance-antipatterns.md §"launch core 数需要重点关注" B）：**适用** — 当前 launch_cores=min(n_num,24) 对小 n_num 已正确；但 single_core_load 过大时每核串行太多 tile，可调 launch_cores 上限。

[#4] [多 tensor batch 化]（参考: cann-bench-elementwise-optimization.md §"解决方案：单输入 split kernel 模式"）：**适用** — 当前 host for-loop 逐 tensor 调 kernel，list_len=4 时 8 次 launch。同 shape tensor 可 reshape 为 (batch,N) 一次 kernel 处理。

[#5] [block_N 自适应]（参考: performance-antipatterns.md §"tile size 过小"）：**部分适用** — fp16/bf16 block_N=8192 仅用 42% UB，可放大到 16384 减少 tile 数；fp32 已接近 UB 上限。

## Part B: [ORDER-PLAN] 实施顺序

1. [#1] P0 Host finalize 内化 — 前置依赖: 无 — 理由: P0 host 反模式，消除每 tensor 2~5 次 launch，对所有 case 即时生效
2. [#2] 双 buffer 流水线 — 前置依赖: [#1] — 理由: 消除 host 开销后再看 kernel compute 是否瓶颈；大 shape case 受益最大
3. [#4] 多 tensor batch — 前置依赖: [#1] — 理由: finalize 内化后 launch 数从 2/tensor 降为 1/tensor 仍多；batch 化可把 list_len 次 launch 降为 1 次
4. [#5] block_N 自适应 — 前置依赖: [#2] — 理由: 双 buffer 后 UB 占用翻倍，需重新评估 block_N 上限
5. [#3] launch_cores 调优 — 前置依赖: [#5] — 理由: block_N 变化影响 n_num 和 launch_cores 计算
