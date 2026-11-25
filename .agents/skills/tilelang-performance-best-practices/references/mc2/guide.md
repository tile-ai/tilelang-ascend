# 当前仓库通算组合优化

## 目标实现

本节讨论通过 host wrapper 编排的通信与计算组合：计算使用 examples/gemm/example_gemm.py，通信由当前运行时的正式 distributed API 承担。Python host wrapper 编排 local GEMM、异步 collective、remote GEMM 与 stream/event；不得在 T.prim_func 中伪造通信或跨核同步。

## 流程

1. 单独验证 local GEMM 精度与 latency。
2. 单独验证 collective 的 tensor layout、rank offset、stream 与完成语义。
3. 建立不融合的通信→计算或计算→通信 reference。
4. 按 chunk 启动异步通信并在独立 stream 执行对应 GEMM；只使用运行时公开 API。
5. 多 rank 同步验证后，再搜索 chunk size 与 local-first 排布。

精度覆盖全部 rank、非整 chunk、不同 world size 和 fp32 GEMM accumulator。性能同时报告通信、计算、端到端 latency、重叠率和膨胀；缺少运行时 API 时明确拒绝该优化。

## 当前参考范围

上述 runtime collective + GEMM 组合为 `DESIGN_ONLY`，只有确认目标运行时的正式 API、完成语义与设备支持后才能实施。目标仓库另有 `tilelang/language/ascend.py` 中的 `T.shmem_*`、`examples/shmem/` 与 `examples/dispatch_combine/dispatch_combine_shmem.py`；这些是独立的通信原语/用例，不等于已验证本节 collective pipeline，不能据此省略 rank、生命周期和同步核验。

进一步阅读 [Local GEMM 与通信重叠](local_matmul_design.md) 和 [流水配平](pipeline_balancing_design.md)。
