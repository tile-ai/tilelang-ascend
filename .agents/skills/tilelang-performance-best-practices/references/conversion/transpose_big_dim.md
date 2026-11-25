# 大维度分块 Transpose

## 适用场景

任一交换维超过 UB 容量，需要二维 tile 化。

## TileLang Ascend Kernel 结构

每个 task 对应一个 batch/x/y 矩形；T.Persistent 在核间均匀分配。选择 block_x/block_y 使输入、输出、padding 和必要多版本均满足 UB 预算。

当前仓库的 [transpose 测试](../../../../../testing/python/language/test_tilelang_ascend_language_transpose.py) 可用于核对转置语义，[copy_op.py](../../../../../tilelang/language/copy_op.py) 与 [Ascend 模板](../../../../../src/tl_templates/ascend/common.h) 用于核对搬运和布局限制；它们不等于通用批量转置 kernel。复用当前算子或新实现时，明确任务分配、stride、dtype、尾块和 UB footprint。复用 [transpose 实现](../../../../../examples/cann-bench/transpose/transpose.py) 的 kernel factory 前，按 [多阶段复用审计](multi_stage_contiguous.md) 核验适用范围。

## 精度要求

验证跨 tile 边界、最后一块和 64 位总偏移；动态 stride 不得截断为 32 位。

按当前公开接口覆盖 dtype、batch、stride、允许的对齐与余数类及非连续输入；接口承诺通用尾块时覆盖 tile-1/tile/tile+1。T.assume 只能表达调用方保证；输出与 torch.permute/transpose reference 比较，并使用 canary 检查越界。

## 性能要求

搜索二维 tile 的长宽比并观察 gather 指令效率、copy burst 与负载均衡。

先通过当前后端定向精度测试，再以相同条件比较 DMA、已验证的 Vector 重排和 UB-local fallback。记录 latency、有效 GM 带宽、标量索引开销、UB 占用和生成代码长度；未实测不得宣称某路径更快。
