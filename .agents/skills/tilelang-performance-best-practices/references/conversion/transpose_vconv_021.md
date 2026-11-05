# 0-2-1 三维变换

## 适用场景

三维 layout 的后两轴交换，常见于 batch/sequence/channel 重排。

## TileLang Ascend Kernel 结构

将第 0 维作为 batch task，后两维复用二维 transpose；对固定余数类生成 block_x/block_y。若同时发生 dtype conversion，在 UB 中以 fp32/目标 dtype 完成并保持转换顺序。

当前仓库的 [transpose 测试](../../../../../testing/python/language/test_tilelang_ascend_language_transpose.py) 可用于核对转置语义，[copy_op.py](../../../../../tilelang/language/copy_op.py) 与 [Ascend 模板](../../../../../src/tl_templates/ascend/common.h) 用于核对搬运和布局限制；它们不等于通用批量转置 kernel。复用当前算子或新实现时，明确任务分配、stride、dtype、尾块和 UB footprint。复用 [transpose 实现](../../../../../examples/cann-bench/transpose/transpose.py) 的 kernel factory 前，按 [多阶段复用审计](multi_stage_contiguous.md) 核验适用范围。

## 精度要求

覆盖三个维度的 size=1、动态 batch 和两个交换轴尾部。

按当前公开接口覆盖 dtype、batch、stride、允许的对齐与余数类及非连续输入；接口承诺通用尾块时覆盖 tile-1/tile/tile+1。T.assume 只能表达调用方保证；输出与 torch.permute/transpose reference 比较，并使用 canary 检查越界。

## 性能要求

比较融合 conversion 与单独 transpose+cast 的 GM 字节和 latency。

先通过当前后端定向精度测试，再以相同条件比较 DMA、已验证的 Vector 重排和 UB-local fallback。记录 latency、有效 GM 带宽、标量索引开销、UB 占用和生成代码长度；未实测不得宣称某路径更快。
