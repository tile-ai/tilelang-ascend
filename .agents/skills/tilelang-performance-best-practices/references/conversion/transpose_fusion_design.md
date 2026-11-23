# Transpose 融合

## 适用场景

transpose 前后存在 elementwise、cast、scale 或量化操作，可消除中间 GM round-trip。

## TileLang Ascend Kernel 结构

在 out_ub 写回前执行融合表达式；规则连续表达式使用当前模式的 T.Parallel/T.tile 原语。保持 transpose 索引与数值操作分层，factory 决定是否生成融合变体。

当前仓库的 [transpose 测试](../../../../../testing/python/language/test_tilelang_ascend_language_transpose.py) 可用于核对转置语义，[copy_op.py](../../../../../tilelang/language/copy_op.py) 与 [Ascend 模板](../../../../../src/tl_templates/ascend/common.h) 用于核对搬运和布局限制；它们不等于通用批量转置 kernel。复用当前算子或新实现时，明确任务分配、stride、dtype、尾块和 UB footprint。复用 [transpose 实现](../../../../../examples/cann-bench/transpose/transpose.py) 的 kernel factory 前，按 [多阶段复用审计](multi_stage_contiguous.md) 核验适用范围。

## 精度要求

严格保持 reference 的运算与 cast 顺序；分别验证仅 transpose、仅 elementwise 和融合结果。

按当前公开接口覆盖 dtype、batch、stride、允许的对齐与余数类及非连续输入；接口承诺通用尾块时覆盖 tile-1/tile/tile+1。T.assume 只能表达调用方保证；输出与 torch.permute/transpose reference 比较，并使用 canary 检查越界。

## 性能要求

只有减少总 GM 字节且没有因寄存器/UB 压力造成回退时保留融合。

先通过当前后端定向精度测试，再以相同条件比较 DMA、已验证的 Vector 重排和 UB-local fallback。记录 latency、有效 GM 带宽、标量索引开销、UB 占用和生成代码长度；未实测不得宣称某路径更快。
