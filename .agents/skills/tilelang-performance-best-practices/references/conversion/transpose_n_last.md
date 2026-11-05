# N-last Layout Conversion

## 适用场景

把指定维移动到最后或从最后维移出，适合先合并其它连续维。

## TileLang Ascend Kernel 结构

factory 将输入归一化为 outer×move×inner；inner 连续时使用矩形 T.copy，move/inner 交换使用当前已验证的 Vector 重排。无法安全表达的 dtype 回退块 DMA + UB-local 重排。

当前仓库的 [transpose 测试](../../../../../testing/python/language/test_tilelang_ascend_language_transpose.py) 可用于核对转置语义，[copy_op.py](../../../../../tilelang/language/copy_op.py) 与 [Ascend 模板](../../../../../src/tl_templates/ascend/common.h) 用于核对搬运和布局限制；它们不等于通用批量转置 kernel。复用当前算子或新实现时，明确任务分配、stride、dtype、尾块和 UB footprint。复用 [transpose 实现](../../../../../examples/cann-bench/transpose/transpose.py) 的 kernel factory 前，按 [多阶段复用审计](multi_stage_contiguous.md) 核验适用范围。

## 精度要求

验证 rank、负 axis、size=1 维和非连续 stride。

按当前公开接口覆盖 dtype、batch、stride、允许的对齐与余数类及非连续输入；接口承诺通用尾块时覆盖 tile-1/tile/tile+1。T.assume 只能表达调用方保证；输出与 torch.permute/transpose reference 比较，并使用 canary 检查越界。

## 性能要求

按 inner 长度选择当前可执行的 Vector 或 UB-local 路径，比较维度合并前后的地址计算与 GM burst。

先通过当前后端定向精度测试，再以相同条件比较 DMA、已验证的 Vector 重排和 UB-local fallback。记录 latency、有效 GM 带宽、标量索引开销、UB 占用和生成代码长度；未实测不得宣称某路径更快。
