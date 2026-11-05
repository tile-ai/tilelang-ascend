# 规则 Tensor Move

## 适用场景

布局变化可由连续矩形重解释与批量 T.copy 完成，无需逐元素 gather。

## TileLang Ascend Kernel 结构

合并连续维后直接对矩形 slice 调用 T.copy；`T.npu_copy_v2(..., transpose=True)` 对应当前接口的 L1→L0 转置参数，只在目标示例已验证的 shape 上使用（当前 `T.copy` 没有该参数），不能用它代替通用 GM/UB 转置。其余场景回到通用 transpose kernel。

当前仓库的 [transpose 测试](../../../../../testing/python/language/test_tilelang_ascend_language_transpose.py) 可用于核对转置语义，[copy_op.py](../../../../../tilelang/language/copy_op.py) 与 [Ascend 模板](../../../../../src/tl_templates/ascend/common.h) 用于核对搬运和布局限制；它们不等于通用批量转置 kernel。复用当前算子或新实现时，明确任务分配、stride、dtype、尾块和 UB footprint。复用 [transpose 实现](../../../../../examples/cann-bench/transpose/transpose.py) 的 kernel factory 前，按 [多阶段复用审计](multi_stage_contiguous.md) 核验适用范围。

## 精度要求

验证源/目标 stride、重叠和原位语义；不支持原位时在接口层明确拒绝。

按当前公开接口覆盖 dtype、batch、stride、允许的对齐与余数类及非连续输入；接口承诺通用尾块时覆盖 tile-1/tile/tile+1。T.assume 只能表达调用方保证；输出与 torch.permute/transpose reference 比较，并使用 canary 检查越界。

## 性能要求

以 copy transaction、burst 长度和有效带宽为准，不能仅凭减少 DSL 行数判断。

先通过当前后端定向精度测试，再以相同条件比较 DMA、已验证的 Vector 重排和 UB-local fallback。记录 latency、有效 GM 带宽、标量索引开销、UB 占用和生成代码长度；未实测不得宣称某路径更快。
