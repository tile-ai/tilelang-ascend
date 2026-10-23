# SIMD Gather Transpose

## 适用场景

dtype、源索引、向量访问范围和当前 TileLang lowering 均可表达目标索引重排时，评估该设计。A2/A3 本页状态为 DESIGN_ONLY，不提供未经验证的 gather transpose kernel。

## TileLang Ascend Kernel 结构

按 lane offset × padded stride + 列偏移构造目标索引，核对索引单位、dtype、mask 与每次访问的合法范围。当前仓库 [T.tile.gather_mask](../../../../../tilelang/language/ascend_tile.py) 具有自己的语义，不能视为任意逐 lane gather 的同义替代；无匹配的 API/lowering 证据时保留为设计，并使用 [dtype 适配](dtype_adaptation.md) 中已验证的块 DMA + UB-local 路径。

当前仓库的 [transpose 测试](../../../../../testing/python/language/test_tilelang_ascend_language_transpose.py) 可用于核对转置语义，[copy_op.py](../../../../../tilelang/language/copy_op.py) 与 [Ascend 模板](../../../../../src/tl_templates/ascend/common.h) 用于核对搬运和布局限制；它们不等于通用批量转置 kernel。复用当前算子或新实现时，明确任务分配、stride、dtype、尾块和 UB footprint。复用 [transpose 实现](../../../../../examples/cann-bench/transpose/transpose.py) 的 kernel factory 前，按 [多阶段复用审计](multi_stage_contiguous.md) 核验适用范围。

## 精度要求

索引 dtype 必须覆盖最大 UB 偏移；mask 与 dtype lane 数一致。尾部未使用 lane 不得写回 GM。

按当前公开接口覆盖 dtype、batch、stride、允许的对齐与余数类及非连续输入；接口承诺通用尾块时覆盖 tile-1/tile/tile+1。T.assume 只能表达调用方保证；输出与 torch.permute/transpose reference 比较，并使用 canary 检查越界。

## 性能要求

检查生成代码中的 gather 数、索引重算和 unroll 体积；索引准备应在列循环外复用。

先通过当前后端定向精度测试，再以相同条件比较 DMA、已验证的 Vector 重排和 UB-local fallback。记录 latency、有效 GM 带宽、标量索引开销、UB 占用和生成代码长度；未实测不得宣称某路径更快。
