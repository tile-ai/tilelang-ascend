# 小 Shape Transpose

## 适用场景

独立 tile 很少、总字节数较小，启动和索引开销占主导。

## TileLang Ascend Kernel 结构

使用 min(按当前 Kernel 映射确定的可用 launch 数, tile 数)；能放入单个 UB tile 时一次 T.copy 搬入、一次已验证的 Vector/UB-local 转置、一次写回，不启用多 stage。按 dtype 和余数类在 factory 中生成少量静态变体。

当前仓库的 [transpose 测试](../../../../../testing/python/language/test_tilelang_ascend_language_transpose.py) 可用于核对转置语义，[copy_op.py](../../../../../tilelang/language/copy_op.py) 与 [Ascend 模板](../../../../../src/tl_templates/ascend/common.h) 用于核对搬运和布局限制；它们不等于通用批量转置 kernel。复用当前算子或新实现时，明确任务分配、stride、dtype、尾块和 UB footprint。复用 [transpose 实现](../../../../../examples/cann-bench/transpose/transpose.py) 的 kernel factory 前，按 [多阶段复用审计](multi_stage_contiguous.md) 核验适用范围。

## 精度要求

覆盖单元素、单行、单列、64/128 边界和 batch=1。

按当前公开接口覆盖 dtype、batch、stride、允许的对齐与余数类及非连续输入；接口承诺通用尾块时覆盖 tile-1/tile/tile+1。T.assume 只能表达调用方保证；输出与 torch.permute/transpose reference 比较，并使用 canary 检查越界。

## 性能要求

比较单核与少核；避免启动大量空核和为小 shape 构造复杂索引表。

先通过当前后端定向精度测试，再以相同条件比较 DMA、已验证的 Vector 重排和 UB-local fallback。记录 latency、有效 GM 带宽、标量索引开销、UB 占用和生成代码长度；未实测不得宣称某路径更快。
