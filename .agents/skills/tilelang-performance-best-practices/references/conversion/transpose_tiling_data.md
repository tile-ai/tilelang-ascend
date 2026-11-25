# Transpose 配置选择

## 适用场景

用 Python kernel factory 生成可缓存的静态配置，不引入额外的 host 配置结构。

## TileLang Ascend Kernel 结构

输入为 shape 余数类、dtype、stride 类别和 layout；输出为 block_x、block_y、num_cores、execution domain 和 stage。动态维只保留在当前 T.symbolic 表达，硬约束通过 T.assume 表达。

当前仓库的 [transpose 测试](../../../../../testing/python/language/test_tilelang_ascend_language_transpose.py) 可用于核对转置语义，[copy_op.py](../../../../../tilelang/language/copy_op.py) 与 [Ascend 模板](../../../../../src/tl_templates/ascend/common.h) 用于核对搬运和布局限制；它们不等于通用批量转置 kernel。复用当前算子或新实现时，明确任务分配、stride、dtype、尾块和 UB footprint。复用 [transpose 实现](../../../../../examples/cann-bench/transpose/transpose.py) 的 kernel factory 前，按 [多阶段复用审计](multi_stage_contiguous.md) 核验适用范围。

## 精度要求

每个 dispatch 分界两侧都做精度测试，fallback 覆盖所有合法输入。

按当前公开接口覆盖 dtype、batch、stride、允许的对齐与余数类及非连续输入；接口承诺通用尾块时覆盖 tile-1/tile/tile+1。T.assume 只能表达调用方保证；输出与 torch.permute/transpose reference 比较，并使用 canary 检查越界。

## 性能要求

离线枚举候选并在代表 shape 实测；控制变体数量和编译缓存占用。

先通过当前后端定向精度测试，再以相同条件比较 DMA、已验证的 Vector 重排和 UB-local fallback。记录 latency、有效 GM 带宽、标量索引开销、UB 占用和生成代码长度；未实测不得宣称某路径更快。
