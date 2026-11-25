# Transpose 与 Layout Conversion

## 适用场景

根据连续性、dtype 字节数、tile footprint 和目标 layout 选择 DMA、当前已验证的 Vector 重排或 UB-local fallback。

## 决策顺序

从 [transpose 实现](../../../../../examples/cann-bench/transpose/transpose.py)、[语言测试](../../../../../testing/python/language/test_tilelang_ascend_language_transpose.py) 与目标公开接口还原 shape、permutation、stride、dtype、storage offset 与输出布局。分别判断逻辑 view、host 物理重排、kernel 内离散 fallback，不能混为一种开销。

1. 合并可证明相邻连续的轴，查是否存在输入/输出共同连续 suffix record。
2. 规则 tensor move 可按合法矩形 copy 完成时，优先比较直接搬运；不要为减少 DSL 行数引入隐式重排。
3. 需要交换两个轴时，在预算内枚举完整一轴、双轴切分、tile 长宽比与 task waves。
4. 按实际 dtype 分支选择 [硬件适配](dtype_adaptation.md)；数据只能重排时要求 bit-exact。
5. 连续 record 很短且通用离散 fallback 慢时，评估 [多阶段连续化](multi_stage_contiguous.md)，先核验 stage kernel 再接 planner。

## TileLang Ascend Kernel 结构

先合并相邻连续维并选择 block_x/block_y。输入按当前仓库已验证的 BufferRegion/stride 表达动态 stride；输出使用连续 T.Tensor。常见二维交换直接复用基线 kernel，复杂 layout 通过 factory 生成静态索引变体。

当前仓库的 [transpose 测试](../../../../../testing/python/language/test_tilelang_ascend_language_transpose.py) 可用于核对转置语义，[copy_op.py](../../../../../tilelang/language/copy_op.py) 与 [Ascend 模板](../../../../../src/tl_templates/ascend/common.h) 用于核对搬运和布局限制；它们不等于通用批量转置 kernel。复用当前算子或新实现时，明确任务分配、stride、dtype、尾块和 UB footprint。复用 [transpose 实现](../../../../../examples/cann-bench/transpose/transpose.py) 的 kernel factory 前，按 [多阶段复用审计](multi_stage_contiguous.md) 核验适用范围。

### 分块与布局语义

一侧完整驻留可减少重复加载，但输入/输出/transpose scratch 和版本均须纳入 UB。双轴 tile 使用输入 `[BM,BN]`、输出 `[BN,BM]`，GM 两侧有界 slice；不能只修输入尾块。小 shape 应比较单核/少核、无流水与简单地址映射，避免大量空核及复杂索引表初始化。

N-last/0-2-1 可以先归一化 outer/move/inner 再处理，仍须证明该 reshape 不复制、真实 stride 与映射一致。5HD 必须保留 N、C1、C0 和无效 channel padding：`C1=ceildiv(C,C0)`，`out[n,c//C0,h,w,c%C0]=x[n,c,h,w]`；不能用简单二维 transpose 替代。未有当前可执行 kernel 的布局仅列设计候选。

### 融合与输出布局

transpose 前后的 cast/scale/elementwise 若能在同一 UB tile 完成，可减少中间 GM；同时核算 cast 扩宽、scratch、顺序和舍入。目标只要求元素重排时，禁止将值转换的误差混入“布局等价”。若生产者能直接写最终布局，比较省下的中间写读与新增地址/同步成本。

host `.contiguous()` 和 `reshape` 的诊断见 [Host 数据流](../common/host_device_dataflow.md)。把转换移到另一个 kernel 不等于消除成本；所有 launch、workspace 和末尾恢复都计入。

## 精度要求

转换不能改变元素值或重复/遗漏元素；低精度与整数执行逐元素精确比较，浮点仅在同时发生数值转换时使用项目既有容差。

按当前公开接口覆盖 dtype、batch、stride、允许的对齐与余数类及非连续输入；接口承诺通用尾块时覆盖 tile-1/tile/tile+1。T.assume 只能表达调用方保证；输出与 torch.permute/transpose reference 比较，并使用 canary 检查越界。

用唯一编号输入逐元素比对，支持的整数/仅重排浮点按 bit-exact；需要数值转换时用原容差并单独检查 NaN mask。覆盖 batch>1、M/N<tile、双尾块、各允许 stride、storage offset、size=1 轴和 dtype。对原位/alias 仅按接口承诺支持，不能默认为安全。

## 性能要求

性能报告有效/填充字节、事务、索引开销、UB、所有阶段 latency。

优先减少 GM transaction 和地址计算；对小 shape 控制 launch/初始化开销，对大 shape 优化有效带宽。

先通过当前后端定向精度测试，再以相同条件比较 DMA、已验证的 Vector 重排和 UB-local fallback。记录 latency、有效 GM 带宽、标量索引开销、UB 占用和生成代码长度；未实测不得宣称某路径更快。

## 策略文档路由

- 规则矩形搬运：[Tensor Move](transpose_tensor_move.md)。
- 大维度与单/双轴切分：[大维度](transpose_big_dim.md)、[单轴](transpose_cut_one_axis.md)、[双轴](transpose_cut_two_axis.md)。
- 小 shape：[小形状](transpose_small_shape.md)。
- 末轴、0-2-1 与 5HD：[N-last](transpose_n_last.md)、[0-2-1](transpose_vconv_021.md)、[5HD](transpose_vconv_5hd.md)。
- 索引 gather 路线：[Gather](transpose_gather.md)。
- 前后处理融合：[融合设计](transpose_fusion_design.md)。
- 统一记录参数与校验：[Tiling 数据](transpose_tiling_data.md)。
