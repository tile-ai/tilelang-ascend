# Host 预处理内化、stride 与输出驻设备

## 先测完整入口

分别记录 host 准备、物理布局/类型转换、kernel chain、输出后处理与必要同步。NPU 算子通常异步，host Python 调用耗时不等于设备执行耗时；计时边界须正确等待完成。kernel 已快但公开入口慢时，优先找额外 pad、copy、cast、D2H 和 launch。

`transpose/permute/movedim` 一般只是 view；`reshape` 可能返回 view，也可能复制，非 contiguous 不等于必然复制。核对 shape/stride/storage offset 以及是否共享 storage。`.contiguous()` 在已连续时可以无操作；不要仅按出现次数估算搬运。

## 输入侧 pad / contiguous 内化

Host 直接传原 tensor，kernel 对真实范围做整块/余数块，按 [尾块规则](tail_block_design.md) 初始化完整 UB footprint。GM 输入声明原始维度，循环次数从原始尺寸计算。输出可按真实 valid 写回；若沿用 padded-output 设计，必须实际分配对应 storage 并核验裁剪后 view 的 stride。

GroupNorm `gamma[G, rows]` 使用单行有效 slice `gamma[g, 0:rows]`，配对齐 staging、cast、broadcast；该已知路径比误用二维 slice 更易核验，但不能推广为“所有 2D copy 禁止”。详细模式见 [广播](../broadcast/broadcast_design.md)。

## 减少重复布局转换

连续两次轴交换若逻辑上抵消，先合成最终 permutation，再比较物理拷贝次数。对 scan 原始 `[M,R,N]`，当 N 为可利用的连续并行轴，可直接按原 stride 处理，避免先物化成 `[R,M*N]`；当 N 太窄，转置带来的并行收益可能仍大于搬运成本。输入与输出布局要一起设计，不能只消除一侧。

`x.movedim(...).reshape(...)` 中的 reshape 仍可能先复制，后续 `.contiguous()` 不能证明只发生一次转换。用 trace 和实际 storage 判定。输出 `reshape(...).movedim(...)` 若接口允许非连续 view，可减少物化；若公开接口要求 contiguous，则相关 copy 必须计入。

## 非连续输入归一化

少量静态 shape/stride 参数可作为 JIT Python int/tuple 输入，kernel 做地址推导，不额外创建元数据 GM tensor。真实动态 stride 或元数据过多时保留合理运行时表示，不能为省搬运破坏接口。

参见 [stride API 模式](../../../tilelang-custom-skill/tilelang-api-best-practices/references/api-compute.md)。每次搬运一段实际连续范围，或者使用当前 lowering 已支持的有 stride 的矩形。不要编造 `T.copy(stride=...)` 参数；核验 source/destination pitch、storage offset、负 stride/重叠是否合法、总偏移位宽。`alloc_var` 的定义与使用必须在正确执行侧；自动划分失配按 [配置专题](pipeline_and_cv_overlap.md) 排查。

## 输出侧 D2H 与同步

对 NPU 输入，输出类型转换、index int32→int64、where 合并与布局恢复优先留在设备；wrapper 中间 `.cpu()` 再 `.npu()` 会引入数据搬运和同步。公共接口若同时支持 CPU 输入，可保留兼容入口，在边界做一次必要的输入/输出搬运；不要未经接口授权新增 CPU 语义。

移除调用链中无依赖依据的 `torch.npu.synchronize()`，由计时器/调用方或真实跨流事件建立必要等待。测试拿回 CPU 与 golden 比较属于验证，不是产品热路径中的同类问题。torch_npu 操作不可用时记录缺口，不把不存在的 NPU 能力当作方案。

**数据相关 JIT 分支不是免费元数据**：`bool(torch.isnan(x).any())` 在 NPU 输入上通常需要一次全量扫描和 host 可见结果同步。比较扫描、同步、额外 kernel 与少执行向量指令的总成本；默认保留 kernel 内正确 NaN 路径。只有接口已保证有限值或已有可靠元数据时，才可无扫描地使用无 NaN 特化。类似 `.item()`、Python `if tensor` 和 CPU 索引读取均需审计。

## 验证

覆盖 contiguous/非连续输入、storage offset、允许的 stride、输出 dtype/device、alias 与尾块。比较完整入口和 kernel 两个口径，并明确 CPU 兼容边界是否计入；不能把转到 host 的工作排除后称为加速。
