# 特定 dtype 的指令路径适配

## 先证明当前路径

本节迁移旧指南 §2.16。按符号定位 [AscendC common.h](../../../../../src/tl_templates/ascend/common.h) 的 `transpose` 模板，并在目标 `get_kernel_source()` 中核对调用和真实 specialization。源码分支是特定后端事实，不能代表其他 target 或未来版本；`SetValue` 也应定位到实际操作，不能全文件出现一次便认定整 kernel 是标量瓶颈。

`GetValue/SetValue` 可能封装在被调用的 C++ 模板中，生成调用点未展开时不会直接出现这些字符串；此时沿 dtype/静态 shape 推导模板分支，必要时查看更低层编译产物，不能以搜索不到字符串就否定 fallback。

当前该模板的主要分支：

| 条件 | 路径 |
|---|---|
| 16×16、2 字节且非 bf16 | `AscendC::Transpose` |
| M/N 均为 16 倍数、2/4 字节且非 bf16 | `transpose_block` / `TransDataTo5HDImpl` |
| 其他形态 | UB-local `GetValue/SetValue` 循环 |

因此 bf16/int8/int64 在这段模板中需要特别审查；fp16/int16/fp32/int32 也必须满足实际静态 tile 条件，不能只凭 dtype 宣称硬件路径。若 planner 已选完整对齐 UB tile，尾部 GM 的 valid 维度不会自动改变其静态模板尺寸；反之也不能假定任意 tile 天然 16 对齐。

只处理算子接口实际支持且本次改变影响的 dtype，不扩张产品范围。完整原语签名与已有建议见 [API 计算参考](../../../tilelang-custom-skill/tilelang-api-best-practices/references/api-compute.md)。

## bf16：同宽位解释

仅搬运/转置 bit pattern 时，评估 host `view(torch.int16)` 的同宽零复制 reinterpret，再使用 int16 物理重排，输出恢复 bf16 view。它不是 `.to(int16)` 数值转换；保留 storage offset、shape、stride、alignment、alias 与输出 dtype，确认运行时不会隐式复制。不适用于其中夹杂 bf16 算术的路径。

同宽 reinterpret 是可复用候选，不凭名字保证每个非连续输入均可 view。每个允许 stride/尾块都要验证 bit-exact，包括 NaN payload、±0 与极值。

## int8：kernel 内扩宽

int8 值能精确表示为 fp16，可在 UB 内 cast 到 fp16、硬件 transpose、再 cast 回 int8。核验当前 CAST mode、饱和/舍入、原 dtype 全值域、扩宽 scratch、转换总成本和同步。不要把旧 `CAST_NONE/CAST_RINT` 当所有后端固定组合，先查实际生成模板。

扩宽/收窄都放在 kernel 数据流内比较；host `.to()` 会引入独立 cast launch，`.contiguous()`/`torch.stack` 可能引入 copy/cat，不能排除这些成本后宣称改进。数据仅重排必须精确恢复。

## int64：保留全部位

禁止以 fp32/fp16 或更窄整数承载通用 int64 值。选择以下两条当前可证明路径：

- **路线 A，record-aware DMA**：输入/输出共享连续 suffix record，record 本身就是正确的最终搬运单位；地址、长度、gap 满足当前 DataCopy 要求。旧方案采用 `record_bytes>=32` 作为合法块路径的候选边界，具体对齐与非对齐支持仍查当前 lowering。小于合法粒度时聚合相邻记录，不能每个 8B 元素独立做 strided GM DMA。
- **路线 B，块 DMA + UB-local 重排**：输入/输出都块搬运，`T.tile.transpose` 的标量循环仅在 UB 内部发生。即便生成 `SetValue`，它也不同于逐元素 GM 访问；可以是正确且更快的 fallback。无收益时如实记录，不为消除标量字样强制换成更差的短 DMA。

把 int64 拆成两个 int32 lane 只能作为独立实验：必须证明 kernel 内提取/交织、字节序、配对、UB 规划和尾部完整，不能用 host split/stack 绕过核心问题。

## 验证与复用门禁

1. 为每个受影响 dtype 写明当前分支、候选路径、静态 tile、所有 buffer 和实际 GM 事务。
2. 使用高 32 位非零、符号位与交错 bit pattern 验证 int64；只用小随机数不能发现截断。
3. 输入/输出两侧都使用有效 BufferRegion；batch、双尾块与小于 tile 尺寸是必测，详见 [stage 审计](multi_stage_contiguous.md)。
4. 同步后检查设备异常，并运行全部受影响 canonical cases，不能只跑 profile 子集。
5. 用完整入口比较，不把 host 新增 cast/copy/cat、workspace 或多阶段 launch 排除。

本页仅核对了当前源码分支与迁移方法，未在本次重构中验证候选 kernel 性能。
