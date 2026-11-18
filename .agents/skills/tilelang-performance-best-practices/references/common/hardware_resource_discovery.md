# Ascend 硬件资源与编译器有效容量发现

Tiling、驻留和多版本流水必须区分硬件物理规格、平台用户可用容量、当前 lowering 的编译器预算和 kernel 显式分配，不能继承历史机器或单个算子的“安全上限”。

## 1. 记录当前硬件

记录实际 SoC、NpuArch、CANN、TileLang 导入路径/版本、编译 target、设备绑定和 kernel 编程模式。硬件知识统一查询 [npu-arch](../../../npu-arch/SKILL.md)，复用调用方已经确认的设备证据：

1. 用其 [产品映射与硬件参数](../../../npu-arch/references/npu-hardware-params.md) 核对 Ascend910B / Ascend910_93 与 DAV_2201 的组合；不能用静态表反推当前设备，也不能仅凭产品系列字样选择架构。
2. 按需读取 [架构指南中的参数获取方式](../../../npu-arch/references/npu-arch-guide.md)，核对实际可用 AIC/AIV、buffer、L2 等资源。典型 SKU 数字、INI 静态配置和运行时返回分别记录；vNPU 切分或设备裁剪后不能沿用整卡核数。`DAV_RESV` / `RESERVED_VERSION` 是查询失败；AIC/AIV 数量还须按本仓 Kernel 的 `(cid, vid)` 映射解释，不能将 C++ launch 示例直接替换成同数值的 `T.Kernel`。
3. 只消费与实际 SoC / NpuArch 对应的条目，其他架构的执行域、数据通路及指令能力不能直接移入候选。架构能力存在还须核对当前 TileLang 前端和 lowering 是否支持。
4. 在分析阶段，设备证据缺失、失败或互相冲突时，列出缺口交调用方补充；本参考不替代环境探测或 profiling。

同一产品系列不代表所有机器具有相同核数；不把历史机器的 24 核、L2 容量或理论带宽作为本机配置。硬件规格与能力以 npu-arch 的来源和目标运行时证据为依据，编译器可用容量仍须从实际使用的后端核验。

分别记录物理规格、平台用户可用容量和编译器预算。INI 或运行时接口值不自动等于物理容量；最终 footprint 必须同时满足平台用户可用量与当前后端限制。已扣除的预留不能再次扣减，未解释的差异保持为待核验项。

## 2. 查询编译器有效容量

先定位任务实际导入的 TileLang，而不是按目录名猜测版本：

```bash
python -c 'import pathlib, tilelang; print(pathlib.Path(tilelang.__file__).resolve()); print(tilelang.__version__)'
```

在对应源码中查找资源上限和执行域预留：

```bash
rg -n "ASCEND_A2A3_.*SIZE|MemoryPlanning|memory_limit" <tilelang_root>/src/target <tilelang_root>/src/transform
```

本仓有多个后端；不能仅凭设备系列切换 target。保留目标已有后端，从实际编译入口确认资源限制。当前 [AscendC codegen](../../../../../src/target/codegen_ascend.cc) 中 `ASCEND_A2A3_*_SIZE` 是**编译器预算**：

| 层级 | 当前源码字节数 | 使用方式 |
|---|---:|---|
| L0A | 65536 | 操作数、K 子块和版本合计 |
| L0B | 65536 | 操作数、K 子块和版本合计 |
| L0C | 131072 | fp32 累加及仍存活的输出版本 |
| L1 | 524032 | 常驻侧、流式侧、多版本和临时量 |
| UB | 196352 | 显式 buffer、scratch、padding 和版本 |

这些数字是该源码路径的限制，不是对物理容量或其他后端的声明。例如 npu-arch 的 DAV_2201 参数表列出 UB 196608 B、L1 524288 B，而此后端预算分别为 196352 B、524032 B；硬件规格不能直接作为可分配上限。核对二者差异的实现依据，不把差值未经证明推广为其他后端的固定预留。L2 及核数也不能用某个典型 SKU 替代当前环境证据。更新 target 或编译器后重查；其他后端应读其资源规划实现。

## 3. 建立资源账本

```text
物理规格/平台用户可用容量：已确认的设备资料或任务环境工具，分别记录口径
编译器有效上限：当前 lowering 对最终 kernel body 的计算结果
显式 footprint：sum(aligned_extent * dtype_bytes * buffer_versions)
常驻 footprint：LUT / index / state / metadata
隐式或临时开销：仅记录当前源码、IR 或编译报错可证明的部分
安全余量：具体保护对象和数值依据
剩余容量：编译器有效上限 - 上述 footprint
```

物理容量与编译器上限不一致时先核对架构、导入路径和版本。安全余量必须有具体保护对象，不能用任意 cap 代替公式。用账本枚举完整连续单元、最大合法容量及相邻 SIMD/DMA 对齐候选；执行模式、buffer version、padding、LUT/index 或临时物化变化后重算，并用生成 IR/运行结果验证。

### 按生命周期计算峰值

逐 buffer 列出 `shape / padded shape / dtype bytes / versions / scope / producer / last consumer / live interval`：

```text
buffer_bytes = product(padded_shape) * dtype_bytes * versions
peak_live = 同一时刻仍存活 buffer 的字节合计
required = peak_live + 已知 scratch/隐式临时量 + 有依据的预留
remaining = 当前后端有效容量 - required
```

没有生成代码证明内存复用时，用保守的分配总量预算。开启 `TL_ASCEND_MEMORY_PLANNING` 只允许编译器按生命周期复用，不能保证所有理论上不同时使用的对象都合并。检查 [内存规划 pass](../../../../../src/transform/ascend_memory_planning.cc) 和实际生成地址，异步 consumer 尚未结束时 buffer 仍存活。

`rows` 与 `rows_padded`、输入/计算/输出 dtype、compare mask、cast scratch、broadcast 完整临时量、尾块 buffer、常驻参数都要单独计数。不能以固定 `N_cal=6` 或从 dtype 字符串尾部猜字节宽度；int8/int32/int64 尤其容易被这类公式误算。

## 4. 从预算生成候选

先按实际指令约束确定对齐：普通 DMA 的字节粒度、cast 的 dtype 组合、compare 的输入字节对齐、transpose 的静态 tile 约束不同，不能统一为“16 元素”。从剩余容量求合法上界，再枚举整除边界、完整连续单元、相邻对齐值和任务波次边界。若最小合法 tile 也超限，返回无合法配置并调整数据流；禁止 `max(min_tile, candidate)` 强行返回超限配置。

容量上界不是性能最优值。保留更小 tile 以比较核并行度、尾部浪费、固定开销和流水。每次改变 rows、stage、驻留、融合或临时布局均重新预算；不把预算拖到所有优化结束才第一次计算。

验证包含最大 footprint、最小任务、尾块及不同 dtype；编译成功和资源公式均不能代替设备正确性或性能结果。
