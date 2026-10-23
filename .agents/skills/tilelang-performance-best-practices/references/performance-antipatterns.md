# A2/A3 性能关注项与错误诊断

本清单用于静态筛查；命中后需当前代码、生成结果与 profiling 支持，不能直接下 bound 或收益结论。改动遵守原公开接口、reference、容差和用户已选模式。详细知识集中在专题，避免不同文件维护冲突规则。

## 1. launch 数与工作量不匹配

逻辑 block 远多于硬件可同时运行数时，逐任务 launch 可能重复初始化、扩大 workspace；任务很少却固定大量核可能浪费初始化与同步。比较 Fixed Core 连续/跨步分配，workspace 按其真实生命周期复用，尾 task 加有效 guard。

不写死 A2/A3=24 核，也不混淆 AIV 总数与 `(cid,vid)` 的 launch 单位。任务时长不均时静态连续分配可能更慢；MIX 不一律 `vid==0`。见 [任务映射](common/task_mapping.md)。

## 2. Vector 侧逐行发射相同操作

`for row: tile.sub/mul/div(...)` 的独立行可以尝试 broadcast+整 tile，或在 UB 允许范围内多行批处理。检查新增 broadcast 的完整 footprint、临时量与实际指令次数。存在真实行间依赖时不能去循环。

如果内层是标量 if/GetValue/SetValue、扫描轴有前缀依赖，则是 [正交轴向量化](scan/orthogonal_axis_vectorization.md) 问题：保留扫描顺序，向量化无依赖轴，计入布局转换成本。

## 3. 全局 barrier 过密或删同步无依据

循环内每个 copy/算术后 `barrier_all/sync_all` 可能扩大等待，但首先区分核内 pipe barrier、跨核通知和真正全局同步，不能把所有 API 视为同一范围。只有证明没有生产消费依赖才删除。

Developer/混合模式沿用自动依赖；Expert 用当前 LOCK 源和预处理器表达 ownership，已有 generated/legacy flag 用于核验。输出写回前等待最终 cast/copy 完成，槽位复用前等待最后 consumer。仅分配 `[2,...]`、交替 dst 或移除 barrier 均不能证明同步正确。见 [流水与 CV](common/pipeline_and_cv_overlap.md)。

## 4. 基础指令有合法融合但未比较

`acc+=x*w` 可评估 `mul_add_dst`，`dst+=alpha*src` 可评估 `axpy`，激活与 `rsqrt` 路径按数值合约比较。`x=x*scale-m` 不等价于 `axpy(x,m,scale)`；错误融合不是精度可容忍的近似。每条候选检查 operand、alias、dtype、舍入、NaN/±0 与原容差。

已有授权的优化范围内按验证执行，不沿用旧资料“所有指令融合必须额外询问”的流程。具体例子见 [Elementwise](elementwise/guide.md)。

## 5. tile 太小或预算模型错误

低利用率、小 DMA、任务/同步太多时评估更大 tile；也保留小 tile 以提高并行度。总量必须包含实际输入/输出/计算 dtype、padding、broadcast、mask、scratch、常驻和全部存活版本。

旧固定 UB=192 KiB、固定六份 buffer、固定 16 元素对齐、任意 tile cap 与固定 L2 容量不作为当前结论。当前后端预算、生命周期及内存规划生成结果见 [资源发现](common/hardware_resource_discovery.md)。编译器复用不允许提前覆盖异步 consumer；最大可装 tile 也不等于最快 tile。

## 6. 纯 AIV 搬算完全串行

按 MTE2→Vector→MTE3 串行且每核多迭代时，比较当前可识别自动流水或显式 ownership。不要只根据算子类别判 memory-bound，不能统一要求关闭 AUTO_SYNC。自动插入同步是否阻塞 overlap 要从生成代码确认。

首次预取 slot 0 后首次消费必须是 slot 0；输入/输出分别版本化，V-only 临时量按生命周期判断。测试 0/1/奇偶迭代、尾波和 case 顺序。见 [Vector 双缓冲](elementwise/double_buffer_design.md)。

## 7. CV workspace handoff 串行

真实 AIC/AIV 数据交互可尝试 `T.Pipelined`，核验自动 CV sync 确实识别 workspace。stage 越大资源成本越高，不按耗时差距机械增加；`cross_interval` 1→2 是待验证候选，必须保留最终尾组通知。

避免未经支持验证的嵌套 `T.Pipelined`；外层 CV、内层 flat ownership 沿用当前教程。workspace 可受益于 L2，但“小于历史 L2 容量”不证明命中或延迟收益。

## 8. 多行 tile 每块重复归约

对于可结合总 sum/sumsq，比较每块 reduce+小状态与先累积 `[rows,block_S]` 后循环外归约。后者减少 reduce 次数，但多用 UB、改变求和顺序；不能禁止所有循环内 reduce。共享统计量可两步归约到标量，独立行统计只能保持每行输出。

旧“dim=0 随机”“reduce 输出必须 1D”是局部旧经验；当前合法 rank/dim/keepdim 及后端 dtype 限制见 [Reduction](reduce/guide.md)。诊断需具体 target、shape、real_shape、对齐和生成代码。

## 9. 多行 layout / padding / 广播错误

| 现象 | 检查与修复方向 |
|---|---|
| 多行 copy 输出错位 | 检查实际 src/dst pitch、slice 与 lowering；reshape 成 4D 不改变物理 stride |
| 大 cpg 内存超限 | 按 rows×tile×版本重算，不沿用单行 tile 或任意 512 cap |
| gamma/beta cast 报错 | 对齐 staging、初始化范围、cast count 与模式一致 |
| broadcast mismatch / fill 异常 | 区分共享标量/逐行统计/GM 参数，核对合法 rank 与真实/填充 rows |
| `StructuralEqual` | 不用 copy 假装跨 rank 值扩展；按支持的 fill/broadcast/视图语义处理 |
| 连续 tile 操作异常 | 检查 alias、pipe hazard 与 ownership，不能只替换 dst |
| 第二遍输入破坏参数 | 核对广播参数最后 consumer、规划重用与同步；不默认追加全局 barrier |
| 尾部随机值 | GM 两侧 valid、UB 幺元、真实 output storage 与签名一致 |
| MIX 重复结果/丢半结果 | 检查 cid/vid owner 与分片，而非机械加/删 guard |

见 [广播](broadcast/guide.md)、[尾块](common/tail_block_design.md) 和 [多行归约](reduce/guide.md)。

## 10. host 拷贝或同步掩盖 kernel 收益

逐项审计 `pad/contiguous/reshape/.to/stack/.cpu/.item/bool(tensor)`；区分 view、设备 copy/cast 和 D2H。非连续 reshape 可能复制但并非必然，trace 与 storage 是证据。输入/输出重排一起设计，后处理尽量驻设备，并保持公共 device/dtype 契约。

特别是 `bool(torch.isnan(x).any())` 包含设备扫描与 host 同步，不能称作零成本编译期特化。见 [Host 数据流](common/host_device_dataflow.md)。

## 11. compare/select 的值正确、index 错误

mask 是 bit-packed；compare 输入对齐按 dtype 字节推导。多次 compare 后 SPR 可能只保存最后 mask，大 tile/保存多 mask 用已验证 tensor 模式。int32 compare 仅 EQ，不默认将 int32 全值域转 fp32；浮点索引也有 2^24 精确范围。

tie、NaN、旧状态与更新顺序显式建表；不能默认 min 传播或所有 compare 的 NaN 结果相同。value 和 index 分开验证。见 [Scan 的类型与 NaN](scan/orthogonal_axis_vectorization.md)。

## 12. dtype 慢路径被错误处理

当前模板的硬件分支与生成调用共同证明路径。bf16 位重排可评估 int16 reinterpret，int8 可评估 UB 内扩宽，int64 保留所有位并比较 record DMA 与块 DMA+UB-local fallback。UB 内 `SetValue` 不等于逐元素 GM DMA；不能只为消除该字符串引入更差路径。

复用 layout kernel 必须证明双尾、多 batch、M/N<tile、每 stage tile/core/dtype；planner 正确不等于 kernel 安全。见 [dtype 适配](conversion/dtype_adaptation.md) 和 [多阶段审计](conversion/multi_stage_contiguous.md)。

## 13. 自动划分被全局误判

纯 Vector + `alloc_var` + AUTO_CV_COMBINE 不一定失败；当前仓已有正向测试。只有生成代码/最小复现证明定义与使用被错误分到 C/V，才定位 [CombineCV](../../../../src/transform/ascend_combinecv.cc) 与共享分类规则。删除配置不等于关闭默认自动划分；诊断时显式 False 必须补齐正确 scope，不能跳过严格验证。

## 14. mask reuse fallback 被当作长期优化

默认失败而 `TL_ASCEND_VECTOR_MASK_REUSE=False` 恢复时，它可能隔离 compiler-managed terminal 的 requires/ensures、路径合流或 opaque call invalidation 缺口。该开关仍使用同一 selected raw terminal，只使每条 terminal 重新建立完整 mask，不等于更快，也不关闭严格 C/V scope 验证。

保留最小复现、生成代码和当前 pass 配置，修复 contract 后重新验证默认路径。说明见 [compiler-managed mask](../../../../docs/ascend/compiler_managed_vector_mask.md)；不能只报禁用后通过并掩盖根因。

## 评审记录

```text
关注项 / 文件函数 / case与dispatch：
当前结构、生成代码或profiling证据：
候选与适用约束：
暂不修改原因（容量/语义/不支持/实测无收益/证据不足）：
已验证范围、原始结果、下一步或回退：
```

发现潜在方向而未实施不等于优化失败；历史测量与本轮结果分开。报告当前实际激活的 kernel，不能用 fallback 或旧轮通过填充候选验证状态。
