# 索引短记录变换

适用于大量独立短记录上的 gather/scatter、仿射组合及参数梯度归约。

**证据等级：源平台历史结构参考。** 源资料记录下列结构已在真实 Ascend kernel 中落地，并有编译运行、精度和同口径性能对比证据；该等级只表示候选可信度，不构成 A2/A3 当前实现或性能证明，不替代当前瓶颈排序和目标 case 复验。

## 优化结构

- 沿外层维打包多个记录，使用矩形 DMA 和当前已验证的 Vector/UB-local 重排，避免逐记录小访问。
- 共享同一短记录、有效区和索引族的相邻字段，优先在一次 SIMD traversal 中联合处理；拆成多轮热循环时须核对重复索引、load/store 和循环成本。
- 将索引、常量和循环不变量移出 Persistent 热循环，并从生成代码确认没有按 tile 重算。
- 前序阶段已物化可恢复导数或中间表达式的值时，比较新增读取与删除重算；结合当前记录搬运、任务映射和生命周期完成 lowering 与实测，不能只按输入字节或算术量取舍。
- 参数梯度等跨 tile 状态分别核对当前 API 可表达的分组归约与 UB 索引重排 + 向量归约；源平台 register permutation 不作为 A2/A3 已有 API；按 accumulator 物理布局、循环后指令、精度和实测选择。
- 消费者、归约轴和生命周期兼容的 partial，在 Persistent 后统一完成 lane reduction，并打包为一个连续 partial record；核对是否真正减少为一个 GM buffer、一次 copy 和一次对应维度的 host reduction。
- 索引外提、中间值复用、accumulator 布局、归约 lowering、partial 布局/copy/host reduction 数和 launcher JIT 配置分别作为候选核对；实现其中一项不能批量标记其余项完成或无效。
- 仅对确需跨迭代重叠的流式 buffer 配置多版本；累积状态及单生命周期数据保持单版本，版本组合逐项 A/B。
- 静态整 tile 候选须联合核对编译期 extent、尾块 mask/分支消除、索引生成、tile 和核数；用受控 A/B 或小规模组合搜索验证，负向结果只否决已测组合，其他 shape 保留通用尾块路径。
- 流水、fast-math 和大 tile 指令必须分别通过 lowering、精度及同口径性能验证。

## 适用性判断

| 优化候选 | 适用条件 | 重点核对 |
|---|---|---|
| 多记录合并搬运与 SIMD | 单条记录短、访问规律，逐记录调度或小 DMA 占比明显 | UB 容量、尾块有效区、索引映射与实际 DMA 数 |
| 共享字段热循环融合 | 多个字段共享记录、有效区或索引族 | traversal 数、重复索引和 load/store，live state 与 lowering |
| 循环不变量外提 | 索引或常量在热循环中不变 | 生成位置、热循环指令和每 tile 重算次数 |
| 中间值复用 | 相邻阶段已物化可恢复当前表达式的值 | 接口与生命周期、新增读取、删除重算及组合后的 lowering |
| accumulator 与归约 lowering | 跨 tile 累加状态短且字段分组固定 | lane 布局、热循环重排、循环后指令、精度与实测 |
| 合并 partial | 多个 partial 的消费者、归约轴和生命周期兼容 | 连续 partial 布局、GM buffer/copy 与 host reduction 数 |
| 静态整 tile 与 JIT tiling | 部分输入可证明无尾块，且不同规模需要不同 tile/core 参数 | extent、尾块、索引 lowering、tile/core 组合及通用尾块路径 |
| 多级流水 | 热循环存在可重叠的搬运和计算，且迭代数足以摊薄流水开销 | buffer version、UB 占用、lowering 和同口径 A/B 性能 |

候选按当前瓶颈独立排序。只有物理布局、lowering、精度和同口径性能均验证有效，才保留对应优化。

当前指令表达、AXPY 等价性及短记录 partial 合并的 A2/A3 补充见 [Elementwise 指南](guide.md)。
