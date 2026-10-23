# A2/A3 GEMM 优化索引

> 迁移来源：Ascend950 同路径文档。保留可复用原文结构与门禁，仅改写 A2/A3 平台、后端、API、路径和证据范围。本页是设计/源码参考；本次未进行 NPU 编译、精度或性能验证，不继承950模板的验证状态。

## 适用条件

bf16/fp16/fp32 GEMM；block-scaled GEMM 的设计另行核验当前量化接口，不因该索引存在就宣称 A2/A3 已有同名模板。先建立正确 Cube 基线，再按瓶颈选择 tile、流水、驻留或 shape 专用调度。

## TileLang A2/A3 实现流程

Python factory 枚举 BM/BN/BK、MAD tile、核数和 stage；约束 L0A/L0B/L0C/L1 容量与 shape 可整除性。普通输出 tile 使用当前已验证的 cid 映射、grid-stride 或 T.Persistent；不引入源仓 T.AscendTileScheduler。动态或尾块落入正确 fallback。

基础实现参考 [当前 GEMM 示例](../../../../../examples/gemm/example_gemm.py) 与 [显式 MMA 示例](../../../../../examples/gemm/example_gemm_intrinsic.py)：按当前 B 的物理布局设置 transpose_B，使用 T.gemm_v0 或 T.mma；浮点 GEMM 的 L0C 使用 fp32，并且只在第一个有效 K tile 设置 init=True。输出 tile 的核数不超过独立任务数；M/N/K 尾块必须走已验证的 padded-copy 或专用 fallback。

## 精度门禁

任何 GEMM 路径只有在当前 TileLang/当前仓库版本完成 lowering、有限值检查和目标容差验证后才能作为优化基线；历史现象若无提交、命令和原始日志，不写入结论。

所有输入 dtype 与输出 dtype 分开测试。覆盖 M/N/K 的 tile±1、长 K 消除误差、正负抵消、大小量混合、0、NaN/Inf 契约。浮点 GEMM 的任何 K 分片的 partial 和最终归约保持 fp32；不得以放宽容差掩盖累加顺序或输出转换错误。

## 性能门禁

依次比较基础、2/3 stage、full-load、L2 控制和 shape specialization；一次只改变一个变量。

对每个候选运行编译、目标 A2/A3 后端的定向精度测试和统一 benchmark。报告 latency、TFLOPS、Cube/MTE2 时间、GM 字节、L1/L0/UB 占用、核利用率和生成代码体积。只有完整算子端到端更快且没有精度回退时进入 dispatch。

## A2/A3 专属细节与既有纠错

完整保留上一版 A2/A3 内容于 [A2/A3 补充](a2a3_guide.md)，包括当前接口、边界、历史记录及验证限制。

## 专题入口

- [A2/A3 分层 K 与容量](tiling_design.md)
- [多级流水](pingpong_design.md)
- [提前发射与预取](mte2_preload_design.md)
- [一侧常驻](fullload_design.md)
- [scale 合并](scale_coalescing_design.md)
- [两阶段 K 分区](streamk_design.md)
- [shape-aware 调度](swat_design.md)
- [A2/A3 任务映射](task_mapping_design.md)
- [A2/A3 历史记录与验证范围](historical_cases.md)
