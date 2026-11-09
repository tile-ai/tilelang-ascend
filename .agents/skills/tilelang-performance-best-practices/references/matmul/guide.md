# GEMM 优化索引

## 适用条件

适用于 GEMM、Batch Matmul，以及融合算子的 Cube 子路径。

bf16/fp16/fp32 GEMM；block-scaled GEMM 的设计另行核验当前量化接口，不因该索引存在就宣称已有同名模板。先建立正确 Cube 基线，再按瓶颈选择 tile、流水、驻留或 shape 专用调度。

### 根据证据选择专题

| 实现特征或已有证据 | 阅读 | 可形成的候选 |
|---|---|---|
| L0 溢出、M/N/K tile 利用不足、gemm_v0 K 粒度不合适 | [分层 tiling](tiling_design.md) | 联立容量与分形约束，比较可行 tile |
| 搬运与 MMA 串行、stage 复用等待暴露 | [Ping-Pong](pingpong_design.md) | L1/L0 独立槽数与完整 ownership |
| 循环每轮等待 GM 输入，K 方向至少两次搬运 | [MTE2 预取](mte2_preload_design.md) | 首块预取、稳态重叠、尾部排空 |
| 一侧矩阵在对侧多个输出 tile 中重复读取 | [Full-load](fullload_design.md) | 小侧 L1 常驻 |
| scale/bias 等小块频繁搬运 | [随路数据合并](scale_coalescing_design.md) | 合并连续范围，减少 DMA 发射 |
| 任务远多于核数、分配不均或 L2 访问顺序可调整 | [任务映射](task_mapping_design.md) | Fixed Core、persistent、swizzle 对照 |
| 需要理解历史 GEMM 实验的参数与负结果 | [历史案例](historical_cases.md) | 作为候选种子，不推广成硬件定律 |

## TileLang 实现流程

Python factory 枚举 BM/BN/BK、MAD tile、核数和 stage；约束 L0A/L0B/L0C/L1 容量与 shape 可整除性。普通输出 tile 使用当前已验证的 cid 映射、grid-stride 或 T.Persistent。动态或尾块落入正确 fallback。

基础实现参考 [当前 GEMM 示例](../../../../../examples/gemm/example_gemm.py) 与 [显式 MMA 示例](../../../../../examples/gemm/example_gemm_intrinsic.py)：按当前 B 的物理布局设置 transpose_B，使用 T.gemm_v0 或 T.mma；浮点 GEMM 的 L0C 使用 fp32，并且只在第一个有效 K tile 设置 init=True。输出 tile 的核数不超过独立任务数；M/N/K 尾块必须走已验证的 padded-copy 或专用 fallback。

### 先识别控制方式

沿用算子已经选择的 Developer / Expert / 混合模式，依据
[编程模式指南](../../../tilelang-custom-skill/tilelang-programming-model-guide/SKILL.md)
确定 pass 和 scope。`T.gemm_v0`、`T.mma` 是不同控制粒度的接口，不能仅凭 API 名称
替用户切换模式。

- `T.gemm_v0` 接收 L1 A/B 和 L0C C，内部管理 L1→L0 搬运。当前前端公开
  `kL0Size`；无需为一次算子调优修改 `common.h` 的全局默认值。
- `T.mma` 使用显式 L0A/L0B/L0C，适合需要独立安排 L0 位置、槽数与生命周期的设计。
  接口更底层并不证明一定更快。
- 自动同步、内存规划和 C/V 划分分别管理不同问题。需要手动流水时，先检查当前
  同步合同与生成代码；不能把关闭 pass 当成独立性能收益。

当前接口和模板分别见 [ascend.py](../../../../../tilelang/language/ascend.py)
中的 `gemm_v0` / `mma`、[common.h](../../../../../src/tl_templates/ascend/common.h)
中的 `gemm_v0`。后者还包含 N 方向分块和容量断言，不能假定其内部只有固定 K 切片。

## 精度门禁

任何 GEMM 路径只有在当前 TileLang/当前仓库版本完成 lowering、有限值检查和目标容差验证后才能作为优化基线；历史现象若无提交、命令和原始日志，不写入结论。

所有输入 dtype 与输出 dtype 分开测试。覆盖 M/N/K 的 tile±1、长 K 消除误差、正负抵消、大小量混合、0、NaN/Inf 契约。浮点 GEMM 的任何 K 分片的 partial 和最终归约保持 fp32；不得以放宽容差掩盖累加顺序或输出转换错误。

### 适配已有 kernel 前的检查

核对 dtype、transpose_A/B、M/N/K 尾块、batch 索引、L0 累加初始化、workspace 复用、
静态槽数和线程分工。扩大任何调用域时先列出 kernel 尚未满足的条件，再规划修复或
新实现；单个整除大矩阵通过不能证明任意形状可复用。

可从 [基础 GEMM](../../../../../examples/gemm/example_gemm.py)、
[intrinsic GEMM](../../../../../examples/gemm/example_gemm_intrinsic.py)、
[persistent GEMM](../../../../../examples/gemm/example_gemm_intrinsic_persistent.py)
理解结构。后两者保留历史固定核数、固定 `kk == 3` 归还条件及示例级尾块假设，
不是可不经审计直接复制的通用模板。

## 性能门禁

依次比较基础、2/3 stage、full-load、L2 控制和 shape specialization；一次只改变一个变量。

对每个候选运行编译、目标后端的定向精度测试和统一 benchmark。报告 latency、TFLOPS、Cube/MTE2 时间、GM 字节、L1/L0/UB 占用、核利用率和生成代码体积。只有完整算子端到端更快且没有精度回退时进入 dispatch。

## 专题入口

- [分层 K 与容量](tiling_design.md)
- [多级流水](pingpong_design.md)
- [提前发射与预取](mte2_preload_design.md)
- [一侧常驻](fullload_design.md)
- [scale 合并](scale_coalescing_design.md)
- [两阶段 K 分区](streamk_design.md)
- [shape-aware 调度](swat_design.md)
- [任务映射](task_mapping_design.md)
- [历史记录与验证范围](historical_cases.md)
