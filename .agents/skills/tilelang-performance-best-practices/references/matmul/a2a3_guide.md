# A2/A3 Matmul 性能方案索引

适用于当前仓库 Ascend A2/A3 的 GEMM、Batch Matmul，以及融合算子的 Cube 子路径。
本组文档提供分析依据和候选方案，不负责采集性能数据或实施修改。本次迁移只核对
源码和既有记录，未重新上板验证；源码中的接口存在不等于候选性能收益已验证。

## 根据证据选择专题

| 实现特征或已有证据 | 阅读 | 可形成的候选 |
|---|---|---|
| L0 溢出、M/N/K tile 利用不足、gemm_v0 K 粒度不合适 | [分层 tiling](tiling_design.md) | 联立容量与分形约束，比较可行 tile |
| 搬运与 MMA 串行、stage 复用等待暴露 | [Ping-Pong](pingpong_design.md) | L1/L0 独立槽数与完整 ownership |
| 循环每轮等待 GM 输入，K 方向至少两次搬运 | [MTE2 预取](mte2_preload_design.md) | 首块预取、稳态重叠、尾部排空 |
| 一侧矩阵在对侧多个输出 tile 中重复读取 | [Full-load](fullload_design.md) | 小侧 L1 常驻 |
| scale/bias 等小块频繁搬运 | [随路数据合并](scale_coalescing_design.md) | 合并连续范围，减少 DMA 发射 |
| 任务远多于核数、分配不均或 L2 访问顺序可调整 | [任务映射](task_mapping_design.md) | Fixed Core、persistent、swizzle 对照 |
| 需要理解旧 GEMM 文档的参数与负结果 | [历史案例](historical_cases.md) | 作为候选种子，不推广成硬件定律 |

## 先识别控制方式

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
中的 `gemm_v0`。后者还包含 N 方向分块和容量断言，不能继续用旧版“内部只有固定 K
切片”的描述推导全部行为。

## 适配已有 kernel 前的检查

核对 dtype、transpose_A/B、M/N/K 尾块、batch 索引、L0 累加初始化、workspace 复用、
静态槽数和线程分工。扩大任何调用域时先列出 kernel 尚未满足的条件，再规划修复或
新实现；单个整除大矩阵通过不能证明任意形状可复用。

可从 [基础 GEMM](../../../../../examples/gemm/example_gemm.py)、
[intrinsic GEMM](../../../../../examples/gemm/example_gemm_intrinsic.py)、
[persistent GEMM](../../../../../examples/gemm/example_gemm_intrinsic_persistent.py)
理解结构。后两者保留历史固定核数、固定 `kk == 3` 归还条件及示例级尾块假设，
不是可不经审计直接复制的通用模板。
