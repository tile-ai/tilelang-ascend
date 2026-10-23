# TileLang A2/A3 热路径编码原则

1. 用 Python factory 生成模式专用 T.prim_func，消除运行时 mode 分支。
2. 用局部标量缓存重复的 task offset、tile id 和 stride 运算。
3. 把主循环与尾块拆开，主循环保持静态完整 tile。
4. 只有固定小循环使用 T.unroll；其余使用 T.serial 或 T.Pipelined。
5. 缩短 fragment、局部向量和标量的活跃范围，降低 spill。
6. 连续规则计算使用目标仓库已实现的 T.tile.* 或 T.Parallel；离散索引、复杂分支和 reduction 使用目标仓库同类实现，核对生成的 Ascend C。
7. 常驻 buffer 单版本，流水 tile 按 T.Pipelined 的当前编译行为或显式 stage buffer 管理多版本。
8. 64 位总偏移不得为了省指令截断；只有证明范围后才降位。
9. 每次结构修改检查生成源码、精度和端到端延迟。
