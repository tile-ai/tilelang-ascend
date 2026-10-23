# Persistent 多轴任务映射

## 目标与适用条件

用于可自然拆成多个 tile 轴、且扁平 task ID 的除法、取模或地址准备成为可见开销的任务域。仅在 profiling 或生成代码支持该判断时采用。

## TileLang Ascend 实现

- 当前版本支持时，直接使用 `T.Persistent([tiles_0, tiles_1, ...], ...)` 获取各轴 tile ID，避免先展平再在热循环中用 `//`、`%` 还原坐标。
- `T.Kernel` 的网格与 `(cid, vid)` 映射按当前执行模式核对；多轴列表只描述任务域。轴顺序和 `group_size` 按连续访存、负载均衡选择。当前 `T.Persistent` 不接收 `num_stages`，流水另按已验证的 `T.Pipelined` 或显式 ownership 表达。
- 若多轴 lowering 不可用、无法表达目标遍历顺序或生成代码没有改善，保留扁平映射。

## 验证门禁

验证任务笛卡尔积无遗漏、无重复，并覆盖每个轴的 1、tile±1、尾块、动态范围和空任务。对比生成代码中的整数除法/取模、地址指令、每核任务尾差和同口径 latency。

当前 [Persistent 签名](../../../../../tilelang/language/persistent.py) 要求 domain 各维、wave_size、group_size 为正，`0 <= index < wave_size`，末维可被 `min(group_size, domain[-1])` 整除；空任务在进入 Persistent 前处理，不能将验证覆盖的空任务直接传作零 extent。多轴映射见 [Persistent 测试](../../../../../testing/python/language/test_tilelang_ascend_language_persistent.py)；已有用例不替代目标 kernel 的精度与性能验证。

A2/A3 的 Fixed Core、空任务、workspace 复用与 owner 约束见 [任务映射补充](task_mapping.md)。
