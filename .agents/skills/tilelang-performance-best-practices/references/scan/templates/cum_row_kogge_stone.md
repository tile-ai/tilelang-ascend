# Row Kogge-Stone Scan

> 迁移来源：Ascend950 同路径文档。保留可复用原文结构与门禁，仅改写 A2/A3 平台、后端、API、路径和证据范围。本页是设计/源码参考；本次未进行 NPU 编译、精度或性能验证，不继承950模板的验证状态。

## 适用条件

扫描轴静态且 tile 可驻留 UB；更重视低逻辑深度，能够接受每层较多 combine 和寄存器流量。

## 算法

第 `k` 层距离为 `d=2^k`，所有 `i>=d` 的元素并行执行：

```text
next[i] = combine(prev[i - d], prev[i])
```

每层必须从上一层快照读取，不能原地让本层新值污染后续 lane。共执行 `ceil(log2(R))` 层；非整尾部只更新真实元素。上面的公式给出每层依赖边；源仓 `level_pairs` helper 不作为本仓已实现 kernel。

跨 tile carry 与 Sklansky 相同，由唯一 row owner 保持浮点 fp32 状态；整数值/索引保持精确 dtype。若当前 A2/A3 向量路径无法安全表达层间快照或产生寄存器 spill，回退到 streaming correctness baseline。

## 精度与性能门禁

覆盖 `R=1`、每个二次幂边界、tile±1、多 tile carry、抵消、大小量和 dtype 溢出契约。与 Sklansky 比较 latency、SIMD 指令数、临时 UB/寄存器和代码体积。

## 实现状态

本文件保留算法流程和验收门禁，状态为 `DESIGN_ONLY`；链接的 A2/A3 示例只提供局部结构依据，不证明本策略已有端到端 kernel。源仓 dav310 Python 模板没有复制到本库。
