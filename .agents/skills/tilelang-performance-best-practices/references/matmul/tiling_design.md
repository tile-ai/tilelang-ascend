# A2/A3 分层 K tiling 与容量预算

## 适用条件与参数

用于为 `gemm_v0` 或显式 `mma` 形成可行 tile 候选。这里的 K 切分发生在一个核内部，
称为“分层 K tiling”；它不等于跨核计算 K 分片、随后归并输出的 Split-K / Stream-K。

| 参数 | 含义 |
|---|---|
| `block_M`, `block_N` | 一次输出 tile 的行、列 |
| `K_L1` | GM→L1 的 K 范围 |
| `block_K` | 显式 MMA 的 L1→L0 K 粒度 |
| `kL0Size` | gemm_v0 前端提供的内部 K 粒度；不与完整 `K_L1` 混用 |
| `S1`, `S2`, `SC` | L1、L0A/B、L0C 同时保留的物理槽数 |

当前 [前端 gemm_v0](../../../../../tilelang/language/ascend.py) 明确提供
`kL0Size=128`，要求正数、16 的倍数且不超过 4095。后端还有与 dtype 的 C0 和容量
有关的限制，见 [模板](../../../../../src/tl_templates/ascend/common.h)。例如 int8
多 K 分片路径还需要满足 C0 对齐；不能只检查前端 `%16` 即宣布合法。

## 联立容量约束

以实际分配的 padded shape、输入字节数 `d`、累加字节数 `a` 计算：

```text
L0A: S2_A * block_M * block_K * d + 其他活跃 L0A ≤ 可用 L0A
L0B: S2_B * block_K * block_N * d + 其他活跃 L0B ≤ 可用 L0B
L0C: SC   * block_M * block_N * a + 其他活跃 L0C ≤ 可用 L0C
L1 : S1_A * block_M * K_L1 * d
   + S1_B * K_L1 * block_N * d + 常驻/随路数据 ≤ 可用 L1
```

当前 A2/A3 codegen 定义 L0A/B 为 65536 B、L0C 为 131072 B、可分配 L1 为
524032 B，见 [codegen_ascend.cc](../../../../../src/target/codegen_ascend.cc)。标称
512 KiB 不应直接当成可分配 L1；内存规划复用也不能免除真实同时活跃状态的预算。

以 fp16 输入、fp32 累加、单 L0C 为例：`128×256` 恰占 128 KiB L0C；
`block_K=128` 时 B 占 64 KiB，不能再双份。若 A/B 都双缓冲，`block_K=64`
使 B 两槽合计 64 KiB。取 `K_L1=256` 时 A/B 单套共 192 KiB、两套共 384 KiB，
仍须计入其他 L1 数据。此组合是可行候选，不是所有 shape 的最优值。

## 分形、对齐与小形状

旧 A2/A3 案例的分形粒度是 fp16/bf16 `(M,N,K)` 为 `(16,16,16)`，int8 的 K
为 32，fp32 的 K 为 8；具体接口可能施加更强限制（如 gemm_v0 的 `kL0Size%16`）。
布局和搬运须同时满足当前接口的要求，参见
[API 计算参考](../../../tilelang-custom-skill/tilelang-api-best-practices/references/api-compute.md)。
旧容量表的地址对齐参考为 UB/L1 32 B、L0A/B 512 B、L0C 64 B；不要把地址对齐
换算成所有操作都相同的元素数量。

规划候选时同时考虑：

- dtype/累加 dtype 改变时重新计算全部容量和分形；增加任一 stage 时重算。
- tile 变大可能减少可并行任务并扩大尾块浪费；容量打满只是候选边界。
- `K_L1/block_K` 的整除值便于安排内层流水；实际 K 尾块仍需有效 extent 和初始化。
- `K_L1` 更大可摊薄 GM 搬运开销，但扩大 L1 驻留、延长首块等待并减少并发空间。
- gemm_v0 内部 N 分块与 ping-pong 由当前模板决定；评估 `kL0Size` 时检查整个生成路径。

## 可形成的方案与后续验证

先列出有限的可行 `(block_M,block_N,K_L1,block_K/ kL0Size,S1,S2,SC)` 组合，说明
候选改善的是搬运粒度、计算粒度还是任务数。保留原配置作对照。后续执行阶段需验证
全部声明 dtype、transpose 组合和 M/N/K 尾块，并用同口径数据比较收益。
本文件只有源码静态核对，未在本轮编译或上板测量候选。
