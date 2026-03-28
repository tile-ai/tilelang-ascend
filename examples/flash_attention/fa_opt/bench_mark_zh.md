**中文** | [English](bench_mark.md)

Flash Attention 是 Transformer 模型中的核心算子，其性能直接影响模型训练和推理效率。

### 4.1 性能测试

输入参数定义：

| 参数 | 取值 | 说明 |
|------|------|------|
| B | 1 | Batch大小 |
| Q_N | 12 | Query序列长度 |
| KV_N | 1 | KV序列数 |
| D | 128 | 隐藏层维度 |
| S | 32K/64K/128K | 序列长度 |
| block_size | 128 | 块大小 |

最佳性能结果：
| S | AscendC | tileLang | 性能百分比（AscendC/tileLang） |
|------|------|------|------|
| 32K | 37555u | 46643u | 80.52% |
| 64K | 149578u | 185188u | 80.77% |
| 128K | 600018u | 741211u | 80.95% |

### 4.2 优化策略及收益分析

针对 FA 算子，我们采用了以下优化组合：

1. **L1 内存常驻**：Q 矩阵在 L1 中持续多个基本块，减少 GM 访问
2. **指令向量化**：将 scalar 操作转换为 tile 操作，减少scalar指令  **--- 原生AscendC算子36%**
3. **多Buffer**：核内流水并行，掩盖数据搬运延迟
4. **核内冗余同步消除**：**--- 原生AscendC算子50%**
5. **T.pipelined 原语**：开启核间 CV 流水，最大化 Cube 和 Vector 核并行度，调整num_stages为8 **--- 原生AscendC算子60%**
6. **优化核间同步下发次数**：每两次任务进行一次核间同步  **--- 原生AscendC算子72%**
7. **减少指令数**：使用axpy代替mul和sub，减少指令下发数  **--- 原生AscendC算子80%**

### 4.3 优化效果

通过系统性优化，FA 算子在保持 TileLang 的高开发效率前提下，达到了 AscendC 原生算子80%的性能，混合编程模式下性能达到原生算子60%。

| 优化项 | L1 内存常驻 | 指令向量化 | 多Buffer | 核内冗余同步消除 | CV pipelined | 优化核间同步下发次数 | 减少指令数 | 性能（A3）|
|------|:----------:|:----------:|:------------:|:-----------:|:------------------:|:---------:|:---------:|:---------:|
| flash_attn_bhsd_expert_h16_d128.py | √ | √ | √ | √ | √ | √ | √ | 80% |
| flash_attn_bhsd_auto_pipeline_h16_d128.py | √ | √ | √ | × | √ | × | √ | 60% |
算子实现：https://github.com/tile-ai/tilelang-ascend/tree/ascendc_pto/examples/flash_attention/fa_opt

| 文件名 | 说明 |
|--------|------|
| flash_attn_bhsd_expert_h16_d128.py | Expert 模式最佳性能实现（80%） |
| flash_attn_bhsd_auto_pipeline_h16_d128.py | 混合编程模式实现（60%） |
| flash_attn_bhsd_ascendc.py | AscendC 原生算子调用脚本 |
| bench_test.sh | 性能精度测试脚本 |