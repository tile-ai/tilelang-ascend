**English** | [中文](benchmark_zh.md)

Flash Attention is a core operator in Transformer models, and its performance directly impacts model training and inference efficiency.

### 4.1 Performance Testing

Input parameter definitions:

| Parameter | Value | Description |
|-----------|-------|-------------|
| B | 1 | Batch size |
| Q_N | 12 | Query sequence length |
| KV_N | 1 | KV sequence count |
| D | 128 | Hidden dimension |
| S | 32K/64K/128K | Sequence length |
| block_size | 128 | Block size |

Best performance results:
| S | AscendC | tileLang | Performance Ratio (AscendC/tileLang) |
|------|------|------|------|
| 32K | 37555u | 46643u | 80.52% |
| 64K | 149578u | 185188u | 80.77% |
| 128K | 600018u | 741211u | 80.95% |

### 4.2 Optimization Strategies and Impact Analysis

For the FA operator, we adopted the following optimization combinations:

1. **L1 Memory Residency**: Q matrix stays in L1 across multiple basic blocks, reducing GM access
2. **Instruction Vectorization**: Convert scalar operations to tile operations, reducing scalar instructions  **--- Native AscendC 36%**
3. **Multi-Buffer**: Intra-core pipeline parallelism to hide data transfer latency
4. **Intra-core Redundant Synchronization Elimination**: **--- Native AscendC 50%**
5. **T.pipelined Primitive**: Enable inter-core CV pipeline to maximize Cube and Vector core parallelism, with num_stages set to 8 **--- Native AscendC 60%**
6. **Optimized Inter-core Synchronization Frequency**: Perform inter-core synchronization every two tasks  **--- Native AscendC 72%**
7. **Reduced Instruction Count**: Use axpy instead of mul and sub to reduce instruction dispatch count  **--- Native AscendC 80%**

### 4.3 Optimization Results

Through systematic optimization, the FA operator achieves 80% of native AscendC performance while maintaining TileLang's high development efficiency. In hybrid programming mode, it reaches 60% of native operator performance.

| Optimization | L1 Residency | Vectorization | Multi-Buffer | Sync Elimination | CV pipelined | Optimized Sync Frequency | Reduced Instructions | Performance (A3)|
|------|:----------:|:----------:|:------------:|:-----------:|:------------------:|:---------:|:---------:|:---------:|
| `expert_v1/kernel.py` | √ | √ | √ | √ | √ | √ | √ | 80% |
| `auto_pipeline/h16_d128.py` | √ | √ | √ | × | √ | × | √ | 60% |

Operator implementation: https://github.com/tile-ai/tilelang-ascend/tree/ascendc_pto/examples/flash_attention/fa_opt

| File Name | Description |
|--------|------|
| `expert_v1/kernel.py` | Expert-mode implementation measured in the historical table (80%) |
| `expert_v2/kernel.py` | Three-stage online-softmax Expert implementation; see its colocated README |
| `auto_pipeline/h16_d128.py` | Hybrid programming mode implementation (60%) |
| `reference/ascendc.py` | AscendC native operator invocation script |
| bench_test.sh | Performance and accuracy test script |
