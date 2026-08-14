# Sigmoid Stage 3 Performance Tuning Log

## Iteration 1 — 2026-08-03T15:50:00Z

- bottleneck_type: transfer + sync (memory-bound element-wise, host launch overhead dominates)
- optimization: [#1] 关闭 AUTO_CV_COMBINE（纯 Vector 算子消除空 AIC）+ [#3] Fixed Core（launch min(block_num,24) 核 + T.serial 每核串行处理 ceildiv(block_num,launch_cores) 个 tile，striped 分配）
- baseline_time: 0.2147 ms (bench fp16 1024×8192) / 71.9 us (msprof NPU task duration)
- candidate_time: 0.2153 ms (bench fp16 1024×8192) / 53.6 us (msprof NPU task duration)
- improvement: +0.3% (bench 端到端，噪声范围) / -25.5% (msprof NPU kernel task duration)
- precision: pass (L0/L1/L2/Boundary 全过，max_abs=4.883e-04 fp16 / 0.0 fp32)
- adopted: yes (基于 msprof kernel 级提升 25.5% > 3% 阈值；bench 端到端无提升是 host runtime 开销掩盖)
- rollback_reason: N/A
- next_hint: 下轮可试 [#4] Vector Double Buffer + 关闭 AUTO_SYNC 让 T.serial 循环内 MTE2/V/MTE3 流水重叠；或 [#5] 增大 tile size (128,512) 减少 tile 数。host 侧 runtime 开销 (~160 us) 是当前 bench 端到端瓶颈，超出 kernel 优化范围。

### 详细数据

| 指标 | baseline (iter1 start) | candidate (iter1 end) | 变化 |
|------|----------------------|----------------------|------|
| bench fp16 (1024,8192) median | 0.2147 ms | 0.2153 ms | +0.3% |
| bench fp32 (512,512) median | 0.1895 ms | 0.1923 ms | +1.5% |
| msprof NPU task duration (fp16) | 71.9 us | 53.6 us | -25.5% |
| msprof Block Dim | 512 | 24 | -95.3% |
| msprof Mix Block Dim | 1024 | 48 | -95.3% |
| torch.sigmoid bench (fp16) | 0.0556 ms | 0.0545 ms | — |
| speedup vs torch.sigmoid (bench) | 0.259x | 0.253x | — |
| speedup vs torch.sigmoid (msprof NPU) | ~0.77x | ~1.02x | NPU 侧已接近持平 |

### 瓶颈诊断 (msprof iter1 baseline)
- Op Type: mix (KERNEL_TYPE_MIX_AIC_1_2)
- aicore compute usage < 20% (空 AIC，Cube utilization ~0.5%)
- aivector MTE2/MTE3 bandwidth < 80% when active
- Vector ratio ~0.94% (极低，单 block 仅 ~2 us 计算)
- 每 block: MTE2 ~0.4-1.3 us, V ~0.67 us, MTE3 ~0.2-0.7 us
- host 开销 ~143 us (bench 214.7 - msprof 71.9) = tilelang runtime launch + synchronize

### 瓶颈诊断 (msprof iter1 candidate)
- Block Dim: 24 (Fixed Core, 每核 T.serial 处理 ~22 tile)
- NPU task duration: 53.6 us (-25.5%)
- host 开销 ~162 us (bench 215.3 - msprof 53.6) = tilelang runtime launch + synchronize
- 结论: NPU kernel 已优化 25.5%，但 host runtime 开销 dominates bench 端到端时间
