# Lightning Indexer 算子（TileLang-Ascend 实现）

**中文**

Lightning Indexer 是 NSA（Native Sparse Attention）稀疏注意力中的索引算子：对 GQA 加权打分矩阵在 S2 维做在线 TopK，输出稀疏注意力位置索引。本目录提供基于 TileLang DSL 的 Expert 模式高性能实现（单 MIX kernel，持久化 + C/V 流水线 + 并行归并），在 A3（Ascend910，24 Cube 核，CANN 9.1.0-beta.1）上相对初版 TileLang 实现累计提速 **64.5x**（42,231us → 654us，B2/S1=512/S2=4096/D=128/K=1024）。

## 目录结构

```text
examples/lightning_indexer/improved/
└── example_lightning_indexer_dynamic_shape_improved.py   # kernel + 精度测试套件（本目录自包含）
```

## 运行方式

假设已完成 [tilelang-ascend 安装](../../../README.md#tilelang-ascend-installation) 并 source 环境（`source set_env.sh`）。

```bash
# 快速精度验证（小形状）
python example_lightning_indexer_dynamic_shape_improved.py --quick

# 全量精度套件：18 用例（B/S1/S2/K/BN/D 网格 + 边界），门限：索引多重集合匹配率 > 0.99
python example_lightning_indexer_dynamic_shape_improved.py --suite

# 性能采集（msprof，单次 kernel 启动，warm-up / launch-count 控制采样）
msprof op --application="python <bench 目标脚本>" \
    --aic-metrics=PipeUtilization --kernel-name=main_kernel \
    --launch-count=5 --warm-up=3 --output=<输出目录>
```

正确性 oracle 为 `index_golden`（PyTorch einsum + relu + 加权 + topk）。

## 输入参数定义

| 参数 | 取值 | 说明 |
|------|------|------|
| B | 1~4（测试过） | Batch 大小 |
| S1 | 64~1024（%64==0） | Query 序列长度 |
| S2 | 512~8192（%BN==0） | Key/Value 序列长度 |
| N2 | 1 | GQA 组数 |
| G | 32（[8,248] 且 %8==0） | 注意力头组数 |
| D | 64 / 128（==BLOCK_K） | 头维度 |
| K (TOP_K) | 256~2048（≤S2） | 输出索引数 |
| BLOCK_N | 64/128/256/512 | trunk 列块（512 走双 256 列子 GEMM） |
| S2_SPLITS | auto / 手动 | S2 切分数（成本模型 v4 自动选择） |

输入 `Query[B,S1,N2,G*D]` fp16、`KEY[B,S2,N2,D]` fp16、`WEIGHTS[B,S1,N2,G]` fp32；输出 `OUT[B,N2,S1,K]` int32。

## 性能测试

测试方法：msprof `task duration`（设备侧主 kernel 时长），A3 真机，warm-up=3 / launch-count=5；
baseline 为 ACLNN 算子。
### 版本演进（B2/S1=512/S2=4096/D=128/K=1024）

| 版本 | us | 加速（vs 初版）|
|------|----|------|
| 初版 TileLang | 42,231 | 1.0x |
| P1（在线 TopK + BN512 + 双 AIV）| 1,127 | 37.5x |
| P2a+b（单 G DMA + 首 trunk 跳 merge）| 946 | 44.6x |
| P2e（Phase 拆分 + 4-way 归并）| 837 | 50.5x |
| P3b（持久化 + S2 切分 + 并行 Phase 2）| 676 | 62.5x |
| P4（列块 fused 广播乘）| 675 | 62.6x |
| P5（trunk 截断 + 跨行预取）| **654** | **64.5x** |

### 全配置结果（G=32/D=128 除注明）

| 配置 | splits | baseline (us) | P5 (us) | vs baseline |
|---|---|---|---|---|
| B2/S1=512/S2=512/K=512 | 1 | 95.4 | 149.9 | 0.64x |
| B2/S1=512/S2=1024/K=1024 | 1 | 141.5 | 248.2 | 0.57x |
| B2/S1=512/S2=2048/K=1024 | 1 | 231.4 | 430.9 | 0.54x |
| B2/S1=512/S2=4096/K=1024 | 4 | 407.5 | 654.2 | 0.62x |
| B2/S1=512/S2=4096/K=1024/D=64 | 4 | 8.4* | 546.0 | —* |
| B4/S1=512/S2=4096/K=1024 | 1 | 783.9 | 1,602.0 | 0.49x |
| B1/S1=512/S2=4096/K=1024 | 8 | 224.6 | 455.5 | 0.49x |
| B1/S1=1024/S2=8192/K=1024 | 4 | 766.9 | 1,252.6 | 0.61x |

\* D=64 的 baseline 8.4us 为历史采集异常值（ACLNN 路径疑与 G=32/D=64 形状不匹配），不参与比值。

### 收益分布

- **V-bound 配置收益最大**（−9%~−16%）：短序列（splits=1）、B1/splits=8、D=64——Phase 1
  V 循环在关键路径占比高，P4/P5 直接见效
- **cube-bound / 大规模配置收益小**（−1%~−4%）：B1/S1=1024/S2=8192 等——GEMM 与归并主导
- **管线证据**（B2/S2=4096）：aiv_scalar_ratio 28.8%→14.2%，aiv_vec_ratio 55.2%→51.6%

## 优化策略及收益分析

针对本算子，按演进顺序采用以下优化组合（累积收益以 B2 主配置计）：

1. **在线 TopK**：放弃全量 `QK_RES`（512MB GM）落盘后 topk，改为逐 trunk 排序 +
   归并（TopK(A∪B,K)=TopK(TopK(A,K)∪TopK(B,K))），配合 BN=512 与双 AIV 分行
   **--- 37.5x**
2. **单 G DMA**：G 循环内 QK 逐头搬运改为 trunk 级 2D 连续大块拷贝；首 trunk 直载
   history 跳过一次 merge **--- 44.6x**
3. **Phase 拆分 + 4-way 归并**：Phase 1 只算+排序+存储（trunk 主序），Phase 2 行主序
   批量 4-way MrgSort（history 常驻 UB）**--- 50.5x**
4. **持久化 kernel + S2 切分 + 并行 Phase 2**：grid 固定为核数、双缓冲 slot 信用协议
   （READY/FREE 跨核 flag）、成本模型 v4 自动选切分、barrier 后全部 AIV 均分行归并
   **--- 62.5x**
5. **列块 fused 广播乘**：原 G 循环每迭代 `PipeBarrier + GetValue`（V↔S 双 flag 往返）
   + 4 条 MOVEMASK（simulator 指令 trace：占 VECTOR 管线 56%），改为 256B 列块
   `brcb + mul_mask` 一次覆盖全部 G 行（aiv_scalar_ratio 28.8%→14.2%）
6. **trunk 截断存储**：每 trunk 只存 top `min(K, BN)` 对（数学等价），K<BN 形状 GM
   读写、MrgSort 源、-inf fill 等比缩短；同时修复 BN>K 时 Phase 2 越界写 UB 的隐患
7. **跨行 MTE2 预取**：行尾（sort/axpy 已发射后）发起下一行 64KB 拷贝，MTE2 与
   sort 执行 + MTE3 store 重叠（splits>1 配置 −2.7%）**--- 64.5x**

## 关键设计决策

### 整体架构

```text
Host 侧（JIT 特化参数）：
  BLOCK_N ∈ {64,128,256,512}（512 走双 256 列子 GEMM）
  S2_SPLITS：auto（成本模型 v4）或手动覆盖
  GRID_SIZE = min(B×N2×S1_TILES×S2_SPLITS, cube_core_num)

Device 侧（单 MIX kernel，Expert 模式，MIX_AIC_1_2 = 1 AIC + 2 AIV/block）：

  for wi in ITEMS_PER_BLOCK:                    ← 持久化：每 block 串行工作项
    C: K/Q GM→L1 → gemm_v0 → L0C → QK_SLOT[cid] (ReLU)  ─┐ 双缓冲 slot
       READY(slot) ──────────────────────────────────────┤ mode=2 跨核
    V: wait READY → 首行预取 → [wait 拷贝 →                │ READY/FREE
        row_expand_mul 列块乘 + G 归约 → sort → axpy       │ 信用协议
        → 存 top min(K,BN) 对 → 行尾预取下一行]×32         ┘
       FREE(slot)
  drain 本地事件（set/wait 严格配平，trunk 末补收尾 wait）
  if S2_SPLITS > 1: sync_all()

  Phase 2（并行行均分）：
    所有 GRID_SIZE×2 个 AIV 平分 TASK_COUNT×BLOCK_M 行
    每行：读全部 trunk 的 top-min(K,BN) 对 → 每 3 trunk 一批 4-way merge_sort → 输出 OUT
```

数据路径：`GM Query/Key → L1 → L0C → GM QK_SLOT → UB → 加权归约 → sort → GM SORTED_WS`
（A3 无 L0C→UB Fixpipe 直通）。`QK_SLOT[GRID,2,BM,G,BN]` 双缓冲中转；`SORTED_WORKSPACE
[TASK,BM,TRUNKS,2×min(K,BN)]` 跨 block 共享（splits>1 时因此需要 sync_all）。

### 同步协议（Expert 模式，自动化全关）

- **本地事件**（set_flag/wait_flag）：ID 值域 0..7，按有向管线对独立分配（FIX↔M、
  MTE2↔M、V↔MTE2、V↔MTE3 各自独立命名空间）；**set/wait 数量严格相等（含终端
  surplus token）**，不配对 = 507014 超时
- **跨核 flag**（set_cross_flag/wait_cross_flag）：mode=2（同组 AIC↔AIV），READY=0/1、
  FREE=2/3；FFTS 信号量语义，surplus credit 无害，但**切勿在 set 方对自己的 flag 加 wait**
- **全核 barrier**（sync_all）：仅 splits>1 时使用；前提 GRID ≤ 核数且每核 1 block
- **P5 预取顺序敏感**：阻塞 `wait(V→MTE2)` 必须在 sort/axpy 发射之后（否则 V 管线在
  flag 唤醒期空转，实测 +14us 回退）；trunk 末补收尾 wait 保持事件逐 trunk 配平

### UB 布局与上界（A2/A3 UB = 196,352B）

- Phase 2 显式缓冲 ~102KB + 隐藏 tmp 共 ~146KB
- **K 上界**：TOP_K ≤ 1536（BN=512）/ 1920（BN=256）/ 2048（BN=128）
- **为什么列块乘而非 (G,BASEN) 广播**：weight_2d 需 64KB（BN=512），叠加 K≥512 形状
  的 Phase 2 缓冲后超 196,352B → 运行期越界 → aicore exception 507015 挂死；列块方案
  额外 UB 仅 G×8 fp32（1KB）

### 其他

- **双 AIV 分工**：Phase 1 每个 AIV 处理 `VID_ROWS = BLOCK_M/2` 行；Phase 2 按
  `aiv_flat = cid*2 + vid` 扁平化行均分
- **BLOCK_N=512 双 GEMM 分裂**：`gemm_v0` 单次 L0B 64KB 限制，拆两个 256 列子 GEMM
  （k_l1_a/k_l1_b 双 K 缓冲），L0C 单缓冲串行复用；trunk 数减半 = MergeSort 调用减半
- **auto S2 切分（成本模型 v4）**：`cost(sp) = items×trunks + rows_per_aiv×merge_calls
  + barrier`，门槛 crit 降幅 ≥2 单位、items ≤4、整除 S2；典型：B1/S2=4096→8、
  B2/S2=4096→4、B2/S2≤2048→1、B4→1

## 支持域与约束

| 约束 | 说明 |
|------|------|
| S1 % 64 == 0 | BLOCK_M=64，未实现尾块 |
| S2 % (S2_SPLITS × BLOCK_N) == 0 | 整除（auto 会自动降级）|
| G % 16 == 0；G ∈ [8,248] 且 %8 | VECTOR_BASEG / 列块乘 brcb 行数域 |
| VECTOR_BASEN % 64 == 0 | 256B 列块（fp32）|
| calc_dtype == float | 列块乘仅 fp32（mul_mask repeat ≤255 亦由此满足）|
| D == BLOCK_K | 已验证 D=64/128 |
| TOP_K ≤ S2 且 ≤ {1536,1920,2048} | 按 BN ∈ {512,256,128} 的 UB 上界 |


