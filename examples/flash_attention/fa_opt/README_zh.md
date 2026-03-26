# CV融合算子性能分析与调优方法

## 目录

- [一、性能分析工具](#一性能分析工具)
- [二、性能优化总体思路](#二性能优化总体思路)
- [三、核内优化](#三核内优化)
- [四、核间优化（CV融合）](#四核间优化cv融合)
- [五、TileLang 性能优化原语](#五tilelang-性能优化原语)
- [六、常见性能问题与解决方案](#六常见性能问题与解决方案)

---

## 一、性能分析工具

### 1.1 msprof 工具概述

CANN 提供内置性能分析工具 `msprof`，支持两种模式：

| 模式 | 用途 | 输出 |
|------|------|------|
| **op 模式** | 采集真实硬件性能数据 | kernel 耗时、带宽利用率等 |
| **simulator 模式** | 采集流水线仿真数据 | 可视化流水图 |

### 1.2 采集性能数据

```bash
msprof op --kernel-name="main_kernel" --output=<输出路径> python3 xxx.py
```

**示例脚本**：
- [性能对比批量测试脚本](bench.sh)

### 1.3 采集流水图（Simulator 模式）

```bash
msprof op simulator --soc-version=Ascend910B4 --kernel-name="main_kernel" --output=<输出路径> python3 xxx.py
```

**注意**：
- `--soc-version` 需替换为实际 NPU 型号（如 Ascend910B1、Ascend910B4 等）
- 可通过工具提示获取支持的型号列表

**流水图查看方式**：
1. Chrome 浏览器访问 `chrome://tracing/`，加载输出文件
2. 下载 [MindStudio Insight](https://gitcode.com/Ascend/msinsight/releases/tag_MindStudio_26.0.0-alpha.1) 工具

---

## 二、性能优化总体思路
以FA算子为例，介绍CV融合类算子性能优化的总体思路。

### 2.1 优化层次架构

```
┌─────────────────────────────────────────────┐
│              算子性能优化                     │
├──────────────────┬──────────────────────────┤
│    核间优化      │       核内优化            │
│  (Inter-Core)    │     (Intra-Core)         │
├──────────────────┼──────────────────────────┤
│  · num_stages    │  Cube核: L1常驻、DB      │
│  · 任务均衡      │  Vector核: MTE2/VEC/MTE3 │
│  · 同步优化      │  · Double Buffer         │
│                  │  · 指令向量化             │
└──────────────────┴──────────────────────────┘
```

### 2.2 优化流程

```
1. 性能基准测试 → 2. 识别瓶颈 → 3. 针对性优化 → 4. 验证收益
        ↑                                              ↓
        └──────────── 迭代优化 ←───────────────────────┘
```

### 2.3 核心原则

| 原则 | 说明 |
|------|------|
| **掩盖短流水** | 将耗时短的流水尽量用耗时长的流水掩盖 |
| **减少气泡** | 优化任务排布，减少核间等待时间 |
| **单一 Bound** | 理想情况下优化至单一类型流水 bound（如 fixPipe bound） |

---

## 三、核内优化

### 3.1 Double Buffer 原理

**背景**：AI Core 的指令队列相互独立、可并行执行：
- **MTE 队列**：内存搬运指令
- **Vector 队列**：向量计算指令
- **Cube 队列**：矩阵计算指令

**示例**：Vector 核执行顺序 `MTE2 → VEC → MTE3`

| 模式 | 执行方式 | 特点 |
|------|----------|------|
| **串行模式** | 数据块依次执行 | 单块耗时长，存在等待 |
| **Double Buffer** | 数据块切分，流水并行 | 总耗时降低，资源利用率高 |

```
串行模式:
  Block0: [MTE2][VEC][MTE3]
  Block1:        ----------[MTE2][VEC][MTE3]
  
Double Buffer:
  Block0: [MTE2][VEC][MTE3]
  Block1:   [MTE2][VEC][MTE3]
```

### 3.2 Cube 核优化

#### 3.2.1 L1 内存常驻

减少 GM 与 L1 之间的数据搬运次数。

| 策略 | 适用场景 | 实现方式 |
|------|----------|----------|
| **大复用** | L1 内存充足 | Q 在 L1 中持续多个基本块，P@V 时不释放 |
| **小复用** | L1 内存紧张 | Q 在 L1 中持续一个基本块，P@V 时释放 |

**代码示例（大复用）**：
```python
T.copy(Q[bz, by, bx * block_M:(bx + 1) * block_M, :], q_l1)
for k in T.Pipelined(T.ceildiv(seq_len, block_N), num_stages=num_stages):
    for n_i in T.serial(n_num):
        T.copy(K[bz, by, k * block_N + n_i * block_K : k * block_N + (n_i + 1) * block_K, :], k_l1)
        T.gemm_v0(q_l1, k_l1, acc_s_l0c, transpose_B=True, init=True)
        T.copy(acc_s_l0c, workspace_1[cid, :, n_i * block_K : (n_i + 1) * block_K])
```

#### 3.2.2 L1 → L0 Double Buffer

当 L0 空间小于 L1 时，需分多次搬运，轮次间做 overlap。

#### 3.2.3 优化到单一 Bound

**目标**：将耗时最长的流水作为 bound，其余流水被其掩盖。

```
优化前: MTE  ████████
       M     ████████████
       FIX   ████████████████████  ← 耗时最长

优化后: MTE   ████████████████████
       M     ████████████████████
       FIX   ████████████████████  ← 全部被 FIX 掩盖
```

### 3.3 Vector 核优化

#### 3.3.1 核内 Double Buffer

不同数据块间 MTE2、VECTOR、MTE3 互相掩盖。

#### 3.3.2 算法优化

**（1）Scalar 向量化**

将 for 循环下的多次 scalar 运算改造为 tile 操作：

```python
# 优化前：循环中多次 scalar 运算
for h_i in range(block_M // 2):
    T.tile.sub(acc_s_ub[h_i, :], acc_s_ub[h_i, :], m_i[h_i])

# 优化后：单次 tile 操作
T.tile.broadcast(m_i_2d, m_i, tmp_ub)
T.tile.sub(acc_s_ub, acc_s_ub, m_i_2d)
```

**（2）减少指令下发次数**

使用 Axpy 算法合并指令：

```python
# 优化前：两条指令
T.tile.mul(acc_s_ub, acc_s_ub, sm_scale)
T.tile.sub(acc_s_ub, acc_s_ub, m_i_2d)

# 优化后：一条指令（如适用）
# 结合实际场景使用 Axpy 等融合指令
```

---

## 四、核间优化（CV融合）

### 4.1 找到最佳 num_stages

**原理**：块数越多，CV 任务执行时间不均匀时，调大 `num_stages` 可减少气泡。

```
num_stages = 2:
  C核: [C0]----[C1]--------[C2]
  V核:      [V0]----[V1]--------[V2]
              ↑ C1 等待 V0 完成

num_stages = 3:
  C核: [C0][C1]----[C2]----[C3]
  V核:     [V0][V1]----[V2]----[V3]
              ↑ 气泡减小
```

**调优建议**：
- 从 `num_stages=2` 开始，逐步增加
- 观察 C/V 核耗时比例，选择使气泡最小的值
- 注意 `num_stages` 过大会增加内存占用

### 4.2 核间同步优化

**问题**：核间同步过多可能导致 scalar bound。

**解决**：减少同步次数，如每两次任务同步一次：

```python
# 优化前：每次任务都同步
for i in range(n):
    process()
    sync()

# 优化后：多次任务后同步
for i in range(n):
    process()
    if i % 2 == 1:
        sync()
```

---

## 五、TileLang 性能优化原语

### 5.1 T.pipelined 原语

用于开启核内或核间流水掩盖。

**语法**：
```python
for i in T.pipelined(loop_range, num_stages=N):
    # 任务处理
```

**参数说明**：
| 参数 | 说明 |
|------|------|
| `loop_range` | 循环范围 |
| `num_stages` | 流水级数，控制任务并行度 |

**详细文档**：[T.pipelined 使用教程](https://github.com/tile-ai/tilelang-ascend/blob/ascendc_pto/docs/tutorials/t_pipelined.md)

### 5.2 使用限制

**重要**：`T.pipelined` 不支持嵌套使用。

```python
# 方式一：开启核间流水（推荐）
for i in T.pipelined(loop, num_stages):
    process_on_cube()
    process_on_vector()

# 方式二：分别开启核间/核内流水
for i in T.pipelined(loop, num_stages):  # 核间
    process_on_cube()
for i in T.pipelined(loop, num_stages):  # 核内
    process_on_vector()

# 错误：嵌套使用
for i in T.pipelined(outer, num_stages):
    for j in T.pipelined(inner, num_stages):  # 不支持！
        ...
```

### 5.3 推荐实践

| 场景 | 推荐方式 |
|------|----------|
| **核间流水** | 使用 `T.pipelined` 原语 |
| **核内流水** | 前端手动书写（手动拆分数据块 + 同步） |
| **同步间隔控制** | 使用 `cross_interval` 参数（即将上线） |

---

## 六、常见性能问题与解决方案

### 6.1 问题诊断流程

```
性能不及预期
    │
    ├─→ 采集 msprof 数据
    │       │
    │       ├─→ C/V 核耗时不均 → 调整 num_stages
    │       │
    │       ├─→ 核内流水气泡大 → 开启 Double Buffer
    │       │
    │       └─→ scalar 操作多 → 向量化改造
    │
    └─→ 检查核间同步
            │
            └─→ 同步次数过多 → 减少 sync 频率
```

### 6.2 常见问题速查表

| 现象 | 可能原因 | 解决方案 |
|------|----------|----------|
| C 核大量气泡 | V 核耗时长，`num_stages` 太小 | 增大 `num_stages` |
| 内存溢出 | `num_stages` 过大或 buffer 过大 | 减小分块参数或 `num_stages` |
| 指令下发慢 | scalar 操作过多 | 改用 `T.tile` 向量化操作 |
| GM 带宽未打满 | 数据搬运效率低 | 开启 L1 常驻、Double Buffer |

### 6.3 调优 Checklist

- [ ] 采集 msprof 性能数据
- [ ] 分析 C/V 核耗时比例
- [ ] 尝试不同 `num_stages` 值
- [ ] 检查 L1/L0 内存利用率
- [ ] 确认 Double Buffer 已开启
- [ ] 优化 scalar 操作为向量化
- [ ] 减少不必要的核间同步

---

## 附录：相关资源

- [T.pipelined 详细教程](https://github.com/tile-ai/tilelang-ascend/blob/ascendc_pto/docs/tutorials/t_pipelined.md)
- [TileLang-Ascend Programming Guide](../docs/TileLang-Ascend%20Programming%20Guide.md)
- [MindStudio Insight 下载](https://gitcode.com/Ascend/msinsight/releases/tag_MindStudio_26.0.0-alpha.1)