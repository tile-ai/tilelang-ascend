# TileLang-Ascend 算子设计速查

## 1. 编程模式选型

### 1.1 模式判定规则

**重要**：`T.reduce_sum/max/min` 和 `T.tile.*` 在 Developer 和 Expert 模式下**都可使用**。模式选择取决于是否需要手动控制内存层级和同步，而非使用了哪个 API。

| 条件 | 推荐模式 | 理由 |
|------|----------|------|
| 纯 element-wise 单步运算 | Developer | `T.Parallel` + 运算符即可，编译器自动处理内存和同步 |
| 纯 element-wise 多步运算（无需精细内存控制） | Developer | `T.Parallel` 内链式运算 + `T.reduce_*` |
| 多步 Vector 运算，需精细控制 buffer 分配和复用 | Expert | 显式 `T.alloc_ub` + `T.Scope("V")` + 手动同步 |
| 含 matmul（GEMM/MMA） | Expert | 需精确控制 L0A/L0B/L0C 分配和 Cube 计算流程 |
| 融合算子（Cube + Vector） | Expert + 核间流水线 | 需要 `T.Scope("C"/"V")`、核间同步、CV 分离 |
| 需要流水线优化 | Expert | `T.Pipelined` 需要手动管理 buffer 和同步 |

### 1.2 模式 API 对照

| 功能 | Developer 模式 | Expert 模式 |
|------|---------------|-------------|
| 内存分配 | `T.alloc_shared` / `T.alloc_fragment` | `T.alloc_ub` / `T.alloc_L1` / `T.alloc_L0A/L0B/L0C` |
| 计算 | `T.Parallel` 内用运算符 | `T.tile.*` / `T.gemm_*` / `T.mma` |
| 作用域 | 编译器自动分离 | `with T.Scope("C")` / `with T.Scope("V")` |
| 同步 | 自动（pass_configs 开启） | `T.barrier_all` / `T.set_flag` / `T.wait_flag` |

---

## 2. 内存层级与数据搬运

### 2.1 Ascend NPU 内存层级

```
GM（全局内存 / HBM）
  ↕ T.copy        — MTE 传输
L1（Cube 核缓存）  /  UB（Vector 核统一缓冲区）
  ↕ T.copy        — DMA 传输
L0A / L0B（矩阵输入寄存器）
  ↓ T.gemm / T.mma
L0C（矩阵输出/累加寄存器）
```

### 2.2 硬约束

- **不可跨级访问**：GM 不能直接到 L0，必须经过 L1/UB 中转
- **UB 容量约束**：~128KB/核，所有 UB buffer 总和不可超出
- **L1 容量约束**：~32KB/核（Cube 专用）
- **L0 容量约束**：~8KB（共享临时寄存器）
- **对齐约束**：尾轴需 32B 对齐（fp16/bf16 需 16 的倍数，fp32 需 8 的倍数）
- **T.copy 自动选路**：编译器根据 src/dst 所在层级自动选择搬运路径

### 2.3 典型搬运路径

| 计算类型 | 数据搬运路径 |
|----------|-------------|
| 纯 Vector | GM → UB → 计算 → UB → GM |
| 纯 Cube | GM → L1 → L0A/L0B → L0C → UB → GM |
| 融合 (Cube→Vector) | GM → L1 → L0A/L0B → L0C → UB(Cube) → UB(Vector) → GM |

---

## 3. Tiling 策略

### 3.1 Block 划分原则

- **Block 数量** = 逻辑划分的独立计算单元数，映射到 NPU 的 AI Core
- Kernel 启动：`T.Kernel(block_num, is_npu=True) as (cid, vid)`
  - `cid`：block ID（AI Core 编号）
  - `vid`：vector unit ID（0 或 1，双 vector 核时有效）
- 典型划分：`block_num = (M // block_M) * (N // block_N)`

### 3.2 Tile Shape 设计要点

- **尾轴对齐**：DMA 搬运最小单位 32B
  - fp16/bf16：尾轴需为 16 的倍数
  - fp32：尾轴需为 8 的倍数
- **维度数 ≤ 输入张量维度**
- **UB 容量限制**：所有 tile 的 UB buffer 总和 < 128KB
- **L0 容量限制**：Cube 模式下 L0A/L0B/L0C 各 tile 不超过 ~8KB

### 3.3 常用 Block 大小参考

| 算子类型 | 典型 block_M | 典型 block_N | 说明 |
|----------|-------------|-------------|------|
| element-wise | 128~256 | 128~256 | 受 UB 容量限制 |
| GEMM | 128 | 256 | 受 L0 容量限制，需配合 K 方向循环 |
| Reduction | 整行/整列 | block_N | 归约维度不可分块 |
| Softmax | 1~8 行 | 整列 | 行内归约，不可沿列分块 |

---

## 4. 循环与调度

### 4.1 循环类型选择

| 场景 | 推荐 API | 说明 |
|------|----------|------|
| 逐元素并行计算 | `T.Parallel(M, N)` | 编译器向量化，Developer 模式核心 |
| 固定次数迭代 | `T.serial(N)` | 普通 for 循环，如 K 方向分块迭代 |
| 编译期展开 | `T.unroll(N)` | 小循环体展开，减少循环开销 |
| 软件流水线 | `T.Pipelined(N, num_stages=2)` | 搬运与计算重叠，提升吞吐 |
| 持久化调度 | `T.Persistent(domain, wave_size, idx)` | 大规模任务的波次调度 |

### 4.2 动态 Shape 处理

- 编译期已知维度 → Python `for` / `T.serial` / `T.unroll`
- 运行时才确定维度 → `T.dyn['K']` 或 `T.dynamic('K', 'int32')` 声明
- 动态维度上的循环 → `T.serial(dynamic_var)` 配合边界检查

### 4.3 硬约束

- `T.Parallel` 仅用于 Developer 模式的元素级运算
- `T.Pipelined` 需要 Expert 模式，手动管理 double buffer
- 避免在 `T.Parallel` 内部嵌套 `T.serial`（会破坏向量化）

---

## 5. 同步策略

### 5.1 自动同步（Developer 模式）

通过 `pass_configs` 启用编译器自动插入同步指令：

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}
```

### 5.2 手动同步（Expert 模式）

| 场景 | API | 说明 |
|------|-----|------|
| 核内全管线屏障 | `T.barrier_all()` | 等待所有操作完成 |
| 核内细粒度同步 | `T.set_flag(src, dst, id)` + `T.wait_flag(src, dst, id)` | 管线间同步 |
| 指定管线屏障 | `T.pipe_barrier(pipe)` | 仅等待特定管线 |
| 核间同步（融合算子） | `T.set_cross_flag` + `T.wait_cross_flag` | Cube→Vector 核间通信 |
| 全局同步 | `T.sync_all()` | 所有核同步 |

### 5.3 融合算子同步模式

融合算子需启用核间流水线相关 pass：

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
}
```

---

## 6. JIT 编译配置

### 6.1 @tilelang.jit 常用参数

```python
@tilelang.jit(
    out_idx=[-1],         # 输出参数索引（-1 表示最后一个）
    supply_type=tilelang.TensorSupplyType.Normal,  # 输入数据生成方式
    pass_configs={...},   # 编译 pass 配置
)
```

### 6.2 常用 pass_configs

| 配置项 | 默认 | 说明 |
|--------|------|------|
| `TL_ASCEND_AUTO_SYNC` | False | 自动同步插入 |
| `TL_ASCEND_MEMORY_PLANNING` | False | 自动内存规划 |
| `TL_ASCEND_AUTO_CV_COMBINE` | False | CV 分离（核间流水线需要） |
| `TL_ASCEND_AUTO_CV_SYNC` | False | 核间自动同步（核间流水线需要） |

---

## 7. 数据类型支持

| 类型 | 说明 | 尾轴对齐要求 |
|------|------|-------------|
| `"float16"` | 半精度浮点 | 16 的倍数 |
| `"bfloat16"` | BF16 | 16 的倍数 |
| `"float32"` | 单精度浮点 | 8 的倍数 |
| `"int8"` | 8 位整数 | 32 的倍数 |
| `"int32"` | 32 位整数 | 8 的倍数 |

---

## 8. 验证标准

### 8.1 精度标准

| 数据类型 | atol | rtol |
|----------|------|------|
| float16 | 1e-2 | 1e-2 |
| bfloat16 | 1e-1 | 1e-1 |
| float32 | 1e-4 | 1e-4 |

### 8.2 测试级别

| 级别 | 目标 | 数据规模 |
|------|------|----------|
| Level 0 | 基础功能 | 最小合法 shape |
| Level 1 | 正确性 | 典型配置 shape |
| Level 2 | 鲁棒性 | 边界值、极值 |
| Level 3 | 性能 | 大规模数据 |
