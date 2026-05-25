# TileLang Ascend 性能关注项清单

本文件用于生成、改写或评审 TileLang Ascend 算子时快速排查常见潜在性能劣化模式。遇到性能关注项时，优先按“替代写法”调整；如果必须临时保留，需要记录 shape、dtype、原因和后续优化计划。

## 使用方式

- 写新算子前先扫一遍本清单，尽量避免为了功能正确引入明显性能风险。
- 性能优化前用本清单做第一轮静态检查，再结合 msprof 数据定位瓶颈。
- 修改后重新检查精度和性能；没有收益或引入内存超限时回退本轮修改。

## 硬件 buffer size 信息表（A2/A3，用于评估内存）

下表容量信息适用于 Ascend A2/A3 硬件；其他硬件型号需要按对应规格重新确认。

| 存储层级 | 容量（字节） | 典型用途 |
|----------|--------------|----------|
| L0A | 65536 | Cube A 操作数 |
| L0B | 65536 | Cube B 操作数 |
| L0C | 131072 | Cube 累加结果 |
| L1 | 524032 | Cube 侧数据缓存、GM 到 L0 的中间层 |
| UB | 196352 | Vector 侧计算 buffer |
| L2 | 201326592 | GM 访问的片上缓存层，AIC/AIV 通过 GM workspace 交互时可受益 |

容量评估时按“所有同层 buffer + pipeline stage/buffer num 倍数 + 临时 buffer”计算。`T.tile.broadcast`、双 buffer、`T.Pipelined(num_stages>1)` 都会增加片上内存占用，必须留出余量。

注意：可以开启 memory planning pass，编译器会在必要时做片上 buffer 复用。因此理论内存值只作为评估参考，最终需要通过实际编译和运行验证是否存在内存不足。

开启方式：在 JIT 的 `pass_configs` 中设置 `TL_ASCEND_MEMORY_PLANNING=True`。

```python
pass_configs = {
	tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

@tilelang.jit(pass_configs=pass_configs)
def kernel(...):
	...
```

如果代码中使用 `import tilelang as tl`，则写作 `tl.PassConfigKey.TL_ASCEND_MEMORY_PLANNING`。

---

## launch core 数需要重点关注

### 关注项 A：任务数高于物理 AI Core 数，按任务数 launch

**识别特征**：逻辑任务数 `block_num` 明显大于 A2/A3 的 24 个 AI Core，但代码使用：

```python
with T.Kernel(block_num, is_npu=True) as (cid, vid):
	# 每个 cid 只处理一个 block
	...
```

**性能原因**：逻辑任务数远高于物理核数时，按任务数 launch 会放大 kernel 初始化、workspace 分配、地址计算和同步开销。很多临时 buffer 还会随 `block_num` 线性膨胀，而同一时刻实际只会有物理 core 数量的任务并行执行。

**替代写法**：使用 Fixed Core，按物理核数 launch，再在每个 core 内手动分配多个逻辑任务。

```python
core_num = 24

with T.Kernel(core_num, is_npu=True) as (cid, vid):
	single_core_load = T.ceildiv(block_num, core_num)
	for block_idx in T.serial(cid * single_core_load, (cid + 1) * single_core_load):
		if block_idx < block_num:
			# 处理逻辑任务 block_idx
			...
```

**检查点**：
- workspace 优先按 `core_num` 维度分配，再通过 `cid` 复用。
- 尾块必须处理 `block_idx < block_num`，避免越界。
- 如果每个逻辑任务耗时差异很大，静态连续分配可能负载不均，需要结合任务形状重新设计映射。

### 关注项 B：任务数低于物理 AI Core 数，仍按 24 核 launch

**识别特征**：`block_num < 24`，但代码固定使用：

```python
with T.Kernel(24, is_npu=True) as (cid, vid):
	...
```

**性能原因**：空闲 core 不做有效计算，但仍会进入 kernel、执行部分初始化和分支判断；如果存在全局同步或 workspace 初始化，空 core 还可能放大等待和资源占用。

**替代写法**：按实际任务数 launch。

```python
launch_core_num = T.min(block_num, 24)

with T.Kernel(launch_core_num, is_npu=True) as (cid, vid):
	# cid 天然落在有效任务范围内
	...
```

如果 `block_num` 是 Python 侧静态整数，也可以直接在 host 侧计算：

```python
core_num = min(block_num, 24)

with T.Kernel(core_num, is_npu=True) as (cid, vid):
	...
```

---

## Vector Core 内逐元素/逐行 for loop 计算

**识别特征**：在 Vector 侧用 Python `range` 或 `T.serial` 循环对 UB 中的小切片反复执行同一类 scalar/tile 操作，常见于 softmax、归一化、mask、逐行缩放。

```python
for row in range(block_M):
	T.tile.sub(acc_ub[row, :], acc_ub[row, :], max_ub[row])
```

**性能原因**：每一行/每个元素单独发起一次指令，会降低 Vector 指令利用率，并引入大量 scalar 地址计算、循环控制和指令下发开销。Vector Core 更适合一次处理连续 tile。

**替代写法**：先把低维标量/向量 broadcast 到与目标 tile 相同形状，再一次性调用向量化 tile 指令。

```python
max_2d = T.alloc_ub([block_M, block_N], dtype)

T.tile.broadcast(max_2d, max_ub, axis=1)
T.tile.sub(acc_ub, acc_ub, max_2d)
```

**收益和代价**：
- 收益：减少循环控制、scalar 指令和多次 tile 指令下发，提升 Vector 计算利用率。
- 代价：broadcast 后的变量需要额外 UB 空间，broadcast 本身也需要执行并可能产生临时 buffer。
- 适用：UB 有余量、原始循环次数较多、每轮计算量偏小的场景。
- 不适用：存在真实迭代依赖（例如前一轮结果影响后一轮）且无法数学等价重排的场景。

**评审建议**：看到 `for row in range(...)`、`for col in range(...)` 里反复调用 `T.tile.add/sub/mul/div/max/min` 时，优先确认能否改成 broadcast + 整 tile 计算。

---

## 冗余全局同步

**识别特征**：循环体内或每个小步骤后频繁出现：

```python
T.barrier_all()
...
T.barrier_all()
...
T.sync_all()
```

**性能原因**：`barrier_all` / `sync_all` 会扩大等待范围。即使只有局部数据依赖，也会让无关 pipeline 或 core 一起等待，造成 MTE、Vector、Cube 的气泡。同步放在内层循环时，开销会被循环次数放大。

**替代方向**：
- 删除没有生产者/消费者依赖的同步。
- 用 `T.set_flag(src, dst, event_id)` / `T.wait_flag(src, dst, event_id)` 约束具体 pipeline 之间的依赖。
- AIC/AIV 跨核交互用 `T.set_cross_flag` / `T.wait_cross_flag` 或交给 `T.Pipelined` 自动管理。
- 对多次任务后才需要一致性的场景，考虑增加同步间隔，参考 `T.Pipelined(..., cross_interval=N)`。

**性能劣化模式示例**：每轮 gather 后都全局同步。

```python
for i in T.serial(num_blocks):
	T.copy(src[i, :], ub_tmp)
	T.barrier_all()
	T.tile.add(out_ub, out_ub, ub_tmp)
	T.barrier_all()
```

**改写示意**：只在真正需要消费搬运结果的位置等待，或改成流水/双 buffer。

```python
for i in T.serial(num_blocks):
	side = i % 2
	T.copy(src[i, :], ub_tmp[side, :])
	# 若开启自动同步且生成代码正确，可不手写 barrier。
	T.tile.add(out_ub, out_ub, ub_tmp[side, :])
```

**检查点**：先验证同步是否必需，再删改；如果关闭自动同步或进入 Expert 模式，必须通过生成的 Ascend C 代码和精度测试确认同步语义没有被破坏。

---

## 基础指令拼接未融合

**识别特征**：连续出现多个基础 element-wise 指令，且整体等价于硬件/TileLang 已有复合指令或激活函数。

### `mul + add` / 累加 pattern

**性能劣化模式示例**：

```python
T.tile.mul(tmp_ub, x_ub, w_ub)
T.tile.add(acc_ub, acc_ub, tmp_ub)
```

**替代写法**：

```python
T.tile.mul_add_dst(acc_ub, x_ub, w_ub)  # acc_ub = x_ub * w_ub + acc_ub
```

如果是 `dst = scalar * src + dst`，使用 `T.tile.axpy`：

```python
T.tile.axpy(dst_ub, src_ub, scale)  # dst_ub = scale * src_ub + dst_ub
```

注意：PTO 后端场景若没有 `axpy`，优先确认能否使用 `mul_add_dst` 等价表达；所有接口以 `tilelang/language/ascend_tile.py` 为准。

### `max(x, 0)` pattern

**性能劣化模式示例**：

```python
T.tile.max(out_ub, x_ub, 0.0)
```

**替代写法**：

```python
T.tile.relu(out_ub, x_ub)
```

其他激活函数也按“数学等价优先”原则尝试替换，例如 `leaky_relu` 等。替换前确认 dtype、边界值和 NaN 行为是否满足算子精度要求。

### `sqrt + div` pattern

**性能劣化模式示例**：

```python
T.tile.sqrt(tmp_ub, x_ub)
T.tile.div(out_ub, one_ub, tmp_ub)
```

**替代写法**：

```python
T.tile.rsqrt(out_ub, x_ub)  # out_ub = 1 / sqrt(x_ub)
```

如果后续计算是 `y / sqrt(x)`，优先尝试 `rsqrt` 后再乘：

```python
T.tile.rsqrt(inv_sqrt_ub, x_ub)
T.tile.mul(out_ub, y_ub, inv_sqrt_ub)
```

**检查点**：融合会改变中间舍入路径，尤其是 fp16/bf16 场景，必须重新跑精度。

---

## tile size 过小导致片上内存浪费

**识别特征**：L0A/L0B/L0C/L1/UB 实际使用量远小于容量。

**性能原因**：tile 太小会增加逻辑任务数、循环次数、同步次数和 GM 往返次数；片上缓存没有充分复用，单次搬运/计算粒度也可能不足以打满带宽或算力。

**替代方向**：在不超过片上容量的前提下，优先成倍扩大 tile size。

```python
# 性能劣化模式示例：UB 占用很小，任务数很多
block_M = 16
block_N = 64

# 尝试：按 2 倍递增，并重新计算 UB/L1/L0 占用
block_M = 32
block_N = 128
```

**容量估算示意**：

```text
UB 使用量 = sum(buffer_elements * dtype_bytes * buffer_num)
L1 使用量 = sum(l1_tile_elements * dtype_bytes * stages)
L0C 使用量 = block_M * block_N * accum_dtype_bytes * stages
```

**AIC/AIV 交互说明**：AIC 和 AIV 之间通常通过 GM workspace 交互。当 workspace 访存量小于 L2 容量（A2/A3 约 201 MB）时，数据更可能命中 L2 cache，从而获得高于普通 GM 往返的带宽收益。设计 workspace 时应尽量让交互数据连续、按 core 复用，并避免无意义地扩大到超过 L2 cache 容量。

**检查点**：
- 扩大 tile size 后确认 `T.Pipelined` stage、双 buffer、broadcast 临时 buffer 的总占用仍不超限。
- 理论内存估算只作参考；开启 memory planning pass 后可能复用部分 buffer，最终以内存规划结果和实际运行是否报内存不足为准。
- tile 变大可能降低并行任务数；当任务数低于物理 core 数时，需要同步调整 launch core 数。
- 尾块处理和 mask 逻辑要随 tile size 一起复查。

---

## AIC/AIV 混合算子未开启 CV overlap

**识别特征**：通过 `get_kernel_source()` 看到生成代码同时包含 `IS_ASCEND_AIC` 和 `IS_ASCEND_AIV`，但主循环仍是 Cube 写 workspace、Vector 读 workspace 的串行结构，未使用 `T.Pipelined`。

```python
for k in T.serial(loop_k):
	# AIC: compute and write workspace
	...
	# AIV: read workspace and vector compute
	...
```

**性能原因**：CV 融合算子的 AIC 和 AIV 通过 workspace 串接。如果没有核间流水，Vector 往往要等 Cube 产出，Cube 也可能等 Vector 消费，形成明显核间气泡。

**替代写法**：用 `T.Pipelined` 表达核间流水，并开启自动 CV combine/sync。

```python
pass_configs = {
	tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
	tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
}

for k in T.Pipelined(loop_k, num_stages=2):
	# AIC: write workspace for iteration k
	...
	# AIV: read previous/ready workspace and compute
	...
```

**调参建议**：
- 从 `num_stages=2` 开始，逐步增大；`num_stages` 不能超过循环次数。
- 循环次数较多、Cube/Vector 耗时差距大时，较大的 `num_stages` 可能收益更好。
- 如果同步开销占比高，尝试 `cross_interval=2`，但需要验证并行度损失是否抵消收益。
- 不要嵌套多个 `T.Pipelined`；核间用 `T.Pipelined` 时，核内 double buffer 推荐手写 flat pattern。

---

## 纯 AIV memory bound 算子未做流水/双 buffer

**识别特征**：`get_kernel_source()` 只包含 `IS_ASCEND_AIV`；每轮循环都按 `GM -> UB -> Vector -> GM` 串行执行。

```python
for i in T.serial(loop_n):
	T.copy(x[i, :], x_ub)
	T.tile.exp(y_ub, x_ub)
	T.copy(y_ub, y[i, :])
```

**性能原因**：纯 AIV 算子常常 memory bound。如果搬入、计算、搬出完全串行，Vector 计算无法掩盖 GM/UB 搬运延迟，MTE 和 Vector pipeline 都容易出现空泡。

**替代写法 A：使用 `T.Pipelined`**

```python
for i in T.Pipelined(loop_n, num_stages=2):
	T.copy(x[i, :], x_ub)
	T.tile.exp(y_ub, x_ub)
	T.copy(y_ub, y[i, :])
```

**替代写法 B：手动双 buffer**

手动双 buffer 只分配双份 buffer 不够，还需要手动控制每个 stage 的同步关系，确保搬入、计算、搬出不会读写同一份未就绪 buffer。下面代码只展示 buffer 轮转位置；实际实现中需要按 pipeline 使用 `T.set_flag` / `T.wait_flag` 明确控制 stage 依赖。

```python
x_ub = T.alloc_ub([2, block_N], dtype)
y_ub = T.alloc_ub([2, block_N], dtype)

for i in T.serial(loop_n):
	side = i % 2
	# 手动双 buffer 时，需要在每个 stage 之间配套 set_flag / wait_flag。
	T.copy(x[i, :], x_ub[side, :])
	T.tile.exp(y_ub[side, :], x_ub[side, :])
	T.copy(y_ub[side, :], y[i, :])
```

**检查点**：
- 双 buffer 会让 UB 占用乘以 2；再叠加 broadcast 或临时 buffer 时尤其要重新估算。
- 如果 `loop_n` 很小，pipeline 启停开销可能抵消收益，需要实测。

---

## 评审记录模板

发现性能关注项但暂不修改时，在优化记录中写清楚：

```text
- 关注项：Vector for loop 逐行计算
- 位置：examples/<op>/<file>.py::<kernel>
- shape/dtype：...
- 暂不修改原因：UB 余量不足，broadcast 后可能超限
- 后续方案：减小其他临时 buffer 或改用分块 broadcast，再验证精度和性能
```
