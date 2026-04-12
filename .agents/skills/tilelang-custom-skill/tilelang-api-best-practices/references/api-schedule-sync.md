# 调度、同步与调试

---

## 1. 循环原语

### T.serial(N) / T.serial(start, end, step)

普通 for 循环。

```python
for i in T.serial(N):        # 0..N-1
for i in T.serial(0, N, 2):  # 0, 2, 4, ...
```

### T.unroll(N)

针对小循环次数进行循环展开。TileLang 将展开提示传递给 TIR。

```python
for k in T.unroll(K_TILE):
    acc += a[k] * b[k]
```

### While 循环

循环条件需要是 TIR expression。TileLang 检测出死循环会编译报错。

```python
i = 0
while i < N:
    ...
    if done:
        break
    i += 1
```

**Break 和 Continue**：在 T.serial/T.unroll/T.Parallel/while 循环中均可使用。

---

## 2. T.Pipelined

实现计算/搬运的流水线并行，通过预取来掩盖内存访问延迟。

### 语法

```python
for var in T.Pipelined(range, num_stages=N):
    ...
```

- `range`：迭代次数
- `num_stages`：预取阶段数（小于 range-1 的正整数）

### 核内流水线（Intra-core）

```python
for k in T.Pipelined(loop_k, num_stages=2):
    T.copy(A[bx * block_M, k * block_K], A_L1)
    T.copy(B[k * block_K, by * block_N], B_L1)

    T.barrier_all()
    if k == 0:
        T.gemm_v0(A_L1, B_L1, C_L0, init=True)
    else:
        T.gemm_v0(A_L1, B_L1, C_L0)

    T.barrier_all()
```

`num_stages=2` 时执行顺序：

| Time | Copy A/B | Compute |
|------|----------|---------|
| t₀ | copy_A_0, copy_B_0 | |
| t₁ | copy_A_1, copy_B_1 | |
| t₂ | copy_A_2, copy_B_2 | gemm_0 |
| t₃ | copy_A_3, copy_B_3 | gemm_1 |
| t₄ | | gemm_2 |
| t₅ | | gemm_3 |

### 核间流水线（Inter-core）

Cube 和 Vector 核之间的流水并行：

```python
for k in T.Pipelined(T.ceildiv(seq_len, block_N), num_stages=2):
    T.copy(K[bz, by, k * block_N:(k + 1) * block_N, :], k_l1)
    T.gemm_v0(q_l1, k_l1, acc_s_l0c, transpose_B=True, init=True)
    T.copy(acc_s_l0c, workspace_1[cid, :, :])

    T.tile.fill(acc_s_ub, 0.0)
    T.copy(workspace_1[cid, vid * block_M // 2:vid * block_M // 2 + block_M // 2, :],
           acc_s_ub_)
    T.tile.add(acc_s_ub, acc_s_ub, acc_s_ub_)
    T.tile.mul(acc_s_ub, acc_s_ub, sm_scale)
    ...
```

**注意**：
- 核间流水线与核内流水线不能同时开启
- 使用核间流水线必须开启：`"tl.ascend_auto_cv_combine": True`, `"tl.ascend_auto_cross_core_sync": True`

---

## 3. T.Persistent

优化数据块在 AI Core 间的调度，使相邻数据块交由同一 AI Core 处理，提高缓存命中率。

```python
for bx, by in T.Persistent(domain, wave_size, index):
    ...
```

**参数**：
- `domain`：迭代空间
- `wave_size`：wave 大小（通常为 core_num）
- `index`：当前核的索引（通常为 cid）

**示例**（来自 Programming Guide）：

```python
with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
    A_L1 = T.alloc_shared((block_M, K_L1), dtype)
    B_L1 = T.alloc_shared((K_L1, block_N), dtype)
    C_L0 = T.alloc_fragment((block_M, block_N), accum_dtype)

    for bx, by in T.Persistent([T.ceildiv(M, block_M), T.ceildiv(N, block_N)],
                                core_num, cid):
        loop_k = T.ceildiv(K, K_L1)
        for k in T.serial(loop_k):
            T.copy(A[bx * block_M, k * K_L1], A_L1)
            T.copy(B[k * K_L1, by * block_N], B_L1)
            T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))
            T.copy(C_L0, C[bx * block_M, by * block_N])
```

---

## 4. 同步原语

### 核内同步

| API | 说明 |
|-----|------|
| `T.set_flag(src, dst, eventId)` | 设置核内流水线同步标志（producer 完成通知） |
| `T.wait_flag(src, dst, eventId)` | 等待核内流水线同步标志（consumer 阻塞等待） |
| `T.barrier_all()` | 所有管线的全局屏障 |
| `T.pipe_barrier(pipe)` | 特定管线的屏障（如 `"MTE3"`, `"V"`） |
| `T.sync_all()` | 全局同步 |

**管线名称**：`"fix"`, `"mte1"`, `"mte2"`, `"mte3"`, `"m"`, `"v"`

```python
T.set_flag("mte2", "v", 0)
T.wait_flag("mte2", "v", 0)
```

### 核间同步

| API | 说明 |
|-----|------|
| `T.set_cross_flag(pipe, flag)` | 设置核间同步标志 |
| `T.wait_cross_flag(flag)` | 等待核间同步标志 |

```python
# Cube 核完成后通知 Vector 核
T.set_cross_flag("MTE3", 0)
T.wait_cross_flag(0)
```

> `set_cross_flag` 源码（`ascend.py:114`）还支持第三个参数 `mode`（默认 2），控制同步范围：0=所有 AIC/AIV 之间，1=同组 AIV 之间，2=同组 AIC 和 AIV 之间。

---

## 5. T.Scope

用于标注代码块的执行域。

```python
with T.Scope("C"):   # Cube 域
    ...
with T.Scope("V"):   # Vector 域
    ...
```

---

## 6. 调试工具

### T.printf(format_str, *args)

设备端格式化打印，类似 C 语言 printf。Buffer 参数自动转换为 access pointer。

**格式说明符**：`%d`/`%i`（整数）, `%f`（浮点）, `%x`（十六进制）, `%s`（字符串）, `%p`（指针，建议使用 `%x`）

```python
T.printf("fmt %s %d\n", "string", 0x123)
```

### T.dump_tensor(tensor, desc, dump_size, shape_info=())

转储指定 Tensor 的内容。

**参数**：
- `tensor`：要转储的张量（支持 ub_buffer、l1_buffer、l0c_buffer、global_buffer）
- `desc`：用户自定义附加信息（uint32，如行号，方便区分多处 dump）
- `dump_size`：转储的元素数量
- `shape_info`：shape 信息元组（可选，用于格式化输出）

```python
T.printf("A_L1:\n")
T.dump_tensor(A_L1, 111, 64)               # 不带 shape
T.dump_tensor(A_L1, 111, 64, (8, 8))       # 带 shape_info 格式化输出
```

### 查看生成的 AscendC 代码

```python
func = tile_add(M, N, block_M, block_N)
print(f"{func.get_kernel_source()}")
```

> 注意：T.printf 和 T.dump_tensor 是设备端调试工具，主机端直接使用 Python `print`。调试完成后应移除，避免影响性能。

---

## 7. 性能调优工具

### msProf

```bash
# 上板性能分析
msprof op --kernel-name="your_kernel_func_name" python your_kernel_script.py

# 仿真性能分析
msprof op simulator --soc-version=<ascend_version> --kernel-name="your_kernel_func_name" python your_kernel_script.py
```

msProf 可展示：计算内存热力图、Roofline 瓶颈分析图、Cache 热力图、通算流水图、算子代码热点图。
