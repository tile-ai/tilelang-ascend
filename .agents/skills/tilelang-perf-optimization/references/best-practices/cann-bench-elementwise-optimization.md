# cann-bench 元素级算子接入优化最佳实践

本文档总结 SwiGLU 算子接入 cann-bench 评测框架的性能优化经验。核心发现：**cann-bench 计时包含 host 侧开销，接口层全量 host 拷贝（`chunk+contiguous`、`permute+contiguous`）是主要瓶颈，而非 kernel 计算本身**。通过"单输入 split kernel"模式消除 host 拷贝，可将平均加速比从 0.36x 提升到 0.68x（+89%）。

适用于所有元素级/激活融合算子接入 cann-bench 的场景。

## 目录

- [场景背景](#场景背景)
- [核心反模式：接口层全量 host 拷贝](#核心反模式接口层全量-host-拷贝)
- [解决方案：单输入 split kernel 模式](#解决方案单输入-split-kernel-模式)
- [动态 tiling 自适应](#动态-tiling-自适应)
- [fp16 精度与性能权衡](#fp16-精度与性能权衡)
- [实测数据](#实测数据)
- [可复用决策清单](#可复用决策清单)

---

## 场景背景

cann-bench 评测流程：`run_evaluation.sh` → 每个用例在独立子进程执行 → 计时含 `cann_bench.<op>(input)` **完整调用**（host 接口层 + kernel launch + kernel 执行 + synchronize）。

与纯 kernel 性能调优（`msprof op` 只测 kernel 执行时间）不同，cann-bench 的 HAP 评分基于**端到端耗时**，因此 host 侧开销直接计入性能得分。

关键差异：

| 计时方式 | 范围 | host 开销 |
|---------|------|----------|
| `msprof op` | 仅 kernel 执行 | 不含 |
| cann-bench HAP | host 接口层 + kernel + sync | **含** |

**结论**：接入 cann-bench 时，仅优化 kernel 计算不够，必须同时消除接口层 host 开销。

---

## 核心反模式：接口层全量 host 拷贝

### 问题

cann-bench 的算子接口签名通常是单输入（如 `swi_glu(input, dim=-1)`），而 TileLang kernel 可能是双输入（如 `swiglu(x, gate)`）。常见适配方式是在接口层做 `chunk + contiguous` 拆分：

```python
# ❌ 反模式：接口层 chunk+contiguous 产生 2 次全量拷贝
def swi_glu(input, dim=-1):
    x_2d = input.permute(perm).contiguous().reshape(outer, k_full)  # 拷贝1: permute+contiguous
    half_k = k_full // 2
    x0 = x_2d[:, :half_k].contiguous()   # 拷贝2: chunk 前半
    x1 = x_2d[:, half_k:].contiguous()   # 拷贝3: chunk 后半
    out = kernel(x0, x1)                 # 双输入 kernel
    ...
```

**实测影响**（SwiGLU case4，8192×16384 fp16，268MB 张量）：

| 开销来源 | 耗时占比 |
|---------|---------|
| permute+contiguous（拷贝1） | 20% |
| chunk+contiguous×2（拷贝2+3） | 39% |
| kernel 计算 | 41% |

host 拷贝占总耗时 59%，是主要瓶颈。大 shape case 尤为严重。

### 适用判断

此反模式在以下条件触发：
1. cann-bench 接口是单输入 split 语义（`chunk(2)` → 融合计算）
2. kernel 是双输入/多输入
3. 接口层做 `chunk` + `.contiguous()` 适配

---

## 解决方案：单输入 split kernel 模式

将 split 逻辑从接口层移入 kernel 内部，用 **GM 列 offset 索引**直接读取两半数据，消除 chunk+contiguous 的 2 次全量拷贝。

### 实现模式

```python
# ✅ 优化：单输入 kernel，内部 offset 索引读 x0/x1
@tilelang.jit(out_idx=[1], pass_configs=PASS_CONFIGS)
def swiglu_single(K_full, half_k, block_M, block_K, dtype="float16"):
    """单输入 kernel: input[M, K_full] -> output[M, half_k]"""
    M = T.symbolic("M")
    VEC_NUM = 2
    rows_per_vec = block_M // VEC_NUM
    m_num = T.ceildiv(M, block_M)
    k_num = T.ceildiv(half_k, block_K)   # block 按 half_k 切分

    @T.prim_func
    def main(
        X: T.Tensor((M, K_full), dtype),   # 单输入，含 x0|x1 拼接
        Y: T.Tensor((M, half_k), dtype),   # 输出，列数折半
    ):
        with T.Kernel(m_num * k_num, is_npu=True) as (cid, vid):
            bx = cid // k_num
            by = cid % k_num
            row = bx * block_M + vid * rows_per_vec
            col = by * block_K              # x0 的列起点

            x_ub = T.alloc_shared((rows_per_vec, block_K), dtype)
            gate_ub = T.alloc_shared((rows_per_vec, block_K), dtype)
            y_ub = T.alloc_shared((rows_per_vec, block_K), dtype)

            # offset 索引：x0=X[:, col:col+block_K], x1=X[:, col+half_k:col+half_k+block_K]
            T.copy(X[row, col], x_ub)                    # 读 x0 半
            T.copy(X[row, col + half_k], gate_ub)        # 读 x1 半（offset half_k）

            # 计算 silu(x0) * x1
            T.tile.silu(silu_ub, x_ub)
            T.tile.mul(y_ub, silu_ub, gate_ub)

            T.copy(y_ub, Y[row, col])                    # 写输出（half_k 列）
    return main
```

### 接口层简化

```python
# ✅ 接口层零拷贝（dim=-1 时连 reshape 都是 view）
def swi_glu(input, dim=-1):
    dim = dim % input.ndim
    # dim=-1: input 直接 reshape 成 2D，零拷贝
    # dim!=-1: 仅 1 次 permute+contiguous（不可避免）
    if dim != input.ndim - 1:
        perm = [i for i in range(input.ndim) if i != dim] + [dim]
        x_2d = input.permute(perm).contiguous().reshape(outer, k_full)
    else:
        x_2d = input.reshape(outer, k_full)   # 零拷贝 view
    # 无需 chunk！直接调单输入 kernel
    out_2d = kernel(x_2d)   # kernel 内部 offset 读 x0/x1
    # reshape 回 + inverse permute
    ...
```

### 收益来源

| 消除的拷贝 | 节省 | 适用条件 |
|-----------|------|---------|
| `chunk` + `.contiguous()` ×2 | 2 次全量拷贝 | 所有单输入 split 语义算子 |
| dim=-1 的 `permute+contiguous` | 1 次全量拷贝 | split 维恰为末维时 |
| dim=-1 的 `reshape` | 0（view 操作） | 非连续张量需 contiguous，连续则零拷贝 |

### dim≠-1 的残留开销

dim≠-1 时仍需 1 次 `permute+contiguous`（使 split 维到末尾）。这是 TileLang 2D kernel 的 layout 限制，无法在接口层完全消除。后续可探索 kernel 支持 stride 输入（非 contiguous tensor 直接索引），但风险较高。

---

## 动态 tiling 自适应

cann-bench 20 个 case 的 K 从 1 到 32768 跨度极大，固定 `block_K=128` 在小 K 场景严重浪费。

### block_K 自适应

```python
def _block_sizes(half_k, tl_dtype):
    # 32B 对齐（DataCopyNd 粒度要求，不对齐会数据损坏）
    dtype_bytes = 2 if tl_dtype in ("bfloat16", "float16") else 4
    align = max(1, 32 // dtype_bytes)   # fp32→8, bf16/fp16→16
    block_k = min(half_k, 128)
    block_k = min(128, ((block_k + align - 1) // align) * align)  # 向上对齐
    ...
```

**关键约束**：block_K 必须对齐到 32 字节（DataCopyNd 搬运粒度）。实测不对齐（如 fp32 block_K=50、bf16 block_K=7/1）会导致数据损坏/精度失败。

### block_M 反向放大

block_K 缩小时，按 `block_M = target_area / block_K` 反向增大 block_M，保持 tile 面积，减少 block 总数：

```python
    target_area = 128 * 128   # 基准 tile 面积（K=128 时的 footprint）
    block_m = target_area // block_k
    block_m = (block_m // 32) * 32   # 向下对齐到 32（VEC_NUM=2 需偶数）
    block_m = max(128, min(1024, block_m))
```

**收益示例**：case12（M=1000003, half_k=1），block_K 128→16，block_M 64→1280，block 数 977→782。

### UB 预算校验

放大 block_M 前必须校验 UB 不溢出（196352B）：

```
cast 路径（bf16/fp16, 7 buffer: 3 native×2B + 4 fp32×4B = 22B/elem）:
    rows_per_vec × block_K × 22 ≤ 196352
direct 路径（fp32, 4 buffer × 4B = 16B/elem）:
    rows_per_vec × block_K × 16 ≤ 196352
```

`MEMORY_PLANNING=True` 会复用 dead buffer（如 cast 后的 native buffer），实际 live UB 更低，但仍需按 worst-case 校验。

---

## fp16 精度与性能权衡

### 问题

`T.tile.silu` 硬件原语文档声明支持 half/float（`ascend_tile.py:1003`）。fp16 走直接 half 路径（4-buffer，16B/elem）比 f32 cast detour（7-buffer，22B/elem）快 30-40%，block_M 可从 128 放到 256。

### 精度风险

fp16 直接走 half 路径时，`silu(x) = x * sigmoid(x) = x / (1 + exp(-x))`，当 `x < -11` 时 `exp(-x) > 65504` 溢出为 inf，导致 `sigmoid(x) = 1/(1+inf) = 0`，`silu(x) = x * 0 = 0`。

而 golden 在 f32 下计算：`exp(-x)` 不溢出，`sigmoid(x)` 为极小非零值，`silu(x)` 为小非零值。两者不一致 → 精度失败。

**实测**：SwiGLU case10（±65504）、case17（±1000）因 fp16 exp 溢出导致 MERE 超阈值，精度失败。

### 决策规则

| dtype | 推荐路径 | 理由 |
|-------|---------|------|
| float32 | 直接原语（4-buffer） | 无精度风险，最高效 |
| bfloat16 | f32 cast detour（7-buffer） | `T.tile.silu` 不支持 bf16，必须 cast |
| **float16** | **f32 cast detour（7-buffer）** | **直接 half 在大负 x 时 exp 溢出精度失败；f32 与 golden 逐位一致** |

**例外**：若输入值域有保证（如 `|x| < 5`），fp16 可安全走 half 路径。但 cann-bench cases 含极端值域（±65504、±1000），无法保证，故 fp16 统一走 f32 cast detour。

### 启用 half 路径的条件（后续优化方向）

若要启用 fp16 half 路径恢复 4-buffer 加速，需手写**溢出安全 sigmoid**：
- `x >= 0`：`sigmoid(x) = 1 / (1 + exp(-x))`（正常）
- `x < 0`：`sigmoid(x) = exp(x) / (1 + exp(x))`（避免 `exp(-x)` 溢出）

用 `T.tile.exp` + 条件分支实现，但 TileLang 条件分支可能引入 PipeBarrier 性能退化，需实测权衡。

---

## 实测数据

### SwiGLU 三版优化对比

| 版本 | 策略 | 平均加速比 | 综合得分 | 精度 |
|------|------|-----------|---------|------|
| V1 原版 | 固定 tiling + 接口层 chunk×2 | 0.32x | 59.79 | 20/20 |
| V2 动态tiling | block_K/M 自适应 | 0.36x | 61.02 | 20/20 |
| **V3 单输入kernel** | **A: 单输入 split + C: K=1 特化** | **0.68x** | **66.83** | **20/20** |

### host 开销占比实测（V2，定位 V3 优化方向）

| case | shape | host 占比 | 主要 host 开销 |
|------|-------|----------|--------------|
| case4 | 8192×16384 fp16 dim=0 | 59% | permute+contiguous + chunk×2 |
| case5 | 2039×65520 fp32 | 41% | chunk×2（256MB） |
| case3 | 4096×8192 bf16 | 23% | chunk×2 |
| case1 | 1024×2048 fp16 | 14% | chunk×2 |
| case12 | 1000003×2 bf16 | 1% | kernel 计算（K=1 block 数） |

### V3 主要受益 case（方案 A 驱动）

| case | V2 耗时 | V3 耗时 | 改善 | 驱动 |
|------|---------|---------|------|------|
| case15 (fp32 dim=-1) | 26.12μs | 9.84μs | -62% | A: dim=-1 零拷贝 |
| case11 (fp32 dim=-1) | 18.56μs | 7.80μs | -58% | A: dim=-1 零拷贝 |
| case10 (fp16 dim=1) | 22.82μs | 10.76μs | -53% | A: 消除 chunk×2 |
| case1 (fp16 dim=-1) | 21.84μs | 10.38μs | -52% | A: dim=-1 零拷贝 |
| case5 (fp32 dim=-1) | 1958μs | 1127μs | -42% | A: chunk 256MB 省掉 |

---

## 可复用决策清单

接入 cann-bench 的元素级算子性能优化，按以下顺序决策：

### 1. 接口层 host 开销排查（P0，必做）

```
检查接口层是否有：
  □ chunk + .contiguous()     → 消除：改单输入 kernel，offset 索引
  □ permute + .contiguous()   → 减少：dim=-1 时零拷贝 view
  □ .to(dtype) 全量转换        → 移入 kernel：T.tile.cast
  □ F.pad / torch.cat          → 移入 kernel 或改 kernel 边界处理
若任一存在 → 实施单输入 kernel 模式（本文件核心方案）
```

### 2. 动态 tiling 自适应（P1）

```
检查 cases 的 K 维度跨度：
  □ K 跨度大（如 1 ~ 32768）→ block_K 自适应（min(K,128) 对齐 32B）
  □ 小 K case 存在           → block_M 反向放大（减少 block 数）
  □ UB 预算校验              → rows_per_vec × block_K × B/elem ≤ 196352
```

### 3. dtype 路径选择（P1）

```
  □ float32  → 直接原语（4-buffer）
  □ bfloat16 → f32 cast detour（7-buffer，T.tile.silu 不支持 bf16）
  □ float16  → f32 cast detour（7-buffer，避免 exp 溢出精度失败）
              例外：值域 |x|<5 有保证时可用 half 路径
```

### 4. 极小 K 特化（P2）

```
  □ half_k ≤ 2 的 case → block_M 放大到 1024+（UB 允许）
  □ block 数仍多       → 考虑 1D 纯元素级 kernel（M 维一维 grid）
```

### 5. 大 shape 优化（P2，V3 未实施，后续方向）

```
  □ block 数 >> 核数（如 2048 blocks vs 24 核）→ Fixed Core + T.serial 多 block
  □ bf16 cast detour 限 block_M=128           → 减少 cast 中间 buffer 或 1D kernel
```

---

## 参考资料

- [optimization-guide.md](../optimization-guide.md) §2.12 P0 Host 侧优化
- [performance-antipatterns.md](../performance-antipatterns.md) — tile size 过小、纯 AIV memory bound
- SwiGLU 优化案例：`tilelang_cann_examples/SwiGLU/cann_bench_compare.md`（三版完整对比）
- SwiGLU 调优日志：`tilelang_cann_examples/SwiGLU/cann_bench/perf_tuning/perf_log.md`
