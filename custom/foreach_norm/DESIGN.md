# ForeachNorm 算子设计文档

## 1. 概述

### 1.1 算子名称

foreach_norm

### 1.2 功能描述

对输入张量列表（TensorList）的每个张量独立求 p 范数，输出为与输入列表等长的标量张量列表。典型应用场景：梯度裁剪中的梯度范数计算、优化器参数范数监控、模型正则化权重范数约束。

### 1.3 数学公式

**通用 p 范数**（p 为正实数且 p ≠ 1, 2, inf）：

$$
y = \left(\sum_i |x_i|^p\right)^{1/p}
$$

**特化公式**（按 scalar 分支）：

| scalar (p) | 公式 | 含义 |
|------------|------|------|
| 0 | $\sum_i \mathbb{1}(x_i \neq 0)$ | L0 范数（非零元素个数） |
| 1 | $\sum_i |x_i|$ | L1 范数（绝对值之和） |
| 2 | $\sqrt{\sum_i x_i^2}$ | L2 范数（欧氏距离） |
| +inf | $\max_i |x_i|$ | 无穷范数（最大绝对值） |
| -inf | $\min_i |x_i|$ | 负无穷范数（最小绝对值，要求非零） |
| 负实数 p | $\left(\sum_i |x_i|^p\right)^{1/p}$ | 负阶范数（要求输入元素非零） |
| 分数阶 p (如 1.5, 2.5) | $\left(\sum_i |x_i|^p\right)^{1/p}$ | 分数阶范数（通用公式） |

### 1.4 算法描述

本算子为 **TensorList → TensorList** 的逐张量归约算子。由于 TileLang kernel 处理单个 tensor，采用 **host 侧 for-loop** 遍历 TensorList，对每个张量独立调用同一组范数 kernel。每个张量的计算步骤分解如下：

**Step 0（host 预处理）**：将 ND 张量 flatten 为 1D 连续张量（`x.contiguous().reshape(-1)`），简化 kernel 内部归约逻辑。TensorList 长度动态、ND shape 动态均在 host 侧处理，kernel 内部只面对静态 1D shape。

**Step 1（host 分派）**：根据 `scalar` 值分派到特化 kernel 或通用 kernel：
- `scalar == 0.0` → L0-count kernel（compare + reduce_sum 计非零数）
- `scalar == 1.0` → L1 kernel（abs + reduce_sum）
- `scalar == 2.0` → L2 kernel（mul + reduce_sum + sqrt）
- `scalar == +inf` → Linf kernel（abs + reduce_max）
- `scalar == -inf` → Lneg-inf kernel（abs + reduce_min）
- `scalar > 0` 且非 1/2 → 通用正阶 kernel（exp(p·ln|x|) + reduce_sum + exp(ln(sum)/p)）
- `scalar < 0` 且非 -inf → 通用负阶 kernel（同通用公式，要求输入非零）

**Step 2（kernel 归约）**：单 block kernel 以 `T.serial` 循环遍历 1D 张量的所有 tile，每个 tile 搬入 UB → cast 到 FP32（FP16/BF16 输入时升精度计算，与 golden.py 一致）→ 计算 |x|^p（或特化等价运算）→ `T.reduce_sum`（或 `T.reduce_max/min`）归约到 UB 标量 → `T.tile.add/max/min` 累加到 accumulator。

**Step 3（kernel finalize）**：所有 tile 处理完后，对 accumulator 做最终运算（sqrt for p=2 / exp(ln(sum)/p) for 通用 / identity for p=1,0,inf）→ cast 回原 dtype → 写出标量到 GM。

**Step 4（host 后处理）**：将 [1] shape 的输出 reshape 为 0-dim 标量张量，追加到结果列表。

### 1.5 数据流图

```
TensorList x[0..L-1]
    │
    ├─ host for-loop: for each tensor x[i]
    │     │
    │     ├─ flatten ND → 1D (contiguous)
    │     │
    │     ├─ host dispatch by scalar → select kernel
    │     │
    │     ├─ kernel (single block, serial tile loop):
    │     │     GM[x_flat] ─T.copy→ UB[x_ub]
    │     │         │ cast (FP16/BF16→FP32)
    │     │         ▼
    │     │     UB[x_cal] ─abs─→ UB[abs_ub]
    │     │         │ pow: exp(p·ln|x|)  [或特化: mul/identity]
    │     │         ▼
    │     │     UB[pow_ub] ─T.reduce_sum─→ UB[tile_sum]
    │     │         │ T.tile.add
    │     │         ▼
    │     │     UB[acc_ub] ─finalize: sqrt/exp(ln/p)/identity─→ UB[out_ub]
    │     │         │ cast back (FP32→FP16/BF16)
    │     │         ▼
    │     │     GM[y[1]] ─T.copy← UB[out_ub]
    │     │
    │     └─ reshape [1] → 0-dim scalar → append to results
    │
    └─ return TensorList y[0..L-1]
```

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**：Developer

### 2.2 选型理由

| 算子特征 | 分析 | 结论 |
|---------|------|------|
| 计算类型 | 纯归约（reduction to scalar），无 matmul | 纯 Vector，仅需 UB，不涉及 Cube 核/L1/L0 |
| 复杂度 | 多步（cast → abs → pow → reduce → accumulate → finalize → cast back），但单核内流水完成，无核间协作 | 单核多步，无 CV 融合需求 |
| 内存层级 | 仅 GM ↔ UB，不涉及 L1/L0A/L0B/L0C | 编译器自动映射 shared→UB 即可 |
| 同步 | 单 block 内串行 tile 循环，无跨 block 依赖 | 自动同步足够 |
| 参考实现 | `examples/pos_embedding/rms_norm.py`（Developer 模式 + `T.alloc_shared` + `T.reduce_sum` + FP16→FP32 cast）已验证通过，结构与本算子的 L2 范数分支高度相似 | 同模式可复用 |
| 用户指定 | 用户明确指定 Developer 模式 | 遵循用户选择 |

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_shared(shape, dtype)` — 编译器自动映射到 UB（Vector 核缓冲） |
| 计算方式 | `T.tile.xxx` Buffer 级 SIMD 原语（cast/abs/ln/mul/exp/sqrt/reduce_sum/max/min/add/fill） |
| 作用域 | 编译器自动分离 Cube/Vector（本算子无 Cube 计算，纯 V 核执行） |
| 同步方式 | 自动同步（`TL_ASCEND_AUTO_SYNC=True`），无需手动 `T.barrier_all` / `T.Scope` |

---

## 3. API 映射设计

### 3.1 公式拆解（以通用正阶 p 为例）

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `x_cal = cast_fp32(x)` | FP16/BF16 升 FP32 计算（FP32 输入则直接 copy） |
| 2 | `a = \|x_cal\|` | 取绝对值 |
| 3 | `t = ln(a)` | 对数（为 pow 做准备） |
| 4 | `t = p * t` | 乘以范数阶数 |
| 5 | `t = exp(t)` | 得到 `|x|^p` |
| 6 | `s = reduce_sum(t)` | tile 内归约求和 |
| 7 | `acc += s` | 累加到全局 accumulator |
| 8（finalize） | `acc = exp(ln(acc) / p)` | `acc^(1/p)` = 最终范数 |
| 9（finalize） | `y = cast_orig(acc)` | 转回原 dtype |

### 3.2 TileLang API 映射

#### 3.2.1 通用正阶 kernel（scalar > 0, scalar ≠ 1, 2）

| 步骤 | 数学表达 | TileLang API | 参数 | 来源确认 |
|------|----------|-------------|------|----------|
| 搬入 | `x_ub = X[k*block_N]` | `T.copy(X[k*block_N], x_ub)` | src=GM slice, dst=UB | `rms_norm.py:40` ✓ / api-kernel-memory.md §3 |
| 升精度 | `x_cal = cast(x_ub)` | `T.tile.cast(x_cal, x_ub, "CAST_NONE", block_N)` | dst=UB, src=UB, mode=CAST_NONE, count | `group_norm.py:91` ✓ / api-compute.md §4.9 |
| 取绝对值 | `abs_ub = \|x_cal\|` | `T.tile.abs(abs_ub, x_cal)` | dst=UB, src=UB | api-compute.md §4.2 ✓ |
| 对数 | `abs_ub = ln(abs_ub)` | `T.tile.ln(abs_ub, abs_ub)` | dst=UB（原地）, src=UB | api-compute.md §4.2 ✓ |
| 乘 p | `abs_ub = p * abs_ub` | `T.tile.mul(abs_ub, abs_ub, scalar)` | dst=UB, src0=UB, src1=scalar(float) | `rms_norm.py:48` ✓（mul with eps scalar）/ api-compute.md §4.1 |
| 指数 | `pow_ub = exp(abs_ub)` | `T.tile.exp(pow_ub, abs_ub)` | dst=UB, src=UB | api-compute.md §4.2 ✓ / `softmax.py:83` |
| tile 归约 | `tile_sum = sum(pow_ub)` | `T.reduce_sum(pow_ub, tile_sum_ub, dim=-1, clear=True)` | src=UB[block_N], dst=UB[1], dim=-1 | `rms_norm.py:45` ✓ / api-compute.md §2 |
| 累加 | `acc += tile_sum` | `T.tile.add(acc_ub, acc_ub, tile_sum_ub)` | dst=UB[1], src0=UB[1], src1=UB[1] | api-compute.md §4.1 ✓ / `group_norm.py:94` |
| 初始化 acc | `acc = 0.0` | `T.tile.fill(acc_ub, 0.0)` | dst=UB[1], value=0.0 | api-compute.md §4.10 ✓ / `group_norm.py:75` |
| finalize 对数 | `acc = ln(acc)` | `T.tile.ln(acc_ub, acc_ub)` | dst=UB[1]（原地） | api-compute.md §4.2 ✓ |
| finalize 除 p | `acc = acc / p` | `T.tile.div(acc_ub, acc_ub, scalar)` | dst=UB[1], src1=scalar | api-compute.md §4.1 ✓ |
| finalize 指数 | `acc = exp(acc)` | `T.tile.exp(acc_ub, acc_ub)` | dst=UB[1]（原地） | api-compute.md §4.2 ✓ |
| 降精度 | `out_ub = cast(acc)` | `T.tile.cast(out_ub, acc_ub, "CAST_RINT", 1)` | dst=UB[1], mode=CAST_RINT | `group_norm.py:165` ✓ / api-compute.md §4.9 |
| 搬出 | `Y[0] = out_ub` | `T.copy(out_ub, Y[0])` | src=UB, dst=GM | api-kernel-memory.md §3 ✓ |

#### 3.2.2 L2 特化 kernel（scalar == 2.0）

| 步骤 | 数学表达 | TileLang API | 与通用版的差异 |
|------|----------|-------------|---------------|
| pow | `pow_ub = x_cal * x_cal` | `T.tile.mul(pow_ub, x_cal, x_cal)` | 用 mul 替代 exp(p·ln|x|)，**精度更高**（无 ln/exp 舍入） |
| finalize | `acc = sqrt(acc)` | `T.tile.sqrt(acc_ub, acc_ub)` | 用 sqrt 替代 exp(ln/p)，**精度更高** |

#### 3.2.3 L1 特化 kernel（scalar == 1.0）

| 步骤 | 数学表达 | TileLang API | 与通用版的差异 |
|------|----------|-------------|---------------|
| pow | `pow_ub = \|x_cal\|` | `T.tile.abs(pow_ub, x_cal)` | 仅 abs，无 pow 运算 |
| finalize | identity | 无（跳过 root 步骤） | `acc` 即为结果，直接 cast 输出 |

#### 3.2.4 Linf 特化 kernel（scalar == +inf）

| 步骤 | 数学表达 | TileLang API | 与通用版的差异 |
|------|----------|-------------|---------------|
| tile 归约 | `tile_max = max(\|x_cal\|)` | `T.reduce_max(abs_ub, tile_max_ub, dim=-1, clear=True)` | 用 reduce_max 替代 reduce_sum |
| 累加 | `acc = max(acc, tile_max)` | `T.tile.max(acc_ub, acc_ub, tile_max_ub)` | 用 max 替代 add |
| 初始化 acc | `acc = -inf` | `T.tile.fill(acc_ub, -T.infinity(cal_dtype))` | 初始化为负无穷 |
| finalize | identity | 无 | `acc` 即为结果 |

#### 3.2.5 Lneg-inf 特化 kernel（scalar == -inf）

| 步骤 | 数学表达 | TileLang API | 与 Linf 的差异 |
|------|----------|-------------|---------------|
| tile 归约 | `tile_min = min(\|x_cal\|)` | `T.reduce_min(abs_ub, tile_min_ub, dim=-1, clear=True)` | 用 reduce_min |
| 累加 | `acc = min(acc, tile_min)` | `T.tile.min(acc_ub, acc_ub, tile_min_ub)` | 用 min |
| 初始化 acc | `acc = +inf` | `T.tile.fill(acc_ub, T.infinity(cal_dtype))` | 初始化为正无穷 |

#### 3.2.6 L0-count 特化 kernel（scalar == 0.0）

| 步骤 | 数学表达 | TileLang API | 说明 |
|------|----------|-------------|------|
| 取绝对值 | `abs_ub = \|x_cal\|` | `T.tile.abs(abs_ub, x_cal)` | 先取绝对值 |
| 比较 | `mask = (abs_ub > 0)` | `T.tile.compare(mask_ub, abs_ub, 0.0, "GT")` | 生成 bitmask：非零处 bit=1 | api-compute.md §4.6 ✓ |
| 选择 | `one_ub = select(mask, 1.0, 0.0)` | `T.tile.select(one_ub, mask_ub, 1.0, 0.0, "VSEL_TENSOR_SCALAR_MODE")` | 非零→1.0，零→0.0 | api-compute.md §4.7 ✓ |
| tile 归约 | `tile_count = sum(one_ub)` | `T.reduce_sum(one_ub, tile_count_ub, dim=-1, clear=True)` | 求和=非零计数 |
| 累加 | `acc += tile_count` | `T.tile.add(acc_ub, acc_ub, tile_count_ub)` | 累加计数 |
| finalize | identity | 无 | `acc` 即为非零个数（FP32 表示） |

#### 3.2.7 通用负阶 kernel（scalar < 0, scalar ≠ -inf）

与通用正阶 kernel 结构完全相同（`exp(p·ln|x|) + reduce_sum + exp(ln(sum)/p)`），但 `scalar` 为负值。要求输入元素非零（desc.md 约束），否则 `ln(0) = -inf` 导致 `exp(p·-inf) = exp(+inf) = +inf`，结果为 inf（与 `torch.norm(x, p=-1)` 行为一致）。

### 3.3 计算伪代码（以 L2 特化 kernel 为代表）

```python
import tilelang
import tilelang.language as T

VEC_NUM = 2  # 默认 V 核数（本算子单 block 内 vid 切分可选，L0 可不切分）

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,   # 自动 CV 分离（无 Cube 时退化为纯 V）
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,         # 自动同步（核内）
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,   # 自动内存规划
}


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def l2_norm_kernel(N, block_N, dtype="float16"):
    """L2 范数 kernel: y = sqrt(sum(x_i^2))"""
    n_num = T.ceildiv(N, block_N)
    cal_dtype = "float32" if dtype in ["float16", "bfloat16"] else dtype
    use_upcast = dtype in ["float16", "bfloat16"]

    @T.prim_func
    def main(
        X: T.Tensor((N,), dtype),       # 输入 1D 张量
        Y: T.Tensor((1,), dtype),       # 输出标量（[1] shape，host 侧 reshape 为 0-dim）
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            # 1. 分配 buffer（Developer: alloc_shared 自动映射 UB）
            x_ub = T.alloc_shared((block_N,), dtype)
            x_cal = T.alloc_shared((block_N,), cal_dtype)
            pow_ub = T.alloc_shared((block_N,), cal_dtype)
            tile_sum_ub = T.alloc_shared((1,), cal_dtype)
            acc_ub = T.alloc_shared((1,), cal_dtype)

            # 2. 初始化 accumulator
            T.tile.fill(acc_ub, 0.0)

            # 3. 串行遍历所有 tile，流式归约
            for k in T.serial(n_num):
                T.copy(X[k * block_N], x_ub)                    # GM → UB
                if use_upcast:
                    T.tile.cast(x_cal, x_ub, "CAST_NONE", block_N)   # FP16/BF16 → FP32
                else:
                    T.copy(x_ub, x_cal)                          # FP32 直接 copy
                T.tile.mul(pow_ub, x_cal, x_cal)                 # pow = x^2（精确，无 ln/exp 舍入）
                T.reduce_sum(pow_ub, tile_sum_ub, dim=-1, clear=True)  # tile 内归约
                T.tile.add(acc_ub, acc_ub, tile_sum_ub)          # 累加到 accumulator

            # 4. finalize: sqrt(sum)
            T.tile.sqrt(acc_ub, acc_ub)

            # 5. cast 回原 dtype 并搬出
            if use_upcast:
                out_ub = T.alloc_shared((1,), dtype)
                T.tile.cast(out_ub, acc_ub, "CAST_RINT", 1)      # FP32 → FP16/BF16
                T.copy(out_ub, Y[0])
            else:
                T.copy(acc_ub, Y[0])                             # FP32 直接输出

    return main
```

### 3.4 API 可行性确认

| API | 来源 | 验证状态 |
|-----|------|---------|
| `T.alloc_shared` | api-kernel-memory.md §2 / `rms_norm.py:33` | ✅ 已验证（rms_norm 测试通过） |
| `T.copy` (GM↔UB) | api-kernel-memory.md §3 / `rms_norm.py:40,56` | ✅ 已验证 |
| `T.tile.cast` (CAST_NONE/CAST_RINT) | api-compute.md §4.9 / `group_norm.py:91,165` | ✅ 已验证（group_norm FP16↔FP32 cast） |
| `T.tile.abs` | api-compute.md §4.2 | ✅ 文档确认 |
| `T.tile.ln` | api-compute.md §4.2 | ✅ 文档确认（`dst = ln(src0)`） |
| `T.tile.mul` (scalar src1) | api-compute.md §4.1 / `rms_norm.py:42,48` | ✅ 已验证（mul with scalar） |
| `T.tile.exp` | api-compute.md §4.2 / `softmax.py:83` | ✅ 已验证 |
| `T.tile.sqrt` | api-compute.md §4.2 / `group_norm.py:111` | ✅ 已验证 |
| `T.tile.div` (scalar src1) | api-compute.md §4.1 / `group_norm.py:105` | ✅ 已验证 |
| `T.tile.add` | api-compute.md §4.1 / `group_norm.py:94` | ✅ 已验证 |
| `T.tile.max/min` | api-compute.md §4.1 / `softmax.py:77` | ✅ 已验证 |
| `T.tile.fill` | api-compute.md §4.10 / `group_norm.py:75` / `softmax.py:66` | ✅ 已验证 |
| `T.tile.compare` | api-compute.md §4.6 | ✅ 文档确认 |
| `T.tile.select` | api-compute.md §4.7 | ✅ 文档确认 |
| `T.reduce_sum` | api-compute.md §2 / `rms_norm.py:45` / `group_norm.py:99` | ✅ 已验证（UB tile 归约） |
| `T.reduce_max` | api-compute.md §2 / `softmax.py:76` | ✅ 已验证 |
| `T.reduce_min` | api-compute.md §2 / `reduce_min.py:27` | ✅ 已验证 |
| `T.infinity(cal_dtype)` | `softmax.py:66` (`-T.infinity(cal_dtype)`) | ✅ 已验证 |
| `T.ceildiv` | `sigmoid.py:16` / `softmax.py:35` | ✅ 已验证（处理非整除） |
| `T.serial` | `group_norm.py:81` / `softmax.py:69` | ✅ 已验证 |

**所有 API 均来自 `examples/` 已验证实现或 `api-best-practices` 文档，无凭记忆猜测。**

---

## 3.5 技术约束确认

### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | No | 一维 `T.Kernel(1)`（单 block 串行 tile 循环），无三维需求 |
| threads 参数限制（仅 1 或 2） | No | 不设 threads 参数（单 block 内默认 vid 处理；L0 不强制 vid 切分） |
| 动态循环边界不支持 | No | `T.serial(n_num)` 中 `n_num = T.ceildiv(N, block_N)` 在 JIT 编译时由 `N` 参数确定（每次 shape 编译一个版本），非 tensor 值依赖 |
| 流水线不支持动态边界 | No | 不使用 `T.Pipelined`（L0 用 `T.serial` 串行 tile 循环；Stage 3 可探索双 buffer 流水线优化） |
| GPU 专用 API | No | 全部使用 `T.tile.xxx` Ascend 原语，无 CUDA API |
| GEMM 非整除风险 | N/A | 本算子非 GEMM，不涉及 matmul 分块 |
| L0C 容量上限 | N/A | 本算子纯 Vector（无 Cube 计算），不分配 L0C |

### 3.5.2 参考实现差异说明

本算子参考 `cann-bench-master/tasks/level1/foreach_norm/golden.py`（PyTorch golden 实现），非 GPU kernel 迁移。差异点：

| 差异项 | golden.py（PyTorch） | 本项目（TileLang-Ascend） | 转换方案 |
|--------|---------------------|-------------------------|----------|
| TensorList 处理 | `torch.norm` 原生支持 list comprehension | TileLang kernel 处理单 tensor | host 侧 for-loop 遍历 TensorList |
| ND shape | `torch.norm` 原生支持 ND | TileLang kernel 用静态 1D shape | host 侧 flatten ND → 1D（contiguous reshape） |
| scalar=inf/-inf | `torch.norm(p=inf)` 原生支持 | inf 不能作为 JIT float 参数 | host 侧分派到特化 reduce_max/min kernel |
| FP16/BF16 计算 | `golden.py` 显式升 FP32 计算 | 同（`T.tile.cast` CAST_NONE → FP32 计算 → CAST_RINT 回原 dtype） | 与 golden.py 一致 |
| p=0 计数 | `torch.norm(p=0)` 返回非零个数 | compare + select + reduce_sum | 用 bitmask → 1.0/0.0 → sum 实现 |
| 通用 p 的 pow | `torch.pow(\|x\|, p)` | `exp(p * ln(\|x\|))` | 用 exp+ln 组合实现（TileLang 无 T.tile.pow 原语） |

### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/pos_embedding/rms_norm.py` | **极高** | L2 范数分支的直接参考：`T.alloc_shared` + `T.tile.cast`(FP16→FP32) + `T.tile.mul`(x*x) + `T.reduce_sum(dim=-1)` + `T.tile.sqrt` + `T.tile.cast`(FP32→FP16)。Developer 模式 + 全 pass_configs 开启。结构与本算子 L2 kernel 几乎一致 |
| `examples/group_norm/example_group_norm.py` | 高 | 多 dtype（FP16/BF16/FP32）cast 模式（`CAST_LOW2HIGH`/`CAST_HIGH2LOW`）、`T.reduce_sum` 级联归约（sum_a → sum_row → total）、MERE/MARE 精度阈值（2^-10/2^-7/2^-13，与本算子 desc.md 一致）、`T.tile.fill` 初始化、`T.tile.add` 累加模式 |
| `examples/softmax/example_online_softmax.py` | 高 | 串行 tile 循环（`T.serial(n_num)`）+ 流式归约累加模式（`T.reduce_max` per tile + `T.tile.max` merge）、`T.tile.fill(prev_max, -T.infinity(cal_dtype))` 初始化、`T.tile.exp`/`T.tile.sub`/`T.tile.mul` 组合运算。Developer 模式参考 |
| `examples/reduce/example_reduce_min.py` | 中 | `T.reduce_min(a_ub, b_ub, dim=-1)` 的直接用法、`T.alloc_ub` + `T.Scope("V")` + `T.barrier_all`（Expert 风格，本算子用 Developer 不直接采用，但 reduce 原语用法可参考） |
| `cann-bench-master/.../foreach_norm/golden.py` | **极高（golden 源）** | PyTorch golden 实现：`torch.norm(tensor, p=scalar)`，FP16/BF16 升 FP32 计算。本算子精度对标的权威 golden |
| `cann-bench-master/.../foreach_norm/desc.md` | **极高（需求源）** | 算子 API 描述、数据类型（float16/float32/bfloat16）、scalar 范围（-1024~1024 含 inf）、精度阈值（MERE < 2^-10/2^-7/2^-13） |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| x（TensorList） | 每个 tensor ND（1D~5D，实测各维 2~1000003，元素总数 1~48M） | float16 / float32 / bfloat16 | 输入张量列表，列表长度 1~64（cases.csv 实测 1~4）。各 tensor dtype 须一致。host 侧 flatten 为 1D 后传给 kernel |

**attr 参数**：

| 参数名 | 类型 | 范围 | 说明 |
|--------|------|------|------|
| scalar | float | -1024.0 ~ 1024.0（含 ±inf、0、负阶、分数阶如 1.5/2.5） | 范数阶数。负阶或 0 阶时输入元素须非零（desc.md 约束） |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| y（TensorList） | 每个元素为标量张量（0-dim，kernel 输出 [1] 后 host reshape） | 与对应输入 tensor dtype 一致 | 每个输入 tensor 的范数结果。列表长度与输入一致 |

### 4.3 中间缓冲区（以通用正阶 kernel 为例，block_N=8192）

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| x_ub | (block_N,) | 输入 dtype | UB（alloc_shared 自动映射） | 输入 tile 缓冲（GM → UB） |
| x_cal | (block_N,) | cal_dtype (FP32) | UB | 升精度后的计算缓冲 |
| abs_ub | (block_N,) | cal_dtype | UB | \|x\| 缓冲（可复用 x_cal） |
| pow_ub | (block_N,) | cal_dtype | UB | \|x\|^p 缓冲 |
| tile_sum_ub | (1,) | cal_dtype | UB | tile 内归约结果 |
| acc_ub | (1,) | cal_dtype | UB | 全局 accumulator |
| out_ub | (1,) | 输入 dtype | UB | 降精度后输出缓冲（仅 FP16/BF16 输入时分配） |

### 4.4 内存搬运路径

```
纯 Vector 路径（单 block 串行 tile 循环）：

GM[X[k*block_N]] --T.copy--> UB[x_ub]
                               |
                    tile.cast(x_cal, x_ub, CAST_NONE)    # FP16/BF16 → FP32
                               |
                    tile.abs(abs_ub, x_cal)              # |x|
                               |
                    tile.ln(abs_ub, abs_ub)              # ln|x|
                               |
                    tile.mul(abs_ub, abs_ub, scalar)     # p·ln|x|
                               |
                    tile.exp(pow_ub, abs_ub)             # |x|^p
                               |
                    T.reduce_sum(pow_ub, tile_sum_ub)    # tile 归约
                               |
                    tile.add(acc_ub, acc_ub, tile_sum_ub)# 累加
                               |
                    ┌────────── k 循环结束 ──────────┐
                               |
                    tile.ln(acc_ub, acc_ub)             # ln(sum)
                    tile.div(acc_ub, acc_ub, scalar)    # ln(sum)/p
                    tile.exp(acc_ub, acc_ub)            # sum^(1/p)
                               |
                    tile.cast(out_ub, acc_ub, CAST_RINT)# FP32 → FP16/BF16
                               |
UB[out_ub] ---------T.copy--> GM[Y[0]]
```

**层级说明**：纯 Vector 算子，数据全程在 UB 上操作，不涉及 L1（Cube 缓存）/ L0A / L0B / L0C。`T.alloc_shared` 在无 Cube 计算时被编译器自动映射到 UB。

### 4.5 UB 内存预算

以主配置 `block_N=8192` 为例：

| Buffer | Shape | dtype | 大小 (Bytes) |
|--------|-------|-------|-------------|
| x_ub | (8192,) | float16 | 8192 × 2 = 16384 (16 KB) |
| x_cal | (8192,) | float32 | 8192 × 4 = 32768 (32 KB) |
| abs_ub | (8192,) | float32 | 32768 (32 KB)（可复用 x_cal，实际可省） |
| pow_ub | (8192,) | float32 | 32768 (32 KB) |
| tile_sum_ub | (1,) | float32 | 4 (4 B，可忽略) |
| acc_ub | (1,) | float32 | 4 (4 B，可忽略) |
| out_ub | (1,) | float16 | 2 (2 B，可忽略) |
| **总计（不复用）** | | | **~114 KB** |
| **总计（复用 x_cal/abs_ub）** | | | **~80 KB** |

- 目标平台 UB 容量：196608 Byte（192 KB，Ascend910B3，见 api-kernel-memory.md §2）
- FP16 输入占用比：114 KB / 192 KB = 59% ✓（不复用）；80 KB / 192 KB = 42% ✓（复用）
- FP32 输入时 x_ub 翻倍至 32 KB，总计 ~130 KB / 192 KB = 68% ✓
- BF16 输入与 FP16 相同（均为 2 Byte）
- **结论**：block_N=8192 在所有 dtype 下 UB 预算充裕 ✓

### 4.6 动态轴定义

| 动态轴 | 声明方式 | 运行时范围 | 说明 |
|--------|----------|-----------|------|
| N（1D flatten 后元素总数） | 作为 `@tilelang.jit` 函数参数传入 | 1 ~ 48M | 每次 shape 编译一个 kernel 版本（参考 `rms_norm.py` 模式：`rms_norm(M, head_dim, block_M, eps, dtype)`） |
| TensorList 长度 L | host 侧 for-loop | 1 ~ 64 | host 侧处理，kernel 不感知 |
| ND shape | host 侧 flatten | 1D~5D | host 侧 `reshape(-1)` 处理，kernel 只面对 1D |

**动态 shape 策略说明**：
- **主方案（采用）**：N 作为 `jit` 函数参数，每次调用 `lp_norm_kernel(N, block_N, scalar, dtype)` 编译一个针对该 N 的 kernel。这是 `examples/pos_embedding/rms_norm.py` 的已验证做法，简单可靠，编译器可充分优化。L0 测试计划中每个 shape 各自编译。
- **scalar 作为 JIT 参数**：通用 kernel 的 `scalar` 作为 JIT float 参数传入（如 `rms_norm.py` 的 `eps` 参数模式）。scalar=inf/-inf 由 host 分派到特化 kernel（inf 不能作为 JIT float 参数）。
- **block_N 自适应**：host 侧 `_choose_block_N(N, dtype)` 根据 N 和 dtype 选择 block_N（小 N 用小 block_N 避免浪费，大 N 用 block_N=8192 平衡 UB 占用与 tile 数）。

### 4.7 JIT 配置

```python
@tilelang.jit(
    out_idx=[1],                # Y（第 2 个参数）为输出
    pass_configs={
        tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,   # 自动 CV 分离（无 Cube 时退化为纯 V）
        tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,         # 自动同步（核内）
        tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,   # 自动内存规划（UB 分配优化 + buffer 复用）
    },
)
def l2_norm_kernel(N, block_N, dtype="float16"):
    ...
    return main
```

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**：纯 Vector（归约到标量）

**判定依据**：算子仅包含 element-wise 运算（cast/abs/ln/mul/exp/sqrt）+ 归约（reduce_sum/max/min），无 matmul。数据全程在 UB 上操作，不涉及 Cube 核（L1/L0）。参考 `rms_norm.py`（同样为纯 Vector 归约）。

### 5.2 Block 划分

```python
# 单 block 串行 tile 循环（L0 主方案）
block_N = 8192   # tile 大小：8192 元素，平衡 UB 占用与 tile 数
                 # - FP16: 8192×2=16KB/buffer，4 buffer ≈ 80KB < 192KB ✓
                 # - FP32: 8192×4=32KB/buffer，4 buffer ≈ 130KB < 192KB ✓
                 # - 8192 是 32 的整数倍，满足 UB 32B 对齐 ✓

n_num = T.ceildiv(N, block_N)   # tile 数（ceildiv 处理非整除尾块）
block_num = 1                    # 单 block，串行遍历所有 tile
```

**block size 选择理由**：
- `block_N=8192`：与 `softmax.py` 的 `block_N=128` 相比更大，因为本算子是 1D 归约（不像 softmax 按 2D tile 分块），大 tile 减少 `T.serial` 循环次数，降低循环开销。UB 预算充裕（80~130KB < 192KB）。
- 小 shape 场景（如 L0 的 (8192,)）：`block_N=8192`，`n_num=1`，单 tile 单 block，最快路径。
- 大 shape 场景（如 48M 元素）：`block_N=8192`，`n_num≈5859`，单 block 串行 5859 次 tile 循环。L0/L1 精度收敛阶段可接受（正确性优先）；Stage 3 性能调优可改为多 block + `T.tile.atomic_add` 跨 block 并行归约。

### 5.3 约束分析

- **UB 对齐约束**：`block_N=8192` × fp16(2B) = 16384B，32B 整除 ✓（fp32: 32768B，同样 32B 整除 ✓）
- **UB 容量**：4 buffer × ~32KB = ~130KB < 192KB（Ascend910B3 UB）✓（详见 §4.5）
- **L0 容量**：无 Cube 计算，不适用
- **V 核切分**：单 block 内 vid 可选切分。L0 不强制 vid 切分（单 block 串行循环已足够）；若 Stage 3 需优化，可将 tile 内 element-wise 计算（cast/abs/pow）按 `vid` 切分到 2 个 V 核并行，但 `reduce_sum` 仍需单核完成。

### 5.4 注意事项（非整除处理）

**非整除场景**：当 `N % block_N ≠ 0` 时（如 N=1000003，block_N=8192，余 4115）：
- 使用 `T.ceildiv(N, block_N)` 计算 tile 数（向上取整，保证覆盖所有元素）
- `T.copy` 已支持动态 shape 切片自动处理尾块（参考 api-kernel-memory.md §3 "T.copy 动态 shape 切片"），**不需要 host 侧 zero-padding**
- 尾块 tile 中超出有效范围的部分：`T.copy` 只搬有效元素（`actual_len = min(block_N, N - k*block_N)`），但 `T.tile.cast/abs/ln/mul/exp` 对全 buffer 操作。若尾块不足 block_N，剩余位置的值不影响 reduce_sum 结果（搬入时 `T.copy` 自动处理，未搬入的 buffer 位置为旧值或 padding，但 reduce_sum 只对有效元素求和——需确认 `T.copy` 尾块行为）。
- **安全做法**：host 侧对 N 非 block_N 整除时，用 `torch.nn.functional.pad(x_flat, (0, block_N - N % block_N))` 补零到 block_N 整数倍。零元素对范数结果无贡献（`|0|^p = 0` for p>0；`0` 不影响 max/min；`0` 不计入 L0 count），安全。这避免了尾块 buffer 未初始化的风险。L0 测试用 block 整除 shape 不涉及此问题。

**L0 测试**只用 block 整除的规则 shape；非整除/尾块/质数 shape 留给 L1（Stage 2 由 `tilelang-op-test-design` 场景 B 扩展）。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| Block 级 | 单 block | `T.Kernel(1)` | 单 block 串行 tile 循环，无跨 block 依赖。归约到标量的自然结构（参考 `rms_norm.py:30` `T.Kernel(m_num, ...)` 按 row 分 block；本算子 1D 归约到标量，单 block 串行最简洁） |
| Tile 级（K 维迭代） | 串行循环 | `T.serial(n_num)` | 串行遍历所有 tile，流式累加到 accumulator。参考 `softmax.py:69` `for by in T.serial(n_num)` |
| 元素级计算 | 无显式循环 | `T.tile.xxx` 原语 | Buffer 级 SIMD 操作，整个 tile 一次性计算（cast/abs/ln/mul/exp），无需 `T.Parallel` 循环 |
| Tile 内归约 | 隐式 | `T.reduce_sum/max/min(pow_ub, tile_sum_ub, dim=-1)` | Ascend fast-path reduce 原语，硬件级归约 |

**说明**：本算子用 `T.tile.xxx` 原语（Buffer 级 SIMD）+ `T.reduce_sum` 原语，而非 `T.Parallel` + 符号 API。原因：`T.tile.xxx` 直接触发 Ascend Vector 指令，性能更优且与 `rms_norm.py`/`group_norm.py`/`softmax.py` 已验证实现一致。

### 6.2 循环伪代码

```python
# 单 block 串行 tile 循环（L0 主方案）
with T.Kernel(1, is_npu=True) as (cid, vid):
    # 分配 buffer
    x_ub = T.alloc_shared((block_N,), dtype)
    x_cal = T.alloc_shared((block_N,), cal_dtype)
    pow_ub = T.alloc_shared((block_N,), cal_dtype)
    tile_sum_ub = T.alloc_shared((1,), cal_dtype)
    acc_ub = T.alloc_shared((1,), cal_dtype)

    # 初始化 accumulator（sum 类: 0.0; max 类: -inf; min 类: +inf）
    T.tile.fill(acc_ub, 0.0)

    # 串行遍历所有 tile，流式归约
    for k in T.serial(n_num):
        T.copy(X[k * block_N], x_ub)                        # GM → UB
        T.tile.cast(x_cal, x_ub, "CAST_NONE", block_N)     # FP16/BF16 → FP32
        T.tile.mul(pow_ub, x_cal, x_cal)                   # x²（L2 特化）
        T.reduce_sum(pow_ub, tile_sum_ub, dim=-1, clear=True)  # tile 归约
        T.tile.add(acc_ub, acc_ub, tile_sum_ub)             # 累加

    # finalize
    T.tile.sqrt(acc_ub, acc_ub)                             # sqrt(sum)（L2 特化）
    T.tile.cast(out_ub, acc_ub, "CAST_RINT", 1)             # FP32 → FP16/BF16
    T.copy(out_ub, Y[0])                                    # UB → GM
```

### 6.3 流水线优化

**L0 不使用 `T.Pipelined`**。理由：
- 单 block 内串行 tile 循环，搬入/计算/归约三步存在天然串行依赖（后一步依赖前一步结果）
- L0 优先正确性，流水线优化留待 Stage 3
- `group_norm.py` 的 serial kernel 用 `T.serial` + double-buffer（`data_buf_p1[0/1]`）实现 MTE2/V/MTE3 3-stage 手动流水线，Stage 3 可参考此模式优化大 shape 性能

**Stage 3 优化预留**（不在 L0 实现）：
- 双 buffer 流水线：`T.Pipelined(n_num, num_stages=2)` 预取下一 tile 同时计算当前 tile
- 多 block 并行：`T.Kernel(n_num)` 每个 block 处理一个 tile + `T.tile.atomic_add(PartialSum[0], tile_sum_ub)` 跨 block 累加 + 独立 finalize kernel

### 6.4 尾块处理

见 §5.4。L0 用 block 整除 shape，不涉及尾块；非整除由 `T.ceildiv` + host 侧补零处理。

---

## 7. 同步策略

### 7.1 同步模式

**模式**：自动同步（Developer 模式）

### 7.2 同步点说明

Developer 模式下由 `TL_ASCEND_AUTO_SYNC=True` 自动插入同步点，无需手动 `T.barrier_all` / `T.set_flag` / `T.wait_flag`。

| 位置 | 同步方式 | 理由 |
|------|----------|------|
| `T.copy` 搬入后 | 自动同步 | 确保数据写入 UB 完成后再 cast 计算 |
| `T.tile.cast` 后 | 自动同步 | 确保 cast 完成后再 abs/mul |
| `T.tile.xxx` 之间（ln→mul→exp） | 自动同步 | 原地复用 buffer，需保证前一步写完成 |
| `T.reduce_sum` 后 | 自动同步 | 确保 tile_sum 写完成后再 add 到 acc |
| `T.tile.add` 后 | 自动同步 | 确保 accumulator 更新完成后再进下一 tile 循环 |
| finalize 步骤间 | 自动同步 | ln→div→exp 串行依赖 |
| `T.copy` 搬出前 | 自动同步 | 确保 cast 结果写完成后再搬出 GM |

### 7.3 pass_configs 配置

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,   # 自动 CV 分离（无 Cube 时退化为纯 V，无 workspace）
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,         # 自动同步（核内，含上述同步点）
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,   # 自动内存规划（UB 分配优化 + buffer 复用）
}
```

**未开启的配置**：
- `TL_ASCEND_AUTO_CV_SYNC`：核间同步，本算子单 block 无核间依赖，不需要

---

## 8. 融合算子设计

### 8.1 融合算子判定

**判定结果**：否

**判定依据**：ForeachNorm 是纯归约（reduction to scalar）算子，无 GEMM（matmul）计算，不存在 Cube↔Vector 核间协作需求。Developer 模式下 `TL_ASCEND_AUTO_CV_COMBINE=True` 对纯 Vector 算子退化为无操作（不产生 workspace/vid 开销）。

本章节不适用，无 workspace 规格、无 CV 交互设计。

---

## 9. 验证方案

### 9.1 Golden 函数

```python
import torch
from typing import List

def golden_foreach_norm(x: List[torch.Tensor], scalar: float) -> List[torch.Tensor]:
    """ForeachNorm 参考实现（PyTorch，与 cann-bench golden.py 一致）。
    
    对输入张量列表的每个张量求 p 范数。FP16/BF16 输入升 FP32 计算保证精度，
    结果转回原 dtype。
    
    Args:
        x: 输入张量列表（TensorList），各 tensor dtype 须一致
        scalar: 范数阶数（支持 0/1/2/inf/-inf/负阶/分数阶）
    Returns:
        y: 输出张量列表，每个元素为标量张量，dtype 与对应输入一致
    """
    input_dtype = x[0].dtype if x else torch.float32
    
    # FP16/BF16 输入需要升到 FP32 计算以保证精度（与 golden.py 一致）
    if input_dtype in (torch.float16, torch.bfloat16):
        compute_dtype = torch.float32
    else:
        compute_dtype = input_dtype
    
    x_compute = [t.to(compute_dtype) for t in x]
    y = [torch.norm(tensor, p=scalar) for tensor in x_compute]
    
    # 转回原始 dtype
    if input_dtype in (torch.float16, torch.bfloat16):
        return [t.to(input_dtype) for t in y]
    return y
```

**Golden 选择说明**：直接采用 `cann-bench-master/.../foreach_norm/golden.py` 的实现（`torch.norm(tensor, p=scalar)` + FP16/BF16 升 FP32）。`torch.norm` 是 PyTorch 标准实现，数值稳定，作为本算子的权威 golden。对 `scalar=inf/-inf/0/负阶`，`torch.norm` 原生支持。

### 9.2 L0 门槛测试计划

> 设计阶段**只给出 L0 门槛用例**（规则 shape，block 整除），供 Stage 2 快速精度收敛。
> L1（功能，含不规则/尾块/质数 shape）/ L2（异常输入）/ Boundary（INF/NAN/极值）的**完整分层套件由 `tilelang-op-test-design` 场景 B 在 Stage 2 L0 通过后扩展**——不在此枚举。

**算子类别判断**（由 `tilelang-op-test-design` 场景 A 生成）：
- 计算类型：纯 Vector（归约到标量，无 matmul）
- 复杂度：Multi（cast → abs → pow → reduce → accumulate → finalize → cast back，6~9 步分解）
- 数学特征：`sum/max + pow + root` → Reduction 类
- 综合类别：Reduction（多步归约到标量）
- 测试策略：dtype 组合（float16 + float32 + bfloat16）+ scalar 组合（1/2/3/inf/0/-1）+ 规则 shape 组合 + TensorList 长度覆盖

**L0 用例集**（规则 shape，block_N=8192 整除；≤50 用例）：

| 用例名 | 级别 | Shape（每个 tensor） | dtype | scalar | block_N | TensorList 长度 | 说明 |
|--------|------|---------------------|-------|--------|---------|-----------------|------|
| l0_l2_fp16_small | L0 | (8192,) | float16 | 2.0 | 8192 | 1 | L2 最小规则，单 tile，基础功能验证 |
| l0_l2_fp16_mid | L0 | (32768,) | float16 | 2.0 | 8192 | 1 | L2 多 tile（4 tiles），串行循环验证 |
| l0_l2_fp16_large | L0 | (131072,) | float16 | 2.0 | 8192 | 1 | L2 大规模（16 tiles），accumulator 累加验证 |
| l0_l1_fp16 | L0 | (8192,) | float16 | 1.0 | 8192 | 1 | L1 范数分支（abs + reduce_sum，无 pow/root） |
| l0_l3_fp16 | L0 | (8192,) | float16 | 3.0 | 8192 | 1 | 通用正阶 p=3（exp(p·ln\|x\|) 路径） |
| l0_linf_fp16 | L0 | (8192,) | float16 | inf | 8192 | 1 | inf 范数分支（reduce_max + tile.max 累加） |
| l0_l0_count_fp16 | L0 | (8192,) | float16 | 0.0 | 8192 | 1 | L0 计数分支（compare + select + reduce_sum） |
| l0_lneg1_fp16 | L0 | (8192,) | float16 | -1.0 | 8192 | 1 | 负阶分支（通用公式 p<0，输入须非零） |
| l0_l2_fp32_mid | L0 | (32768,) | float32 | 2.0 | 8192 | 1 | float32 L2（不升精度，验证 FP32 直接计算路径） |
| l0_l1_fp32 | L0 | (8192,) | float32 | 1.0 | 8192 | 1 | float32 L1 |
| l0_l2_bf16_mid | L0 | (32768,) | bfloat16 | 2.0 | 8192 | 1 | bfloat16 L2（升 FP32 计算，验证 BF16 cast 路径） |
| l0_l1_bf16 | L0 | (8192,) | bfloat16 | 1.0 | 8192 | 1 | bfloat16 L1 |
| l0_l2_2d_fp16 | L0 | (1024, 1024) | float16 | 2.0 | 8192 | 1 | 2D shape → host flatten 1D（1048576 元素，128 tiles），验证 ND flatten 路径 |
| l0_l2_tl2_fp16 | L0 | (8192,) × 2 | float16 | 2.0 | 8192 | 2 | TensorList 长度 2，验证 host for-loop |
| l0_l1_tl3_fp32 | L0 | (8192,) × 3 | float32 | 1.0 | 8192 | 3 | TensorList 长度 3，验证 host for-loop |

**L0 覆盖维度**：
- **dtype 全集**：float16（8 用例）、float32（3 用例）、bfloat16（2 用例）= 3/3 dtypes ✓
- **scalar 分支全集**：p=1（3 用例）、p=2（6 用例）、p=3 通用正阶（1 用例）、p=inf（1 用例）、p=0 计数（1 用例）、p=-1 负阶（1 用例）= 6/7 分支 ✓（p=-inf 与 p=inf 结构对称，L1 补充）
- **shape 规模**：单 tile（8192）、多 tile（32768/131072）、2D flatten（1024×1024）= 4 种规模 ✓
- **TensorList 长度**：1（12 用例）、2（1 用例）、3（1 用例）= 3 种长度 ✓

**L0 输入数据生成**：
- 默认 `torch.randn(shape, dtype=..., device='npu')`，标准正态分布
- `scalar=0.0`（L0 count）：含约 0% 精确零值（randn 极少精确零），count ≈ N，验证计数正确性
- `scalar=-1.0`（负阶）：randn 极少精确零，满足"非零"约束；保险起见 host 侧 `x[x == 0] = 1.0`
- 极值/INF/NAN 输入留给 Boundary 测试（Stage 2 扩展）

**L0 验证流程**（供 Stage 2 落地参考）：
```python
def test_foreach_norm_l0():
    """L0 门槛测试：规则 shape，block_N 整除。返回是否全过。"""
    test_configs = [
        # (dtype, shape_per_tensor, scalar, block_N, list_len)
        ("float16", (8192,),       2.0,    8192, 1),  # l0_l2_fp16_small
        ("float16", (32768,),      2.0,    8192, 1),  # l0_l2_fp16_mid
        ("float16", (131072,),     2.0,    8192, 1),  # l0_l2_fp16_large
        ("float16", (8192,),       1.0,    8192, 1),  # l0_l1_fp16
        ("float16", (8192,),       3.0,    8192, 1),  # l0_l3_fp16
        ("float16", (8192,),       float("inf"), 8192, 1),  # l0_linf_fp16
        ("float16", (8192,),       0.0,    8192, 1),  # l0_l0_count_fp16
        ("float16", (8192,),       -1.0,   8192, 1),  # l0_lneg1_fp16
        ("float32", (32768,),      2.0,    8192, 1),  # l0_l2_fp32_mid
        ("float32", (8192,),       1.0,    8192, 1),  # l0_l1_fp32
        ("bfloat16", (32768,),     2.0,    8192, 1),  # l0_l2_bf16_mid
        ("bfloat16", (8192,),      1.0,    8192, 1),  # l0_l1_bf16
        ("float16", (1024, 1024),  2.0,    8192, 1),  # l0_l2_2d_fp16
        ("float16", (8192,),       2.0,    8192, 2),  # l0_l2_tl2_fp16
        ("float32", (8192,),       1.0,    8192, 3),  # l0_l1_tl3_fp32
    ]
    ok = True
    for dtype, shape, scalar, block_N, list_len in test_configs:
        torch_dtype = getattr(torch, dtype)
        x_list = [torch.randn(shape, dtype=torch_dtype, device="npu") for _ in range(list_len)]
        if scalar == -1.0:  # 负阶须非零
            for t in x_list:
                t[t == 0] = 1.0
        y_list = foreach_norm(x_list, scalar)
        ref_list = golden_foreach_norm(x_list, scalar)
        # 逐 tensor 校验精度
        for i, (y, ref) in enumerate(zip(y_list, ref_list)):
            passed, ratio, max_abs = check_precision(y, ref, dtype)
            tag = "PASS" if passed else "FAIL"
            print(f"[PRECISION_{tag}] l0 case shape={shape} dtype={dtype} scalar={scalar} "
                  f"tl_idx={i}/{list_len} matched_ratio={ratio:.4f} max_abs={max_abs:.3e}")
            ok &= passed
    return ok
```

### 9.3 精度标准

> 采用**混合容差**：逐元素 `|actual-golden| ≤ atol + rtol·|golden|`，整体判定 `matched_ratio ≥ required_matched_ratio` **且** `max_abs_error ≤ max_abs_error_limit`。
> 阈值**仅按 dtype**（与算子类别无关），L0/L1/Boundary 套用精度比对（L2 为非法输入负向测试，不比精度）；整型按 0 误差精确匹配。完整定义见 `tilelang-op-test-design/references/precision-standard.md`。

本算子支持 float16 + float32 + bfloat16 三个 dtype（与 §4 数据规格、cann-bench desc.md §3 一致）：

| dtype | atol | rtol | max_abs_error_limit | required_matched_ratio |
|-------|------|------|---------------------|------------------------|
| float16 | 2⁻¹⁴ (6.10e-5) | 2⁻⁹ (1.95e-3) | 1e-1 | 0.99 |
| bfloat16 | 2⁻¹⁰ (9.77e-4) | 2⁻⁶ (1.56e-2) | 1e0 | 0.99 |
| float32 | 2⁻¹⁶ (1.53e-5) | 2⁻¹⁰ (9.77e-4) | 1e-2 | 0.99 |

> 阈值取自 `precision-standard.md §二`，与算子类别无关。`required_matched_ratio` 浮点统一 0.99；本算子无整型 dtype，不列整型行。
>
> **与 cann-bench desc.md §4 精度标准的对应关系**：desc.md 采用 MERE（平均相对误差）< Threshold 且 MARE（最大相对误差）< 10×Threshold，其中 Threshold: float16=2⁻¹⁰, bfloat16=2⁻⁷, float32=2⁻¹³。本设计采用 `precision-standard.md` 的混合容差标准（atol+rtol·|golden| + max_abs_error_limit + required_matched_ratio），这是测试框架 `check_precision()` 的统一判定方式。混合容差对小值场景（golden 接近 0）更鲁棒（atol 兜底），对大值场景用 rtol 保证相对精度。两个标准在标准正态输入下等价性良好；`group_norm.py` 示例同时用 MERE/MARE 阈值（2^-10/2^-7/2^-13）且测试通过，说明本算子的混合容差阈值（2^-14/2^-10/2^-16 atol + 2^-9/2^-6/2^-10 rtol）在精度上足够严格。
>
> **FP16/BF16 升精度策略**：与 golden.py 一致，FP16/BF16 输入升 FP32 计算（`T.tile.cast` CAST_NONE → FP32 运算 → CAST_RINT 回原 dtype），中间计算在 FP32 完成，只有最终 cast 引入一次舍入，精度有保障。L2 特化 kernel 用 `mul`(x²) + `sqrt` 替代通用 `exp(p·ln|x|)` + `exp(ln/p)`，避免 ln/exp 舍入，精度更高。

---

## 10. 风险点与注意事项

### 10.1 已知约束（技术约束检测结论）

| 约束 | 本算子状态 | 说明 |
|------|-----------|------|
| 三维 Kernel | 不涉及 | 一维 `T.Kernel(1)`，单 block 串行 tile 循环 |
| threads 参数 | 不涉及 | 不设 threads，单 block 内默认 vid 处理 |
| 动态循环边界 | 不涉及 | `T.serial(n_num)` 中 `n_num` 由 JIT 参数 `N` 编译期确定，非 tensor 值依赖 |
| 流水线动态边界 | 不涉及 | L0 不使用 `T.Pipelined`；Stage 3 优化预留 |
| GPU 专用 API | 不涉及 | 全部 `T.tile.xxx` Ascend 原语 |
| L0C 容量 | N/A | 纯 Vector 算子，无 Cube 计算，不分配 L0C |
| GEMM 非整除 | N/A | 非 GEMM 算子 |
| UB 容量 | 已预算 | block_N=8192 时 ~80~130KB < 192KB ✓ |
| UB 对齐 | 已满足 | block_N=8192 × fp16(2B) = 16384B，32B 整除 ✓ |

### 10.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| `T.tile.ln(0) = -inf` | 通用 p kernel 中输入含零元素（p>0 时 `exp(p·-inf)=0` 正确；p<0 时 `exp(p·-inf)=+inf` 结果为 inf） | p<0 负阶时零元素产生 inf 结果 | host 侧校验：负阶 scalar 时输入须非零（desc.md 约束）；kernel 内不特殊处理（与 `torch.norm(p=-1)` 行为一致） |
| `T.reduce_sum` 1D→标量 shape 不支持 | 1D buffer `[block_N]` reduce dim=-1 到 `[1]` 可能不被前端支持 | 编译错误 | 改用 2D buffer `[1, block_N]` reduce dim=-1 到 `[1]`（参考 `rms_norm.py` 2D 模式）；或 reduce 到 `[]` 后 cast |
| 尾块 buffer 未初始化 | N 非 block_N 整除时，尾块 tile 不足 block_N，剩余 buffer 位置为旧值 | reduce_sum 包含垃圾值，结果偏大 | host 侧补零到 block_N 整数倍（零元素对范数无贡献）；或用 `T.copy` 动态切片 `actual_len` 限制搬入量 + `real_shape` 参数限制 reduce 范围 |
| FP16/BF16 精度不足 | 通用 p kernel 的 `exp(p·ln|x|)` 在 FP16 下 ln/exp 舍入累积 | 精度超阈值 | 已通过升 FP32 计算规避（CAST_NONE → FP32 运算 → CAST_RINT 回原 dtype）；L2 特化用 mul+sqrt 避免 ln/exp |
| `T.tile.atomic_add` 跨 block 竞态 | Stage 3 多 block 优化时，多个 block 同时 atomic_add 到同一 GM 标量 | 累加结果不确定 | L0 不涉及（单 block）；Stage 3 用 atomic_add（硬件保证原子性）或改用两 pass（block 各写 GM[block_id] + finalize kernel 归约） |
| 读写索引不一致 | vid 切分时读写偏移不匹配 | 结果错乱 | L0 单 block 不涉及 vid 切分；Stage 3 若切分须读写索引一致（参考 api-kernel-memory.md §4.3） |
| scalar=inf 作为 JIT float 参数 | inf 不能作为 JIT 编译期 float 参数 | 编译错误或异常 | host 侧分派到特化 reduce_max/min kernel，不将 inf 传入通用 kernel |
| 通用 p kernel 的 `1/scalar` 除零 | scalar=0 时通用 kernel 的 finalize `exp(ln(sum)/scalar)` 除零 | inf/nan 结果 | scalar=0 由 host 分派到 L0-count 特化 kernel，不走通用路径 |

### 10.3 特殊场景处理

| 场景 | 处理 | 归属层级 |
|------|------|---------|
| 非整除 shape（N%block_N≠0） | host 侧补零到 block_N 整数倍（零元素无贡献）；或 `T.ceildiv` + `T.copy` 动态切片 | L1（Stage 2 扩展） |
| 极小 shape（如 N=1） | `T.ceildiv(1, block_N)=1`，单 tile 单 block；block_N 可调小至 32（UB 对齐最小值） | L1（Stage 2 扩展） |
| 大 shape（48M 元素） | block_N=8192，n_num≈5859，单 block 串行 5859 tiles。L0/L1 精度可接受；Stage 3 优化为多 block + atomic_add | L1（Stage 2）/ Stage 3（性能） |
| INF/NAN 输入 | `ln(inf)=inf`→`exp(p·inf)=inf`；`ln(nan)=nan`→结果 nan。与 `torch.norm` 行为一致 | Boundary（Stage 2 扩展） |
| 全零输入 | p>0: `sum(0^p)=0`→`exp(ln(0)/p)=exp(-inf)=0` ✓；p=0: count=0 ✓；p=inf: max=0 ✓；p<0: inf（约束违反） | Boundary（Stage 2 扩展） |
| 负阶 + 零元素 | `ln(0)=-inf`→`exp(p·-inf)=+inf`（p<0），结果 inf。与 `torch.norm(p=-1)` 一致 | Boundary（Stage 2 扩展，记录不阻塞） |
| TensorList 空列表（L=0） | host 侧返回空列表，不调用 kernel | L2（Stage 2 扩展） |
| TensorList 各 tensor dtype 不一致 | host 侧校验拒绝（desc.md 约束"各张量 dtype 须一致"） | L2（Stage 2 扩展） |
| 分数阶 scalar（1.5/2.5） | 走通用正阶 kernel（exp(p·ln\|x\|)），无特殊处理 | L0 已覆盖 p=3，L1 补充 1.5/2.5 |
| FP32 dtype | 直接用 FP32 计算（不升精度），精度更高，UB 占用翻倍（~130KB < 192KB ✓） | L0 已覆盖 |

---

## 11. 交付清单

### 11.1 目录结构

```
custom/foreach_norm/
├── DESIGN.md            # 本设计文档
├── proto.yaml           # 算子接口规格（dtype/attr），供覆盖门禁派生应覆盖维度
├── foreach_norm.py      # 纯 kernel（@tilelang.jit，多分支 kernel + host dispatch）— Stage 2 产出
├── test_foreach_norm.py # from foreach_norm import foreach_norm + golden + 分层测试 + main — Stage 2 产出
└── README.md            # 使用说明（可选）— Stage 2 产出
```

### 11.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | 已完成 | 本设计文档（11 章 + L0 门槛测试计划） |
| `proto.yaml` | 已完成 | 算子接口规格（dtype 全集 float16+float32+bfloat16，attrs=[scalar]），覆盖门禁 `coverage_check.py --proto` 用 |
| `foreach_norm.py` | 待实现 | 纯 kernel（多分支 @tilelang.jit + host `foreach_norm(x_list, scalar)` dispatch），Stage 2 产出 |
| `test_foreach_norm.py` | 待实现 | `from foreach_norm import foreach_norm` + golden + L0 用例 + L1/L2/Boundary 桩 + main（`--level` 分发），Stage 2 产出 |

### 11.3 命名规范

- 目录名：`foreach_norm`（snake_case，与算子名一致）
- kernel 文件：`foreach_norm.py`
- 测试文件：`test_foreach_norm.py`（顶部 `from foreach_norm import foreach_norm`）
- kernel 入口函数名：`foreach_norm`（host dispatch 函数，可 import；内部 JIT kernel 命名为 `l2_norm_kernel`/`l1_norm_kernel`/`lp_norm_kernel`/`linf_norm_kernel` 等）

### 11.4 实现顺序

1. ✅ 设计文档（DESIGN.md）+ proto.yaml + L0 门槛测试计划（本文件 §9.2）
2. ⬜ kernel 实现（`foreach_norm.py`，多分支 @tilelang.jit + host dispatch，参考 §3.3 伪代码 + `examples/pos_embedding/rms_norm.py`）
3. ⬜ 测试文件（`test_foreach_norm.py`）：`from foreach_norm import foreach_norm` + golden 函数 + L0 用例 + L1/L2/Boundary 桩 + main（`--level` 分发）
4. ⬜ L0 门槛测试通过（精度收敛，按 §9.3 精度标准）
5. ⬜ 扩展分层套件（L1 功能含不规则 shape / L2 异常 / Boundary 特殊值，由 `tilelang-op-test-design` 场景 B 生成）+ 覆盖门禁 `coverage_check.py` 全 PASS/N/A
6. ⬜ 全量套件运行（L0/L1 须通过；L2/Boundary 失败仅记录不阻塞）

### 11.5 算子 proto.yaml（覆盖门禁用，Stage 1 产出）

> **dtype 全集取自本文档 §9.3 精度表**（float16 + float32 + bfloat16）+ **§4/§1** 的 attr/shape 机械派生，是覆盖门禁 `coverage_check.py --proto` 的**权威 dtype/attr 来源**。checker 只读 `operator.inputs[].dtype` 与 `operator.attrs[].name`。

```yaml
operator:
  name: ForeachNorm
  category: Reduction
  formula: |
    y = (sum |x_i|^p)^(1/p)
  attrs:
    - name: scalar
      type: Float
      default: 2.0
      required: true
  inputs:
    - name: x
      dtype: [float16, float32, bfloat16]    # 与 §9.3 精度表 dtype 行一致（全集）
  outputs:
    - name: y
      dtype: [float16, float32, bfloat16]    # 输出 dtype 与输入一致
  schema: foreach_norm(Tensor[] x, float scalar) -> Tensor[] y
```

> **一致性约束**：`inputs[].dtype` = `[float16, float32, bfloat16]` 与 §9.3 精度表的 dtype 行一致（全集）；`attrs` = `[scalar]`（范数阶数，影响计算路径分派，派生 `D-PARAM-scalar` 覆盖维度）。

---

## 12. 性能目标（Stage 3，用户追加）

> 本章节由 Orchestrator 在 Stage 2 精度通过、用户确认进入 Stage 3 后追加，不覆盖既有内容。

### 12.1 调优目标

| 字段 | 值 |
|------|-----|
| 性能目标类型 | `baseline_compare`（与 PyTorch 对比） |
| 目标数值 | **平均加速比 ≥ 0.6x**（即 `baseline_time / our_time ≥ 0.6`，等价 `our_time ≤ 1.67 × baseline_time`） |
| Baseline | `torch.norm(tensor, p=scalar)` 逐 tensor 调用（与本项目 host for-loop 策略一致；FP16/BF16 升 FP32 计算与 golden 一致） |
| 测试 shape | cann-bench cases.yaml 的 20 个代表性用例（覆盖 3 dtype × 6 scalar 分支 × 1D~5D shape × TensorList len 1~4 × 各种值域） |
| 噪声阈值 | 3%（perf-tuner 默认采纳门槛） |
| 最大迭代数 | 10 |

### 12.2 测试方法

- **Baseline 测量**：对每个 cann-bench 用例，用 `torch.norm(tensor.to(fp32), p=scalar)` 逐 tensor 调用，warmup 5 次 + 测量 20 次取中位数。
- **本项目测量**：对每个用例调用 `foreach_norm(x_list, scalar)`，warmup 5 次 + 测量 20 次取中位数。
- **加速比**：`speedup = baseline_time / our_time`（>1 表示比 baseline 快，<1 表示比 baseline 慢）。
- **平均加速比**：所有用例 speedup 的算术平均，目标 ≥ 0.6。
- **精度复验**：每次优化后必须重跑 `test_foreach_norm.py --level all` 确认精度不退化（L0/L1 全过 + 覆盖门禁全 PASS）。

### 12.3 已知性能特征（来自 DESIGN.md §5.2 / §6.3）

- **当前实现**：单 block 串行 tile 循环，block_N=8192。
- **大 shape 瓶颈**：48M 元素需 5859 tiles 串行处理，性能差。
- **优化方向**（Stage 3 探索）：
  1. 多 block 并行归约：`T.Kernel(n_num)` 每个 block 处理一个 tile + `T.tile.atomic_add` 跨 block 累加 + 独立 finalize kernel
  2. 双 buffer 流水线：`T.Pipelined(n_num, num_stages=2)` 预取下一 tile 同时计算当前 tile
  3. 多核并行：`T.Kernel(launch_cores)` 跨核分配 tile（参考 `examples/pos_embedding/rms_norm.py` 的多 block 模式 + sigmoid 的 `launch_cores` 策略）
  4. V 核切分：`VEC_NUM=2` 双 vector 子核并行处理 tile 内 element-wise 计算

### 12.4 中止条件

满足任一即结束 Stage 3：
1. 迭代次数达到 10
2. 连续 3 次无性能提升（speedup 改善 < 3%）
3. 达到平均加速比 ≥ 0.6x 目标

### 12.5 最终产出（用户额外要求）

性能调优完成后，生成类似 `custom/sigmoid/Sigmoid/Sigmoid.zip` 的压缩包，结构：
```
ForeachNorm/
├── build.sh                    # 打包脚本（python setup.py bdist_wheel）
├── setup.py                    # setuptools 配置（name="cann_bench"）
└── cann_bench/                 # Python 包
    ├── __init__.py             # 导出 foreach_norm + tilelang.cache.clear_cache()
    ├── _common.py              # 共享配置（pass_configs + dtype 映射）
    ├── foreach_norm.py         # adapter（host dispatch + tiling 选择 + kernel cache）
    └── _foreach_norm_kernel.py # JIT kernel 实现（7 个特化 kernel）
```
压缩为 `custom/foreach_norm/ForeachNorm/ForeachNorm.zip`。
