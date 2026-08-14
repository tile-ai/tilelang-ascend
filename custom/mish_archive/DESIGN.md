# Mish 算子设计文档

## 1. 概述

### 1.1 算子名称

mish

### 1.2 功能描述

Mish 是一种自正则化的非单调神经网络激活函数，具有平滑、非单调的特性。对输入张量逐元素计算 `y = x * tanh(softplus(x))`，在 YOLOv4/v5 等目标检测模型和深层卷积网络中常作为 ReLU/Swish 的替代激活层。单输入单输出，输出 shape/dtype 与输入完全一致。

### 1.3 数学公式

$$
\text{mish}(x) = x \cdot \tanh(\text{softplus}(x)) = x \cdot \tanh(\ln(1 + e^x))
$$

**特殊情况**：

| 输入 | 输出 |
|------|------|
| x = 0 | y = 0 |
| x → +∞ | y → x（趋近恒等） |
| x → -∞ | y → 0 |

### 1.4 算法描述

Mish 是逐元素（element-wise）激活算子，计算步骤分解为两个阶段：

**阶段一：稳定 softplus 计算**

直接公式 `ln(1 + exp(x))` 在大正数 x 时 `exp(x)` 溢出（float16 下 x > 11.09 即溢出）。采用数值稳定等价公式：

```
softplus(x) = max(x, 0) + ln(1 + exp(-|x|))
```

- 对 x >> 0：`max(x,0) = x`，`exp(-|x|) → 0`，`ln(1+0) = 0`，softplus ≈ x（无 exp 溢出）
- 对 x << 0：`max(x,0) = 0`，`exp(-|x|) → 0`，`ln(1+0) = 0`，softplus ≈ 0
- 对 x = 0：`max(0,0) = 0`，`exp(0) = 1`，`ln(2) ≈ 0.693`，softplus = ln(2) ✓

步骤分解（7 步）：
1. `abs_x = |x|`（T.tile.abs）
2. `neg_abs = -|x|`（T.tile.mul 标量 -1.0）
3. `exp_neg = exp(-|x|)`（T.tile.exp）
4. `one_plus = 1 + exp(-|x|)`（T.tile.add 标量 1.0 —— 经 one_ub 缓冲）
5. `ln_val = ln(1 + exp(-|x|))`（T.tile.ln）
6. `max_x0 = max(x, 0)`（T.tile.max 标量 0.0）
7. `softplus = max_x0 + ln_val`（T.tile.add）

**阶段二：tanh(s) 计算（恒等变换）**

`T.tile.tanh` 在本项目中不存在（`ascend_tile.py` 未定义；`examples/activation/tanh.py` 手动用 exp/sub/div 实现）。本设计采用更稳定的恒等式：

```
tanh(s) = 2 * sigmoid(2s) - 1
```

- `T.tile.sigmoid` 已验证可用（`ascend_tile.py:980`，`examples/activation/sigmoidv2.py:29`）
- 对 s 大正数：`sigmoid(2s) → 1`，`tanh → 1`（无 exp 溢出，sigmoid 内部稳定）
- 对 s = 0：`sigmoid(0) = 0.5`，`tanh = 0` ✓
- 对 s 大正数导致 `2s` float16 溢出为 inf：`sigmoid(inf) = 1`，`tanh = 1`（正确极限值）✓

步骤分解（5 步）：
8. `s2 = 2 * softplus`（T.tile.mul 标量 2.0）
9. `sig = sigmoid(2 * softplus)`（T.tile.sigmoid）
10. `two_sig = 2 * sig`（T.tile.mul 标量 2.0）
11. `tanh_s = 2*sig - 1`（T.tile.sub —— 经 one_ub 缓冲，因 sub 不接受标量 PrimExpr）
12. `y = x * tanh_s`（T.tile.mul）

**总计 12 步 T.tile 计算 + 1 步 fill（one_ub 初始化）+ 2 步 T.copy（搬入/搬出）**。

**数值稳定性总结**：
- 全程无 `exp(x)` 直接计算（最危险操作改为 `exp(-|x|)`，结果 ∈ [0, 1]，无溢出）
- `sigmoid(2s)` 中 `2s` 可能为 inf（float16 大数），但 `sigmoid(inf) = 1` 是正确极限值
- inf/nan 输入传播行为与 `torch.nn.functional.mish` 一致（见 §10.3）

### 1.5 数据流图

```
输入 GM[A] → T.copy → UB[a_ub]
                        │
          ┌─────────────┼──────────────────────────────────────┐
          │             │                                        │
   fill(one_ub,1.0)     │                                        │
          │             │                                        │
   abs(t0,a_ub)         │                                        │
   mul(t0,t0,-1.0)      │                                        │
   exp(t0,t0)           │                                        │
   add(t0,t0,one_ub)    │           ← 1 + exp(-|x|)              │
   ln(t0,t0)            │           ← ln(1+exp(-|x|))            │
   max(t1,a_ub,0.0)     │           ← max(x, 0)                  │
   add(t0,t0,t1)        │           ← softplus                    │
   mul(t0,t0,2.0)       │           ← 2*softplus                  │
   sigmoid(t0,t0)       │           ← sigmoid(2*softplus)         │
   mul(t0,t0,2.0)       │           ← 2*sigmoid                   │
   sub(t0,t0,one_ub)    │           ← tanh = 2*sig - 1            │
   mul(b_ub,a_ub,t0)    │ ←──────────────────────────────────────┘
                        │           ← y = x * tanh(softplus(x))
UB[b_ub] → T.copy → 输出 GM[B]
```

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**：Developer

### 2.2 选型理由

| 算子特征 | 分析 | 结论 |
|---------|------|------|
| 计算类型 | 纯 element-wise，无 matmul、无归约 | 纯 Vector，仅需 UB |
| 复杂度 | 12 步分解（abs/mul/exp/add/ln/max/add/mul/sigmoid/mul/sub/mul），无核间协作 | 单核内多步，无 CV 融合需求 |
| 内存层级 | 仅 GM ↔ UB，不涉及 L1/L0A/L0B/L0C | 编译器自动映射 shared→UB 即可 |
| 同步 | 单 block 内 V 核 vid 切分，无跨 block 依赖 | 自动同步足够 |
| 参考实现 | `examples/activation/tanh.py` / `sigmoid.py` / `gelu_mul.py` 均用 Developer 模式（`T.alloc_shared` + 全 pass_configs 开启）已验证通过 | 同模式可复用 |

用户明确指定 Developer 模式，且算子特征与同类激活函数示例完全契合，无需 Expert/混合模式的手动内存层级控制。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_shared(shape, dtype)` — 编译器自动映射到 UB（Vector 核缓冲） |
| 计算方式 | `T.tile.xxx` Buffer 级 SIMD 原语（abs/mul/exp/add/ln/max/sigmoid/sub） |
| 作用域 | 编译器自动分离 Cube/Vector（本算子无 Cube 计算，纯 V 核执行） |
| 同步方式 | 自动同步（`TL_ASCEND_AUTO_SYNC=True`），无需手动 `T.barrier_all` / `T.Scope` |

---

## 3. API 映射设计

### 3.1 公式拆解

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `abs_x = \|x\|` | 取绝对值（stable softplus 的预处理） |
| 2 | `neg_abs = -\|x\|` | 取负（mul 标量 -1.0） |
| 3 | `exp_neg = exp(-\|x\|)` | 指数，结果 ∈ [0, 1]，无溢出 |
| 4 | `one_plus = 1 + exp(-\|x\|)` | 加一（经 one_ub 缓冲） |
| 5 | `ln_val = ln(1 + exp(-\|x\|))` | 自然对数 |
| 6 | `max_x0 = max(x, 0)` | 取正部（max 标量 0.0） |
| 7 | `softplus = max_x0 + ln_val` | 稳定 softplus 结果 |
| 8 | `s2 = 2 * softplus` | 乘二（mul 标量 2.0） |
| 9 | `sig = sigmoid(2 * softplus)` | sigmoid（T.tile.sigmoid 原语） |
| 10 | `two_sig = 2 * sig` | 乘二（mul 标量 2.0） |
| 11 | `tanh_s = two_sig - 1` | 减一（经 one_ub 缓冲，因 sub 不接受标量） |
| 12 | `y = x * tanh_s` | 最终乘法 |
| 搬入 | `a_ub = A[...]` | GM → UB |
| 搬出 | `B[...] = b_ub` | UB → GM |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 来源确认 |
|------|----------|-------------|------|----------|
| 搬入 | `a_ub = A[...]` | `T.copy(A[...], a_ub)` | src=GM slice, dst=UB | `tanh.py:33` ✓ / `sigmoid.py:31` ✓ |
| 填一 | `one_ub = 1.0` | `T.tile.fill(one_ub, 1.0)` | dst=UB, value=1.0 scalar | `tanh.py:34` ✓ (fill 用法) / `ascend_tile.py:221` ✓ |
| 取绝对值 | `t0 = \|a\|` | `T.tile.abs(t0_ub, a_ub)` | dst=UB, src=UB | `ascend_tile.py:1049` ✓ / `fused_gdn_gating.py:229` ✓ |
| 取负 | `t0 = -\|a\|` | `T.tile.mul(t0_ub, t0_ub, -1.0)` | dst=UB, src0=UB, src1=-1.0 scalar | `fused_gdn_gating.py:230` ✓ (完全相同模式) / `gelu_mul.py:47` ✓ (mul 负标量) |
| 指数 | `t0 = exp(-\|a\|)` | `T.tile.exp(t0_ub, t0_ub)` | dst=UB（原地）, src=UB | `tanh.py:36` ✓ / `fused_gdn_gating.py:231` ✓ |
| 加一 | `t0 = 1 + exp(-\|a\|)` | `T.tile.add(t0_ub, t0_ub, one_ub)` | dst=UB, src0=UB, src1=UB(one_ub) | `fused_gdn_gating.py:232` ✓ (add 标量 1.0；本设计用 one_ub 缓冲等价) |
| 对数 | `t0 = ln(1+exp(-\|a\|))` | `T.tile.ln(t0_ub, t0_ub)` | dst=UB（原地）, src=UB | `ascend_tile.py:1039` ✓ / `fused_gdn_gating.py:233` ✓ (完全相同 softplus) |
| 取正部 | `t1 = max(a, 0)` | `T.tile.max(t1_ub, a_ub, 0.0)` | dst=UB, src0=UB, src1=0.0 scalar | `ascend_tile.py:900` ✓ (max 接受 PrimExpr) |
| 加和 | `t0 = t1 + t0` (softplus) | `T.tile.add(t0_ub, t0_ub, t1_ub)` | dst=UB, src0=UB, src1=UB | `tanh.py:39` ✓ (add buffer) |
| 乘二 | `t0 = 2*t0` | `T.tile.mul(t0_ub, t0_ub, 2.0)` | dst=UB, src0=UB, src1=2.0 scalar | `gelu_mul.py:43` ✓ (mul 标量) |
| sigmoid | `t0 = sigmoid(2*softplus)` | `T.tile.sigmoid(t0_ub, t0_ub)` | dst=UB（原地）, src=UB | `ascend_tile.py:980` ✓ / `sigmoidv2.py:29` ✓ |
| 乘二 | `t0 = 2*sig` | `T.tile.mul(t0_ub, t0_ub, 2.0)` | dst=UB, src0=UB, src1=2.0 scalar | `gelu_mul.py:43` ✓ |
| 减一 | `t0 = 2*sig - 1` (tanh) | `T.tile.sub(t0_ub, t0_ub, one_ub)` | dst=UB, src0=UB, src1=UB(one_ub) | `tanh.py:38` ✓ (sub buffer) — **注：sub 不接受标量 PrimExpr，须用 one_ub** |
| 最终乘 | `b = a * t0` | `T.tile.mul(b_ub, a_ub, t0_ub)` | dst=UB, src0=UB, src1=UB | `gelu_mul.py:55` ✓ (mul buffer) |
| 搬出 | `B[...] = b_ub` | `T.copy(b_ub, B[...])` | src=UB, dst=GM slice | `tanh.py:41` ✓ / `sigmoid.py:37` ✓ |

**关键 API 约束说明**：
- `T.tile.sub(dst, src0, src1)` 的 `src1` 类型为 `Buffer | BufferRegion | BufferLoad`，**不接受标量 PrimExpr**（`ascend_tile.py:867`）。因此 `tanh = 2*sig - 1` 中的减一操作必须用 `one_ub` 缓冲（预填充 1.0），不能直接写 `T.tile.sub(t0, t0, 1.0)`。`T.tile.add` / `T.tile.mul` / `T.tile.max` / `T.tile.min` 接受标量 PrimExpr，可直接用标量。
- `T.tile.ln`（自然对数）的函数名为 `ln`，不是 `log`（`ascend_tile.py:1039`）。
- `T.tile.tanh` 在本项目中**不存在**（`ascend_tile.py` 未定义）。本设计用 `tanh(s) = 2*sigmoid(2s) - 1` 恒等式绕过，复用已验证的 `T.tile.sigmoid`。

### 3.3 计算伪代码

```python
import tilelang
import tilelang.language as T

VEC_NUM = 2

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,   # 自动 CV 分离（无 Cube 时退化为纯 V）
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,         # 自动同步（核内）
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,   # 自动内存规划
}


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def mish(M, N, block_M, block_N, dtype="float16"):
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            # 1. 分配 buffer（Developer: alloc_shared 自动映射 UB；vid 切分行）
            a_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)
            t0_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)
            t1_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)
            one_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)

            # 2. 数据搬入 GM → UB
            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)

            # 3. 计算：y = x * tanh(softplus(x))
            #    稳定 softplus: max(x,0) + ln(1+exp(-|x|))
            #    稳定 tanh:     2*sigmoid(2s) - 1
            T.tile.fill(one_ub, 1.0)                       # one = 1.0（常量，供 add/sub 使用）
            T.tile.abs(t0_ub, a_ub)                        # t0 = |x|
            T.tile.mul(t0_ub, t0_ub, -1.0)                 # t0 = -|x|
            T.tile.exp(t0_ub, t0_ub)                       # t0 = exp(-|x|)
            T.tile.add(t0_ub, t0_ub, one_ub)               # t0 = 1 + exp(-|x|)
            T.tile.ln(t0_ub, t0_ub)                        # t0 = ln(1 + exp(-|x|))
            T.tile.max(t1_ub, a_ub, 0.0)                   # t1 = max(x, 0)
            T.tile.add(t0_ub, t0_ub, t1_ub)                # t0 = softplus = max(x,0) + ln(1+exp(-|x|))
            T.tile.mul(t0_ub, t0_ub, 2.0)                  # t0 = 2*softplus
            T.tile.sigmoid(t0_ub, t0_ub)                   # t0 = sigmoid(2*softplus)
            T.tile.mul(t0_ub, t0_ub, 2.0)                  # t0 = 2*sigmoid
            T.tile.sub(t0_ub, t0_ub, one_ub)               # t0 = tanh = 2*sigmoid - 1
            T.tile.mul(b_ub, a_ub, t0_ub)                  # b = x * tanh(softplus(x))

            # 4. 数据搬出 UB → GM
            T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main
```

### 3.4 API 可行性确认

| API | 来源 | 验证状态 |
|-----|------|---------|
| `T.alloc_shared` | api-kernel-memory.md §2 / `sigmoid.py:27` | ✅ 已验证（sigmoid.py 测试通过） |
| `T.copy` (GM↔UB) | api-kernel-memory.md §3 / `tanh.py:33,41` | ✅ 已验证 |
| `T.tile.fill` | `ascend_tile.py:221` / `tanh.py:34` | ✅ 已验证 |
| `T.tile.abs` | `ascend_tile.py:1049` / `fused_gdn_gating.py:229` | ✅ 已验证（fused_gdn_gating 中用于 softplus） |
| `T.tile.mul` (scalar src1) | `ascend_tile.py:878` / `gelu_mul.py:43,47` / `fused_gdn_gating.py:230` | ✅ 已验证（含负标量 -1.0） |
| `T.tile.exp` | `ascend_tile.py:970` / `tanh.py:36` / `fused_gdn_gating.py:231` | ✅ 已验证 |
| `T.tile.add` (buffer/scalar src1) | `ascend_tile.py:856` / `tanh.py:39` / `fused_gdn_gating.py:232` | ✅ 已验证 |
| `T.tile.ln` | `ascend_tile.py:1039` / `fused_gdn_gating.py:233` / `cross_entropy_loss/example_cross_entro.py:95` / `testing/python/language/test_tilelang_ascend_language_elementwise.py:2795` | ✅ 已验证（**fused_gdn_gating.py:233 用于完全相同的 softplus 计算**） |
| `T.tile.max` (scalar src1) | `ascend_tile.py:900` / `per_block_cast_lossless_kernel.py:1110-1113` | ✅ 已验证（max 接受 PrimExpr 标量） |
| `T.tile.sigmoid` | `ascend_tile.py:980` / `sigmoidv2.py:29` | ✅ 已验证 |
| `T.tile.sub` (buffer src1) | `ascend_tile.py:867` / `tanh.py:35,38` | ✅ 已验证（**sub 不接受标量，须用 one_ub**） |
| `T.ceildiv` | `sigmoid.py:16-17` / `tanh.py:16-17` | ✅ 已验证（处理非整除） |

**所有 API 均来自 `examples/` 已验证实现或 `ascend_tile.py` 源码定义，无凭记忆猜测。** 关键的 `T.tile.ln` + `T.tile.abs` + `T.tile.exp` 组合在 `examples/xllm_kernels/fused_gdn_gating.py:229-233` 中已用于完全相同的稳定 softplus 计算，可直接交叉验证。

---

## 3.5 技术约束确认

### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | No | 一维 `T.Kernel(m_num * n_num)`，无三维需求 |
| threads 参数限制（仅 1 或 2） | No | 不设 threads 参数（用默认；vid 切分由 VEC_NUM=2 隐式处理） |
| 动态循环边界不支持 | No | 无循环迭代（单次搬入+计算+搬出），不涉及循环边界 |
| 流水线不支持动态边界 | No | 不使用 `T.Pipelined`（element-wise 单步足够） |
| GPU 专用 API | No | 全部使用 `T.tile.xxx` Ascend 原语，无 CUDA API |
| GEMM 非整除风险 | No | 无 GEMM 计算 |
| L0C 容量上限 | No | 无 Cube 计算，不分配 L0C |

### 3.5.2 参考实现差异说明

本算子无外部 GPU 参考实现迁移。用户提供的 cann-bench `golden.py`（`torch.nn.functional.mish`）仅作 golden 参考实现，不涉及 kernel 迁移。cann-bench 的 `desc.md` 提供算子语义、精度标准和测试用例，本设计据此生成 Ascend 兼容方案。

**与 cann-bench 参考的差异**：

| 差异项 | cann-bench 参考 | 本项目（Ascend） | 转换方案 |
|--------|----------------|-----------------|----------|
| Golden 实现 | `torch.nn.functional.mish(x)` | 同（测试用 golden，非 kernel） | 无需转换 |
| Kernel 实现 | 无（cann-bench 只提供 golden + cases） | TileLang DSL + `T.tile.xxx` 原语 | 基于 `examples/activation/` 同类实现 |
| 精度标准 | MERE < Threshold, MARE < 10×Threshold | 混合容差（atol/rtol/max_abs_error_limit/required_matched_ratio），阈值取 cann-bench 值 | 见 §9.3 精度标准表 + 映射说明 |
| 维度支持 | 0~8 维（cases 实测 1~5D） | kernel 接收 2D (M,N)，host 侧 flatten 高维输入 | 见 §4.6 动态轴说明 |

### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/activation/tanh.py` | **极高** | tanh kernel：Developer 模式 + `T.alloc_shared` + `T.tile.fill/sub/exp/add/div` 分解 + VEC_NUM=2 vid 切分 + `T.ceildiv`。tanh.py 手动实现 tanh（因 `T.tile.tanh` 不存在），本设计改用 `2*sigmoid(2s)-1` 恒等式 |
| `examples/activation/sigmoid.py` | **极高** | sigmoid kernel：Developer 模式 + `T.alloc_shared` + pass_configs 三件套 + VEC_NUM=2。完整结构可复用 |
| `examples/xllm_kernels/fused_gdn_gating.py` | **极高（softplus 同源）** | **行 229-233 包含与本设计完全相同的稳定 softplus 计算**：`abs → mul(-1.0) → exp → add(1.0) → ln`。直接交叉验证 API 用法 |
| `examples/activation/sigmoidv2.py` | 高 | `T.tile.sigmoid` 一步原语写法，验证 sigmoid 原语可用 |
| `examples/activation/gelu_mul.py` | 高 | gelu + mul 模式：`T.alloc_ub` + 多步 `T.tile.mul/add/exp/div` + 末尾 `T.tile.mul`（与 mish 的 `x * tanh(...)` 末尾乘法结构一致） |
| `examples/activation/swi_glu.py` | 高 | silu * x2 模式：多步激活 + 末尾乘法，与 mish 的 `x * tanh(softplus(x))` 结构一致 |
| `examples/fused_sigmoid_gating_delta_rule/fused_sigmoid_gating_delta_rule_varlen.py` | 高 | **行 184 包含 `ln(1+exp(x·β))` 即 softplus 计算**（注释明确标注），验证 `T.tile.ln` + `T.tile.exp` + `T.tile.add` 组合用于 softplus |
| `examples/cross_entropy_loss/example_cross_entro.py` | 中 | `T.tile.ln` 用于 log-sum-exp，验证 `T.tile.ln` 原语可用 |
| `custom/sigmoid/DESIGN.md` | 高 | 同类 Activation 算子的设计文档，结构模板参考 |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| A（x） | (M, N) | float16 / float32 / bfloat16 | 输入张量，M 和 N 均为运行时维度。高维输入（1~8 维）由 host 侧 flatten 为 2D |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| B（y） | (M, N) | 同输入 | 输出张量，shape/dtype 与输入一致 |

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| a_ub | (block_M // VEC_NUM, block_N) | 同输入 | UB（alloc_shared 自动映射） | 输入 tile 缓冲，全程保留（末尾 `x * tanh` 需读回 x） |
| t0_ub | (block_M // VEC_NUM, block_N) | 同输入 | UB | 主临时缓冲，原地复用（abs→mul→exp→add→ln→add→mul→sigmoid→mul→sub，共 10 步原地） |
| t1_ub | (block_M // VEC_NUM, block_N) | 同输入 | UB | 辅助临时缓冲，仅用于 `max(x, 0)` 结果 |
| one_ub | (block_M // VEC_NUM, block_N) | 同输入 | UB | 全 1.0 常量缓冲，供 `add(1+exp)` 和 `sub(2*sig-1)` 使用（因 `T.tile.sub` 不接受标量） |
| b_ub | (block_M // VEC_NUM, block_N) | 同输入 | UB | 输出缓冲（`x * tanh` 结果） |

### 4.4 内存搬运路径

```
纯 Vector 路径（element-wise）：

GM[A] --T.copy--> UB[a_ub]
                     │
              fill(one_ub, 1.0)
              abs(t0, a_ub) → mul(t0, t0, -1.0) → exp(t0, t0) → add(t0, t0, one_ub) → ln(t0, t0)
              max(t1, a_ub, 0.0) → add(t0, t0, t1)                # softplus
              mul(t0, t0, 2.0) → sigmoid(t0, t0) → mul(t0, t0, 2.0) → sub(t0, t0, one_ub)  # tanh
              mul(b_ub, a_ub, t0)                                  # y = x * tanh
                     │
UB[b_ub] --T.copy--> GM[B]
```

**层级说明**：纯 Vector 算子，数据全程在 UB 上操作，不涉及 L1（Cube 缓存）/ L0A / L0B / L0C。`T.alloc_shared` 在无 Cube 计算时被编译器自动映射到 UB（Vector 核缓冲）。

### 4.5 UB 内存预算

以主配置 `block_M=128, block_N=128, VEC_NUM=2` 为例（每个 V 核处理 64 行）：

| Buffer | Shape | dtype | 大小 (Bytes) |
|--------|-------|-------|-------------|
| a_ub | (64, 128) | float16 | 64 × 128 × 2 = 16384 (16 KB) |
| t0_ub | (64, 128) | float16 | 16384 (16 KB) |
| t1_ub | (64, 128) | float16 | 16384 (16 KB) |
| one_ub | (64, 128) | float16 | 16384 (16 KB) |
| b_ub | (64, 128) | float16 | 16384 (16 KB) |
| **总计** | | | **81920 (80 KB)** |

- 目标平台 UB 容量：196608 Byte（192 KB，Ascend910B3，见 api-kernel-memory.md §2）
- float16/bfloat16 占用比：80 KB / 192 KB = 41.7% ✓
- float32 时单 buffer 翻倍至 32 KB，总计 160 KB / 192 KB = 83.3% ✓（仍可接受，但余量较小；若 float32 UB 不足可降 block_M 至 64 → 5×16KB=80KB）

### 4.6 动态轴定义

| 动态轴 | 声明方式 | 运行时范围 | 说明 |
|--------|----------|-----------|------|
| M | 作为 `@tilelang.jit` 函数参数传入 | 1 ~ 64K | 行数（高维输入 flatten 后的行数） |
| N | 作为 `@tilelang.jit` 函数参数传入 | 1 ~ 64K | 列数（高维输入 flatten 后的列数） |

**动态 shape 策略说明**：
- **主方案（采用）**：M, N 作为 `jit` 函数参数，每次调用 `mish(M, N, block_M, block_N, dtype)` 编译一个针对该 shape 的 kernel。这是 `examples/activation/sigmoid.py` / `tanh.py` 的已验证做法，简单可靠，编译器可充分优化。L0 测试计划中每个 shape 各自编译。
- **高维输入支持**：cann-bench 要求支持 1~8 维输入。本设计 kernel 接收 2D (M, N)，host 侧（测试 / 调用方）将高维 tensor `view` 或 `reshape` 为 2D 后传入 kernel，输出再 `reshape` 回原始 shape。因 mish 是 element-wise，flatten 不影响计算正确性。
- **可选优化（Stage 2 探索）**：用 `T.dyn['M']` / `T.dyn['N']` 声明真动态 shape，一次编译支持任意 shape。本设计不强制采用，留作 Stage 2 性能调优阶段的可选优化项。

### 4.7 JIT 配置

```python
@tilelang.jit(
    out_idx=[1],                # B（第 2 个参数）为输出
    pass_configs={
        tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,   # 自动 CV 分离
        tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,         # 自动同步（核内）
        tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,   # 自动内存规划
    },
)
def mish(M, N, block_M, block_N, dtype="float16"):
    ...
```

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**：纯 Vector

**判定依据**：算子仅包含 element-wise 运算（abs/mul/exp/add/ln/max/sigmoid/sub），无 matmul、无归约。数据全程在 UB 上操作，不涉及 Cube 核（L1/L0）。

### 5.2 Block 划分

```python
block_M = 128   # M 维分块：与 tanh.py/sigmoid.py 大规模用例一致，平衡 UB 占用与并行度
block_N = 128   # N 维分块：128 × fp16(2B) = 256B，满足 UB 32B 对齐（256/32=8）
VEC_NUM = 2     # V 核数：每个 V 核处理 block_M // 2 = 64 行

m_num = T.ceildiv(M, block_M)   # M 方向 block 数（ceildiv 处理非整除）
n_num = T.ceildiv(N, block_N)   # N 方向 block 数
block_num = m_num * n_num       # 一维 block 总数
```

**block size 选择理由**：
- `block_M=128`：与 `tanh.py` / `sigmoid.py` 的 (1100, 50000, 128, 128) 大规模用例一致，已验证可行
- `block_N=128`：满足 UB 32B 对齐（128 × 2B = 256B），且单 block 数据量适中
- 小 shape 场景（如 L0 的 (256, 256)）可配 `block_M=64, block_N=64`（参考 tanh.py 的 (256,256,64,64)）

### 5.3 约束分析

- **UB 对齐约束**：`block_N=128` × fp16(2B) = 256B，32B 整除 ✓（float32 时 128×4B=512B，bfloat16 同 float16=256B，均 32B 整除 ✓）
- **UB 容量**：5 buffer × 16KB = 80KB < 192KB（float16/bfloat16）✓；float32 时 5 × 32KB = 160KB < 192KB ✓
- **L0 容量**：无 Cube 计算，不适用
- **V 核切分**：`block_M // VEC_NUM = 64`，每 V 核处理 64 行，读写索引一致（`bx*block_M + vid*block_M//VEC_NUM`）✓

### 5.4 注意事项（非整除处理）

**非整除场景**：当 `M % block_M ≠ 0` 或 `N % block_N ≠ 0` 时：
- 使用 `T.ceildiv(M, block_M)` / `T.ceildiv(N, block_N)` 计算 block 数（向上取整，保证覆盖所有元素）
- `T.copy` 已支持动态 shape 切片自动处理尾块（参考 api-kernel-memory.md §3），**不需要 host 侧 zero-padding**
- 尾块 block 中超出有效范围的部分，`T.copy` 会自动处理（参考 `tanh.py` 的 (300, 300, 64, 64) 测试用例，300 不被 64 整除但测试通过）
- 本算子为 element-wise 无归约，尾块计算结果独立，无跨 block 竞态风险

**L0 测试**只用 block 整除的规则 shape；非整除/尾块/质数 shape 留给 L1（Stage 2 由 `tilelang-op-test-design` 场景 B 扩展）。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| Block 级（M×N 分块） | 隐式并行 | `T.Kernel(m_num * n_num)` | 每个 block 处理一个 (block_M, block_N) tile，由硬件调度到不同核 |
| V 核切分 | 隐式 | `vid ∈ {0, 1}` | VEC_NUM=2，每 V 核处理 block_M//2 行（`bx*block_M + vid*block_M//VEC_NUM`） |
| 元素级计算 | 无显式循环 | `T.tile.xxx` 原语 | Buffer 级 SIMD 操作，整个 tile 一次性计算，无需 `T.Parallel` 循环 |

**说明**：本算子用 `T.tile.xxx` 原语（Buffer 级 SIMD）而非 `T.Parallel` + 符号 API。原因：`T.tile.xxx` 直接触发 Ascend Vector 指令，性能更优且与 `tanh.py` / `sigmoid.py` 已验证实现一致。

### 6.2 循环伪代码

```python
# Block 级并行（隐式，由 T.Kernel 管理）；无 K 维迭代循环
with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
    bx = cid // n_num
    by = cid % n_num
    # 单次搬入 → 12 步 tile 计算 → 单次搬出（无循环）
    T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
    T.tile.fill(one_ub, 1.0)
    T.tile.abs(t0_ub, a_ub)
    T.tile.mul(t0_ub, t0_ub, -1.0)
    T.tile.exp(t0_ub, t0_ub)
    T.tile.add(t0_ub, t0_ub, one_ub)
    T.tile.ln(t0_ub, t0_ub)
    T.tile.max(t1_ub, a_ub, 0.0)
    T.tile.add(t0_ub, t0_ub, t1_ub)
    T.tile.mul(t0_ub, t0_ub, 2.0)
    T.tile.sigmoid(t0_ub, t0_ub)
    T.tile.mul(t0_ub, t0_ub, 2.0)
    T.tile.sub(t0_ub, t0_ub, one_ub)
    T.tile.mul(b_ub, a_ub, t0_ub)
    T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N])
```

### 6.3 流水线优化

**不使用 `T.Pipelined`**。理由：
- element-wise 单步计算（搬入→计算→搬出），无 K 维迭代累加
- 单 block 内计算量适中（64×128=8192 元素），流水线开销大于收益
- `tanh.py` / `sigmoid.py` 未使用流水线且性能已验证

若 Stage 3 性能调优发现搬入/计算/搬出存在等待，可探索双 buffer 流水线（`T.Pipelined(num_stages=2)`），但本设计不预设。

### 6.4 尾块处理

非整除时尾块由 `T.ceildiv` + `T.copy` 动态切片自动处理（见 §5.4）。无显式尾块循环逻辑。

---

## 7. 同步策略

### 7.1 同步模式

**模式**：自动同步（Developer 模式）

### 7.2 同步点说明

Developer 模式下由 `TL_ASCEND_AUTO_SYNC=True` 自动插入同步点，无需手动 `T.barrier_all` / `T.set_flag` / `T.wait_flag`。

| 位置 | 同步方式 | 理由 |
|------|----------|------|
| `T.copy` 搬入后 | 自动同步 | 确保数据写入 UB 完成后再计算 |
| `T.tile.xxx` 之间 | 自动同步 | t0_ub 原地复用（abs→mul→exp→add→ln→...），需保证前一步写完成 |
| `T.tile.mul(b_ub, a_ub, t0_ub)` 前 | 自动同步 | 确保 t0_ub（tanh 结果）和 a_ub（输入 x）均就绪 |
| `T.copy` 搬出前 | 自动同步 | 确保 `x * tanh` 结果写完成后再搬出 |

### 7.3 pass_configs 配置

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,   # 自动 CV 分离（无 Cube 时退化为纯 V）
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,         # 自动同步（核内，含上述同步点）
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,   # 自动内存规划（UB 分配优化）
}
```

**未开启的配置**：
- `TL_ASCEND_AUTO_CV_SYNC`：核间同步，本算子无 CV 融合（纯 Vector），不需要

---

## 8. 融合算子设计

### 8.1 融合算子判定

**判定结果**：否

**判定依据**：mish 是纯 element-wise 激活算子，无 GEMM（matmul）计算，不存在 Cube↔Vector 核间协作需求。Developer 模式下 `TL_ASCEND_AUTO_CV_COMBINE=True` 对纯 Vector 算子退化为无操作（不产生 workspace/vid 开销）。

本章节不适用，无 workspace 规格、无 CV 交互设计。

---

## 9. 验证方案

### 9.1 Golden 函数

```python
import torch

def golden_mish(x: torch.Tensor) -> torch.Tensor:
    """Mish 参考实现（PyTorch）。
    
    公式: y = x * tanh(softplus(x)) = x * tanh(ln(1 + e^x))
    
    Args:
        x: 输入张量，float16 / float32 / bfloat16
    Returns:
        y: mish(x)，shape/dtype 与输入一致
    """
    return torch.nn.functional.mish(x)
```

**Golden 选择说明**：`torch.nn.functional.mish` 是 PyTorch 内置 Mish 实现，内部采用数值稳定算法。对 float16/bfloat16 输入，在 NPU 上直接计算对应 dtype 结果。本设计 kernel 采用稳定 softplus（`max(x,0) + ln(1+exp(-|x|))`）+ 稳定 tanh（`2*sigmoid(2s)-1`），在边界值和特殊值处与 `torch.nn.functional.mish` 行为一致（见 §10.3 特殊场景处理）。

**cann-bench 参考来源**：`/mnt/workspace/gitCode/cann/cann-bench-master/tasks/level1/mish/golden.py`，golden 实现与 cann-bench 完全一致（`torch.nn.functional.mish(x)`）。

### 9.2 L0 门槛测试计划

> 设计阶段**只给出 L0 门槛用例**（规则 shape，block 整除），供 Stage 2 快速精度收敛。
> L1（功能，含不规则/尾块/质数 shape）/ L2（异常输入）/ Boundary（INF/NAN/极值）的**完整分层套件由 `tilelang-op-test-design` 场景 B 在 Stage 2 L0 通过后扩展**——不在此枚举。

**算子类别判断**（由 `tilelang-op-test-design` 场景 A 生成）：
- 计算类型：纯 Vector（element-wise，无 matmul）
- 复杂度：Multi（12 步分解：abs/mul/exp/add/ln/max/add/mul/sigmoid/mul/sub/mul）
- 数学特征：mish → Activation 类（含 softplus + tanh + mul）
- 综合类别：Activation（多步激活）
- 测试策略：dtype 组合（float16 + float32 + bfloat16）+ 规则 shape 组合 + 特殊值稳定性验证

**L0 用例集**（规则 shape，block 整除；8 用例。来源：cann-bench cases.yaml 代表性 subset，shape 统一取 block 整除的 1024×1024 或 2048×2048）：

| 用例名 | 级别 | Shape (M, N) | dtype | block | value_range | 来源 cann-bench case | 说明 |
|--------|------|--------------|-------|-------|-------------|----------------------|------|
| l0_fp16_basic | L0 | (1024, 1024) | float16 | (128, 128) | [-1, 1] | case 1 | float16 基本精度，对称小值域 |
| l0_fp32_basic | L0 | (1024, 1024) | float32 | (128, 128) | [-2, 2] | case 2（shape 缩小） | float32 基本精度，对称小值域 |
| l0_bf16_basic | L0 | (1024, 1024) | bfloat16 | (128, 128) | [-3, 3] | case 3（shape 缩小） | bfloat16 基本精度，对称小值域 |
| l0_fp16_mid | L0 | (2048, 2048) | float16 | (128, 128) | [-10, 10] | case 4（shape 缩小） | float16 中等值域，多 block 网格 |
| l0_fp16_maxval | L0 | (1024, 1024) | float16 | (128, 128) | [-65504, 65504] | case 10 | float16 边界值（finite max），验证稳定 softplus 不溢出 |
| l0_bf16_inf | L0 | (1024, 1024) | bfloat16 | (128, 128) | [-inf, inf] | case 12（shape 对齐化） | inf 特殊值，结构比对（inf 位置一致即可） |
| l0_fp32_nan | L0 | (1024, 1024) | float32 | (128, 128) | [nan, nan] | case 13（shape 对齐化） | nan 特殊值，结构比对（nan 位置一致即可） |
| l0_fp16_zero | L0 | (1024, 1024) | float16 | (128, 128) | [0, 0] | case 14（shape 对齐化） | 零值，mish(0) = 0 精确验证 |

**L0 覆盖维度命中标注**（供 Stage 2 覆盖门禁参考）：

| 用例名 | 命中维度 tags |
|--------|--------------|
| l0_fp16_basic | D-DTYPE-fp16, D-SHAPE-ALIGNED, D-VALRANGE-S |
| l0_fp32_basic | D-DTYPE-fp32, D-SHAPE-ALIGNED, D-VALRANGE-S |
| l0_bf16_basic | D-DTYPE-bf16, D-SHAPE-ALIGNED, D-VALRANGE-S |
| l0_fp16_mid | D-DTYPE-fp16, D-SHAPE-ALIGNED, D-VALRANGE-M |
| l0_fp16_maxval | D-DTYPE-fp16, D-SHAPE-ALIGNED, D-SPECIAL-DBOUND |
| l0_bf16_inf | D-DTYPE-bf16, D-SHAPE-ALIGNED, D-SPECIAL-INF |
| l0_fp32_nan | D-DTYPE-fp32, D-SHAPE-ALIGNED, D-SPECIAL-NAN |
| l0_fp16_zero | D-DTYPE-fp16, D-SHAPE-ALIGNED, D-SPECIAL-ZERO |

**L0 输入数据生成**：
- 正常值域用例（l0_fp16_basic / l0_fp32_basic / l0_bf16_basic / l0_fp16_mid）：`torch.empty(shape, dtype=..., device='npu').uniform_(lo, hi)` 均匀分布采样
- float16 边界值用例（l0_fp16_maxval）：`torch.empty(shape, dtype=torch.float16, device='npu').uniform_(-65504, 65504)`
- inf 用例（l0_bf16_inf）：构造含 ±inf 的 tensor（如 `torch.full(shape, float('inf'), ...)` 混合 `-inf` 和有限值）
- nan 用例（l0_fp32_nan）：`torch.full(shape, float('nan'), dtype=torch.float32, device='npu')` 混合有限值
- 零值用例（l0_fp16_zero）：`torch.zeros(shape, dtype=torch.float16, device='npu')`

**L0 验证流程**（供 Stage 2 落地参考）：
```python
def test_mish_l0():
    """L0 门槛测试：规则 shape，block 整除。返回是否全过。"""
    test_configs = [
        # (dtype, shape, block, value_range)
        ("float16",  (1024, 1024), (128, 128), (-1, 1)),
        ("float32",  (1024, 1024), (128, 128), (-2, 2)),
        ("bfloat16", (1024, 1024), (128, 128), (-3, 3)),
        ("float16",  (2048, 2048), (128, 128), (-10, 10)),
        ("float16",  (1024, 1024), (128, 128), (-65504, 65504)),
        ("bfloat16", (1024, 1024), (128, 128), "inf"),      # 特殊：含 ±inf
        ("float32",  (1024, 1024), (128, 128), "nan"),      # 特殊：含 nan
        ("float16",  (1024, 1024), (128, 128), (0, 0)),
    ]
    ok = True
    for dtype, shape, block, vrange in test_configs:
        M, N = shape
        block_M, block_N = block
        kernel = mish(M, N, block_M, block_N, dtype=dtype)
        x = _gen_input(shape, dtype, vrange)          # 按 vrange 生成输入
        y = kernel(x)
        ref = torch.nn.functional.mish(x)
        passed, ratio, max_abs = check_precision(y, ref, dtype)
        tag = "PASS" if passed else "FAIL"
        print(f"[PRECISION_{tag}] l0 shape={shape} dtype={dtype} "
              f"matched_ratio={ratio:.4f} max_abs={max_abs:.3e}")
        ok &= passed
    return ok
```

> **注**：inf/nan 用例的精度判定由 `check_precision` 自动走结构比对路径（`precision-standard.md §3.1`）：inf/nan 位置一致即 PASS，有限值位置仍按混合容差判定。

### 9.3 精度标准

> 采用**混合容差**：逐元素 `|actual-golden| ≤ atol + rtol·|golden|`，整体判定 `matched_ratio ≥ required_matched_ratio` **且** `max_abs_error ≤ max_abs_error_limit`。
> 阈值**仅按 dtype**（与算子类别无关），L0/L1/Boundary 套用精度比对（L2 为非法输入负向测试，不比精度）；整型按 0 误差精确匹配。完整定义见 `tilelang-op-test-design/references/precision-standard.md`。

**精度标准来源**：cann-bench 生态算子精度标准（`desc.md §4`），采用 MERE/MARE 双指标：
- MERE（平均相对误差）< Threshold
- MARE（最大相对误差）< 10 × Threshold
- Threshold 按 dtype：float16=2⁻¹⁰, bfloat16=2⁻⁷, float32=2⁻¹³

**混合容差映射**（将 cann-bench MERE/MARE 标准映射到 `check_precision` 的四元组）：
- `rtol` = cann-bench Threshold（对应 MERE 相对误差阈值）
- `max_abs_error_limit` = 容纳 mish 大值输出的绝对误差硬帽（mish(x) ≈ x 对大 x，故绝对误差随 |x| 线性增长；硬帽设为可容纳最大 |golden| × 10 × Threshold 的值，仅拦截灾难性错误）
- `atol` = Threshold / 16（小值场景的绝对容差，避免 golden≈0 时除零）
- `required_matched_ratio` = 0.99（对应 MERE < Threshold 对 99% 元素成立）

本算子支持 float16 + float32 + bfloat16 三个 dtype（与 §4 数据规格一致）：

| dtype | atol | rtol | max_abs_error_limit | required_matched_ratio |
|-------|------|------|---------------------|------------------------|
| float16 | 2⁻¹⁴ (6.10e-5) | 2⁻¹⁰ (9.77e-4) | 1e2 | 0.99 |
| bfloat16 | 2⁻¹¹ (4.88e-4) | 2⁻⁷ (7.81e-3) | 1e3 | 0.99 |
| float32 | 2⁻¹⁷ (7.63e-6) | 2⁻¹³ (1.22e-4) | 1e0 | 0.99 |

> **max_abs_error_limit 说明**：mish(x) ≈ x 对大正数 x，输出值域与输入值域相当。cann-bench case 10 的 float16 输入达 ±65504，mish(65504) ≈ 65504，rtol=2⁻¹⁰ 时绝对误差可达 65504×2⁻¹⁰≈64。故 float16 的 `max_abs_error_limit=1e2`（100），既容纳大值场景的合理绝对误差，又拦截灾难性错误（如产生 0 而非 65504，abs_err=65504 > 100 → FAIL）。bfloat16 同理取 1e3，float32 的 L0 最大值约 ±100（case 5 缩小后），取 1e0 足够。
>
> **与 tilelang 默认精度的差异**：`precision-standard.md §二` 默认值为 float16 rtol=2⁻⁹/bfloat16 rtol=2⁻⁶/float32 rtol=2⁻¹⁰，本设计采用 cann-bench 更严格的标准（rtol 更小）。Stage 2 测试实现时应在 `get_precision()` 函数中使用本表的 cann-bench 派生值，而非 `precision-standard.md` 默认值。
>
> **INF/NAN 处理**：inf/nan 位置做结构比对（`precision-standard.md §3.1`），不计入 `matched_ratio` / `max_abs_error`。mish(+inf)=+inf, mish(-inf) 的数学极限为 0 但 IEEE 浮点 `-inf * 0 = nan`——与 `torch.nn.functional.mish(-inf)` 行为一致（golden 同样产生 nan），故结构比对 PASS。

---

## 10. 风险点与注意事项

### 10.1 已知约束

| 约束 | 本算子状态 | 说明 |
|------|-----------|------|
| 三维 Kernel | 不涉及 | 一维 `T.Kernel(m_num * n_num)` |
| threads 参数 | 不涉及 | 不设 threads，用默认 + VEC_NUM=2 隐式 vid 切分 |
| 动态循环边界 | 不涉及 | 无循环迭代 |
| L0C 容量 | 不涉及 | 无 Cube 计算 |
| GEMM 非整除 | 不涉及 | 无 GEMM |
| UB 容量 | 已预算 | 80KB（fp16/bf16）/ 160KB（fp32）< 192KB ✓ |
| UB 对齐 | 已满足 | block_N=128 × fp16(2B) = 256B，32B 整除 ✓ |
| `T.tile.tanh` 不存在 | 已规避 | 用 `tanh(s) = 2*sigmoid(2s) - 1` 恒等式，复用 `T.tile.sigmoid` |
| `T.tile.sub` 不接受标量 | 已规避 | 用 `one_ub` 缓冲（fill 1.0）替代标量 1.0 |
| `T.tile.ln` 函数名 | 已确认 | 名为 `ln` 非 `log`（`ascend_tile.py:1039`） |

### 10.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| `T.tile.tanh` 调用 | 误用不存在的 `T.tile.tanh` | 编译错误（AttributeError） | 用 `2*sigmoid(2s)-1` 恒等式（本设计已采用） |
| `T.tile.sub(t0, t0, 1.0)` 标量 | sub 不接受 PrimExpr 标量 | 编译错误 | 用 `one_ub` 缓冲（fill 1.0 后 `sub(t0, t0, one_ub)`） |
| `T.tile.log` 调用 | 误用 `log` 而非 `ln` | AttributeError | 正确函数名为 `T.tile.ln` |
| float16 exp(x) 直接溢出 | naive 公式 `exp(x)` 对 x>11 溢出 | float16 边界值错误 | 本设计用 `exp(-\|x\|)`（结果 ∈ [0,1]，无溢出），已规避 |
| float32 UB 不足 | 5 buffer × 32KB = 160KB 接近 192KB 上限 | 编译警告或 OOM | 降 block_M 至 64（5×16KB=80KB），或 Stage 2 开启 MEMORY_PLANNING 优化 |
| 读写索引不一致 | vid 切分时读写行偏移不匹配 | 结果错乱 | 读写均用 `bx*block_M + vid*block_M//VEC_NUM`（参考 tanh.py/sigmoid.py） |
| block_N 不满足 32B 对齐 | block_N × sizeof(dtype) 不是 32 倍数 | DMA 搬运异常 | block_N=128（fp16: 256B, fp32: 512B, bf16: 256B）均满足 |
| 尾块越界写 | 非整除时尾块 block 写入超出有效范围 | 越界写邻近数据 | `T.copy` 动态切片自动处理；本算子 element-wise 无跨 block 竞态 |
| 高维输入未 flatten | 直接传 3D+ tensor 给 2D kernel | shape 不匹配错误 | host 侧 `x.view(-1, N)` 或 `x.reshape(-1, N)` 后传入，输出 `y.view(original_shape)` |

### 10.3 特殊场景处理

| 场景 | mish 行为 | golden (`torch.nn.functional.mish`) 行为 | 一致性 | 归属层级 |
|------|----------|----------------------------------------|--------|---------|
| x = +inf | softplus(+inf)=+inf, sigmoid(+inf)=1, tanh=1, y=+inf×1=+inf | +inf | ✓ 一致 | L0（l0_bf16_inf）+ Boundary |
| x = -inf | softplus(-inf)=0, sigmoid(0)=0.5, tanh=0, y=-inf×0=nan | nan（IEEE: -inf×0=nan） | ✓ 一致（均 nan） | L0（l0_bf16_inf）+ Boundary |
| x = nan | 全程 nan 传播, y=nan | nan | ✓ 一致 | L0（l0_fp32_nan）+ Boundary |
| x = 0 | softplus(0)=ln(2)≈0.693, tanh(0.693)≈0.6, y=0×0.6=0 | 0 | ✓ 一致（精确 0） | L0（l0_fp16_zero） |
| x = 65504 (float16 max) | softplus≈65504, sigmoid(inf)=1, tanh=1, y=65504 | 65504 | ✓ 一致 | L0（l0_fp16_maxval）+ Boundary |
| x = -65504 | softplus≈0, tanh≈0, y≈0 | ≈0 | ✓ 一致（均≈0） | L0（l0_fp16_maxval）+ Boundary |
| 非整除 shape | `T.ceildiv` + `T.copy` 动态切片 | N/A | ✓ 自动处理 | L1（Stage 2 扩展） |
| 极小 shape（如 (1, 128)） | `T.ceildiv` 保证至少 1 个 block | N/A | ✓ | L1（Stage 2 扩展） |
| 空 tensor（0 元素） | `T.ceildiv(0, block)=0`，无 block 启动 | N/A | ✓ | L2/Boundary（Stage 2 扩展） |
| float32 dtype | 直接用 float32 计算，精度更高，UB 160KB < 192KB ✓ | N/A | ✓ | L0 已覆盖 |
| bfloat16 dtype | 直接用 bfloat16 计算，UB 80KB < 192KB ✓ | N/A | ✓ | L0 已覆盖 |
| 高维输入（3D~5D） | host 侧 flatten 为 2D 后计算 | N/A | ✓（element-wise 不受 shape 影响） | L1（Stage 2 扩展） |

---

## 11. 交付清单

### 11.1 目录结构

```
custom/mish/
├── DESIGN.md            # 本设计文档
├── proto.yaml           # 算子接口规格（dtype/attr），供覆盖门禁派生应覆盖维度
├── mish.py              # 纯 kernel（@tilelang.jit，可 import，无 golden/测试/__main__）— Stage 2 产出
├── test_mish.py         # from mish import mish + golden + 分层测试 + main — Stage 2 产出
└── README.md            # 使用说明（可选）— Stage 2 产出
```

### 11.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | 已完成 | 本设计文档（11 章 + L0 门槛测试计划） |
| `proto.yaml` | 已完成 | 算子接口规格（dtype 全集 float16+float32+bfloat16，attrs=[]），覆盖门禁 `coverage_check.py --proto` 用 |
| `mish.py` | 待实现 | 纯 kernel（@tilelang.jit），Stage 2 产出 |
| `test_mish.py` | 待实现 | `from mish import mish` + golden + L0 用例 + L1/L2/Boundary 桩 + main（`--level` 分发），Stage 2 产出 |

### 11.3 命名规范

- 目录名：`mish`（snake_case）
- kernel 文件：`mish.py`
- 测试文件：`test_mish.py`（顶部 `from mish import mish`）
- kernel 函数名：`mish`（与 `@tilelang.jit` 装饰的函数一致，可 import）

### 11.4 实现顺序

1. ✅ 设计文档（DESIGN.md）+ proto.yaml + L0 门槛测试计划（本文件 §9.2）
2. ⬜ kernel 实现（`mish.py`，纯 @tilelang.jit，参考 §3.3 伪代码 + `examples/activation/tanh.py` / `sigmoid.py`）
3. ⬜ 测试文件（`test_mish.py`）：`from mish import mish` + golden 函数（§9.1）+ L0 用例（§9.2）+ L1/L2/Boundary 桩 + main（`--level` 分发）
4. ⬜ L0 门槛测试通过（精度收敛，按 §9.3 精度标准）
5. ⬜ 扩展分层套件（L1 功能含不规则 shape / L2 异常 / Boundary 特殊值，由 `tilelang-op-test-design` 场景 B 生成）+ 覆盖门禁 `coverage_check.py` 全 PASS/N/A
6. ⬜ 全量套件运行（L0/L1 须通过；L2/Boundary 失败仅记录不阻塞）

### 11.5 算子 proto.yaml（覆盖门禁用，Stage 1 产出）

> **dtype 全集取自本文档 §9.3 精度表**（float16 + float32 + bfloat16）+ **§4/§1** 的 attr/shape 机械派生，是覆盖门禁 `coverage_check.py --proto` 的**权威 dtype/attr 来源**。checker 只读 `operator.inputs[].dtype` 与 `operator.attrs[].name`。

```yaml
operator:
  name: Mish
  category: Activation
  formula: |
    y = x * tanh(softplus(x)) = x * tanh(ln(1 + e^x))
  attrs: []                              # mish 无影响计算路径的属性
  inputs:
    - name: x
      dtype: [float16, float32, bfloat16]   # 与 §9.3 精度表 dtype 行一致（全集）
  outputs:
    - name: y
      dtype: [float16, float32, bfloat16]   # 输出 dtype 与输入一致
  schema: mish(Tensor x) -> Tensor y
```

> **一致性约束**：`inputs[].dtype` = `[float16, float32, bfloat16]` 与 §9.3 精度表的 dtype 行一致（全集）；`attrs` = `[]`（mish 无 dim/axis/epsilon 等参数，无 D-PARAM-* 派生维度）。

---

## 12. 性能目标（Stage 3，由 Orchestrator 在 Stage 2 通过后追加）

> 本章节由 Orchestrator 在 Stage 2 `[PRECISION_PASS]` 后根据用户提供的调优信息追加，不覆盖既有内容。

- **性能目标类型**: `baseline_compare`
- **Baseline**: `torch.nn.functional.mish`（PyTorch NPU 实现，与 §9.1 golden 一致）
- **目标数值**: 平均加速比 ≥ **0.6x**（即 `mean(tilelang_time / torch_time) ≤ 1/0.6 ≈ 1.67`，等价 tilelang 平均时间不超过 torch 的 1.67 倍）
- **测试 shape**: 取 cann-bench `cases.yaml` 的代表性 subset（覆盖 S/M/L 规模 + 三 dtype + 对齐/非对齐）：
  - (1024, 1024) float16 / float32 / bfloat16 — S 对齐
  - (2048, 2048) float16 / float32 — M 对齐
  - (8192, 8192) float16 / float32 — L 对齐
  - (1023, 1023) bfloat16 — S 非对齐
  - (1537, 769) float32 — S 质数非对齐
- **噪声阈值**: 3%（默认）
- **最大迭代数**: 10（默认）
- **中止条件**:
  1. 迭代次数达到 10
  2. 连续三次无性能提升（< 3% 噪声阈值内）
  3. 达到平均加速比 ≥ 0.6x 目标
- **测试方法**:
  - bench 端到端（Python 计时，warmup=30, iters=100, `torch.npu.synchronize()`）
  - 可选 msprof NPU kernel task duration 用于瓶颈定位
- **优化方向建议**（参考 `custom/sigmoid/` 同类算子 Stage 3 经验）:
  - Expert 双缓冲 + Fixed Core 模式（`T.alloc_ub` 3D buffer + `T.set_flag/wait_flag` MTE2→V→MTE3 overlap）
  - 关闭 `AUTO_CV_COMBINE`（pure Vector 退化为 no-op）
  - tiling 搜索（block_N ∈ {128, 256, 512}，block_M 由 UB 预算反推）
  - bf16 cast 路径单独优化（参考 sigmoid `_sigmoid_kernel_bf16` 的 3-buffer in-place 模式）
- **已知风险**:
  - mish 是 12 步分解（vs sigmoid 1 步），中间 fp32 计算增加 UB 压力和 latency
  - host runtime 开销在 sigmoid 中已观察到 ~160us，可能成为端到端瓶颈
  - 0.6x 目标相对宽松（sigmoid 端到端 0.25x，NPU kernel 0.95x），主要看 NPU kernel 提升空间
