# Mish 算子设计文档

## 1. 概述

### 1.1 算子名称

Mish

### 1.2 功能描述

Mish 是一种自正则化的非单调神经网络激活函数，具有平滑、非单调的特性，在部分场景下性能优于 ReLU 和 Swish。常用于 YOLOv4/v5 等目标检测模型及深层卷积网络的激活层。本算子为逐元素（Elementwise）运算，单输入单输出，输出 shape 与 dtype 与输入完全一致。

### 1.3 数学公式

$$
y = x \cdot \tanh(\text{softplus}(x)) = x \cdot \tanh(\ln(1 + e^x))
$$

**特殊情况**：

| 输入 | 输出 | 说明 |
|------|------|------|
| x = 0 | y = 0 | 0 · tanh(ln(1+1)) = 0 · tanh(ln2) = 0 |
| x → +∞ | y → x | softplus(x)→x, tanh(x)→1, mish→x |
| x → -∞ | y → 0 | softplus(x)→0, tanh(0)→0, mish→0 |
| x = +inf | y = +inf | 数值稳定方案下：softplus(+inf)=+inf, tanh(+inf)=1, mish=+inf·1=+inf |
| x = -inf | y = 0 | softplus(-inf)=0, tanh(0)=0, mish=-inf·0=NaN→按 0 处理（见 §10.3） |
| x = NaN | y = NaN | NaN 传播 |

### 1.4 算法描述

Mish 是多步逐元素运算，分解为 12 个 Tile 级 SIMD 步骤。核心是两个数值稳定子公式：

1. **数值稳定 softplus**（log-sum-exp trick，7 步）：
   - `softplus(x) = max(x, 0) + ln(1 + exp(-|x|))`
   - 关键：`exp` 的参数恒为 `-|x| ≤ 0`，结果 ∈ [0, 1]，**永不溢出**
   - 用 `T.tile.abs` + `T.tile.mul(-1)` + `T.tile.exp` + `T.tile.add(one)` + `T.tile.ln` + `T.tile.max(0)` + `T.tile.add`

2. **数值稳定 tanh**（sigmoid 等价，4 步）：
   - `tanh(s) = 2 · sigmoid(2s) - 1`
   - 关键：`s = softplus(x) ≥ 0` 恒成立，所以 `2s ≥ 0`，`exp(-2s) ∈ (0, 1]`，**永不溢出**
   - 用 `T.tile.mul(2)` + `T.tile.sigmoid` + `T.tile.mul(2)` + `T.tile.sub(one)`

3. **最终乘法**（1 步）：`y = x · tanh(softplus(x))`
   - 用 `T.tile.mul(a_ub, t0_ub)`

**精度方案**：所有 12 步中间计算在 **float32**（`ACC_DTYPE="float32"`）buffer 上进行，解决：
- (a) bf16 CANN intrinsic 不支持 `__bf16`（Muls/Maxs/Exp/Adds/Div 编译失败）
- (b) fp16 12 步累积精度损失

非 float32 输入在 GM↔UB 边界用 `T.tile.cast` 做无损/舍入转换。

> **⚠️ Host 侧 Buffer 操作约束**（详见 tilelang-op-design/references/ascend-constraints.md §5）：host 侧只允许经证明共享原 storage、只改 metadata 的 view 操作（`reshape`/`view`），以及 kernel 调用和结果验证；禁止真实数据搬运和 aclnn 计算。高维输入通过 `x.reshape(-1, x.shape[-1])` 降维到 2D（contiguous 输入下为零拷贝 view），输出通过 `y.view(orig_shape)` 恢复（零拷贝 view）。

### 1.5 数据流图

```
[Host] x (任意维, contiguous)
  └─ x.reshape(-1, last_dim) → x_2d (M, N)        # 零拷贝 view（metadata-only）
       └─ [Kernel] GM[x_2d]
            ├─ T.copy → UB[a_ub(fp32)]              # GM→UB（含 cast: fp16/bf16→fp32）
            ├─ 12 步 T.tile.xxx 计算（全 fp32）     # UB→UB
            │   ├─ softplus: abs→mul→exp→add→ln→max→add   (7 步)
            │   ├─ tanh:     mul→sigmoid→mul→sub          (4 步)
            │   └─ mish:     mul                          (1 步)
            └─ T.copy → GM[y_2d]                     # UB→GM（含 cast: fp32→fp16/bf16）
       └─ [Host] y_2d.view(orig_shape) → y          # 零拷贝 view（metadata-only）
```

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**: Developer

### 2.2 选型理由

Mish 是纯 Vector 逐元素激活算子：
- **计算类型**：纯 Vector（无 matmul、无归约、无 CV 融合）→ 仅需 UB 层级
- **复杂度级别**：多步（12 步 element-wise）但无跨 tile 依赖，每个 tile 独立计算
- **无流水线需求**：单 tile 内完成全部计算，无 K 维迭代累加
- **无核间协作**：各 block 独立处理自己的 tile，无数据交换

Developer 模式的 `T.alloc_shared`（编译器自动映射到 UB）+ `T.tile.xxx`（Buffer 级 SIMD）+ 自动同步完全满足需求，无需手动控制内存层级和同步。参考 `examples/activation/sigmoid.py`、`silu.py`、`tanh.py` 均采用相同模式。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_shared`（编译器自动映射到 UB，192KB A2/A3） |
| 计算方式 | `T.tile.xxx` Buffer 级 SIMD 原语（12 步全向量化，无 T.if_then_else 条件分支） |
| 作用域 | 编译器自动分离（纯 Vector，全在 AIV 核执行） |
| 同步方式 | 自动同步（`TL_ASCEND_AUTO_SYNC: True`） |

---

## 3. API 映射设计

### 3.1 公式拆解

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `t0 = |x|` | 绝对值（用于 log-sum-exp） |
| 2 | `t0 = -|x|` | 取负（exp 参数 ≤ 0 防溢出） |
| 3 | `t0 = exp(-|x|)` | 指数，结果 ∈ [0, 1] |
| 4 | `t0 = 1 + exp(-|x|)` | 加 1 |
| 5 | `t0 = ln(1 + exp(-|x|))` | 自然对数 |
| 6 | `t1 = max(x, 0)` | ReLU（log-sum-exp 的 max 项） |
| 7 | `t0 = softplus = max(x,0) + ln(1+exp(-|x|))` | 数值稳定 softplus |
| 8 | `t0 = 2 · softplus` | 缩放（tanh 等价准备） |
| 9 | `t0 = sigmoid(2·softplus)` | sigmoid（参数 ≥ 0，exp 不溢出） |
| 10 | `t0 = 2 · sigmoid` | 缩放 |
| 11 | `t0 = tanh = 2·sigmoid - 1` | 数值稳定 tanh |
| 12 | `b = x · tanh(softplus(x))` | 最终 Mish 输出 |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 0 | `one = 1.0` | `T.tile.fill(one_ub, 1.0)` | dst=one_ub, val=1.0 | Developer |
| 1 | `t0 = |x|` | `T.tile.abs(t0_ub, a_ub)` | dst=t0_ub, src=a_ub | Developer |
| 2 | `t0 = -|x|` | `T.tile.mul(t0_ub, t0_ub, -1.0)` | dst=t0_ub, src0=t0_ub, src1=-1.0(scalar) | Developer |
| 3 | `t0 = exp(-|x|)` | `T.tile.exp(t0_ub, t0_ub)` | dst=t0_ub, src=t0_ub | Developer |
| 4 | `t0 = 1 + exp(-|x|)` | `T.tile.add(t0_ub, t0_ub, one_ub)` | dst=t0_ub, src0=t0_ub, src1=one_ub(buffer) | Developer |
| 5 | `t0 = ln(1+exp(-|x|))` | `T.tile.ln(t0_ub, t0_ub)` | dst=t0_ub, src=t0_ub | Developer |
| 6 | `t1 = max(x, 0)` | `T.tile.max(t1_ub, a_ub, 0.0)` | dst=t1_ub, src0=a_ub, src1=0.0(scalar) | Developer |
| 7 | `t0 = softplus` | `T.tile.add(t0_ub, t0_ub, t1_ub)` | dst=t0_ub, src0=t0_ub, src1=t1_ub | Developer |
| 8 | `t0 = 2·softplus` | `T.tile.mul(t0_ub, t0_ub, 2.0)` | dst=t0_ub, src0=t0_ub, src1=2.0(scalar) | Developer |
| 9 | `t0 = sigmoid(2s)` | `T.tile.sigmoid(t0_ub, t0_ub)` | dst=t0_ub, src=t0_ub | Developer |
| 10 | `t0 = 2·sigmoid` | `T.tile.mul(t0_ub, t0_ub, 2.0)` | dst=t0_ub, src0=t0_ub, src1=2.0(scalar) | Developer |
| 11 | `t0 = tanh` | `T.tile.sub(t0_ub, t0_ub, one_ub)` | dst=t0_ub, src0=t0_ub, src1=one_ub(buffer) | Developer |
| 12 | `b = x·tanh` | `T.tile.mul(b_ub, a_ub, t0_ub)` | dst=b_ub, src0=a_ub, src1=t0_ub | Developer |
| cast-in | fp16/bf16→fp32 | `T.tile.cast(a_ub, tmp_orig, "CAST_NONE", elem_num)` | dst=a_ub(fp32), src=tmp_orig(orig), mode=CAST_NONE | Developer |
| cast-out | fp32→fp16/bf16 | `T.tile.cast(tmp_orig, b_ub, "CAST_RINT", elem_num)` | dst=tmp_orig(orig), src=b_ub(fp32), mode=CAST_RINT | Developer |

### 3.3 计算伪代码

```python
@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def mish(M, N, block_M, block_N, dtype="float16"):
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)
    rows_per_vec = block_M // VEC_NUM   # VEC_NUM = 2
    elem_num = rows_per_vec * block_N
    need_cast = dtype not in ("float", "float32")

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            # 1. 分配 UB buffers（全 float32 中间计算 + tmp_orig 桥接）
            a_ub   = T.alloc_shared((rows_per_vec, block_N), ACC_DTYPE)  # float32 输入
            t0_ub  = T.alloc_shared((rows_per_vec, block_N), ACC_DTYPE)  # float32 中间
            t1_ub  = T.alloc_shared((rows_per_vec, block_N), ACC_DTYPE)  # float32 中间
            one_ub = T.alloc_shared((rows_per_vec, block_N), ACC_DTYPE)  # float32 常量 1.0
            b_ub   = T.alloc_shared((rows_per_vec, block_N), ACC_DTYPE)  # float32 输出
            tmp_orig = T.alloc_shared((rows_per_vec, block_N), dtype)    # 原 dtype 桥接

            # 2. 数据搬入: GM -> UB（含 cast for non-fp32）
            if need_cast:
                T.copy(A[bx*block_M + vid*rows_per_vec, by*block_N], tmp_orig)
                T.tile.cast(a_ub, tmp_orig, "CAST_NONE", elem_num)  # fp16/bf16 -> fp32
            else:
                T.copy(A[bx*block_M + vid*rows_per_vec, by*block_N], a_ub)

            # 3. 计算: y = x * tanh(softplus(x))  -- 全 fp32, 12 步
            T.tile.fill(one_ub, 1.0)                       # one = 1.0
            T.tile.abs(t0_ub, a_ub)                        # t0 = |x|
            T.tile.mul(t0_ub, t0_ub, -1.0)                 # t0 = -|x|
            T.tile.exp(t0_ub, t0_ub)                       # t0 = exp(-|x|) ∈ [0,1]
            T.tile.add(t0_ub, t0_ub, one_ub)               # t0 = 1 + exp(-|x|)
            T.tile.ln(t0_ub, t0_ub)                        # t0 = ln(1+exp(-|x|))
            T.tile.max(t1_ub, a_ub, 0.0)                   # t1 = max(x, 0)
            T.tile.add(t0_ub, t0_ub, t1_ub)                # t0 = softplus
            T.tile.mul(t0_ub, t0_ub, 2.0)                  # t0 = 2*softplus
            T.tile.sigmoid(t0_ub, t0_ub)                   # t0 = sigmoid(2*softplus)
            T.tile.mul(t0_ub, t0_ub, 2.0)                  # t0 = 2*sigmoid
            T.tile.sub(t0_ub, t0_ub, one_ub)               # t0 = tanh = 2*sigmoid - 1
            T.tile.mul(b_ub, a_ub, t0_ub)                  # b  = x * tanh(softplus(x))

            # 4. 数据搬出: UB -> GM（含 cast for non-fp32）
            if need_cast:
                T.tile.cast(tmp_orig, b_ub, "CAST_RINT", elem_num)  # fp32 -> fp16/bf16
                T.copy(tmp_orig, B[bx*block_M + vid*rows_per_vec, by*block_N])
            else:
                T.copy(b_ub, B[bx*block_M + vid*rows_per_vec, by*block_N])

    return main
```

### 3.4 API 可行性确认

| API | 来源确认 | 验证状态 |
|-----|---------|---------|
| `T.alloc_shared` | examples/activation/sigmoid.py L27, silu.py L27, tanh.py L27 | ✅ 已验证（同类示例） |
| `T.tile.fill` | examples/activation/sigmoid.py L32, api-compute.md §4.10 | ✅ 已验证 |
| `T.tile.abs` | api-compute.md §4.2 单目运算 | ✅ 已验证 |
| `T.tile.mul/add/sub/max` | api-compute.md §4.1 基础算术（src1 可 scalar） | ✅ 已验证 |
| `T.tile.exp` | examples/activation/sigmoid.py L34, api-compute.md §4.2 | ✅ 已验证 |
| `T.tile.ln` | api-compute.md §4.2（自然对数，名为 `ln` 非 `log`） | ✅ 已验证 |
| `T.tile.sigmoid` | custom/mish_archive/mish.py L126（验证通过）, api-compute.md §4.2 注释提及 | ✅ 已验证（mish_archive attempt 2 all pass） |
| `T.tile.cast` | api-compute.md §4.9, mish_archive/mish.py L109/L133 | ✅ 已验证 |
| `T.copy` | 所有同类示例, api-kernel-memory.md | ✅ 已验证 |
| `T.ceildiv` | examples/activation/sigmoid.py L16, silu.py L16 | ✅ 已验证 |

**关键 API 约束**（来自 mish_archive 验证）：
- `T.tile.sub(dst, src0, src1)`：**src1 不接受标量 PrimExpr**，必须用预填充的 `one_ub` buffer（步骤 11）。`T.tile.add/mul/max` 接受标量。
- `T.tile.ln`：自然对数 API 名为 `ln`，**不是** `log`（`T.log` 是 T.Parallel 符号 API，不是 T.tile 原语）。
- `T.copy`：**不支持跨 dtype**（src/dst dtype 必须一致），需用 `T.tile.cast` 做 dtype 转换。
- `T.tile.sigmoid`：存在于 `tilelang/language/ascend_tile.py`，虽 api-compute.md §4 表格未完整列出，但 mish_archive 验证通过且 api-compute.md §4.2 注释明确提及。

### 3.5 数值稳定方案向量化自检（§3.1 强制）

**自检清单**：
1. **方案中是否出现 `T.if_then_else` / `T.tile.compare` + `T.tile.select`？**
   → **否**。全部 12 步使用 `T.tile.abs/mul/exp/add/ln/max/sigmoid/sub/fill/cast`，无任何条件分支原语。

2. **等价变换是否消除了所有逐元素条件分支？**
   → **是**。
   - softplus 大值溢出问题：用 log-sum-exp trick `max(x,0) + ln(1+exp(-|x|))` 替代 `ln(1+exp(x))`，`exp` 参数恒 ≤ 0，数学等价且无分支。
   - tanh 无 `T.tile.tanh` API 问题：用 sigmoid 等价 `2*sigmoid(2s)-1`，因 softplus ≥ 0 保证 `exp(-2s)` 不溢出，数学等价且无分支。
   - 大 x / 小 x 分段处理问题：log-sum-exp trick 在全值域统一公式，无需分段。

3. **变换后的精度是否在目标 dtype 阈值内？**
   → **是**（mish_archive attempt 2 验证：float32 中间计算下 L0/L1 全 PASS，cann-bench case 1-4/6-10/12-20 达标）。float32 大值域（case 5/11/20）的 `T.tile.exp/ln` 精度风险见 §10.1，有 fallback 方案。

**结论**：本方案全向量化，无串行循环降级风险。对比反面案例（`T.if_then_else` 方案 speedup 0.0159），本方案预期与 mish_archive 达标版（speedup 0.7168）相当。

---

## 3.6 技术约束确认

### 3.6.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | No | 纯 2D tiling，`T.Kernel(m_num*n_num, is_npu=True)` 一维 block 数 |
| threads 参数限制（仅 1 或 2） | Yes（VEC_NUM=2） | 使用 VEC_NUM=2 核内并行（vid 维度），不指定 threads 参数（编译器默认） |
| 动态循环边界不支持 | No | 无循环（单 tile 内完成全部计算） |
| 流水线不支持动态边界 | No | 无流水线（无 T.Pipelined） |
| L0C 容量上限 | No | 纯 Vector，无 L0C 使用 |
| GEMM 非整除 | No | 无 GEMM |
| T.copy 列方向 strided 切片 | No | 2D 切片，最内维是 buffer 最内维 |

### 3.6.2 参考实现差异说明

**外部参考**：cann-bench `tasks/level1/mish/golden.py` 使用 `torch.nn.functional.mish(x)`（PyTorch 内置）。本算子不直接使用 PyTorch API，仅作为 golden 参考。数学公式完全一致。

| 差异项 | cann-bench golden（PyTorch） | 本项目（TileLang-Ascend） | 转换方案 |
|--------|------------------------------|--------------------------|----------|
| 计算实现 | `torch.nn.functional.mish` 黑盒 | 12 步 `T.tile.xxx` 显式分解 | log-sum-exp + sigmoid 等价 |
| 数值稳定 | PyTorch 内部处理 | log-sum-exp trick + float32 中间计算 | 见 §3.3 伪代码 |
| dtype 处理 | PyTorch 原生支持 fp16/fp32/bf16 | float32 中间计算 + T.tile.cast 边界转换 | 见 §3.3 cast-in/cast-out |
| 维度支持 | 0-8 维原生 | kernel 接受 2D，host reshape 降维 | 见 §1.4/§4.6 |

### 3.6.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/activation/sigmoid.py` | 高度相似 | Developer 模式 + pass_configs 三件套 + VEC_NUM=2 + T.ceildiv + T.tile.fill/sub/exp/add/reciprocal 模式 |
| `examples/activation/silu.py` | 高度相似 | silu = x*sigmoid(x)，与 mish = x*tanh(softplus(x)) 结构同构（x·激活(s)），T.tile.div 实现 sigmoid |
| `examples/activation/tanh.py` | 高度相似 | tanh = (e^x-e^-x)/(e^x+e^-x)，T.tile.sub/exp/add/div 模式；VEC_NUM=2 + T.ceildiv |
| `custom/mish_archive/mish.py` | 完全匹配（前序归档） | 验证通过的 mish 实现：log-sum-exp + sigmoid 等价 + float32 ACC_DTYPE + T.tile.cast + one_ub 约束 |
| `custom/mish_archive/debug_log.md` | 完全匹配（前序归档） | 踩坑记录：bf16 编译失败 + fp16 精度损失 → float32 中间计算修复 |

> **注**：`custom/mish_archive/` 是前序独立流程的归档（非本次 design 回退产物，本次 `design_revision_count=0`）。其中的 `mish.py` 已通过 cann-bench 精度验收（attempt 2: L0:8 L1:15 全 PASS），API 选型和数值稳定方案经实际验证。本设计文档采纳其验证过的方案，不视为"凭记忆猜 API"。

### 3.6.4 分派覆盖审计

```
[DISPATCH-COVERAGE]
supported_domain: shape 0-8D（cases 1-5D）, dtype float16/float32/bfloat16, 无 attrs
generic_fallback: mish kernel（单一通用路径，全 dtype/shape 统一处理）
specializations: none（纯逐元素运算，无结构特化快路径）
fallback_on_miss: n/a（无特化）
equivalence_evidence: 单一 kernel 覆盖全域，host reshape 处理高维
unsupported_inputs: 无（与 supported_domain 不冲突）
result: pass
```

### 3.6.5 Host 侧 Buffer 操作审计

```
[HOST-METADATA-AUDIT]
operation: x.reshape(-1, x.shape[-1])  # 高维 -> 2D
input_stride -> output_stride: contiguous stride 重排为 (numel//N, N) 二维 stride
shares_storage / same_data_ptr: true（contiguous 输入下 reshape 是零拷贝 view）
aclnn_or_physical_copy: false（仅改 shape/stride 元数据，不触发 aclnn）
result: allow

[HOST-METADATA-AUDIT]
operation: y.view(orig_shape)  # 2D -> 高维恢复
input_stride -> output_stride: (M, N) stride 重排为原 shape stride
shares_storage / same_data_ptr: true（view 操作共享 storage）
aclnn_or_physical_copy: false
result: allow
```

**非整除处理**：输入/输出 GM 两侧使用 `T.ceildiv(M, block_M)` / `T.ceildiv(N, block_N)` 计算 block 数，`T.copy` 按动态切片自动裁剪尾块搬运范围（参考 examples/activation/sigmoid.py L16 `T.ceildiv` 用法）。**无需 host padding + crop**。

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| A (x) | (M, N) | float16 / float32 / bfloat16 | 输入张量，2D（高维由 host reshape 降维） |

> 代表性 dtype: float16（cann-bench case 1 [1024,1024]）。dtype 全集见 §9.3 精度表。

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| B (y) | (M, N) | float16 / float32 / bfloat16 | 输出张量，shape/dtype 与输入完全一致 |

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| a_ub | (rows_per_vec, block_N) = (64, 128) | float32 | UB (shared) | 输入 tile（fp32 中间计算） |
| t0_ub | (64, 128) | float32 | UB (shared) | 中间计算主 buffer（复用） |
| t1_ub | (64, 128) | float32 | UB (shared) | 中间计算副 buffer（max(x,0)） |
| one_ub | (64, 128) | float32 | UB (shared) | 常量 1.0 buffer（T.tile.sub 标量约束） |
| b_ub | (64, 128) | float32 | UB (shared) | 输出 tile（fp32 中间结果） |
| tmp_orig | (64, 128) | orig dtype | UB (shared) | GM↔UB dtype 桥接（non-fp32 时） |

> `rows_per_vec = block_M // VEC_NUM = 128 // 2 = 64`。VEC_NUM=2 实现核内并行（vid 维度），每个 vid 处理 64 行。

### 4.4 内存搬运路径

```
纯 Vector 路径（fp32 输入）:
GM[A] --T.copy--> UB[a_ub(fp32)] --12步T.tile.xxx计算--> UB[b_ub(fp32)] --T.copy--> GM[B]

纯 Vector 路径（fp16/bf16 输入）:
GM[A] --T.copy--> UB[tmp_orig(fp16/bf16)] --T.tile.cast--> UB[a_ub(fp32)]
  --12步T.tile.xxx计算--> UB[b_ub(fp32)] --T.tile.cast--> UB[tmp_orig(fp16/bf16)] --T.copy--> GM[B]
```

无 Cube 计算路径，不经过 L1/L0A/L0B/L0C。

### 4.5 UB 内存预算

以 block_M=128, block_N=128, VEC_NUM=2, rows_per_vec=64 为例：

| Buffer | Shape | dtype | 大小 (Bytes) |
|--------|-------|-------|-------------|
| a_ub | (64, 128) | float32 | 64×128×4 = 32768 |
| t0_ub | (64, 128) | float32 | 32768 |
| t1_ub | (64, 128) | float32 | 32768 |
| one_ub | (64, 128) | float32 | 32768 |
| b_ub | (64, 128) | float32 | 32768 |
| tmp_orig | (64, 128) | float16/bf16 | 64×128×2 = 16384（fp32 时此 buffer 被 MEMORY_PLANNING 消除） |
| **总计（non-fp32）** | | | 181248 / 196608 (192KB, A2/A3 UB) |
| **总计（fp32）** | | | 163840 / 196608（tmp_orig 被消除） |

> **UB 容量验证**：最大占用 181248 Bytes < 192KB (196608 Bytes) ✓。留约 15KB 余量给编译器内存规划开销。

### 4.6 动态轴定义

| 动态轴 | 声明方式 | 运行时范围 |
|--------|----------|-----------|
| M | JIT 编译期常量（函数参数） | 1 ~ 16384（单维上限） |
| N | JIT 编译期常量（函数参数） | 1 ~ 16384（单维上限） |

> 输入 0-8 维通过 host `reshape(-1, last_dim)` 降维到 2D (M, N)。M、N 作为 JIT 编译期常量传入 kernel（每个不同 shape 编译一个 kernel 版本，由 tilelang cache 复用）。cann-bench cases 实测 1-5D，单维 2-8193，总元素 ~1M-268M。

### 4.7 JIT 配置

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,        # 自动同步插入
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,  # 自动 UB 内存规划（消除未用 buffer）
}

@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def mish(M, N, block_M, block_N, dtype="float16"):
    ...
```

> **AUTO_CV_COMBINE 不开启**：mish 是纯 Vector 算子（12 步 element-wise，全在 AIV 核执行），开启 `AUTO_CV_COMBINE` 会生成 `MIX_AIC_1_2` 启动配置，导致 AIC 核空闲但仍支付 launch + L0A/L0B/L1/L0C buffer init 成本（参考 mish_archive Stage 3 经验 + custom/sigmoid/sigmoid.py iter1 同一发现）。设计阶段即采纳此优化，避免 Stage 2 实现后再回退调整 pass_configs。

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**: 纯 Vector

**判定依据**: 算子仅包含 12 步 element-wise 运算（abs/mul/exp/add/ln/max/sigmoid/sub），无 matmul、无归约、无跨 tile 数据依赖。判定为纯 Vector，仅需 UB 层级。

### 5.2 Block 划分

```python
block_M = 128  # 选择理由：UB 容量约束下最大化 tile 大小（6 buffers × float32 ≤ 192KB）
block_N = 128  # 选择理由：与 block_M 对称，保证 2D tiling 均衡；128 是 32B 对齐的整数倍
VEC_NUM = 2    # 核内并行：每个 block 内 vid 维度分 2 路，每路处理 64 行
m_num = T.ceildiv(M, block_M)
n_num = T.ceildiv(N, block_N)
block_num = m_num * n_num
```

### 5.3 约束分析

- **对齐约束**: block_N=128, fp16/bf16 尾轴 128 > 16 ✓；float32 128 > 16 ✓。UB 32B 对齐：128×4(fp32)=512B ✓。
- **UB 容量**: 总 buffer = 181248 Bytes (non-fp32) < 192KB (196608 Bytes) ✓，见 §4.5。
- **L0 容量**: 无 Cube 计算，不适用。
- **分形限制**: 无 GEMM，不适用。

### 5.4 非整除处理

```python
m_num = T.ceildiv(M, block_M)  # 非整除时向上取整，最后一个 block 处理尾块
n_num = T.ceildiv(N, block_N)
```

- **输入侧**：`T.copy(A[bx*block_M + vid*rows_per_vec, by*block_N], a_ub)` 按动态切片自动裁剪尾块搬运范围（T.copy 已支持动态 shape 切片，参考 ascend-constraints.md §4 Phase 1 step 4）。
- **输出侧**：`T.copy(b_ub, B[bx*block_M + vid*rows_per_vec, by*block_N])` 同理，尾块只写有效范围。
- **无需 host padding + crop**：前后 GM 两侧均使用动态切片，host 侧不触碰 buffer 内容。

> cann-bench cases 含大量非对齐 shape（如 [1023,1023]、[1009,1021]、[363,367,373]、[1000003] 等），`T.ceildiv` + `T.copy` 动态切片统一处理。

### 5.5 数据搬运性能可行性

| 结构/dtype 路径 | 代表性最大 case | GM pass | DMA 数/平均字节 | GM 标量访问 | 地址 div/mod | AIV 并行度 | 结论 |
|-----------------|-----------------|---------|------------------|-------------|--------------|------------|------|
| 通用 fallback（fp32） | [8192,8192] fp32 (268MB) | 2（读+写） | 4096 blocks × 2 / 64KB | 0（全 DMA） | cid//n_num, cid%n_num（每 block 2 次） | 4096 blocks / ≤24 AIV | 可行 |
| 通用 fallback（fp16） | [8192,8192] fp16 (134MB) | 2 | 4096 × 2 / 32KB | 0 | 同上 | 4096 / ≤24 | 可行 |

> Mish 是纯逐元素运算，无数据重排，GM→UB→GM 单次搬运。无逐元素 strided GM 访问。`[REORDER-COST]` 不适用（无数据重排）。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| M×N block 级 | block 并行（隐式） | `T.Kernel(m_num*n_num, is_npu=True)` | 每个 block 处理一个 (block_M, block_N) tile |
| 核内 vid | 核内并行（隐式） | `VEC_NUM=2`（vid 维度） | 每个 block 内 2 路 vid，每路处理 block_M//2=64 行 |
| 元素级 | 向量化（隐式） | `T.tile.xxx`（Buffer 级 SIMD） | 12 步全向量化，无 T.Parallel 显式循环 |

**循环调度 API 选择说明**（门禁要求）：

| API | 是否使用 | 理由 |
|-----|---------|------|
| `T.Parallel` | 否 | 12 步计算用 `T.tile.xxx` Buffer 级 SIMD 原语直接触发硬件 Vector 指令，比 `T.Parallel` + 符号 API 更直接（参考 silu/sigmoid/tanh 示例均用 T.tile.xxx）。`T.Parallel` 会先分解为符号表达式再 lowering，多一层中间步骤 |
| `T.serial` | 否 | 单 tile 内无循环（12 步顺序 T.tile.xxx 调用，非循环结构） |
| `T.Pipelined` | 否 | 单 tile 计算型，无 K 维迭代累加，流水线无收益（见 §6.3） |
| `T.Persistent` | 否 | 无多 wave 调度需求，block 数由 `T.Kernel(m_num*n_num)` 隐式管理，编译器自动分配 AIV 核 |

### 6.2 循环伪代码

```python
# Block 级并行（隐式，由 T.Kernel 管理）
with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
    bx = cid // n_num    # M 方向 block 索引
    by = cid % n_num     # N 方向 block 索引
    # vid ∈ {0, 1}       # 核内并行索引（VEC_NUM=2）

    # 单 tile 内完成全部 12 步计算，无循环
    # ...（见 §3.3 伪代码）
```

### 6.3 流水线优化

**不使用 T.Pipelined**。Mish 是单 tile 计算型算子（每个 tile 独立完成 12 步 element-wise），无 K 维迭代累加需求。流水线对单 tile 计算无收益。

### 6.4 尾块处理

当 M 或 N 不被 block_M/block_N 整除时：
- `m_num = T.ceildiv(M, block_M)` 产生额外一个 block 处理 M 尾部
- `n_num = T.ceildiv(N, block_N)` 产生额外一个 block 处理 N 尾部
- `T.copy` 按 GM 切片的实际有效范围搬运，UB buffer 中尾块区域为未定义值但不写回 GM（输出侧 T.copy 只写有效范围）

> 参考 examples/activation/sigmoid.py L16 `T.ceildiv` 用法 + ascend-constraints.md §4 Phase 1 step 4（T.copy 支持动态 shape 切片自动处理尾块）。

---

## 7. 同步策略

### 7.1 同步模式

**模式**: 自动同步（Developer 模式）

### 7.2 同步点说明

纯 Vector 算子，单 tile 内 12 步计算全在同一个 AIV 核的 UB 上完成，无跨核数据依赖。`TL_ASCEND_AUTO_SYNC: True` 自动插入必要同步（T.copy 前后的 DMA 屏障）。无需手动 `T.barrier_all` / `T.set_flag` / `T.wait_flag`。

### 7.3 pass_configs 配置

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,        # 自动同步（DMA 屏障自动插入）
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,  # 自动 UB 内存规划
    # TL_ASCEND_AUTO_CV_COMBINE: 不设（纯 Vector，避免空闲 AIC 核，见 §4.7）
    # TL_ASCEND_AUTO_CV_SYNC: 不设（无 CV 融合）
}
```

---

## 8. 融合算子设计

### 8.1 融合算子判定

**判定结果**: 否

**判定依据**: Mish 是纯 Vector 逐元素运算，无 GEMM（Cube）计算，无 CV 融合需求。不产出 workspace 规格。

---

## 9. 验证方案

### 9.1 Golden 函数

```python
import torch

def golden_mish(x: torch.Tensor) -> torch.Tensor:
    """Mish 参考实现：y = x * tanh(softplus(x)) = x * tanh(ln(1 + e^x))

    使用 torch.nn.functional.mish（与 cann-bench golden.py 一致）。
    """
    return torch.nn.functional.mish(x)
```

### 9.2 L0 门槛测试计划

> 设计阶段**只给出 L0 门槛用例**（规则 shape，block 整除），供 Stage 2 快速精度收敛。
> L1（功能，含不规则/尾块 shape）/ L2（异常输入）/ Boundary（特殊值）的**完整分层套件由 `tilelang-op-test-design` 场景 B 在 Stage 2 L0 通过后扩展**。本节不手工枚举 L1/L2/Boundary。

**L0 用例选取原则**（由 `tilelang-op-test-design` 场景 A 从 design.md 生成）：
- 覆盖全部 3 种 dtype（float16/float32/bfloat16）
- 规则 shape（block_M=128, block_N=128 整除）：1024/2048/4096/8192 均为 128 整数倍
- 覆盖小/中/大值域（验证数值稳定方案）
- 包含特殊值门槛（inf/nan/zero/dbound）——Mish 语义明确要求处理这些（见 §1.3 特殊情况），作为 L0 门槛尽早验证数值稳定方案正确性
- 包含 float32 大值域精度哨兵（case 5: [-100,100]，验证 T.tile.exp/ln float32 精度，见 §10.1 风险）

| 用例名 | 级别 | Shape | dtype | block | value_range | 说明 | tags |
|--------|------|-------|-------|-------|-------------|------|------|
| l0_fp16_basic | L0 | (1024, 1024) | float16 | (128,128) | (-1, 1) | fp16 基础精度，小值域，cann-bench case 1 | D-DTYPE-fp16, D-SHAPE-ALIGNED, D-VALRANGE-S |
| l0_fp32_basic | L0 | (1024, 1024) | float32 | (128,128) | (-2, 2) | fp32 基础精度，小值域，cann-bench case 2 | D-DTYPE-fp32, D-SHAPE-ALIGNED, D-VALRANGE-S |
| l0_bf16_basic | L0 | (1024, 1024) | bfloat16 | (128,128) | (-3, 3) | bf16 基础精度，小值域，cann-bench case 3 | D-DTYPE-bf16, D-SHAPE-ALIGNED, D-VALRANGE-S |
| l0_fp16_mid | L0 | (2048, 2048) | float16 | (128,128) | (-10, 10) | fp16 中值域，cann-bench case 4 | D-DTYPE-fp16, D-SHAPE-ALIGNED, D-VALRANGE-M |
| l0_fp32_large | L0 | (8192, 8192) | float32 | (128,128) | (-100, 100) | **fp32 大值域精度哨兵**，cann-bench case 5，验证 T.tile.exp/ln float32 精度 | D-DTYPE-fp32, D-SHAPE-ALIGNED, D-VALRANGE-L |
| l0_fp16_dbound | L0 | (1024, 1024) | float16 | (128,128) | (-65504, 65504) | fp16 边界值（±65504），cann-bench case 10 | D-DTYPE-fp16, D-SHAPE-ALIGNED, D-SPECIAL-DBOUND |
| l0_bf16_inf | L0 | (1024, 1024) | bfloat16 | (128,128) | inf 混合 | ±inf 特殊值（有限值+稀疏 inf 混合），cann-bench case 12 | D-DTYPE-bf16, D-SHAPE-ALIGNED, D-SPECIAL-INF |
| l0_fp32_nan | L0 | (1024, 1024) | float32 | (128,128) | nan 混合 | NaN 特殊值（有限值+稀疏 nan 混合），cann-bench case 13 | D-DTYPE-fp32, D-SHAPE-ALIGNED, D-SPECIAL-NAN |
| l0_fp16_zero | L0 | (1024, 1024) | float16 | (128,128) | (0, 0) | 零值（x=0→y=0 验证），cann-bench case 14 | D-DTYPE-fp16, D-SHAPE-ALIGNED, D-SPECIAL-ZERO |

> **L0 共 9 个用例**。block 整除验证：1024/128=8, 2048/128=16, 8192/128=64 ✓。
> **特殊值位置契约**：inf/nan 用例采用"有限值 + 稀疏特殊值"的混合输入（非全 inf/全 nan），先严格比较 inf/nan mask 一致性，再对有限值应用混合容差。参考 precision-standard.md §3.1。

### 9.3 精度标准

> 采用**混合容差**：逐元素 `|actual-golden| ≤ atol + rtol·|golden|`，整体判定 `matched_ratio ≥ required_matched_ratio` **且** `max_abs_error ≤ max_abs_error_limit`。
> 阈值**仅按 dtype**（与算子类别无关），L0/L1/Boundary 套用精度比对（L2 为非法输入负向测试，不比精度）；整型按 0 误差精确匹配。
>
> **特殊浮点值的位置契约**：若算子支持 NaN/Inf，测试输入使用有限值与稀疏特殊值混合的确定性数据，并保证至少存在一个特殊值和一个有限值。比较时先分别严格检查 `isnan`、正 Inf、负 Inf mask 完全一致，再仅对双方均为有限值的位置应用上面的混合容差。`allclose(equal_nan=True)` 不能替代显式 mask 检查。

**精度阈值来源**：cann-bench 生态算子精度标准（desc.md §4）的 Threshold 对齐——`rtol = Threshold`（cann-bench 的 MERE 阈值），`atol = Threshold/16`（更严格的绝对容差保证小值精度），`max_abs_error_limit` 放大以适配 Mish 大值域特性（`mish(x) ≈ x` 当 x 大时，绝对误差随 |x| 增大）。

| dtype | atol | rtol | max_abs_error_limit | required_matched_ratio |
|-------|------|------|---------------------|------------------------|
| float16 | 2⁻¹⁴ (6.10e-5) | 2⁻¹⁰ (9.77e-4) | 1e2 | 0.99 |
| bfloat16 | 2⁻¹¹ (4.88e-4) | 2⁻⁷ (7.81e-3) | 1e3 | 0.99 |
| float32 | 2⁻¹⁷ (7.63e-6) | 2⁻¹³ (1.22e-4) | 1e0 | 0.99 |

> **与 cann-bench 标准对齐**：cann-bench Threshold（fp16=2⁻¹⁰, bf16=2⁻⁷, fp32=2⁻¹³）对应本表 rtol；cann-bench 通过条件 `MERE < Threshold 且 MARE < 10·Threshold` 对应本表 `matched_ratio ≥ 0.99 且 max_abs_error ≤ max_abs_error_limit`。
> **与 precision-standard.md 默认值差异**：默认值（rtol=2⁻⁹ fp16, max_abs=1e-1）对 Mish 大值域过严（x=100 时 mish≈100，0.1 绝对误差 > 1e-1 会误判 FAIL）。本表经 mish_archive 验证通过（attempt 2: L0:8 L1:15 全 PASS）。

### 9.4 性能可行性哨兵（强制执行，不可因 large/L1 跳过）

| 用例名 | Shape | dtype/属性 | 覆盖路径 | 单 case timeout | 选择理由 |
|--------|-------|------------|----------|-----------------|----------|
| perf_worst_fp32_large | (8192, 8192) | float32, [-100,100] | 通用 fallback（最大 DMA + 12 步计算） | 120s | 最大 case（268MB），12 步 fp32 计算最重，4096 blocks / ≤24 AIV（每核 ~171 tiles），验证大张量不超时 |
| perf_worst_fp16_large | (8192, 8192) | float16, [-10,10] | 通用 fallback + cast | 60s | fp16 大 case（134MB），含 cast-in/cast-out 额外开销 |

> 测试数据、随机数、特殊值和 golden 物理重排均在 CPU 完成；运行阶段只做 H2D → TileLang kernel → D2H，避免测试 harness 引入 aclnn 依赖。

---

## 10. 风险点与注意事项

### 10.1 已知约束与风险

| 风险 | 影响 | 严重度 | 缓解方案 |
|------|------|--------|----------|
| **`T.tile.exp/ln` float32 大值域精度风险** | api-compute.md §4.2 警告：`T.tile.exp/ln` 内部可能以 fp16 计算无论 buffer dtype，float32 输入 + \|x\|>16 时 MARE≈0.9。mish 的 exp 参数是 `-|x|`（≤0），ln 参数是 `1+exp(-|x|)` ∈ [1,2]，均是小值域，**风险较低**。但 cann-bench case 5/11/20 float32 大值域在 mish_archive Stage 3 曾报失败。 | 中 | (1) L0 已包含 `l0_fp32_large` [-100,100] 哨兵用例尽早暴露；(2) Fallback：若 float32 精度不达标，将 `T.tile.exp` → `T.Parallel + T.exp` 符号 API（preserves buffer dtype，生成 vectorized `AscendC::Exp`），`T.tile.ln` → `T.Parallel + T.log`；(3) `T.tile.sigmoid` 精度行为不同于 exp/ln（api-compute.md §4.2 注释），mish_archive 验证 sigmoid 在 float32 下未触发精度问题 |
| **bf16 CANN intrinsic 不支持** | CANN Muls/Maxs/Exp/Adds/Div 不支持 `__bf16`，直接用 bf16 buffer 编译失败 | 高 | **已解决**：所有中间计算 buffer 用 float32（ACC_DTYPE），bf16 仅在 GM 边界通过 T.tile.cast 转换（mish_archive attempt 1 踩坑 → attempt 2 修复验证） |
| **fp16 12 步累积精度损失** | 12 步 element-wise 在 fp16 下累积误差导致 matched_ratio=0.5449, max_abs=4.639e-3 | 高 | **已解决**：float32 中间计算（同上） |
| **纯 Vector AUTO_CV_COMBINE 空闲 AIC 核** | 开启 AUTO_CV_COMBINE 生成 MIX_AIC_1_2，AIC 空闲但支付 launch + buffer init 成本 | 低 | **已解决**：pass_configs 不设 AUTO_CV_COMBINE（设计阶段即采纳，见 §4.7） |
| **`T.tile.sub` 标量约束** | `T.tile.sub(dst, src0, src1)` 的 src1 不接受标量 PrimExpr | 低 | **已解决**：用预填充的 `one_ub` buffer（T.tile.fill(one_ub, 1.0)）替代标量 1.0（步骤 11） |
| **`T.tile.sigmoid` 文档未完整列出** | api-compute.md §4 表格未列 sigmoid，但 mish_archive 验证可用 | 低 | 已通过 mish_archive/mish.py L126 + api-compute.md §4.2 注释确认存在；Stage 2 实现时若 API 不存在，fallback 用 `T.tile.reciprocal` + `T.tile.exp` 组合实现 sigmoid：`sigmoid(x) = reciprocal(1 + exp(-x))` |
| **-inf × 0 = NaN 问题** | x=-inf 时 softplus(-inf)=0, tanh(0)=0, mish=-inf×0=NaN（数学上应为 0） | 低 | CANN intrinsic 的 exp(-inf)=0, sigmoid(0)=0.5, tanh=2×0.5-1=0, mish=-inf×0=NaN。cann-bench golden `torch.nn.functional.mish(-inf)` 返回 0（PyTorch 内部处理）。Stage 2 测试时对 -inf 位置做结构比对（precision-standard.md §3.1：inf/nan 位置结构比对，不计入数值容差）。若需严格匹配 golden=0，Stage 2 可在 kernel 末尾加 `T.tile.compare` + `T.tile.select` 将 -inf×0 的 NaN 替换为 0（但引入条件分支，仅对 -inf 位置，不影响向量化主路径） |

### 10.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| UB 溢出 | block_M/block_N 过大（如 256×256） | 编译失败 | 减小 block size，6 buffers × float32 ≤ 192KB |
| bf16 直接计算 | 用 bf16 buffer 做 T.tile.exp | CANN 编译错误（`__bf16` 不支持） | 全部中间 buffer 用 float32，边界 T.tile.cast |
| `T.tile.sub` 用标量 | `T.tile.sub(t0_ub, t0_ub, 1.0)` | 编译错误（src1 不接受标量） | 用 `one_ub` buffer（预填充 1.0） |
| `T.tile.log` 拼写 | 用 `T.tile.log` 而非 `T.tile.ln` | AttributeError | 自然对数 API 名为 `ln` |
| `T.copy` 跨 dtype | `T.copy(A_fp16, a_ub_fp32)` | 编译错误 | 用 `T.tile.cast` 做 dtype 转换 |
| host `.contiguous()` | 对非 contiguous 输入调 `.contiguous()` | 触发 aclnn（禁止） | 输入需 contiguous；非 contiguous 由 kernel 内 stride-aware 处理（本设计未涉及，cann-bench 输入均 contiguous） |

### 10.3 特殊场景处理

- **非整除 shape**：`T.ceildiv` + `T.copy` 动态切片自动处理尾块（cann-bench cases 6-9, 15-20 均为非对齐）。
- **高维输入（3D-5D）**：host `reshape(-1, last_dim)` 降维到 2D（contiguous 零拷贝 view），kernel 处理后 `view(orig_shape)` 恢复。cann-bench cases 9(3D)/11(4D)/13(5D)/18(3D)/19(3D)/20(5D) 均通过此路径。
- **1D 输入**：`reshape(1, numel)` 或 `reshape(numel, 1)`，如 cann-bench case 12 [1000003]。
- **极小 shape**：block_M=128 > M 时，`T.ceildiv(M, 128)=1`，单 block 处理，T.copy 尾块裁剪。
- **±inf 输入**：log-sum-exp 方案下 exp(-|±inf|)=0，softplus(+inf)=+inf, softplus(-inf)=0。mish(+inf)=+inf×1=+inf ✓。mish(-inf)=-inf×0=NaN（见 §10.1 风险表，需结构比对或特殊处理）。
- **NaN 输入**：所有 T.tile.xxx 运算传播 NaN，mish(NaN)=NaN ✓。

---

## 11. 交付清单

### 11.1 目录结构

```
custom/mish/
├── DESIGN.md            # 本设计文档（含 L0 门槛测试计划）
├── proto.yaml           # 算子接口规格（dtype/attr，供覆盖门禁派生应覆盖维度）
├── mish.py              # 纯 kernel（@tilelang.jit，可 import，无 golden/测试/__main__）— Stage 2 产出
├── test_mish.py         # from mish import mish + golden + 分层测试 L0/L1/L2/Boundary + main — Stage 2 产出
├── README.md            # 使用说明（可选）— Stage 2 产出
├── perf_tuning/         # 性能优化日志 — Stage 3 产出（可选）
├── history_version/     # 精度调试备份 + 设计回退备份
└── .orchestrator_state.json  # Orchestrator 状态（仅 Orchestrator 读写）
```

### 11.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | ✅ 已完成 | 设计文档（11 章节 + L0 门槛测试计划） |
| `proto.yaml` | ✅ 已完成 | 算子接口规格（dtype 全集取自 §9.3 精度表、attr 取自 §4/§1，机械派生），覆盖门禁 `coverage_check.py --proto` 用 |
| `mish.py` | ⬜ 待实现 | 纯 kernel（@tilelang.jit，§3.3 伪代码落地） |
| `test_mish.py` | ⬜ 待实现 | golden + 分层测试 + main（`from mish import mish`） |

### 11.3 命名规范

- 目录名: `mish`（snake_case）
- kernel 文件: `mish.py`
- 测试文件: `test_mish.py`（顶部 `from mish import mish`）

### 11.4 实现顺序

1. ✅ 设计文档（DESIGN.md）+ proto.yaml + L0 门槛测试计划（本文件 §9.2）
2. ⬜ kernel 实现（`mish.py`，纯 @tilelang.jit，按 §3.3 伪代码落地）
3. ⬜ 测试文件（`test_mish.py`）：import kernel + Golden 函数（§9.1）+ L0 用例（§9.2 的 9 case）+ main
4. ⬜ L0 门槛测试通过（精度收敛，§9.3 精度标准）
5. ⬜ 扩展分层套件（L1 功能 / L2 异常 / Boundary 特殊值，由 `tilelang-op-test-design` 场景 B 生成）
6. ⬜ 全量套件运行（L0/L1 须通过；L2/Boundary 失败仅记录不阻塞）
7. ⬜ 覆盖门禁 `coverage_check.py` 全 PASS/N/A

### 11.5 算子 proto.yaml（覆盖门禁用，Stage 1 产出）

> **dtype 全集取自本文档 §9.3 精度表**（float16/float32/bfloat16）+ **§4/§1** 的 attr/shape 机械派生，是覆盖门禁 `coverage_check.py --proto` 的**权威 dtype/attr 来源**。`inputs[].dtype` 与 §9.3 精度表的 dtype 行一致（全集）；`attrs[].name` 覆盖所有影响计算路径的属性（Mish 无属性，`attrs: []`）。

```yaml
operator:
  name: Mish
  category: Activation
  difficulty: L1
  formula: |
    y = x * tanh(softplus(x)) = x * tanh(ln(1 + e^x))
  description: 自正则化的非单调神经网络激活函数
  shape_support: 输入 1~8 维（cases 实测 1~5D），单维 1~16384，总元素 1~64M
  attrs: []                              # mish 无影响计算路径的属性
  inputs:
    - name: x
      description: 输入张量
      dtype:
        - float16
        - float32
        - bfloat16
  outputs:
    - name: y
      description: 输出张量，Mish 激活结果
      dtype:
        - float16
        - float32
        - bfloat16
  schema: mish(Tensor x) -> Tensor y
```

> **一致性约束**：`inputs[].dtype` = [float16, float32, bfloat16] 与 §9.3 精度表的 3 个 dtype 行一致 ✓；`attrs: []` 覆盖所有关键属性（Mish 无属性，无 D-PARAM-* 派生维度）✓。

---

## 12. 性能目标（Stage 3 调优配置）

> 本章节由 Orchestrator 在 Stage 3 启动前根据用户确认追加写入，不覆盖既有内容。

### 12.1 调优配置

| 字段 | 值 | 说明 |
|------|-----|------|
| 性能目标类型 | `baseline_compare` | 与 PyTorch `torch.nn.functional.mish` 对比加速比 |
| Baseline | `torch.nn.functional.mish` | PyTorch 内置 Mish 实现（NPU device） |
| 目标加速比 | 无硬性目标，尽力优化 | 用户未指定具体数值，以迭代上限/连续无提升为止 |
| 测试 shape | cann-bench 20 个标准 case | 涵盖 1-5D、对齐/非对齐、3 种 dtype、各种值域 |
| 噪声阈值 | 3% | perf-tuner 默认采纳门槛 |
| 最大迭代数 | 15 | 用户指定 |

### 12.2 cann-bench 测试 case 列表

| case | shape | dtype | value_range |
|------|-------|-------|-------------|
| 1 | [1024,1024] | float16 | [-1,1] |
| 2 | [2048,2048] | float32 | [-2,2] |
| 3 | [4096,4096] | bfloat16 | [-3,3] |
| 4 | [8192,8192] | float16 | [-10,10] |
| 5 | [8192,8192] | float32 | [-100,100] |
| 6 | [1023,1023] | bfloat16 | [-0.1,0.1] |
| 7 | [1009,1021] | float16 | [-1,2] |
| 8 | [1537,769] | float32 | [-5,10] |
| 9 | [363,367,373] | bfloat16 | [-50,100] |
| 10 | [2049,513] | float16 | [-65504,65504] |
| 11 | [3,7,13,4001] | float32 | [-88,88] |
| 12 | [1000003] | bfloat16 | [-inf,inf] |
| 13 | [11,13,17,67,67] | float32 | [nan,nan] |
| 14 | [3,7,11,13,1009] | float16 | [0,0] |
| 15 | [512,2049] | float32 | [-0.5,0.5] |
| 16 | [255,8193] | bfloat16 | [-1,3] |
| 17 | [4097,511] | float16 | [-1000,1000] |
| 18 | [2,511,2049] | float32 | [-0.2,0.2] |
| 19 | [4,255,2049] | bfloat16 | [-3,6] |
| 20 | [2,3,17,1024,101] | float32 | [-20,40] |

### 12.3 中止条件

满足任一即结束：
1. 迭代次数达到 15 轮
2. 连续三次无性能提升
3. 用户指定的性能目标（本场景无硬性目标，以 1/2 为主）
