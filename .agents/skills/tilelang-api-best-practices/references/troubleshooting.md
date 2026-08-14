# TileLang Ascend API 故障排查

> 本文件收录 TileLang Ascend API 使用中遇到的具体故障及其解决方案。每条按"症状 → 原因 → 解决"三段式组织，可独立查阅。

---

## 编译时错误

### 1. T.tile.compare + T.tile.select mask 损坏（128 元素时后 64 位反转）

**错误信息**:
```
# 无编译错误，运行时精度失败
# 现象：使用 T.tile.compare + T.tile.select 实现条件分支后，
# L1 测试全部 matched_ratio 下降，负值元素被错误地选择 src0 而非 src1
```

**原因**: Ascend codegen 在生成 `T.tile.compare` 的 mask 时，当 buffer 元素数为 128 的整数倍时，后 64 个元素的 mask 位被反转。这导致 `T.tile.select` 对这些元素选择错误的 source。通过最小测试 kernel 确认：128 元素 buffer，设置 `compare(mask, x, 0.0, "GT")`，前 64 个元素 mask 正确（x>0 时 mask=1），后 64 个元素 mask 反转（x>0 时 mask=0）。

**解决方案**:

1. **优先用数值稳定公式替代条件分支**（推荐，完全消除 compare/select 依赖）：

```python
# ❌ 错误：用 compare + select 实现条件分支
mask_ub = T.alloc_shared((rows, cols), "uint8")
T.tile.compare(mask_ub, x_ub, 20.0, "GT")  # x > 20?
T.tile.select(y_ub, mask_ub, x_ub, mish_naive_ub)  # bug: 后 64 位 mask 反转

# ✅ 正确：用数值稳定公式替代，无需条件分支
# 例如 tanh 用 1 - 2/(exp(2x)+1) 替代 (e^x - e^-x)/(e^x + e^-x)
# 当 x 大时 exp(2x)→inf, 2/inf=0, tanh=1（无 NaN）
T.tile.exp(tmp_ub, sp_ub)          # tmp = exp(softplus)
T.tile.add(tmp2_ub, tmp_ub, tmp_ub) # tmp2 = exp(2*softplus) via add(t,t)
# 注意：add(t,t) 实现乘法 2x，比 mul(t,2.0) 更稳定
T.tile.add(tmp2_ub, tmp2_ub, 1.0)  # tmp2 = exp(2*sp) + 1
# 用 fill + mul 实现 2.0
T.tile.fill(tmp_ub, 1.0)
T.tile.add(tmp_ub, tmp_ub, tmp_ub) # tmp = 2.0
T.tile.div(tanh_ub, tmp_ub, tmp2_ub) # tanh = 2/(exp(2*sp)+1)
# tanh = 1 - 2/(exp(2*sp)+1)，用 fill+sub 实现
T.tile.fill(one_ub, 1.0)
T.tile.sub(tanh_ub, one_ub, tanh_ub)
```

2. **softplus 数值稳定公式**（避免 exp(x) 溢出，无需 compare/select）：

```python
# softplus(x) = ln(1 + exp(x))
# 当 x > 0：等价变形为 x + ln(1 + exp(-x))，exp(-x) ∈ (0,1] 永不溢出
# 当 x ≤ 0：原始公式 exp(x) ∈ (0,1] 不溢出
# 但可以用统一公式避免分支：
# softplus(x) = max(x, 0) + ln(1 + exp(-|x|))
# 用 T.tile.max 实现（如果支持标量比较），或直接用 naive 公式 + 稳定 tanh
```

3. **必须用 select 时避免 128 元素对齐 buffer**（不推荐，治标不治本）：

```python
# 将 buffer size 设为非 128 整数倍（如 129 或 127）
# 但这会引入对齐问题，不推荐
```

4. **用 fill + mul 组合替代 select**（适用于简单条件赋值）：

```python
# 目标：当 mask=True 时 y=a，mask=False 时 y=b
# 等价：y = mask * a + (1-mask) * b
# 但 Ascend mask 是 uint8，需要先 cast 到 float
# 此方法仍有 128 元素 bug 风险，不推荐
```

**触发条件**: 当 buffer 元素数为 128 的整数倍时（如 block_M=128, block_N=128 的 UB buffer），使用 `T.tile.compare` + `T.tile.select` 组合实现条件分支。

**关联 skill / 章节**: `tilelang-api-best-practices/references/api-compute.md §4` (T.tile.compare / T.tile.select)

**发现来源**: mish 算子开发 Stage 2 attempt 2（2026-08-07），`custom/mish/debug_log.md` attempt 2 详细记录。

---
