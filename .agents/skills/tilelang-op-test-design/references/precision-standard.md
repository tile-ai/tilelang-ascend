# TileLang-Ascend 精度标准体系（混合容差）

**对齐来源**：opbase 生态算子开源精度标准。
本体系采用**混合容差（Mixed Tolerance）+ 通过率 + 最大绝对误差硬帽**的判定方式，阈值**仅按 dtype**划分。

---

## 〇、适用范围

- 本标准适用于**浮点计算类算子**（float16 / bfloat16 / float32 / hifloat32 / float8_e4m3 / float8_e5m2）。
- **整型算子**（int8/int16/int32/int64/uint8）按 **0 误差精确匹配**：逐元素必须完全相等，`required_matched_ratio = 1.0`，一个元素不符即 `[PRECISION_FAIL]`。
- 搬运类 / 其他非计算类算子需按各算子实际业务场景单独制定，不在本标准范围内。

---

## 一、误差指标与通过标准

### 1.1 逐元素通过条件（混合容差）

对输出张量中的每个（有限值）元素，满足下式即判定该元素通过：

```
|actual - golden| ≤ atol + rtol * |golden|
```

- **atol**（绝对容差）：保证小值（golden 接近 0）场景的合理误差范围，天然避免除零。
- **rtol**（相对容差）：保证大值场景的相对精度。

### 1.2 整体通过条件（双门限）

定义**通过率**：

```
matched_ratio = 通过元素数 / 总元素数
```

当**同时**满足以下两个条件时，判定该用例通过：

```
(1) matched_ratio    ≥ required_matched_ratio
(2) max_abs_error    ≤ max_abs_error_limit
```

其中 `max_abs_error` 为该用例中任意元素的最大绝对误差，`max_abs_error_limit` 为绝对误差硬上限。
二者是**与**关系：允许至多 `1 - required_matched_ratio` 比例的元素超出逐元素容差，但**任何单个元素的绝对误差都不得超过硬帽**。整型为逐元素精确匹配（等价 `required_matched_ratio = 1.0`、硬帽 = 0）。

---

## 二、混合容差阈值表（按 dtype）

| 数据类型 | rtol | atol | max_abs_error_limit | required_matched_ratio |
|---|---|---|---|---|
| **float16** | 2⁻⁹ (1.95e-3) | 2⁻¹⁴ (6.10e-5) | 1e-1 | 0.99 |
| **bfloat16** | 2⁻⁶ (1.56e-2) | 2⁻¹⁰ (9.77e-4) | 1e0 | 0.99 |
| **float32** | 2⁻¹⁰ (9.77e-4) | 2⁻¹⁶ (1.53e-5) | 1e-2 | 0.99 |
| **hifloat32** | 2⁻¹⁰ (9.77e-4) | 2⁻¹⁶ (1.53e-5) | 1e-2 | 0.99 |
| **float8_e4m3** | 2⁻² (0.25) | 2⁻⁴ (0.0625) | 1e0 | 0.99 |
| **float8_e5m2** | 2⁻¹ (0.5) | 2⁻³ (0.125) | 1e-1 | 0.99 |
| **int8/16/32/64, uint8** | 0 | 0 | 0 | 1.0（精确匹配） |

**说明**：
- rtol / atol 均以 **2 的幂**给出。
- `required_matched_ratio` 浮点统一 0.99；整型必须 1.0。
- 阈值**只看 dtype，不看算子类别**——GEMM / Softmax / Normalization / Activation / Reduction / Fusion 统一用本表。

---

## 三、通过判定（单标杆）

- **单标杆比对**：与更高精度的实现（CPU 或昇腾小算子拼接）的单一精度标杆直接比较作为 golden。
- 当用例同时满足 `matched_ratio ≥ required_matched_ratio` 且 `max_abs_error ≤ max_abs_error_limit` 时，判定该用例精度通过。

### 3.1 特殊值（INF / NAN）处理

`|inf - inf|` 会得到 `nan`，无法计入数值容差，因此 inf/nan 位置做**结构比对**，不参与 `matched_ratio` / `max_abs_error` 计算：

- `isinf(actual)` 与 `isinf(golden)` 位置一致；
- `isnan(actual)` 与 `isnan(golden)` 位置一致；
- 其余有限值位置按 §1 混合容差判定。

---

## 四、精度标准应用方法

### 4.1 标准查询与判定函数

```python
import torch


def get_precision(dtype):
    """返回 (atol, rtol, max_abs_error_limit, required_matched_ratio)。
    浮点：混合容差；整型：精确匹配（0 误差）。"""
    fp_table = {
        # dtype       : (atol,   rtol,   max_abs_error_limit, required_matched_ratio)
        "float16":     (2**-14, 2**-9,  1e-1, 0.99),   # atol 6.10e-5, rtol 1.95e-3
        "bfloat16":    (2**-10, 2**-6,  1e0,  0.99),   # atol 9.77e-4, rtol 1.56e-2
        "float32":     (2**-16, 2**-10, 1e-2, 0.99),   # atol 1.53e-5, rtol 9.77e-4
        "hifloat32":   (2**-16, 2**-10, 1e-2, 0.99),
        "float8_e4m3": (2**-4,  2**-2,  1e0,  0.99),   # atol 0.0625, rtol 0.25
        "float8_e5m2": (2**-3,  2**-1,  1e-1, 0.99),   # atol 0.125,  rtol 0.5
    }
    int_types = {"int8", "int16", "int32", "int64", "uint8"}
    if dtype in int_types:
        return (0.0, 0.0, 0.0, 1.0)          # 整型：精确匹配，一个元素不符即 FAIL
    return fp_table.get(dtype, (2**-14, 2**-9, 1e-1, 0.99))


def check_precision(actual, golden, dtype):
    """精度判定：返回 (passed, matched_ratio, max_abs_error)。
    浮点双门限：matched_ratio ≥ required 且 max_abs_error ≤ max_abs_error_limit；
    整型：逐元素精确相等。inf/nan 位置做结构比对，不计入数值容差。"""
    atol, rtol, max_abs_limit, required_ratio = get_precision(dtype)
    a = actual.detach().cpu()
    g = golden.detach().cpu()
    if atol == 0.0 and rtol == 0.0:                      # 整型精确匹配
        mism = (a != g).sum().item()
        total = max(a.numel(), 1)
        return mism == 0, 1.0 - mism / total, (0.0 if mism == 0 else float("inf"))
    a = a.float()
    g = g.float()
    special = ~torch.isfinite(g)                         # inf/nan 位置结构比对
    if special.any():
        if not torch.equal(torch.isnan(a[special]), torch.isnan(g[special])) or \
           not torch.equal(torch.isinf(a[special]), torch.isinf(g[special])):
            return False, 0.0, float("inf")
    m = torch.isfinite(g)                                # golden 有限值位置全比：actual 若为 inf/nan 则计为不达标
    if m.sum().item() == 0:
        return True, 1.0, 0.0
    abs_err = (a[m] - g[m]).abs()                        # actual 为 inf/nan 处 abs_err=inf/nan → 逐元素判 False 且拉高 max_abs
    matched_ratio = (abs_err <= (atol + rtol * g[m].abs())).float().mean().item()
    max_abs_error = abs_err.max().item()
    passed = (matched_ratio >= required_ratio) and (max_abs_error <= max_abs_limit)
    return passed, matched_ratio, max_abs_error
```

### 4.2 在测试代码中的应用

```python
def test_silu_float16():
    dtype = "float16"
    x = torch.randn(128, 256, dtype=torch.float16, device="npu")
    y_actual = silu_kernel(x)
    y_expected = torch.nn.functional.silu(x)

    passed, ratio, max_abs = check_precision(y_actual, y_expected, dtype)
    assert passed, f"matched_ratio={ratio:.4f}, max_abs={max_abs:.3e}"
```

### 4.3 在用户交互中的应用

Phase 3 用户交互中询问精度标准（如 design.md §9.3 未定义）：

```
询问精度标准（如 §9.3 未定义）：
[1] 使用默认混合容差标准（按 dtype 自动选择，见 §二）
[2] 自定义（atol / rtol / max_abs_error_limit / required_matched_ratio）
[3] 不验证精度（仅验证功能）
```

---

## 五、总结

1. **混合容差 + 双门限**：逐元素 `|actual-golden| ≤ atol + rtol·|golden|`，整体看 `matched_ratio ≥ required` 且 `max_abs ≤ limit`。
2. **阈值仅按 dtype**：查 §二 表，与算子类别无关。
3. **整型精确匹配**：atol=rtol=0、required=1.0，逐元素必须相等。
4. **单标杆 golden**：CPU 或昇腾小算子拼接的高精度实现。
5. **INF/NAN 结构比对**：不计入数值容差，位置一致即可。
