# tilelang_precision_checker Usage Guide

> Located at `testing/python/precision/tilelang_precision_checker.py`

---

## 1. Quick Start

```python
from tilelang_precision_checker import precision_check, PrecisionLevel, OpCategory

# Simplest usage: default L0 + Float Compute
precision_check(npu_output, torch_reference.cpu())

# Specify level
precision_check(npu_output, torch_reference.cpu(), level=PrecisionLevel.L1)

# Silent mode, decide manually
report = precision_check(npu_output, torch_reference.cpu(), verbose=False)
if report.passed:
    print("Passed")
else:
    print(report.summary())
```

---

## 2. Core Concepts

### 2.1 Precision Levels

| Level | Meaning | MARE ≤ | MERE ≤ | RMSE ≤ |
|-------|---------|--------|--------|--------|
| `L0` | Standard ops, non-critical | 10 | 2.0 | 2.0 |
| `L1` | Important ops (LLM/Recommendation) | 5 | 1.5 | 1.5 |
| `L2` | Critical ops (core business) | 2 | 1.2 | 1.2 |

### 2.2 Operator Categories

| Category | Meaning | Check Method |
|----------|---------|-------------|
| `NON_COMPUTE` | Data movement / Cast | Bitwise match |
| `INTEGER_COMPUTE` | Integer arithmetic | Bitwise or AE=0 |
| `QUANTIZE_COMPUTE` | Quantization | AE ≤ 1 |
| `FLOAT_COMPUTE` | **Floating-point (default)** | MARE/MERE/RMSE + small-value + INF/NAN |

### 2.3 Error Metrics

| Metric | Formula | Meaning |
|--------|---------|---------|
| **MARE** | `max(|NPU - golden| / (|golden| + 1e-7))` | Maximum relative error (worst point) |
| **MERE** | `mean(|NPU - golden| / (|golden| + 1e-7))` | Mean relative error |
| **RMSE** | `sqrt(mean((NPU - golden)^2))` | Root mean square error (overall dispersion) |
| **SmallVal** | Count of elements where `|golden|<threshold` and `|diff|>tolerance` | Small-value error count |
| **Ratio** | `NPU metric / threshold` | Pass if Ratio ≤ 1.0 |

---

## 3. API Reference

### 3.1 `precision_check()`

```python
def precision_check(
    actual: torch.Tensor,                           # NPU operator output
    golden: torch.Tensor,                           # Torch reference (must be on CPU)
    level: PrecisionLevel = PrecisionLevel.L0,      # Precision level
    category: OpCategory = OpCategory.FLOAT_COMPUTE,  # Operator category
    verbose: bool = True,                           # Print report to stdout
) -> PrecisionReport:
```

**Internal execution flow**:

```
actual, golden
    │
    ├─ Shape / Device assertions
    ├─ Branch by category:
    │   ├─ NON_COMPUTE     → Bitwise comparison
    │   ├─ INTEGER_COMPUTE → Bitwise or AE=0
    │   ├─ QUANTIZE        → AE ≤ 1
    │   └─ FLOAT_COMPUTE   → Full flow:
    │
    ├─ ① INF/NAN position match check  _check_inf_nan()
    │     Mismatch → print first 5 positions → return FAIL
    │
    ├─ ② Compute raw MARE/MERE/RMSE  _compute_errors()
    │
    ├─ ③ Small-value check  _check_small_value()
    │     Count elements where |golden|<threshold AND |diff|>tolerance
    │     Ratio = NPU_errs / max(Ref_errs, 1) ≤ 2 → Pass
    │
    ├─ ④ Exclude small-value elements → recompute MARE/MERE/RMSE
    │     (Relative error explodes when golden→0)
    │
    ├─ ⑤ Compare each metric against _THRESHOLDS[level]
    │     All must pass for final PASS
    │
    └─ Return PrecisionReport
```

**Return object `PrecisionReport`**:

| Attribute | Type | Description |
|-----------|------|-------------|
| `passed` | `bool` | Whether the check passed |
| `level` | `PrecisionLevel` | Precision level used |
| `category` | `OpCategory` | Operator category |
| `mare` | `float` | MARE (after excluding small values) |
| `mere` | `float` | MERE (after excluding small values) |
| `rmse` | `float` | RMSE (after excluding small values) |
| `mare_ratio` | `float \| None` | MARE / threshold |
| `mere_ratio` | `float \| None` | MERE / threshold |
| `rmse_ratio` | `float \| None` | RMSE / threshold |
| `small_value_error_count_npu` | `int` | NPU small-value error count |
| `small_value_error_count_ref` | `int` | Reference small-value error count |
| `small_value_passed` | `bool` | Small-value check result |
| `inf_nan_match` | `bool` | INF/NAN position consistency |
| `num_elements` | `int` | Total output element count |
| `message` | `str` | Descriptive message |
| `summary()` | `() -> str` | Generate readable report string |

### 3.2 `multi_seed_precision_check()`

**Retest workflow** triggered when a single `precision_check` fails, using Bootstrap statistical inference.

```python
def multi_seed_precision_check(
    kernel_fn,                          # Compiled tilelang kernel
    *input_args,                        # Input tensors
    level: PrecisionLevel = L0,
    category: OpCategory = FLOAT_COMPUTE,
    num_seeds: int = 1000,              # Number of random seeds
    ref_fn=None,                        # Reference function
    **input_kwargs,                     # Extra kernel args
) -> Dict[str, Any]:
```

**Decision rules**:

```
Run kernel N times with different random seeds, collect MARE Ratio
         ↓
Bootstrap resampling 2000×, compute 95% CI of median Ratio
         ↓
┌─ N < 200  ─→ Fuse: too few samples, FAIL
├─ CI_lower > 1.0 → Confirmed precision anomaly ❌
└─ CI_lower ≤ 1.0 → Sporadic false positive, PASS ✅
```

**Return dict**:

| Key | Type | Description |
|-----|------|-------------|
| `passed` | `bool` | Pass or fail |
| `median_ratio` | `float` | Median MARE Ratio |
| `ci_lower` | `float` | 95% CI lower bound |
| `ci_upper` | `float` | 95% CI upper bound |
| `num_samples` | `int` | Sample count |
| `message` | `str` | Verdict description |

**Note**: Assumes the kernel is purely functional (input → output). Operators with workspaces (e.g., Flash Attention) require custom adaptation.

---

## 4. Usage by Category

### 4.1 Float Compute (default, most common)

```python
report = precision_check(
    npu_out,
    torch_ref.cpu(),
    level=PrecisionLevel.L0,          # L0 for development phase
    category=OpCategory.FLOAT_COMPUTE, # Optional, the default
)
# Output:
# === Precision Check [✅ PASS] ===
#   Level:     L0
#   Category:  FLOAT_COMPUTE
#   Elements:  65536
#   MARE:      9.999e-02  (ratio=0.0100)
#   MERE:      6.075e-04  (ratio=0.0003)
#   RMSE:      4.005e-05  (ratio=0.0000)
```

### 4.2 Non-Compute (data movement / Cast)

```python
report = precision_check(
    npu_moved_data,
    torch_expected_data,
    category=OpCategory.NON_COMPUTE,
)
# → Bitwise comparison, must be identical
```

### 4.3 Integer Compute

```python
report = precision_check(
    npu_int_result,
    torch_int_result,
    category=OpCategory.INTEGER_COMPUTE,
)
# → Bitwise or max absolute error = 0
```

### 4.4 Quantize Compute

```python
report = precision_check(
    npu_quant_result,
    torch_quant_result,
    category=OpCategory.QUANTIZE_COMPUTE,
)
# → Integer output: AE ≤ 1
# → Float output: same as FLOAT_COMPUTE
```

### 4.5 Multi-Seed Retest

```python
result = multi_seed_precision_check(
    my_kernel_func,        # Compiled kernel
    tensor_a, tensor_b,    # Inputs
    level=PrecisionLevel.L2,
    num_seeds=1000,
    ref_fn=torch_reference_func,
)

if result["passed"]:
    print("Retest passed")
else:
    print(f"Precision anomaly confirmed: CI_lower={result['ci_lower']:.4f} > 1.0")
# Output:
# === Multi-Seed Retest (n=1000) ===
#   Median ratio: 0.6534
#   95% CI:       [0.6201, 0.6893]
#   Verdict:      ✅ PASS  (PASS)
```

---

## 5. Internal Parameters

### 5.1 Small-Value Thresholds

Defined in `_SMALL_VALUE_THRESHOLDS`, no user modification needed:

| Data type | Threshold `|golden| <` | Error tolerance `|diff| >` |
|-----------|-----------------------|---------------------------|
| float16 | 2^-11 ≈ 0.00049 | 2^-16 ≈ 0.000015 |
| bfloat16 | 2^-8 ≈ 0.0039 | 2^-16 ≈ 0.000015 |
| float32 | 2^-14 ≈ 0.000061 | 2^-30 ≈ 9.3e-10 |
| float64 | 2^-14 ≈ 0.000061 | 2^-30 ≈ 9.3e-10 |

When `|golden|` falls below the threshold, relative error explodes due to near-zero denominators. The checker automatically excludes these "small-value" elements and uses ErrorCount for separate judgment.

### 5.2 Precision Level Thresholds

Defined in `_THRESHOLDS`, no user modification needed:

| Level | MARE_th | MERE_th | RMSE_th |
|-------|---------|---------|---------|
| L0 | 10.0 | 2.0 | 2.0 |
| L1 | 5.0 | 1.5 | 1.5 |
| L2 | 2.0 | 1.2 | 1.2 |

---

## 6. FAQ

### Q1: Must golden be on CPU?

Yes. `precision_check` asserts that `golden` is on CPU (or same device as `actual`). Downstream computations (error calc, sorting, median) require CPU support. Always call `.cpu()` on NPU tensors.

```python
# Correct
precision_check(npu_out, torch_ref.cpu())

# Wrong: golden on NPU
precision_check(npu_out, torch_ref_on_npu)  # AssertionError
```

### Q2: Returned FAIL but MARE/MERE/RMSE look fine?

Likely a **small-value false positive**. Check the `SmallVal` line in the report — if `Ref_errors=0`, torch (as golden) has no small-value errors while NPU has a few FP16 rounding differences, inflating the ratio.

**Fix**: Switch to **dual-reference** mode (see `ascend_precision_standard_guide.md` §5.3). Have the torch reference also compared against an FP64 golden so small-value errors are counted fairly. Or skip the small-value check and rely on MARE/MERE/RMSE alone.

### Q3: Why does multi_seed_precision_check fail?

This function assumes the kernel is purely functional (`kernel_fn(*args)` returns output directly). Operators with workspaces (e.g., Flash Attention) are incompatible. Implement the Bootstrap loop manually (see `examples/flash_attention/test_precision_flash_attn_dual.py` for a working example).

### Q4: How do I know if dual-reference alignment is needed?

Follow the judgment method in `ascend_precision_standard_guide.md` §5.3:
1. Analyze NPU kernel dataflow to find precision bottlenecks (FP16 intermediate storage)
2. Check if torch reference uses higher precision at the same position
3. If yes → insert `.half().float()` truncation at the corresponding point in the reference

---

## 7. File Layout

```
testing/python/precision/
├── tilelang_precision_checker.py   ← The checker script
└── README.md                       ← 中文使用说明
└── README_EN.md                    ← This file (English)
```

Related documents:
- `docs/ascend_precision_standard_guide.md` — Full precision standard guide (Chinese)
- `examples/flash_attention/README.md` — Flash Attention precision test case study
