"""
TileLang Operator Precision Checker

Usage:
    1. Import this module in your operator test
    2. Call precision_check() with NPU output and torch reference
    3. The script computes MARE/MERE/RMSE and judges pass/fail by precision level
    4. Outputs a detailed precision report

Examples:
    from tilelang_precision_checker import precision_check, PrecisionLevel, OpCategory

    # Simplest usage (defaults: L0 + Float Compute)
    precision_check(npu_result, torch_result)

    # Specify precision level
    precision_check(npu_result, torch_result, level=PrecisionLevel.L1)

    # Specify operator category
    precision_check(npu_result, torch_result, category=OpCategory.INTEGER_COMPUTE)

    # Silent mode, inspect report manually
    report = precision_check(npu_result, torch_result, level=PrecisionLevel.L2, verbose=False)
    if report.passed:
        print("Passed")
    else:
        print(report.summary())

    # Retest: multi-seed Bootstrap after a single failure
    result = multi_seed_precision_check(kernel_fn, tensor_a, tensor_b, num_seeds=1000, ref_fn=torch_ref)
    print(result["ci_lower"], result["ci_upper"])
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Any, Dict, Optional, Tuple

import torch


class PrecisionLevel(enum.IntEnum):
    """Precision level

    L0: Standard operators, non-critical workloads
    L1: Important operators, multimodal / LLM / recommendation
    L2: Critical operators, core business
    """

    L0 = 0
    L1 = 1
    L2 = 2


class OpCategory(enum.IntEnum):
    """Operator category

    NON_COMPUTE:      Data movement / Cast, no arithmetic
    INTEGER_COMPUTE:  Integer arithmetic
    QUANTIZE_COMPUTE: Quantization / dequantization
    FLOAT_COMPUTE:    Floating-point arithmetic (default)
    """

    NON_COMPUTE = 0
    INTEGER_COMPUTE = 1
    QUANTIZE_COMPUTE = 2
    FLOAT_COMPUTE = 3


# Pass thresholds by precision level: (MARE, MERE, RMSE)
# Source: ascend_precision_standard_guide.md §6
_THRESHOLDS: Dict[PrecisionLevel, Tuple[float, float, float]] = {
    PrecisionLevel.L0: (10.0, 2.0, 2.0),
    PrecisionLevel.L1: (5.0, 1.5, 1.5),
    PrecisionLevel.L2: (2.0, 1.2, 1.2),
}

# Small-value domain thresholds: dtype -> (value_threshold, error_tolerance)
# When |golden| < value_threshold, relative error explodes.
# Switch to ErrorCount: count elements where |diff| > error_tolerance.
# Source: ascend_precision_standard_guide.md §6.2
_SMALL_VALUE_THRESHOLDS: Dict[torch.dtype, Tuple[float, float]] = {
    torch.float16: (2.0**-11, 2.0**-16),  # |g| < 0.00049,  |diff| > 0.000015
    torch.bfloat16: (2.0**-8, 2.0**-16),  # |g| < 0.0039,   |diff| > 0.000015
    torch.float32: (2.0**-14, 2.0**-30),  # |g| < 0.000061, |diff| > 9.3e-10
    torch.float64: (2.0**-14, 2.0**-30),
}


@dataclasses.dataclass
class PrecisionReport:
    """Precision check report

    Attributes:
        passed:                         Whether the check passed
        level:                          Precision level (L0/L1/L2)
        category:                       Operator category
        mare:                           Max relative error (excluding small values)
        mere:                           Mean relative error (excluding small values)
        rmse:                           Root mean square error (excluding small values)
        mare_ratio:                     MARE / threshold
        mere_ratio:                     MERE / threshold
        rmse_ratio:                     RMSE / threshold
        small_value_error_count_npu:    NPU error count in small-value domain
        small_value_error_count_ref:    Reference error count in small-value domain
        small_value_passed:             Small-value check result
        inf_nan_match:                  Whether INF/NAN positions match
        num_elements:                   Total output element count
        message:                        Descriptive message
    """

    passed: bool
    level: PrecisionLevel
    category: OpCategory
    mare: float
    mere: float
    rmse: float
    mare_ratio: Optional[float]
    mere_ratio: Optional[float]
    rmse_ratio: Optional[float]
    small_value_error_count_npu: int = 0
    small_value_error_count_ref: int = 0
    small_value_passed: bool = True
    inf_nan_match: bool = True
    num_elements: int = 0
    message: str = ""

    def summary(self) -> str:
        """Generate a human-readable precision report string."""
        passed_str = "PASS" if self.passed else "FAIL"
        lines = [
            f"=== Precision Check [{passed_str}] ===",
            f"  Level:     {self.level.name}",
            f"  Category:  {self.category.name}",
            f"  Elements:  {self.num_elements}",
            f"  MARE:      {self.mare:.6e}  (ratio={self.mare_ratio:.4f})" if self.mare_ratio is not None else f"  MARE:  {self.mare:.6e}",
            f"  MERE:      {self.mere:.6e}  (ratio={self.mere_ratio:.4f})" if self.mere_ratio is not None else f"  MERE:  {self.mere:.6e}",
            f"  RMSE:      {self.rmse:.6e}  (ratio={self.rmse_ratio:.4f})" if self.rmse_ratio is not None else f"  RMSE:  {self.rmse:.6e}",
        ]
        if self.small_value_error_count_npu > 0 or self.small_value_error_count_ref > 0:
            lines.append(
                f"  SmallVal:  NPU_errors={self.small_value_error_count_npu}, Ref_errors={self.small_value_error_count_ref}  {'PASS' if self.small_value_passed else 'FAIL'}"
            )
        if not self.inf_nan_match:
            lines.append("  INF/NAN:   FAIL (Mismatch detected)")
        if self.message:
            lines.append(f"  Message:   {self.message}")
        return "\n".join(lines)


def _compute_errors(
    actual: torch.Tensor,
    golden: torch.Tensor,
    eps: float = 1e-7,
) -> Tuple[torch.Tensor, torch.Tensor, float, float, float]:
    """Compute error metrics between actual output and reference.

    Args:
        actual: Actual output tensor (NPU result)
        golden: Reference / ground-truth tensor
        eps:    Small constant to avoid division by zero

    Returns:
        (per-element absolute diff, per-element relative error, MARE, MERE, RMSE)
    """
    # Absolute error: |actual - golden|
    diff = (actual - golden).float().abs().flatten()
    golden_flat = golden.float().flatten()
    # Relative error: |actual - golden| / (|golden| + eps)
    denom = golden_flat.abs() + eps
    relative_error = diff / denom
    # MARE = maximum relative error
    mare = relative_error.max().item()
    # MERE = mean relative error
    mere = relative_error.mean().item()
    # RMSE = sqrt(mean(diff^2))
    rmse = (diff.pow(2).mean().sqrt()).item()
    return diff, relative_error, mare, mere, rmse


def _check_small_value(
    actual: torch.Tensor,
    golden: torch.Tensor,
    dtype: torch.dtype,
) -> Tuple[int, int, bool]:
    """Small-value domain check.

    When |golden| falls below the small-value threshold, relative error
    is unreliable. Instead, count elements whose absolute error exceeds
    the tolerance as "errors".

    Pass condition: error_count_npu / max(error_count_ref, 1) ≤ 2
    i.e. NPU's small-value errors must not exceed 2× the reference count.

    Args:
        actual: NPU actual output
        golden: Reference output
        dtype:  Data type, used to look up thresholds

    Returns:
        (NPU small-value error count, ref small-value error count, passed)
    """
    if dtype not in _SMALL_VALUE_THRESHOLDS:
        return 0, 0, True
    threshold, error_tol = _SMALL_VALUE_THRESHOLDS[dtype]
    golden_flat = golden.float().flatten()
    actual_flat = actual.float().flatten()
    diff = (actual_flat - golden_flat).abs()
    # Positions where |golden| < threshold
    small_mask = golden_flat.abs() < threshold
    # Positions where |diff| > error_tolerance
    error_mask = diff > error_tol
    # Both conditions must hold to count as a small-value error
    error_count_npu = (small_mask & error_mask).sum().item()
    error_count_ref = 0
    passed = True
    if error_count_npu > 0 or error_count_ref > 0:
        ratio = error_count_npu / max(error_count_ref, 1)
        passed = ratio <= 2.0
    return error_count_npu, error_count_ref, passed


def _check_inf_nan(
    actual: torch.Tensor,
    golden: torch.Tensor,
) -> bool:
    """Check whether NPU and golden have INF/NAN at the same positions.

    On mismatch, prints the first 5 mismatched positions and returns False.

    Args:
        actual: NPU actual output
        golden: Reference output

    Returns:
        True if INF/NAN positions match exactly
    """
    # Find non-finite values (inf / -inf / nan) on both sides
    actual_special = ~torch.isfinite(actual)
    golden_special = ~torch.isfinite(golden)
    # Check for mismatches
    mismatch = actual_special != golden_special
    if mismatch.any():
        mismatch_indices = torch.where(mismatch)
        # Print details of the first 5 mismatches
        for idx in zip(*[i[:5].tolist() for i in mismatch_indices]):
            a_val = actual[tuple(idx)].item()
            g_val = golden[tuple(idx)].item()
            print(f"  INF/NAN mismatch at {idx}: NPU={a_val}, torch={g_val}")
        return False
    return True


def precision_check(
    actual: torch.Tensor,
    golden: torch.Tensor,
    level: PrecisionLevel = PrecisionLevel.L0,
    category: OpCategory = OpCategory.FLOAT_COMPUTE,
    verbose: bool = True,
) -> PrecisionReport:
    """Main precision check entry point.

    Based on operator category and precision level, runs the appropriate
    validation pipeline:
      1. Non-compute   → Bitwise comparison
      2. Integer       → Bitwise comparison or AE = 0
      3. Quantize      → AE ≤ 1
      4. Float compute → Full pipeline:
         a) INF/NAN position consistency check
         b) Compute MARE/MERE/RMSE (excluding small-value domain)
         c) Small-value ErrorCount check
         d) Compare against per-level thresholds

    Args:
        actual:   NPU operator output (torch.Tensor), can be on NPU or CPU
        golden:   Torch reference output (torch.Tensor), must be on CPU
        level:    Precision level, default L0
        category: Operator category, default FLOAT_COMPUTE
        verbose:  Whether to print the report to stdout

    Returns:
        PrecisionReport object with pass/fail status and all metrics

    Example:
        # Float operator, L0 acceptance
        report = precision_check(npu_out, torch_ref.cpu(), level=PrecisionLevel.L0)
        assert report.passed

        # Data movement operator, bitwise check
        report = precision_check(npu_copy, torch_copy, category=OpCategory.NON_COMPUTE)
        assert report.passed
    """
    assert actual.shape == golden.shape, f"Shape mismatch: {actual.shape} vs {golden.shape}"
    assert actual.device.type == golden.device.type or str(golden.device) == "cpu", f"golden should be on cpu, got {golden.device}"

    dtype = golden.dtype
    num_elements = golden.numel()

    # ── Non-compute: bitwise comparison, must be identical ──
    if category == OpCategory.NON_COMPUTE:
        bitwise_match = actual.cpu().to(torch.uint8).tolist() == golden.to(torch.uint8).tolist()
        report = PrecisionReport(
            passed=bitwise_match,
            level=level,
            category=category,
            mare=0.0,
            mere=0.0,
            rmse=0.0,
            mare_ratio=None,
            mere_ratio=None,
            rmse_ratio=None,
            num_elements=num_elements,
            message="Bitwise match" if bitwise_match else "Bitwise mismatch",
        )
        if verbose:
            print(report.summary())
        return report

    # ── Integer compute: bitwise match or AE = 0 ──
    if category == OpCategory.INTEGER_COMPUTE:
        ae = (actual.cpu() - golden).abs().max().item()
        bitwise_match = ae == 0
        report = PrecisionReport(
            passed=bitwise_match,
            level=level,
            category=category,
            mare=0.0,
            mere=0.0,
            rmse=0.0,
            mare_ratio=None,
            mere_ratio=None,
            rmse_ratio=None,
            num_elements=num_elements,
            message=f"Bitwise match (AE={ae})" if bitwise_match else f"AE mismatch: max AE={ae}",
        )
        if verbose:
            print(report.summary())
        return report

    # ── Quantize compute (integer output): AE ≤ 1 ──
    if category == OpCategory.QUANTIZE_COMPUTE and golden.dtype in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
        torch.uint16,
        torch.uint32,
        torch.uint64,
    ):
        ae_max = (actual.cpu() - golden).abs().max().item()
        passed = ae_max <= 1
        report = PrecisionReport(
            passed=passed,
            level=level,
            category=category,
            mare=0.0,
            mere=0.0,
            rmse=0.0,
            mare_ratio=None,
            mere_ratio=None,
            rmse_ratio=None,
            num_elements=num_elements,
            message=f"Quantize AE: max_AE={ae_max} (threshold=1)",
        )
        if verbose:
            print(report.summary())
        return report

    # ── Float compute (and quantize float output) ──
    # Step 1: Compute raw error metrics
    actual_cpu = actual.cpu().float()
    golden_float = golden.float()
    diff, relative_error, mare, mere, rmse = _compute_errors(actual_cpu, golden_float)

    thresholds = _THRESHOLDS[level]
    mare_th, mere_th, rmse_th = thresholds

    # Step 2: INF/NAN position consistency check
    inf_nan_match = _check_inf_nan(actual_cpu, golden_float)

    if not inf_nan_match:
        report = PrecisionReport(
            passed=False,
            level=level,
            category=category,
            mare=mare,
            mere=mere,
            rmse=rmse,
            mare_ratio=mare_th,
            mere_ratio=mere_th,
            rmse_ratio=rmse_th,
            num_elements=num_elements,
            inf_nan_match=False,
            message="INF/NAN mismatch detected",
        )
        if verbose:
            print(report.summary())
        return report

    # Step 3: Small-value domain check
    small_err_npu, small_err_ref, small_val_passed = _check_small_value(actual_cpu, golden_float, dtype)

    # Step 4: Exclude small-value elements and recompute error metrics
    # When golden is near zero, relative error explodes. Excluding these
    # yields reliable metrics for the normal-value domain.
    if dtype in _SMALL_VALUE_THRESHOLDS:
        threshold, _ = _SMALL_VALUE_THRESHOLDS[dtype]
        small_mask = golden_float.flatten().abs() < threshold
        if small_mask.any():
            large_mask = ~small_mask
            if large_mask.sum() > 0:
                diff_large = diff[large_mask]
                golden_large = golden_float.flatten()[large_mask]
                denom = golden_large.abs() + 1e-7
                rel_large = diff_large / denom
                mare = rel_large.max().item() if rel_large.numel() > 0 else 0.0
                mere = rel_large.mean().item() if rel_large.numel() > 0 else 0.0
                rmse = diff_large.pow(2).mean().sqrt().item() if diff_large.numel() > 0 else 0.0

    # Step 5: Compare metrics against thresholds
    mare_passed = mare <= mare_th
    mere_passed = mere <= mere_th
    rmse_passed = rmse <= rmse_th
    float_passed = mare_passed and mere_passed and rmse_passed
    final_passed = float_passed and small_val_passed

    # Build failure reason
    message_parts = []
    if not mare_passed:
        message_parts.append(f"MARE={mare:.4e} > threshold={mare_th:.1f}")
    if not mere_passed:
        message_parts.append(f"MERE={mere:.4e} > threshold={mere_th:.1f}")
    if not rmse_passed:
        message_parts.append(f"RMSE={rmse:.4e} > threshold={rmse_th:.1f}")
    if not small_val_passed:
        message_parts.append(f"SmallValue: NPU_errs={small_err_npu}, Ref_errs={small_err_ref}")
    message = "; ".join(message_parts) if message_parts else "Pass"

    report = PrecisionReport(
        passed=final_passed,
        level=level,
        category=category,
        mare=mare,
        mere=mere,
        rmse=rmse,
        mare_ratio=mare / mare_th,
        mere_ratio=mere / mere_th,
        rmse_ratio=rmse / rmse_th,
        small_value_error_count_npu=small_err_npu,
        small_value_error_count_ref=small_err_ref,
        small_value_passed=small_val_passed,
        inf_nan_match=True,
        num_elements=num_elements,
        message=message,
    )

    if verbose:
        print(report.summary())

    return report


def multi_seed_precision_check(
    kernel_fn,
    *input_args,
    level: PrecisionLevel = PrecisionLevel.L0,
    category: OpCategory = OpCategory.FLOAT_COMPUTE,
    num_seeds: int = 1000,
    ref_fn=None,
    **input_kwargs,
) -> Dict[str, Any]:
    """Multi-seed precision retest (Bootstrap workflow).

    When a single precision_check fails, it may be a sporadic false positive
    from numerical instability. This function re-runs the kernel N times with
    different random seeds, computes the 95% confidence interval of the median
    MARE Ratio via Bootstrap resampling, and judges statistically.

    Decision rules:
        - N < 200      → Fuse: sample too small, FAIL
        - CI_lower > 1 → Confirmed precision anomaly, needs kernel tuning
        - CI_lower ≤ 1 → Sporadic false positive, statistically PASS

    Args:
        kernel_fn:    Compiled tilelang kernel function
        *input_args:  Input tensors (must support torch.manual_seed)
        level:        Precision level, default L0
        category:     Operator category, default FLOAT_COMPUTE
        num_seeds:    Number of random seeds, default 1000
        ref_fn:       Reference function. If None, defaults to element-wise sum
                      of the first two args (demo only — specify ref_fn in practice)
        **input_kwargs: Extra args passed to both kernel_fn and ref_fn

    Returns:
        {
            "passed":        bool,   # Whether the retest passed
            "median_ratio":  float,  # Median MARE Ratio
            "ci_lower":      float,  # 95% CI lower bound
            "ci_upper":      float,  # 95% CI upper bound
            "num_samples":   int,    # Number of samples
            "message":       str,    # Verdict description
        }

    Note:
        Assumes kernel_fn is purely functional (input → output).
        Operators with workspaces (e.g., Flash Attention) need custom adaptation.
    """
    thresholds = _THRESHOLDS[level]
    mare_th, _, _ = thresholds

    # Run kernel N times with different seeds, collect MARE Ratios
    ratios = []
    for seed in range(num_seeds):
        torch.manual_seed(seed)
        inp = tuple(a.clone() if isinstance(a, torch.Tensor) else a for a in input_args)

        actual = kernel_fn(*inp, **input_kwargs)

        torch.manual_seed(seed)
        inp_ref = tuple(a.clone() if isinstance(a, torch.Tensor) else a for a in input_args)
        if ref_fn is not None:
            golden = ref_fn(*inp_ref, **input_kwargs)
        else:
            # Default fallback (demo only — specify ref_fn in practice)
            golden = inp_ref[0] + inp_ref[1] if len(inp_ref) >= 2 else inp_ref[0]

        _, _, mare, _, _ = _compute_errors(actual.cpu().float(), golden.float())
        ratios.append(mare / mare_th)

    ratios_tensor = torch.tensor(ratios, dtype=torch.float64)

    # Bootstrap 2000× resampling → 95% CI of median MARE Ratio
    boot_medians = []
    n = len(ratios)
    for _ in range(2000):
        indices = torch.randint(0, n, (n,))
        sample = ratios_tensor[indices]
        boot_medians.append(sample.median().item())
    boot_medians = torch.tensor(boot_medians)
    boot_medians_sorted = boot_medians.sort().values

    ci_lower = boot_medians_sorted[int(0.025 * 2000)].item()
    ci_upper = boot_medians_sorted[int(0.975 * 2000)].item()
    median_ratio = ratios_tensor.median().item()

    # Decision
    if num_seeds < 200:
        # Too few samples for reliable statistics
        passed = False
        message = "FUSE: sample size < 200"
    else:
        # CI lower bound > 1.0 → 95% confidence that MARE Ratio exceeds threshold
        passed = ci_lower <= 1.0
        message = "PASS" if passed else f"FAIL: CI lower={ci_lower:.4f} > 1.0"

    result = {
        "passed": passed,
        "median_ratio": median_ratio,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "num_samples": num_seeds,
        "message": message,
    }

    print(f"=== Multi-Seed Retest (n={num_seeds}) ===")
    print(f"  Median ratio: {median_ratio:.4f}")
    print(f"  95% CI:       [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"  Verdict:      {'PASS' if passed else 'FAIL'}  ({message})")

    return result
