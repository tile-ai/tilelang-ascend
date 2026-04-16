---
name: tilelang-precision-debug-skill
description: TileLang 算子精度比对与误差分析技能。当用户提及“精度错误”、“结果不一致”、“比对失败”、“ASCII diff map”或需要深入定位数值差异时必须使用本技能。
---

# TileLang Precision Debug Skill

## Mandatory routing rule

Before answering, follow AGENTS.md section "Docs Auto Routing Rules (Mandatory)".

## Trigger Guidance

- When users face accuracy issues, mismatch with reference implementation (e.g., PyTorch), or precision errors.
- When users mention `assert_close` failure, Top-10 errors, relative error too large, or need to see the ASCII diff map.
- When a `.precision_debug` directory is mentioned.

## Instructions SOP

1. **Replace Comparison API:**
   Ensure `assert_close` is imported from `testcommon` (standard in `testing/npuir`) or `tilelang.utils.prec_assert_close`.

2. **Activate Debug Reporting:**
   The detailed reporting is **disabled by default** to avoid clutter. You can activate it in two ways:
   - **Surgical (recommended):** Add `@pytest.mark.precision_debug` to the test function.
     ```python
     import pytest
     from testcommon import assert_close

     @pytest.mark.precision_debug
     def test_my_op():
         ...
           assert_close(actual, expected, dtype=dtype)
       ```
   - **Global:** Set the environment variable `TL_PREC_DEBUG=1` before running `pytest`.
       ```bash
       TL_PREC_DEBUG=1 pytest testing/npuir/test_xxx.py
       ```

2. **Run and Collect:**
   Execute the test script. If there is a mismatch, the tool will automatically generate a `.precision_debug/<test_name>_<timestamp>/` directory containing `report.txt`, `diff_map.txt`, and the serialized tensors.

3. **Analyze the Precision Debug Report:**
   - **`report.txt`:** Check the mismatch ratio, `Max abs/rel diff`, and the `Top-10 largest differences`. Determine if the error is widespread (logic error) or isolated (boundary, overflow).
   - **`diff_map.txt` (ASCII Map):** Identify spatial distribution patterns:
     - **Blocky distribution:** Incorrect tiling or block_M/N parameters.
     - **Periodic stripes:** Incorrect memory layout, strides, or vectorization broadcast issues.
     - **Edge/Boundary spikes:** Padding, masking, or loop boundary conditions not handled correctly.

4. **Root Cause Localization:**
   Use the pattern analysis to trace back to Load (memory alignment), Compute (accumulation precision, type casting), or Store (write-back boundary) stages in the TVM IR or MLIR.

## Related skills

- tilelang-debug-helper
- tilelang-error-fixer
