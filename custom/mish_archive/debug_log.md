# Mish Operator Debug Log

## Attempt 2 — 2026-08-05T00:00:00Z
- mode: precision_fix
- classification: precision_pass
- fail_category: none
- test_level: all
- coverage: L0:8 L1:15 L2:2 Boundary:4
- boundary_warnings: l2 illegal_shape_3d: illegal input not rejected ([BOUNDARY_WARN], non-blocking)
- changes:
  - `custom/mish/mish.py`: Rewrote kernel to use float32 intermediate compute buffers (ACC_DTYPE="float32").
    - All 5 compute UB buffers (a_ub, t0_ub, t1_ub, one_ub, b_ub) allocated as float32.
    - Added tmp_orig buffer (original dtype) for GM↔UB dtype bridging.
    - Cast-in: T.copy(GM→tmp_orig) + T.tile.cast(tmp_orig→a_ub_fp32, "CAST_NONE") for non-fp32.
    - Cast-out: T.tile.cast(b_ub_fp32→tmp_orig, "CAST_RINT") + T.copy(tmp_orig→GM) for non-fp32.
    - Float32 input skips cast (direct T.copy GM↔UB).
    - Fix addresses both: (a) bf16 CANN intrinsic unsupported (Muls/Maxs/Exp/Adds/Div don't support __bf16), (b) fp16 12-step accumulated precision loss.
  - `custom/mish/test_mish.py`: Expanded L1/L2/Boundary stubs (scenario B):
    - L1: 15 cases covering D-SHAPE-{EDGE,TAIL-1,TAIL-MID,PRIME}, D-VALRANGE-{M,L,ASYM}, multi-dtype.
    - L2: 2 cases (D-EXC-DTYPE int8, D-EXC-SHAPE 3D tensor).
    - Boundary: 4 cases (D-SPECIAL-{INF,NAN,ZERO,DBOUND}).
    - Updated COVERAGE_MANIFEST to reflect expanded coverage.
  - `custom/mish/history_version/mish_impl_s2_attempt1.py`: Backup of attempt 1 kernel (pre-fix).
- error_summary: none (all L0/L1 pass, exit 0)
- design_error_reason: none
- rollback: no
- backup_path: custom/mish/history_version/mish_impl_s2_attempt1.py
- instrumentation_cleaned: n/a (no debug instrumentation used)
- next_hint: none — precision passed, ready for Stage 3 (perf tuning) if user requests.

## Attempt 1 — 2026-08-04T00:00:00Z (historical, from Stage 2 first_impl)
- mode: first_impl
- classification: precision_fail
- fail_category: compile (bf16) + precision (fp16)
- test_level: l0
- coverage: L0:8 (4 PASS, 4 FAIL)
- boundary_warnings: none (L0 only)
- changes: Initial implementation — 12-step mish kernel with original-dtype UB buffers.
- error_summary:
  - bf16: 5 CANN compile errors (Muls/Maxs/Exp/Adds/Div don't support __bf16).
  - fp16_basic: matched_ratio=0.8187, max_abs=1.221e-3 (atol=6.1e-5 exceeded 20×).
  - fp16_mid: matched_ratio=0.5449, max_abs=4.639e-3 (atol=6.1e-5 exceeded 75×).
- design_error_reason: none (fixable in implementation layer by using fp32 intermediate)
- rollback: no
- backup_path: n/a
- instrumentation_cleaned: n/a
- next_hint: Use float32 intermediate buffers for all dtypes; cast at GM boundary with T.tile.cast.
