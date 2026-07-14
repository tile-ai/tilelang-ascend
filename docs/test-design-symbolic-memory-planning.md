# Test Design: Symbolic Address/Size in AscendMemoryPlanning

## Reference

- Issue: [#1319](https://github.com/tile-ai/tilelang-ascend/issues/1319)
- Design: [design-symbolic-memory-planning.md](design-symbolic-memory-planning.md)

## Test Categories

### L0 — Crash Regression (must-pass gate)

| ID | Name | Description |
|----|------|-------------|
| L0-1 | `test_symbolic_buffer_size_linear_mode_no_crash` | `T.alloc_ub([symbolic_var], dtype)` in linear mode must not crash |
| L0-2 | `test_symbolic_address_expr_linear_mode_no_crash` | `T.annotate_address({buf: symbolic_expr})` in linear mode must not crash |
| L0-3 | `test_symbolic_buffer_size_with_symbolic_address_no_crash` | Both symbolic size AND symbolic address together in linear mode |

### L1 — Correctness (codegen verification)

| ID | Name | Description |
|----|------|-------------|
| L1-1 | `test_symbolic_address_preserved_in_codegen` | Symbolic address expression appears in generated source |
| L1-2 | `test_constant_address_still_works_linear` | Constant `annotate_address` values still produce correct offsets in linear mode |
| L1-3 | `test_constant_address_still_works_auto` | Constant `annotate_address` values still produce correct offsets in auto mode (regression) |
| L1-4 | `test_symbolic_size_buffer_gets_address` | Symbolic-size buffer without annotate_address gets a valid (symbolic) address |

### L2 — Auto-mode boundary

| ID | Name | Description |
|----|------|-------------|
| L2-1 | `test_auto_mode_symbolic_address_assigned_directly` | Auto mode: symbolic pre-alloc address is assigned directly, constant buffers go through allocator |
| L2-2 | `test_auto_mode_symbolic_size_clear_error` | Auto mode: symbolic buffer size produces ICHECK error (not crash) with helpful message |

### L3 — NPU correctness (if NPU available)

| ID | Name | Description |
|----|------|-------------|
| L3-1 | `test_npu_symbolic_buffer_size_correctness` | End-to-end: symbolic-size buffer produces correct results on NPU |

## Verification Strategy

1. **Crash regression**: Compile programs that previously caused SEGFAULT. Success = no crash.
2. **Codegen verification**: Parse generated source to check symbolic expressions appear in address positions.
3. **Regression**: Existing tests in `test_ascend_memory_planning.py` must all still pass.
4. **NPU correctness**: Run on actual NPU if available, compare against PyTorch reference.

## Key Design Decisions

- **Linear mode only for symbolic sizes**: Auto-plan mode (LinearScanAllocator) requires constant sizes for free-block management. Symbolic sizes in auto mode produce a clear ICHECK error guiding users to linear mode.
- **Symbolic addresses in auto mode**: Pre-alloc buffers with symbolic addresses are assigned directly to `address_map_`, bypassing the allocator entirely. Constant buffers still go through normal allocation.
- **No overflow check for symbolic**: The `check_overflow` flag (already `false` by default) is removed from the linear path since symbolic expressions can't be compared against constant memory limits at compile time.
