# Design: Support Symbolic Address/Size in AscendMemoryPlanning + Codegen

## Issue

[#1319](https://github.com/tile-ai/tilelang-ascend/issues/1319) — `T.alloc_ub([max_segs], "int32")` where `max_segs = T.symbolic("max_segs")`, combined with `T.annotate_address({buf: symbolic_expr})`, causes Segmentation fault in `AscendMemoryPlanning::AscendMemoryPlanner::SetPreAllocBuffer`. Even after fixing the crash, codegen fails because `copy_gm_to_ub<T, dstN>(...)` uses `dstN` as a C++ template parameter requiring a compile-time constant.

## Root Cause

1. **Direct crash (null deref)**: `SetPreAllocBuffer` (line 282) called `kv.second.as<IntImmNode>()->value` without null check. When the address expression contains symbolic variables (e.g. `max_segs`), `.as<IntImmNode>()` returns `nullptr`, causing SEGFAULT.

2. **Architectural limitation**: `pre_alloc_buffer_`, `address_map_`, `buffer_sizes_` were all typed as `int64_t`/`size_t`, assuming compile-time constant addresses and sizes. The entire memory planning pass could not handle symbolic expressions.

3. **Config not skipping pass**: `TL_ASCEND_MEMORY_PLANNING: False` only switches the allocation algorithm (auto-reuse vs linear), it does NOT skip the pass. `SetPreAllocBuffer` is always called when `T.annotate_address` is used.
4. **Codegen template parameter**: `copy_gm_to_ub<T, dstN, dstM>` uses `dstN`/`dstM` as C++ non-type template parameters. When the buffer has symbolic size, `src/op/ascend.cc:210` streams the symbolic Var name into the template position, producing uncompilable C++ like `copy_gm_to_ub<int, max_segs>(...)`.
5. **Buffer allocation size**: `codegen_ascend.cc:811` uses `op->ConstantAllocationSize()` which returns 0 for symbolic extents, producing `GetWithOffset<T>(0, addr)`.

## Design Decision

**Approach B: Full symbolic support** — Change all internal data structures from `int64_t` to `PrimExpr`, enabling symbolic address and size expressions to flow through the entire memory planning pipeline.

### Why not Approach A (just add ICHECK)?

Approach A would only convert the crash to an error message. Users with dynamic-size buffers (a legitimate use case for variable-length inputs) would still be blocked.

### Why not Approach C (skip pass when config=False)?

Changing config semantics would break existing users who rely on "False = linear allocation". Also, the pass is needed to compute `address_map` and `size_map` for codegen regardless of the algorithm.

## Changes

### File: `src/transform/ascend_memory_planning.cc`

#### 1. New helper: `AlignUpExpr` (line 56)

PrimExpr version of `AlignUp`, using `floordiv` for TVM compatibility:
```cpp
static PrimExpr AlignUpExpr(PrimExpr value, int64_t alignment) {
  Integer align(alignment);
  Integer mask(alignment - 1);
  return floordiv(value + mask, align) * align;
}
```

#### 2. Data structure changes

| Variable | Before | After |
|----------|--------|-------|
| `address_map_` | `unordered_map<VarNode*, int64_t>` | `unordered_map<VarNode*, PrimExpr>` |
| `buffer_sizes_` | `unordered_map<VarNode*, size_t>` | `unordered_map<VarNode*, PrimExpr>` |
| `pre_alloc_buffer_` | `unordered_map<string, int64_t>` | `unordered_map<string, PrimExpr>` |
| `GetAddressMap()` return | `...<int64_t>` | `...<PrimExpr>` |
| `GetBufferSizes()` return | `...<size_t>` | `...<PrimExpr>` |

#### 3. `SetPreAllocBuffer` (line 282)

**Before** (crashes on symbolic):
```cpp
int64_t addr_offset = kv.second.as<IntImmNode>()->value;
pre_alloc_buffer_[buf->name_hint] = addr_offset;
```

**After** (stores PrimExpr directly):
```cpp
pre_alloc_buffer_[buf->name_hint] = kv.second;
```

#### 4. `CalculateBufferSize` (line 925)

**Before**: Returns `size_t`, requires `extent.as<IntImmNode>()` (crashes on symbolic extent).

**After**: Returns `PrimExpr`, multiplies extents as PrimExpr:
```cpp
PrimExpr size_elements = Integer(1);
for (const auto &extent : alloc->extents) {
  size_elements = size_elements * extent;  // works with symbolic
}
return AlignUpExpr(size_bytes, 32);
```

PTO 4D shape path still requires constant (unchanged, has ICHECK).

#### 5. `PlanMemoryForScope` (auto-plan path)

- Symbolic pre-alloc addresses: assigned directly to `address_map_`, skipped by `LinearScanAllocator` (which requires constants for conflict detection)
- Symbolic buffer sizes: ICHECK with clear error message guiding user to set `TL_ASCEND_MEMORY_PLANNING: False` (linear mode)
- Constant buffers: unchanged path through `LinearScanAllocator`

#### 6. `PlanMemoryForScopeLinear` (linear path)

- `current_offset`, `max_offset` changed from `int64_t` to `PrimExpr`
- `alloc_buffer` lambda uses `AlignUpExpr` instead of `AlignUp`
- Removed `check_overflow` branch (was already `false` by default, can't compare symbolic > constant)
- `max_offset = max(max_offset, ...)` uses TVM's `max(PrimExpr, PrimExpr)`

#### 7. `Substitute` (output)

**Before**: Wrapped results in `Integer(...)` (forced IntImm):
```cpp
address_map_attr.Set(buffer_var, Integer(kv.second));
size_map_attr.Set(buffer_var, Integer(static_cast<int64_t>(kv.second)));
```

**After**: Passes PrimExpr directly:
```cpp
address_map_attr.Set(buffer_var, kv.second);
size_map_attr.Set(buffer_var, kv.second);
```

## Downstream Compatibility

| Consumer | How it uses address/size | Symbolic-safe? |
|----------|-------------------------|----------------|
| `codegen_ascend.cc` | `PrintExpr(target_expr)` | Yes — handles any PrimExpr |
| `codegen_ascend_pto.cc` | `PrintExpr(...)` for address, `as<IntImmNode>()` for UB end tracking (defensive `if`) | Yes — skips tracking if not IntImm |
| `ascend_sync_insert.cc` | `as<IntImmNode>()` with null check, returns -1 if not constant | Yes — gracefully skips symbolic buffers |

All downstream consumers are already symbolic-safe (they either use `PrintExpr` or have null-checked `as<IntImmNode>()` with graceful fallback).

## What NOT Changed

- `LinearScanAllocator` class: unchanged, only processes constant-size buffers
- `LiveInterval` / `Allocation` structs: unchanged (`size_t`), only used by allocator
- `memory_limits_`: unchanged (`int`), hardware limits are always constant
- PTO 4D shape sizing: unchanged, still requires constant (has ICHECK)

## Limitations

1. Auto-plan mode (`TL_ASCEND_MEMORY_PLANNING: True`) does NOT support symbolic buffer sizes — only linear mode does. ICHECK provides clear error message.
2. Overflow check disabled for symbolic sizes (was already disabled by default `check_overflow = false`).
3. PTO 4D physical shapes still require constants (hardware constraint).

## Codegen Changes (Part 2)

### Problem

Even after memory planning passes, codegen fails for symbolic-size buffers:
- `copy_gm_to_ub<T, dstN>(...)` requires `dstN` as a compile-time template parameter
- `GetWithOffset<T>(size, addr)` gets `size=0` from `ConstantAllocationSize()` for symbolic extents

### File: `src/tl_templates/ascend/common.h`

Added runtime-size overloads that don't use `dstN`/`dstM` as template parameters:

```cpp
template <typename T>
CATLASS_DEVICE void
copy_gm_to_ub_dynamic(LocalTensor<T> dstTensor, GlobalTensor<T> srcTensor,
                      uint32_t realSrcN, uint32_t maskShapeM,
                      uint32_t maskShapeN, T padValue = T(0),
                      uint32_t dstN = 0, uint32_t dstM = 1);

template <typename T>
CATLASS_DEVICE void
copy_ub_to_gm_dynamic(GlobalTensor<T> dstTensor, LocalTensor<T> srcTensor,
                      uint32_t realdstN, uint32_t maskShapeM,
                      uint32_t maskShapeN, uint32_t srcN = 0,
                      uint32_t srcM = 1);
```

The `dstN`/`srcN` values are passed as runtime function arguments instead of template parameters. They default to 0 (which only affects padding calculation, not actual copy size).

### File: `src/op/ascend.cc`

When `dst_extents[last]` (or `compute_blocklen`) is not `IntImmNode`, generates `_dynamic` variant:
- `copy_gm_to_ub_dynamic<dtype>` instead of `copy_gm_to_ub<dtype, dstN, dstM>`
- `copy_ub_to_gm_dynamic<dtype>` instead of `copy_ub_to_gm<dtype, srcN, srcM>`

New config flags: `gm2ub_dynamic`, `ub2gm_dynamic`. Extra args include the buffer shape dimensions as runtime parameters.

### File: `src/target/codegen_ascend.cc`

1. **`print_buffer` lambda**: When `ConstantAllocationSize()` returns 0 (symbolic extents), falls back to `PrintExpr(product of extents)` for the size argument.

2. **`kCopyOpExtraArgs` map**: Added `copy_gm_to_ub_dynamic` (6 args) and `copy_ub_to_gm_dynamic` (6 args) entries. The `CopyCodegen` function already matches by substring, so no other changes needed.
