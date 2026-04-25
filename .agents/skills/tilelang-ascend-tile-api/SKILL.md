---
name: tilelang-ascend-tile-api
description: End-to-end workflow for adding a small Ascend-specific T.tile.xxx API in tilelang/language/ascend_tile.py and making it directly usable. Use this skill whenever the user asks to add, wrap, expose, implement, or test a new Ascend tile primitive or small T.tile API, especially when the change must connect Python frontend, C++ lowering/codegen, Ascend C helpers, docs, and CI-facing tests.
---

# TileLang Ascend Tile API Workflow

Use this skill when the task is to add a small user-facing API under `T.tile.xxx`, especially one implemented from `tilelang/language/ascend_tile.py` and backed by Ascend C codegen.

The goal is not just to add a Python function name. The goal is a usable API that compiles, lowers, generates valid Ascend C, has a stable semantic boundary, and has an appropriate CI-facing test.

## First Pass

Start by reading the current repo shape before choosing an implementation:

1. Read `AGENTS.md`.
2. Read `.agents/skills/tilelang-custom-skill/tilelang-api-best-practices/SKILL.md`.
3. Read `.agents/skills/tilelang-custom-skill/tilelang-api-best-practices/references/api-compute.md`.
4. If mode or `pass_configs` are relevant, read `.agents/skills/tilelang-custom-skill/tilelang-expert-to-developer/SKILL.md`.
5. Inspect the closest existing APIs in `tilelang/language/ascend_tile.py`.
6. Inspect comparable tests in `testing/python/language/`.
7. Inspect comparable lowering and codegen in `src/op/ascend.{h,cc}`, `src/target/codegen_ascend.cc`, and `src/tl_templates/ascend/common.h`.

Do not infer API signatures from memory. Existing local patterns win over a clever new abstraction.

## Scope Decision

Before editing, decide and state the API boundary:

- User API name, for example `T.tile.foo(dst, src, ...)`.
- Whether it is pure tile compute, data movement, reduction-like behavior, or a side-effect writeback.
- Supported buffer scopes, usually GM, UB/shared, L1, L0, or a subset.
- Supported dtypes and ranks.
- Whether `Buffer`, `BufferLoad`, and `BufferRegion` are accepted.
- Unsupported arguments and semantics.
- Whether it should work in Developer pass_configs, Expert mode, or both.

Prefer an Ascend-specific `T.tile.xxx` API when the semantics do not match the main TileLang/GPU global API. Do not add or change global `T.xxx` APIs unless the request explicitly requires it and the semantics truly align.

## Implementation Path

### 1. Python Frontend

Add the user entry in `tilelang/language/ascend_tile.py`.

Use local helper patterns already in that file and in `tilelang/language/copy.py`:

- Resolve let-bound values if existing APIs do so.
- Accept `Buffer`, `BufferLoad`, or `BufferRegion` only when meaningful.
- Convert frontend inputs into `tl.region` calls when the C++ lowering needs region/rank/extent information.
- Validate unsupported scope or argument combinations early with clear error messages.
- Emit a clearly named op, usually `tl.ascend_<api_name>`, when the operation needs C++ lowering.

Avoid reusing `tl.ascend_copy` for different semantics unless the operation really is a normal copy. Side-effecting or mode-changing operations should usually have an explicit op name.

### 2. C++ Operator And Lowering

If the frontend emits a new `tl.ascend_*` op, add the corresponding C++ operator path:

- Declare it in `src/op/ascend.h`.
- Implement parsing and lowering in `src/op/ascend.cc`.
- Register it with `TIR_REGISTER_TL_OP` when it needs tile-op lowering.
- Preserve region information until lowering can compute access pointers, extents, strides, masks, or valid shapes.
- Re-check critical constraints in C++ even if Python already checks them.

Lower to a backend-recognizable `call_extern` or existing codegen pattern. Keep the lowered call name stable and descriptive, such as `tl::ascend::<helper_name><...>`.

### 3. Ascend C Helper

Put reusable Ascend C snippets in `src/tl_templates/ascend/common.h` unless there is already a better local template home.

Keep hardware state changes inside helper functions when possible. For example, if a helper enables a mode, it should also restore or disable that mode before returning.

If CANN version compatibility is needed:

- Search local or target Ascend C headers for real version macros.
- Prefer official macros such as `CANN_MAJOR` only after confirming they exist in the environment or project convention.
- Add a small compatibility helper instead of scattering `#if` branches throughout codegen.
- Document fallback behavior in comments only when it prevents future confusion.

### 4. Codegen And Pipeline Integration

Wire the lowered call through the backend:

- Update `src/target/codegen_ascend.cc` to print the helper call.
- Reuse existing helpers such as `CopyCodegen` only when argument order and pointer printing match exactly.
- If the operation reads/writes GM, update scheduling or pipeline metadata as needed:
  - `src/transform/common/operation_config.h`
  - `src/transform/ascend_combinecv.cc`
  - `src/transform/cross_core_pipeline.cc`
- Treat side-effect GM writes as writes in pipeline analysis.

Do not add PTO support by default. Add it only when the task requires it or an existing PTO path makes the change small and low risk.

## Test Placement

Do not automatically create a new standalone test file. First choose the narrowest existing home that matches the API's behavior.

Use this placement guide:

- Pure elementwise `T.tile.xxx` math APIs: prefer `testing/python/language/test_tilelang_ascend_language_elementwise.py`.
- Compare/select APIs: prefer the existing compare/select themed file when present.
- Cast/copy-like behavior: prefer the existing cast/copy themed file.
- Parallel lowering behavior: use a parallel test file only when the API primarily tests `T.Parallel` or auto-copy behavior.
- New side-effect writeback, new memory pipeline category, or API with no good thematic owner: create a focused standalone file.

Standalone files are acceptable when they reduce cognitive load, but they should be rare and clearly justified. Remember that `examples/bench_test.sh` runs `testing/python/`, so every new file becomes part of CI.

## CI-Facing Test Style

Default to CI-facing correctness tests, not development-stage TDD tests.

Good default test shape:

- Compile and run the kernel on NPU when available.
- Use `pytest.mark.skipif` when `torch.npu` is unavailable.
- Compare against PyTorch or a simple reference with `torch.testing.assert_close`.
- Cover the smallest set of dtype/rank cases that protect the API contract.
- Use Developer or mixed-mode pass configs when the API supports them:

```python
PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}
```

Avoid permanent CI tests that assert generated source strings, helper names, or negative cases unless those are the main public contract. Such checks are useful during development but often become noisy after the implementation stabilizes.

For side-effect writeback APIs, tests should initialize the destination explicitly before running the kernel. For example, an accumulation API should zero GM before checking the accumulated value.

## Documentation Updates

For a user-facing API, update the docs that future users and agents will actually read:

- `docs/language_ref/tilelibrary.md` for the short language reference.
- `docs/TileLang-Ascend Programming Guide.md` for the detailed guide.
- `.agents/skills/tilelang-custom-skill/tilelang-api-best-practices/references/api-compute.md` for agent-facing API usage.
- `.agents/skills/tilelang-custom-skill/tilelang-expert-to-developer/SKILL.md` only if mode guidance changes.
- Avoid updating broad docs just to mention an internal helper.

If an older doc contains a similar global API with different semantics, add a short warning instead of silently rewriting examples that may belong to a GPU/mainline tutorial.

## Verification Checklist

Run the strongest local verification available:

```bash
python -m py_compile <changed-python-files>
conda run -n tilelang_dev ruff check <changed-python-files>
conda run -n tilelang_dev ruff format --check <changed-python-files>
git diff --check
```

When C++ files change, also run:

```bash
conda run -n tilelang_dev clang-format --dry-run --Werror <changed-cpp-files>
```

When a runnable environment is available, run the targeted pytest:

```bash
pytest -q <selected-test-file-or-test-node>
```

If pytest, TVM, CANN, or NPU runtime is unavailable locally, say exactly which layer could not be verified and whether a server-side run is needed.

## Optional Agent Split

Only split work across agents when the user explicitly authorizes parallel agent work.

Good independent slices:

- Frontend API and IR shape in `tilelang/language/ascend_tile.py`.
- C++ op and lowering in `src/op/ascend.{h,cc}`.
- Ascend C helper in `src/tl_templates/ascend/common.h`.
- Codegen and pipeline metadata in `src/target` and `src/transform`.
- Tests and docs.
- Integration verification.

Give each agent a disjoint write scope and tell it not to revert other agents' work.

## Final Response

When done, summarize:

- The new user API and exact supported boundary.
- The frontend, lowering, codegen/helper, and test files touched.
- Where the test was placed and why.
- Which checks passed and which could not run locally.
- Any known unsupported semantics.
