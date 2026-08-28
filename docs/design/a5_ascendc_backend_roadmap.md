# A5 AscendC backend roadmap

Status: architecture decision and implementation roadmap

Scope: `target="ascendc"` on the A5 platform

Non-goal: changing or removing the independent PTO backend

## Decision

The product route for A5 is the **AscendC backend**. A successful product
artifact must retain the generated A5 AscendC source, the exact compile
command, and the resulting loadable binary as one bound generation.

PTO remains a separate backend and a useful implementation/reference route.
Running PTO-generated CCE source on A5 does not satisfy an A5 AscendC-source
acceptance gate. PTO can replace the product route only after an independently
replayable conversion proves all of the following:

1. PTO output is translated into readable A5 AscendC without hand rewriting;
2. the translation preserves host launch, tiling, ABI, precision, and
   performance semantics across the declared denominator;
3. a known-bad translation is rejected by the same consumer; and
4. the translation is maintained as an explicit compiler stage rather than an
   undocumented fallback.

No such PTO-to-AscendC stage exists in the current lowering pipeline.

## Current pipeline and exact gaps

TileLang's Python DSL lowers through TVM TIR. Device code generation then
selects one of two sibling targets:

- `target="ascendc"` calls `target.build.tilelang_ascend`;
- `target="pto"` calls `target.build.tilelang_ascend_pto`.

See [`tilelang/engine/lower.py`](../../tilelang/engine/lower.py). The two
branches emit different C++ dialects and are compiled independently; PTO is
not an intermediate representation consumed by the AscendC code generator.

The repository contains partial A5 awareness in the AscendC route: platform
detection recognizes A5 and the AscendC code generator has A5 memory-capacity
constants. That is not yet an end-to-end A5 backend because:

- [`tilelang/jit/adapter/libgen.py`](../../tilelang/jit/adapter/libgen.py)
  currently compiles every `ascendc` source with `dav-2201` and `-xasc`, while
  only the PTO branch selects the A5-specific `dav-c310`/register-memory mode;
- [`src/tl_templates/ascend/common.h`](../../src/tl_templates/ascend/common.h)
  unconditionally includes Catlass and binds `ArchTag` to `AtlasA2`;
- [`tilelang/carver/arch/ascend.py`](../../tilelang/carver/arch/ascend.py)
  has no A5 chip profile or A5 detection branch; and
- [`src/transform/ascend_host.cc`](../../src/transform/ascend_host.cc)
  publishes `tiling_map`, but its current visitors do not populate a real
  symbolic tiling mapping. Host/tiling correctness therefore requires a
  producer-side proof, not only a code-generation consumer change.

These facts make “compiler-generated code runs on A5” strictly weaker than
“the compiler emits and compiles A5 AscendC source.” The latter is the product
contract.

## Implementation phases

### Phase 0 — direct A5 AscendC PoC

Use a small static-shape vector kernel that does not need Catlass/Cube. The
real consumer must prove, from a fresh build:

1. `target="ascendc"` remains selected with no PTO fallback;
2. the emitted `.cpp` contains AscendC kernel code and is retained;
3. the A5 compile recipe is selected from platform identity, with the exact
   supported compiler flags recorded rather than inferred from PTO flags;
4. the binary loads and launches on the declared A5 target;
5. precision passes against the bound reference; and
6. a known-bad source mutation fails the same end-to-end gate.

The first implementation surfaces are platform-to-compiler-option resolution
in `libgen.py`, explicit target/platform validation, and focused tests that
prove A3 and PTO behavior do not drift.

### Phase 1 — dynamic host/tiling closure

Bind symbolic inputs to a typed, deterministic tiling contract. The producer,
generated host wrapper, kernel arguments, and launch receipt must agree on the
same ordered fields. Missing, extra, stale, or reordered fields fail before
compilation or launch. The consumer must include static and dynamic positive
cases plus a known-bad field-order/type mutation.

Some planning discussions call this stage “ISOC”. The repository does not
currently define that term. Until a concrete definition and acceptance test
are committed, “ISOC” is a label only and must not be used as a completion
claim.

### Phase 2 — A5 Cube/Catlass interface

After the direct vector route and host/tiling contract pass, remove the
unconditional `AtlasA2` assumption and introduce an explicit A5 architecture
interface for Cube kernels. Audit Catlass headers, layouts, memory capacities,
and compiler flags against the actual A5 toolchain before enabling this path.
Keep vector-only A5 kernels independent of Catlass so an unavailable Cube
interface does not block the Phase-0 product route.

The dependency present in this repository is named **Catlass**. “Catalyst” has
no defined compiler surface here; it must be treated as an unresolved naming
question rather than silently substituted for Catlass.

## Acceptance and fail-loud rules

An A5 AscendC milestone is accepted only when one content-addressed receipt
binds:

- TileLang source and lowered TIR;
- generated host and device AscendC source;
- target/platform identity and complete compiler command;
- toolchain and system-header identity;
- binary identity and load/launch identity;
- exact inputs/reference/precision results; and
- the end-to-end known-bad result.

Compilation success, PTO execution, A5 platform detection, source inspection,
or a single helper/unit test is diagnostic evidence, not an A5 AscendC product
result. Unknown platforms, unsupported compiler flags, missing A5 architecture
profiles, and unbound tiling fields must fail loudly; none may fall back to A3
or PTO.

## Evidence that can change this decision

The primary route may be reconsidered only with a fresh, independent consumer
of an explicit PTO-to-AscendC compiler stage satisfying the four conversion
criteria in the Decision section. Performance or feature coverage of PTO by
itself is not evidence for that change, because it does not produce the
required artifact type.
