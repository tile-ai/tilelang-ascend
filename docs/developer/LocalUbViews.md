# Static contiguous local UB views

The NPUIR bridge preserves the shape, element type and element offset of a
Buffer alias sharing a local UB allocation. Sharing a data Var identifies the
allocation; it does not make every Buffer descriptor identical.

Same-type nonidentity views use `memref.reinterpret_cast`. Allocations accessed
through multiple element types use one byte allocation, with typed
`memref.view` operations deriving their addresses from it. No copy or second
allocation is introduced. Single-type identity views retain their existing
representation.

Supported nonidentity views have positive static dimensions, scalar byte-sized
element types, a nonnegative static element offset, contiguous strides, and a
byte extent within the allocation. Unsupported dynamic or noncontiguous local
views are diagnosed. Typed aliases of external or non-UB storage are outside
this implementation.

For example, a local F32 allocation of shape `[8, 64]` has 2048 bytes. An F16
view of shape `[8, 64]` at element offset 16 starts at byte 32 and occupies 1024
bytes. Its allocation identity is retained through both view operations. An
F16 view of shape `[8, 129]` exceeds this allocation and is rejected.

Views reinterpret storage. They do not convert numeric values or establish that
an arbitrary overlapping numeric conversion or asynchronous schedule is safe.
The author must preserve the operation's overlap constraints and explicit
synchronization.

This bridge support is separate from the Python `T.view` helper: constructing a
view of an already-offset Buffer also requires the helper to preserve its
first-element byte address.

For N TIR nodes, function-level type discovery visits N nodes once. A view
materialization examines R dimensions/strides, using O(R) temporary metadata and
emitting O(R) shape/stride metadata. This is a correctness change, with no claim
of kernel speedup or general alias-equivalence proof.

Regression tests are in `testing/npuir/memory_ops/test_local_typed_views.py`.
They use explicit Expert synchronization and check exact bytes for reinterpret
cases, plus bounds/stride rejection and a grouped reduction. See the PR's
validation record for the tested versions and device configuration.
