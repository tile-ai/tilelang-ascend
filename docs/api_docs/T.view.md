# T.view / T.reshape

## 1. What these APIs do

`T.view` gives existing bytes a second Buffer description. The returned Buffer uses the same
storage as `src`; it does not allocate memory, move data, or convert values. A write through either
Buffer changes the bytes observed through the other.

`T.reshape(src, shape)` is the dtype-preserving spelling of `T.view(src, shape)`. Although
`T.view` can also omit `dtype`, `T.reshape` makes the intent explicit and prevents a later edit from
silently turning a shape-only operation into a dtype reinterpretation.

## 2. Signatures

```python
def view(
    src: Buffer,
    shape: list[PrimExpr] | tuple[PrimExpr, ...] | None = None,
    dtype: str | DataType | None = None,
) -> Buffer

def reshape(
    src: Buffer,
    shape: list[PrimExpr] | tuple[PrimExpr, ...],
) -> Buffer
```

| Argument | Meaning |
| --- | --- |
| `src` | The complete source Buffer |
| `shape` | New shape; `None` keeps `src.shape` in `T.view` |
| `dtype` | New dtype; `None` keeps `src.dtype` |

Every dimension must be written explicitly. PyTorch-style `-1` inference is not supported.

## 3. Common rules

### 3.1 The complete Buffer is viewed

`src` must be a compact, zero-offset `tir.Buffer`. The current API rejects `BufferRegion`, slices,
explicit strides, axis separators, and non-zero `elem_offset`.

### 3.2 The number of bits is unchanged

The compiler must prove:

```text
product(src.shape) * src.dtype.bits * src.dtype.lanes
    == product(view.shape) * view.dtype.bits * view.dtype.lanes
```

Two unrelated symbolic variables are rejected even if a caller may give them equal values at
runtime, because the compiler cannot prove the equality.

### 3.3 A view is not a numerical cast

```python
words = T.alloc_ub((64,), "int32")
values = T.view(words, (4, 16), "float32")
```

`values` reads each group of 32 bits as `float32`. It does not turn the integer value `1` into the
floating-point value `1.0`. Use `T.tile.cast` or another supported conversion when values, rather
than their bit patterns, must change.

## 4. Memory-specific rules

| Storage | Supported view |
| --- | --- |
| GM (`global`) | Equal-bit complete-buffer shape/dtype view |
| UB (`shared.ub`) | Equal-bit complete-buffer view; PTO has the row-layout restriction below |
| Dynamic `shared` | Allowed only when scope inference resolves it to a compatible storage |
| L1/L0A/L0B/L0C | Equal-bit view that preserves the physical fractal-block grid |

PTO aligns each UB row to 32 bytes. A UB view must either keep the same row boundary, or have
32-byte-aligned source and destination rows with the same padded footprint. For example,
`uint8[2, 17] -> uint8[1, 34]` has equal logical bits but changes the padding, so PTO rejects it.

PTO local dtype-changing views currently require byte-addressable source and destination dtypes.
Packed INT4 storage can still be viewed from GM, but a local packed dtype view requires an explicit
supported unpack/repack path.

### 4.1 Fractal-block-preserving L1/L0 views

L1 and L0 do not always store a logical matrix as ordinary row-major elements. They divide it into
hardware blocks and assign a fixed byte order inside each block. A view is valid when every old
block maps to exactly one new block at the same position.

For example, one row-oriented L1/L0A inner block can be described as either:

```text
fp16: [16, 16] * 2 bytes = 512 bytes
int8: [16, 32] * 1 byte  = 512 bytes
```

Therefore this 4D view is valid: the first two dimensions still select the same blocks, while the
last two dimensions reinterpret bytes inside each block.

```python
a_fp16 = T.alloc_L1((F0, F1, 16, 16), "float16")
a_int8 = T.view(a_fp16, (F0, F1, 16, 32), "int8")
```

An L1 buffer can instead use the opposite inner-block orientation. Its corresponding view is
`fp16[..., 16, 16] -> int8[..., 32, 16]`. `T.view` keeps the existing zN/nZ layout tag; it does
not choose or change it. L0A uses the `16 x C0` inner block, while L0B uses `C0 x 16`. L0C keeps
a fixed 16x16 accumulator block; same-width reinterpretations such as `int32 -> float32` can
preserve that grid.

The compiler rejects a change that alters leading dimensions, splits one block across several
positions, or merges several blocks into one. `T.view` also does not change zN/nZ or another
physical layout tag; such a change requires a real layout conversion.

Creating the view does not guarantee that every operation accepts the new dtype and shape.
`T.copy`, `T.mma`, `T.tile.*`, reductions, and scalar access still enforce their own contracts.

## 5. Examples and API choice

```python
words = T.alloc_ub((64,), "int32")
values = T.view(words, (4, 16), "float32")

T.copy(A_bits, words)
T.tile.add(values, values, 1.0)
T.copy(values, C)
```

```python
matrix = T.reshape(words, (4, 16))
```

| Need | API |
| --- | --- |
| Interpret the same bytes with another shape or dtype | `T.view` |
| Change only shape and require dtype to stay unchanged | `T.reshape` |
| Convert numerical values | A supported cast/conversion operation |
| Change physical fractal layout | `T.copy` or a layout-conversion operation |
| Alias a slice | Not supported by the current complete-buffer API |

## 6. Migrating from removed T.reinterpretcast

The old API required a separately allocated destination Buffer. Bind the old destination name to
the source view instead, and remove the allocation that existed only for `T.reinterpretcast`:

```python
# Removed API
values = T.alloc_ub((4, 16), "float32")
T.reinterpretcast(values, words, "float")

# Current API
values = T.view(words, (4, 16), "float32")
```

Use the old destination Buffer's TileLang shape and dtype. Before deleting its allocation, verify
that it did not have a separate lifetime or an address/synchronization annotation that remains
necessary.

## 7. T.decl_buffer and upstream TileLang

`T.decl_buffer(data=...)` is the low-level way to construct a Buffer descriptor. It does not run
the complete-buffer, equal-bit, or fractal-grid checks above. Prefer `T.view`; use
`T.decl_buffer` only when the caller intentionally takes responsibility for those rules.

The public meaning matches upstream TileLang: both APIs return another Buffer over the same data.
Upstream implements this mainly as descriptor construction. TileLang-Ascend adds the checks above
and emits the target representation required by each backend: AscendC uses a typed local/global
alias, while PTO uses `TRESHAPE` for a complete local view. These are implementation details, not
additional data movement.

See the migrated [RoPE example](../../examples/pos_embedding/rope.py) for an end-to-end use.
