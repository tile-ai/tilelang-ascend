# T.tile.fill

`T.tile.fill(buffer, value)` fills a UB buffer or a contiguous buffer region
with a scalar value. A region must remain contiguous when flattened; the
compiler rejects windows that span gaps between physical rows.

For a padded buffer, fill the selected part of each row separately:

```python
buffer = T.alloc_ub((64, 40), "float32")
for row in T.serial(64):
    T.tile.fill(buffer[row:row + 1, 0:32], 1.0)
```

A single call on `buffer[:, 0:32]` is rejected: the 32 selected columns in
adjacent rows are separated by 8 unselected elements. Complete dense
buffers, one-dimensional slices, individual row windows, and selections
that include each physical row in full remain supported.
