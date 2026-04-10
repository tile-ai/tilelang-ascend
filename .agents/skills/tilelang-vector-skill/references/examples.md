# Vector Examples (v-prefix style)

## Example pattern: elementwise add

- copy input tiles to local buffer
- call T.vadd
- copy result back

## Example pattern: normalization pieces

- square: T.vmul(x, x, tmp)
- reduce: T.reduce(tmp, sum, dims=[1], reduce_mode="sum")
- scale and rsqrt: T.vdiv, T.vadd, T.vrsqrt
- finalize: T.vmul(x, inv_std, y)

## Rule

When both forms are valid, always generate the v-prefix form first.
