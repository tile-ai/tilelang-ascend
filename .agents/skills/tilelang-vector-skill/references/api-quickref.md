# Vector API Quick Reference (v-prefix first)

## Binary ops

- T.vadd(A, B, C)
- T.vsub(A, B, C)
- T.vmul(A, B, C)
- T.vdiv(A, B, C)
- T.vmax(A, B, C)
- T.vmin(A, B, C)

## Unary ops

- T.vexp(A, B)
- T.vln(A, B)
- T.vsqrt(A, B)
- T.vrsqrt(A, B)
- T.vabs(A, B)
- T.vrelu(A, B)
- T.vsigmoid(A, B)
- T.vtanh(A, B)
- T.verf(A, B)

## Utility ops

- T.vcast(src, dst, round_mode="rint")
- T.vbrc(value, dst)
- T.vcmp(a, b, dst, cmp_mode)
- T.vselect(mask, a, b, dst, mode)
- T.reduce(src, dst, dims=[...], reduce_mode="sum|max|min")

## Compatibility mapping

- T.vmul == T.npuir_mul
- T.vadd == T.npuir_add
- T.vexp == T.npuir_exp
- T.vcast == T.npuir_cast
- T.vbrc == T.npuir_brc
