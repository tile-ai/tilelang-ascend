# NZ Format Notes

NZ format path is intended for Expert mode kernels only.

## Why NZ path matters

Cube compute often benefits from ND to NZ layout conversion for compute-friendly access.

## Typical path

- load_nd2nz for input tiles (Expert mode)
- gemm in cube path
- store_fixpipe with enable_nz2nd=True for output conversion (Expert mode)

## Developer mode note

- Developer mode kernels should keep ND layout.
- Use T.copy/T.alloc_shared path and do not force NZ conversion.

## Validation checklist

- check tile size consistency across load, gemm, and store
- check transpose setting and layout assumptions
- compare outputs with reference implementation
