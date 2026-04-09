# MLIR Test Guide

## Main location

- examples/
- testing/npuir/

## Deprecated location

- unittest/npuir/mlir_files is deprecated and should not be used as the primary correctness baseline.

## Suggested workflow

1. Reproduce with a minimal kernel under examples/ first.
2. Validate operator behavior with tests under testing/npuir/ (CI-protected baseline).
3. Build or regenerate target MLIR for the failing case.
4. Compare IR structure before and after suspect passes.
5. Isolate the first failing transformation stage.

## What to inspect

- operation sequence
- region nesting
- data movement ops and sync ops

## Optional auxiliary checks

- testing/mlir/ can be used for MLIR lit-style checks as a supplementary signal.
