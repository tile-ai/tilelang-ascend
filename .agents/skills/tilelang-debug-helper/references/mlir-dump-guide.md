# IR Dump Guide for npuir branch

## Strategy

- capture IR before and after major pass stages
- compare operation-level diffs
- correlate with failing runtime behavior

## Common checkpoints

- after lower entry
- after tilelangir pass application
- before backend codegen
