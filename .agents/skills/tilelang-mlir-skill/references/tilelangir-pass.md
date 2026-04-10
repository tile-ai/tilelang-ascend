# TileLangIR Pass Notes

## Current pass names

- tilelangir-cv-split
- tilelangir-vectorize

## Source locations

- tilelangir/include/tilelangir/Transforms/Passes.td
- tilelangir/lib/Transforms/CVSplit.cpp
- tilelangir/lib/Transforms/Vectorize.cpp

## Debug method

- run pass one by one
- inspect IR after each pass
- narrow down first divergence point
