# Environment Variables Guide

This document describes all supported environment variables and their effects.

## 📋 Table of Contents
- [Overview](#overview)
- [Debugging](#debugging)
- [Compilation Options](#compilation-options)

## Overview
Environment variables allow you to control various aspects of the project's runtime behavior, debugging output, performance optimizations, and more.

```bash
# Example of setting environment variables
export TILELANG_DUMP_IR=TRUE
```

## Debugging

| Variable | Default | Description | Valid Values |
|----------|---------|-------------|--------------|
| `TILELANG_DUMP_IR` | `FALSE` | Enable print TVM IR and NPUIR | `FALSE`: Disabled<br>`TRUE`:Enabled |
| `TILELANG_ASCEND_WORKSPACE_SIZE` | `32768` | Set workspace size for Ascend CV fusion (in Byte, Single aicore) | Positive integer, e.g., `32768`, `65536` |

## Compilation Options
| Variable | Default | Description | Valid Values |
|----------|---------|-------------|--------------|
| `TILELANG_ASCEND_MODE` | `Expert` | Set the TileLang Mode; currently, Expert mode and Developer mode are supported | `Expert`: Expert Mode<br>`Developer`: Developer Mode |
| `TILELANG_ASCEND_DEVICE_NAME` | `Ascend910B` | Override the target device name for compilation (e.g. for cross-compilation). If not set, runtime hardware detection is used. | String, e.g., `Ascend910B`|

## Autotuner

| Variable | Default | Description | Valid Values |
|----------|---------|-------------|--------------|
| `TILELANG_BENCH_METHOD` | `` | Choose the method for kernel execution evaluation |String `npu`: use torch_npu.profiler<br> Otherwise: use torch.npu.Event
| `TILELANG_CACHE_DIR` | `` | Set the path to store autotuner cache data | String, e.g., `/home/autotune_cache` |
