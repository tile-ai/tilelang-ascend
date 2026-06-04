# GEMM 与 CV 融合编码规范

## 目录

- [1. GEMM 编码规范](#1-gemm-编码规范)
  - [1.1 gemm_v0 初始化](#11-gemm_v0-初始化)
  - [1.2 NPU 分形限制](#12-npu-分形限制)
- [2. CV 融合 pass_configs](#2-cv-融合-pass_configs)

---

## 1. GEMM 编码规范

### 1.1 gemm_v0 初始化

第一次调用必须清零 C_L0：

```python
for k_chunk in T.serial(k_num):
    T.gemm_v0(A_L1, B_L1, C_L0, transpose_B=True, init=(k_chunk == 0))
```

### 1.2 NPU 分形限制

GEMM 的 block size 必须满足 L0A/L0B/L0C 分形限制（详见 [api-compute.md](../../tilelang-custom-skill/tilelang-api-best-practices/references/api-compute.md)）：

- int8 GEMM：`block_M ≥ 16`, `block_N ≥ 16`, `block_K ≥ 32`
- float16 GEMM：`block_M ≥ 16`, `block_N ≥ 16`, `block_K ≥ 16`

---

## 2. CV 融合 pass_configs

CV 融合算子必须开启全部 4 个 pass_configs：

```python
PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,  # 自动分离 Cube/Vector
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}
```
