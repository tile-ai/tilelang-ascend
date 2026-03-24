**English** | [中文](README_zh.md)

## 🎉 Overview
SparseMLA is the core attention mechanism introduced in DeepSeek v3.2. This directory implements the related use cases of this mechanism based on Tilelang-AscendC, and provides specific optimization solutions for different optimized versions for developers' reference and use.

## 📊 Test Inputs
The inputs are defined in `bench_sfa.py`, with the following common parameters:

- T=1
- B=1
- Q_N=128
- KV_N=1
- D=512
- D_rope=64
- sparse_size=2048
- block_size=128
- act_kv_s=2560

Three sets of shapes differ only in KV_S:

| No. | KV_S | Description |
| --- | ---: | --- |
| shape0 | 2560 | Short sequence |
| shape1 | 6400 | Medium sequence |
| shape2 | 48000 | Long sequence |

`act_kv_s` is fixed at 2560, and `sparse_size` is fixed at 2048. The effective sparse window size actually participating in the computation remains basically unchanged.

## 📖 File Description
```
├── sparse_flash_attn_pa_baseline.py            # Baseline version
├── sparse_flash_attn_pa_developer.py           # Developer mode version
├── sparse_flash_attn_pa.py                     # Version with T.Pipeline, hybrid programming in developer mode and expert mode
├── sparse_flash_attn_pa_no_cv_pipeline.py      # Version without T.Pipeline, implements matrix multiplication in expert mode
├── sparse_mla_performance_optimization.zh.md   # SparseMLA optimization scheme document
└── bench_sfa.py                                # Execution entry script
```

The specific optimization details of each version are shown in the following table:

| File | Fixed Core | s2 Split Size | Sparse Memory Access Optimization | CV pipeline | Broadcast and AXPY Optimization | Performance Data (A3) |
|----- |:-----------:  | :----------: | :-----------: | :-----------: | :------------------: | :-----:  |
|[sparse_flash_attn_pa_baseline.py](./sparse_flash_attn_pa_baseline.py)             | √ | 64  | × | × | × | 602us |
|[sparse_flash_attn_pa_developer.py](./sparse_flash_attn_pa_developer.py)           | √ | 64  | × | × | √ | 347us |
|[sparse_flash_attn_pa.py](./sparse_flash_attn_pa.py)                               | √ | 64  | √ | √ | √ | 127us |
|[sparse_flash_attn_pa_no_cv_pipeline.py](./sparse_flash_attn_pa_no_cv_pipeline.py) | √ | 256 | √ | × | √ | 109us |

## 📄 Usage

Run `bench_sfa.py` directly:
```Python
python bench_sfa.py --file="sparse_flash_attn_pa_baseline"
```
* `--file`: Supports `sparse_flash_attn_pa_baseline`, `sparse_flash_attn_pa_developer`, `sparse_flash_attn_pa` and `sparse_flash_attn_pa_no_cv_pipeline`.
