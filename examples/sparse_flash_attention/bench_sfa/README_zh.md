
[English](README.md) | **中文**

## 🎉概述
SparseMLA 是 DeepSeek v3.2 版本中引入的核心注意力机制。本目录基于 Tilelang-AscnedC 完成了该机制的相关用例实现，并提供了不同优化版本的具体优化方案，供开发者参考与使用。

## 📊测试输入
输入定义在 bench_sfa.py 中，公共参数如下：

- T=1
- B=1
- Q_N=128
- KV_N=1
- D=512
- D_rope=64
- sparse_size=2048
- block_size=128
- act_kv_s=2560

三组 shape 只在 KV_S 上不同：

| 编号 | KV_S | 说明 |
| --- | ---: | --- |
| shape0 | 2560 | 短序列 |
| shape1 | 6400 | 中序列 |
| shape2 | 48000 | 长序列 |

act_kv_s 固定为 2560，sparse_size 固定为 2048，实际参与计算的有效稀疏窗口规模基本不变。

## 📖文件说明
```
├── sparse_flash_attn_pa_baseline.py            # 基线版本
├── sparse_flash_attn_pa_developer.py           # 开发者模式版本
├── sparse_flash_attn_pa.py                     # 开启T.Pipeline版本，开发者模式和expert模式混合编程
├── sparse_flash_attn_pa_no_cv_pipeline.py      # 不开T.Pipeline版本，并且专家模式实现矩阵乘法
├── sparse_mla_performance_optimization.zh.md   # SparseMLA优化方案文档
└── bench_sfa.py                                # 运行入口脚本
```

具体各版本优化细节如下表所示：

| 文件 | Fixed Core    | s2切分大小    | 稀疏访存优化   | CV pipeline   |  Broadcast和AXPY优化 |  性能数据（A3） |
|----- |:-----------:  | :----------: | :-----------: | :-----------: | :------------------: | :-----:  |
|[sparse_flash_attn_pa_baseline.py](./sparse_flash_attn_pa_baseline.py)             | √ | 64  | × | × | × | 602us |
|[sparse_flash_attn_pa_developer.py](./sparse_flash_attn_pa_developer.py)           | √ | 64  | × | × | √ | 347us |
|[sparse_flash_attn_pa.py](./sparse_flash_attn_pa.py)                               | √ | 64  | √ | √ | √ | 127us |
|[sparse_flash_attn_pa_no_cv_pipeline.py](./sparse_flash_attn_pa_no_cv_pipeline.py) | √ | 256 | √ | × | √ | 109us |

## 📄使用方法

直接运行bench_sfa.py即可。
```Python
python bench_sfa.py --file="sparse_flash_attn_pa_baseline"
```
* --file：支持sparse_flash_attn_pa_baseline、sparse_flash_attn_pa_developer、sparse_flash_attn_pa和sparse_flash_attn_pa_no_cv_pipeline。
