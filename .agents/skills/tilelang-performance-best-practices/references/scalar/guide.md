# TileLang 标量调度索引

使用 Python kernel factory 与静态 DSL 结构减少运行时标量工作：shape/dtype/mode/tile/stage 在编译前选择，动态值只保留真实输入维和 task id。热循环采用 T.Persistent、T.Pipelined、T.Parallel、T.unroll；地址公共子表达式保存为局部变量。

所有优化以生成源码和 timeline 为证据。检查动态分支、重复除模、64 位地址、寄存器 spill 与 unroll 代码膨胀。不得删除真实尾块、mask 或动态 shape 契约。

精度先跑定向精度测试；性能使用相同 kernel 语义做单变量 A/B。

当前 API 依据：`tilelang/language/persistent.py`、`pipeline.py`、`parallel.py` 和 `examples/gemm/example_gemm_intrinsic_persistent.py`；`T.Persistent` 不接受 `num_stages`。专题见 [编码原则](coding_principles.md)、[Buffer 管理](buffer_management_conversion_guide.md) 和 [Conv/GEMM 标量精简](conv_scalar.md)。
