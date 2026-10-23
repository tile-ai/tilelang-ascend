# A2/A3 Buffer 管理

Developer 模式使用 T.alloc_shared、T.alloc_fragment，由当前 buffer scope 推导确定最终存储层级；不得将 fragment 一律当作寄存器/线程局部。Expert 模式 UB 使用 T.alloc_ub，Cube 路径使用 T.alloc_L1、T.alloc_L0A/L0B/L0C。T.copy 负责目标仓库已验证的层级搬运。

流水 tile 的物理字节为元素数×dtype 字节×版本数；再加入 padding、resident 数据与安全余量。常驻权重和 reduction 状态不参与多版本。只有 lifetime 不重叠的 buffer 才允许依赖后端 merge/reuse，并通过生成代码确认。

尾部 GM 只访问 valid，UB 仍按完整指令访问 footprint 分配。每个 task 重新初始化可变状态。

当前存储映射以 `tilelang/language/allocate.py`、`src/transform/ascend_infer_buffer_scope.cc` 与生成源码为依据。
