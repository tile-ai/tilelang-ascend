# TileLang-Ascend 架构参考

## 编译流程

```
Python DSL (@tilelang.jit)
  → IR 变换 Pass (tilelang/transform/ + src/transform/)
  → 代码生成 (src/target/codegen_ascend_pto.cc)
  → 模板库 (src/tl_templates/)
  → CANN 工具链编译
  → NPU 执行
```

## 关键变换 Pass 链

```
frontend_legalize → layout_inference → flatten_buffer → loop_vectorize
→ inject_pipeline → ascend_lower_parallel_to_vector → ascend_memory_planning
→ ascend_sync_insert → ascend_combinecv → lower_tile_op
```

## 目录结构

```
tilelang-ascend/
├── tilelang/              # Python 前端
│   ├── jit/               # JIT 编译 (@tilelang.jit)
│   ├── language/          # DSL 原语定义
│   │   ├── ascend.py      # Ascend 操作
│   │   ├── ascend_tile.py # Tile 级操作 (56KB, 最大模块)
│   │   ├── pto.py         # PTO 模式
│   │   ├── gemm.py        # 矩阵乘法
│   │   ├── copy.py        # 数据搬运
│   │   ├── reduce.py      # 归约操作
│   │   ├── allocate.py    # 内存分配
│   │   └── parallel.py    # 并行化
│   ├── engine/            # 编译引擎 (lower.py)
│   ├── transform/         # Python 侧 IR 变换
│   ├── autotuner/         # 自动调优
│   ├── carver/            # 调度与资源映射
│   ├── layout/            # 内存布局
│   ├── intrinsics/        # 硬件内建函数
│   └── profiler/          # 性能分析
├── src/                   # C++ 后端
│   ├── target/            # 代码生成
│   │   ├── codegen_ascend_pto.cc  # 主要: PTO 代码生成 (3225行)
│   │   ├── codegen_ascend.cc      # Ascend C 代码生成 (2050行)
│   │   └── rt_mod_ascend*.cc      # 运行时模块
│   ├── transform/         # C++ 侧 IR 变换 (47个文件)
│   │   ├── ascend_storage_rewrite.cc     # 存储重写 (70KB)
│   │   ├── ascend_sync_insert.cc         # 同步插入 (46KB)
│   │   ├── ascend_lower_parallel_to_vector.cc  # 向量化 (49KB)
│   │   ├── ascend_memory_planning.cc     # 内存规划 (24KB)
│   │   └── cross_core_pipeline.cc        # 核间流水 (35KB)
│   ├── tl_templates/      # 代码生成模板
│   └── op/                # 算子定义
├── examples/              # 31类算子示例
├── testing/python/        # 测试
├── docs/                  # 文档
└── 3rdparty/              # 第三方依赖 (TVM, pto-isa)
```

## 硬件原语映射

| DSL 原语 | 硬件资源 |
|---------|---------|
| `T.alloc_L1` | L1 缓存 (Cube 核) |
| `T.alloc_ub` | 统一缓冲区 (Vector 核) |
| `T.alloc_L0A/L0B/L0C` | L0 寄存器 |
| `T.gemm` / `T.mma` | 矩阵乘法加速器 |
| `T.Parallel` | 向量化指令 |
| `T.tile.atomic_add` | UB 到 GM 的 DMA 原子累加写回 |
| `T.Pipelined` | 流水线调度 |

## 核心 API

- `@tilelang.jit` — JIT 编译装饰器
- `T.alloc_L1/ub/L0A/L0B/L0C` — 多级内存分配
- `T.copy` — 数据搬运
- `T.tile.atomic_add` — 本地 tensor 到 GM 的原子累加写回
- `T.gemm` / `T.mma` — 矩阵计算
- `T.Parallel` — 向量化
- `T.Pipelined` — 流水线
- `T.printf` / `T.dump_tensor` — 调试
