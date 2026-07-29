# TileLang-Ascend API 文档完善 — 学习计划指南

> **目标任务**：为对外暴露的 API 编写详细说明文档，明确参数类型、数据范围和支持场景；补充 Docstring；完善测试用例。

---

## 一、项目概览

### 1.1 项目定位

**TileLang-Ascend** 是基于 TVM 编译器基础设施构建的领域特定语言（DSL），专为华为昇腾 NPU 优化。开发者使用 Python 语法编写高性能 AI 算子 kernel，编译器自动生成底层 Ascend C++ 代码。

### 1.2 核心目录结构

```
tilelang-ascend/
├── tilelang/                  ← Python DSL 核心库
│   ├── language/              ← ★ 对外 API 定义（文档工作的核心目录）
│   │   ├── __init__.py        ← API 导出入口 + annotate_layout 等函数
│   │   ├── allocate.py        ← 内存分配 API（alloc_L1/L0A/L0B/L0C/ub）
│   │   ├── ascend.py          ← 同步、GEMM、调试、核间通信 API
│   │   ├── ascend_tile.py     ← ★ 向量计算 API（最大文件，~60 个 API）
│   │   ├── copy_op.py         ← T.copy 数据搬运
│   │   ├── reduce.py          ← 归约操作
│   │   ├── reduce_ascend.py   ← 昇腾专用归约
│   │   ├── kernel.py          ← T.Kernel 启动入口
│   │   ├── parallel.py        ← T.Parallel 自动向量化
│   │   ├── pipeline.py        ← T.Pipelined 软件流水线
│   │   ├── memscope.py        ← T.Scope("C"/"V") 执行域
│   │   └── customize.py       ← mma、atomic_add 等定制 API
│   ├── jit/                   ← JIT 编译引擎
│   └── transform/             ← 编译 Pass
├── src/                       ← C++ 编译器后端
├── examples/                  ← 算子示例（~40+ 种算子）
├── testing/python/            ← 单元测试
├── benchmark/                 ← 性能基准测试
└── docs/                      ← 现有文档
    └── TileLang-Ascend Programming Guide.md  ← ★ 核心参考文档
```

### 1.3 硬件内存层级（理解 API 的前提）

```
                    ┌─────────────────────────┐
                    │    GM (Global Memory)    │  外部存储，容量大，延迟高
                    └──────────┬──────────────┘
                               │ T.copy()
               ┌───────────────┼───────────────┐
               ▼               ▼               │
        ┌────────────┐  ┌────────────┐          │
        │  L1 Buffer │  │  UB Buffer │          │
        │ (Cube 核)  │  │ (Vector核) │          │
        └─────┬──────┘  └────────────┘          │
              │ T.copy()                         │
       ┌──────┼──────┐                           │
       ▼      ▼      ▼                           │
   ┌──────┐┌──────┐┌──────┐                      │
   │ L0A  ││ L0B  ││ L0C  │ ← 矩阵寄存器        │
   └──────┘└──────┘└──────┘                      │
                                                  │
   Cube Core ←── GM/L2 ──→ Vector Core           │
```

**关键理解**：
- **Cube Core**：负责矩阵运算（GEMM），使用 L1 + L0A/L0B/L0C
- **Vector Core**：负责向量运算（add/mul/exp 等），使用 UB
- 两种核之间通过 **GM/L2** 交换数据，需要**显式同步**（set_flag/wait_flag）

---

## 二、对外 API 全景图

所有对外 API 从 `tilelang/language/__init__.py` 导出，用户通过 `import tilelang.language as T` 使用。

### 2.1 内存分配 API（`allocate.py`）

| API | 功能 | 内存层级 | 使用模式 |
|-----|------|---------|---------|
| `T.alloc_L1(shape, dtype)` | 分配 L1 缓存 | Cube 核 | Expert |
| `T.alloc_L0A(shape, dtype)` | 分配 L0A 矩阵寄存器 | Cube 核 | Expert |
| `T.alloc_L0B(shape, dtype)` | 分配 L0B 矩阵寄存器 | Cube 核 | Expert |
| `T.alloc_L0C(shape, dtype)` | 分配 L0C 累加寄存器 | Cube 核 | Expert |
| `T.alloc_ub(shape, dtype)` | 分配 UB 统一缓冲 | Vector 核 | Expert |
| `T.alloc_shared(shape, dtype)` | 分配共享内存 | 自动推断 L1/UB | Developer |
| `T.alloc_fragment(shape, dtype)` | 分配片段内存 | 寄存器 | Developer |
| `T.alloc_var(dtype, init)` | 分配标量变量 | 线程私有 | 通用 |

### 2.2 数据搬运 API（`copy_op.py`）

| API | 功能 | 典型搬运路径 |
|-----|------|------------|
| `T.copy(src, dst)` | 跨层级数据搬运 | GM→L1, L1→L0A, L0C→GM, L0C→UB 等 |

### 2.3 矩阵计算 API（`ascend.py` / `gemm.py`）

| API | 功能 | 关键参数 |
|-----|------|---------|
| `T.gemm_v0(A, B, C, ...)` | 块级矩阵乘法 | transpose_A/B, init, kL0Size, n_actual |
| `T.mma(A, B, C, ...)` | alias for npu_gemm | — |

### 2.4 向量计算 API（`ascend_tile.py` → `T.tile.xxx`）

**这是 API 数量最多、文档缺失最严重的文件。**

#### 二元运算
| API | 功能 | src1 支持的类型 |
|-----|------|---------------|
| `T.tile.add(dst, src0, src1)` | 逐元素加法 | Buffer / BufferLoad / Scalar |
| `T.tile.sub(dst, src0, src1)` | 逐元素减法 | Buffer / BufferLoad |
| `T.tile.mul(dst, src0, src1)` | 逐元素乘法 | Buffer / BufferLoad / Scalar |
| `T.tile.div(dst, src0, src1)` | 逐元素除法 | Buffer / BufferLoad |
| `T.tile.max(dst, src0, src1)` | 逐元素取最大值 | Buffer / BufferLoad / Scalar |
| `T.tile.min(dst, src0, src1)` | 逐元素取最小值 | Buffer / BufferLoad / Scalar |
| `T.tile.pow(dst, src0, src1)` | 逐元素幂运算 | Buffer |
| `T.tile.bitwise_and/or/xor` | 逐元素位运算 | Buffer / BufferLoad / Scalar |

#### 一元运算
| API | 功能 |
|-----|------|
| `T.tile.exp(dst, src)` | 指数函数 |
| `T.tile.ln(dst, src)` | 自然对数 |
| `T.tile.abs(dst, src)` | 绝对值 |
| `T.tile.sqrt(dst, src)` | 平方根 |
| `T.tile.rsqrt(dst, src)` | 平方根倒数 |
| `T.tile.relu(dst, src)` | ReLU 激活 |
| `T.tile.reciprocal(dst, src)` | 倒数 |
| `T.tile.sigmoid(dst, src)` | Sigmoid 激活 |
| `T.tile.silu(dst, src)` | SiLU/Swish 激活 |
| `T.tile.sin(dst, src)` | 正弦 |
| `T.tile.cos(dst, src)` | 余弦 |
| `T.tile.bitwise_not(dst, src)` | 按位取反 |

#### 标量运算
| API | 功能 |
|-----|------|
| `T.tile.leaky_relu(dst, src, scalar)` | Leaky ReLU |
| `T.tile.axpy(dst, src, scalar)` | dst = scalar * src + dst |
| `T.tile.bitwise_lshift/rshift(dst, src, scalar)` | 位移 |

#### 三目/融合运算
| API | 功能 |
|-----|------|
| `T.tile.mul_add_dst(dst, src0, src1)` | dst = src0 * src1 + dst |
| `T.tile.clamp(out, buf, min, max, count)` | 裁剪到 [min, max] |
| `T.tile.clamp_max/min` | 单侧裁剪 |

#### 比较与类型转换
| API | 功能 |
|-----|------|
| `T.tile.compare(dst, src0, src1, mode)` | 比较（EQ/NE/GT/GE/LT/LE） |
| `T.tile.cast(dst, src, mode, count)` | 类型转换（7 种舍入模式） |

#### 数据操作
| API | 功能 |
|-----|------|
| `T.tile.fill(buffer, value)` | 填充值 |
| `T.tile.clear(buffer)` | 清零 |
| `T.tile.sort(dst, src, actual_num)` | 排序 |
| `T.tile.merge_sort(dst, src0..src3)` | 2/3/4 路归并排序 |
| `T.tile.topk(dst, src, K, actual_num)` | TopK 选取 |
| `T.tile.transpose(dst, src)` | 矩阵转置 |
| `T.tile.gather(dst, src, offset, base)` | Gather 操作 |
| `T.tile.gather_mask(dst, src, pattern)` | Gather Mask |
| `T.tile.gatherb(dst, src, offset, ...)` | GatherB |
| `T.tile.select(dst, mask, src0, src1, mode)` | 条件选择 |
| `T.tile.broadcast(dst, src, axis)` | 广播 |
| `T.tile.createvecindex(dst, first)` | 生成索引序列 |
| `T.tile.arith_progression(buf, first, diff, count)` | 等差数列 |
| `T.tile.atomic_add(dst, src)` | 原子加 |
| `T.tile.row_expand_mul(dst, src0, src1)` | 行广播乘法 |

### 2.5 归约 API（`reduce.py` / `reduce_ascend.py`）

| API | 功能 |
|-----|------|
| `T.reduce_max / reduce_min` | 归约最大/最小值 |
| `T.reduce_sum` | 归约求和 |
| `T.reduce_abssum / reduce_absmax` | 绝对值归约 |
| `T.cumsum` | 前缀和 |

### 2.6 同步与控制 API（`ascend.py`）

| API | 功能 | pipe 取值 |
|-----|------|----------|
| `T.set_flag(src, dst, eventId)` | 设置同步信号 | "fix"/"mte1"/"mte2"/"m"/"v"/"s" |
| `T.wait_flag(src, dst, eventId)` | 等待同步信号 | 同上 |
| `T.set_cross_flag(pipe, flag, mode)` | 跨核同步设置 | — |
| `T.wait_cross_flag(flag, pipe)` | 跨核同步等待 | — |
| `T.barrier_all()` | 全流水线屏障 | — |
| `T.pipe_barrier(pipe)` | 指定流水线屏障 | — |
| `T.sync_all()` | 全局同步 | — |

### 2.7 核间通信 API（`ascend.py`）

| API | 功能 |
|-----|------|
| `T.shmem_put_nbi(dst, src, nelems, newPe)` | 本地 GM → 远端 GM |
| `T.shmem_get_nbi(dst, src, nelems, newPe)` | 远端 GM → 本地 GM |
| `T.shmem_ub_put_nbi(ub, dst, nelems, newPe)` | 本地 UB → 远端 GM |
| `T.shmem_ub_get_nbi(dst, src, nelems, newPe)` | 远端 GM → 本地 UB |

### 2.8 调度原语

| API | 功能 | 定义位置 |
|-----|------|---------|
| `T.Kernel(n, threads, is_npu)` | kernel 启动入口 | `kernel.py` |
| `T.Scope("C"/"V")` | Cube/Vector 执行域 | `memscope.py` |
| `T.Parallel(M, N)` | 自动向量化 | `parallel.py` |
| `T.Pipelined(range, num_stages)` | 软件流水线 | `pipeline.py` |
| `T.annotate_layout(map)` | 标注内存布局 | `__init__.py` |
| `T.annotate_address(map)` | 标注内存地址 | `__init__.py` |
| `T.use_swizzle(...)` | L2 Cache 局部性优化 | `__init__.py` |

### 2.9 调试工具（`ascend.py`）

| API | 功能 |
|-----|------|
| `T.printf(format_str, *args)` | 设备端打印 |
| `T.dump_tensor(tensor, desc, dump_size)` | Dump 张量数据 |
| `T._src_code(source_code)` | 注入原始 C++ 代码 |

---

## 三、Docstring 现状评估

### 3.1 各文件覆盖率

| 文件 | API 数量 | Docstring 覆盖率 | 质量评级 | 优先级 |
|------|---------|----------------|---------|--------|
| `ascend_tile.py` | ~60 | ⚠️ ~50% | ⭐⭐ 参差不齐 | **P0** |
| `allocate.py` | 8 | ⚠️ alloc_L1/L0x/ub 完全缺失 | ⭐⭐ | **P0** |
| `__init__.py` | ~8 | ❌ ~30% | ⭐ 严重缺失 | **P1** |
| `ascend.py` | ~18 | ✅ ~85% | ⭐⭐⭐⭐ 最好 | P2（参考模板） |
| `reduce_ascend.py` | 待确认 | ❓ 待确认 | 待定 | P2 |
| `copy_op.py` | 待确认 | ❓ 待确认 | 待定 | P1 |
| `customize.py` | 待确认 | ❓ 待确认 | 待定 | P2 |

### 3.2 典型缺失示例

```python
# ❌ allocate.py — alloc_L1/L0A/L0B/L0C/ub 完全没有 Docstring
def alloc_L1(shape, dtype):
    return T.alloc_buffer(shape, dtype, scope="shared.l1")

# ❌ __init__.py — 核心 API 无有效文档
def use_swizzle(cid, m, n, k, block_m, block_n, off=1, dir=0, in_loop=False):
    """Alias for npu_use_swizzle with proper signature for function hints."""
    # ← 参数含义、取值范围、用法完全缺失

# ⚠️ ascend_tile.py — sigmoid 无 Args 说明
def sigmoid(dst, src):
    ...

# ⚠️ ascend_tile.py — compare 的 mode 参数 6 种取值未逐一说明
def compare(dst, src0, src1, mode):
    ...

# ⚠️ ascend_tile.py — sort 的输出格式描述不充分
def sort(dst, src, actual_num):
    ...
```

### 3.3 已标记废弃的 API（可跳过）

以下 API 已标注 `@deprecated()`，文档工作优先级最低：

- `T.tile.brcb` — 后端未实现
- `T.tile.bilinear_interpolation`
- `T.tile.wholereducemax/min/sum` — 不支持 PTO target
- `T.tile.block_reduce_max/min/sum` — 不支持 PTO target

---

## 四、两天学习计划

### Day 1：建立认知 + 开始产出

#### 上午（~3h）：硬件基础 + 编程模型

| 时段 | 任务 | 预期产出 |
|------|------|---------|
| 0:00-0:45 | 精读 `docs/TileLang-Ascend Programming Guide.md` 前 3 章（硬件架构、编程模型、内存管理） | 硬件内存层级笔记 |
| 0:45-1:15 | 对比阅读 `examples/gemm/example_gemm.py`（Expert）和 `examples/developer_mode/matmul_add_developer.py`（Developer） | Developer vs Expert 差异笔记 |
| 1:15-1:45 | 阅读 `examples/elementwise/`、`examples/softmax/`、`examples/reduce/` | 向量 API 实际用法理解 |
| 1:45-2:15 | 运行 2-3 个 example 验证环境 | 确认环境可用 |

**理解检查清单**（必须搞清楚才能写出好文档）：
- [ ] L1 缓存大小限制是多少？shape 参数有什么对齐约束？
- [ ] L0A/L0B/L0C 分别支持哪些 dtype？shape 对齐要求？
- [ ] UB 和 L1 的区别？什么场景用哪个？
- [ ] `T.copy` 支持的合法搬运路径有哪些？（GM→L1、L1→L0A、L0C→UB 等）
- [ ] `set_flag`/`wait_flag` 的 pipe 参数（"mte1"/"mte2"/"m"/"v"/"fix"）各代表什么流水线阶段？
- [ ] `T.Kernel` 的 `threads` 参数取 1 或 2 时行为有何不同？

#### 下午（~3h）：精读 API 源码 + 开始补 Docstring

| 时段 | 任务 | 预期产出 |
|------|------|---------|
| 0:00-0:45 | 精读 `ascend_tile.py`，按类别分组标记 Docstring 缺失情况 | API 清单 + 缺失列表 |
| 0:45-1:15 | 对照 `examples/` 中实际用法，理解每类 API 的参数范围和约束 | 每个 API 的用法笔记 |
| 1:15-2:15 | 从最简单的 API 开始补 Docstring（优先级：二元运算 → 一元运算 → 标量运算 → 填充/清除） | 10-15 个 API 的完整 Docstring |
| 2:15-3:00 | 对照 Ascend C 官方文档验证参数约束（dtype 支持、shape 对齐等） | 数据范围描述校准 |

#### 晚上（~1.5h）：阅读现有测试 + 理解测试规范

| 时段 | 任务 | 预期产出 |
|------|------|---------|
| 0:00-0:15 | `ls testing/python/` 了解测试目录结构 | 测试目录全景 |
| 0:15-0:45 | 精读 2-3 个测试文件，观察 fixture、assert 模式、命名规范 | 测试编写模式笔记 |
| 0:45-1:00 | 阅读 `examples/bench_test.sh`，理解测试发现和执行机制 | 测试运行机制理解 |
| 1:00-1:30 | 列出完全没有测试覆盖的 API | 测试缺失清单 |

### Day 2：深入补全 + 产出文档

#### 上午（~3h）：继续补 Docstring + 写接口文档

| 时段 | 任务 | 预期产出 |
|------|------|---------|
| 0:00-1:00 | 继续补 `ascend_tile.py` Docstring（重点：sort/merge_sort/topk、compare/cast、gather/gather_mask/select） | 15-20 个 API 的完整 Docstring |
| 1:00-1:30 | 补 `__init__.py` 核心函数 Docstring（annotate_layout、use_swizzle、annotate_address） | 3-5 个 API 的完整 Docstring |
| 1:30-2:00 | 补 `allocate.py` 中 alloc_L1/L0A/L0B/L0C/ub 的 Docstring | 5 个内存分配 API 的完整 Docstring |
| 2:00-3:00 | 开始编写独立接口说明文档（按类别组织，明确"支持场景"） | 文档框架 + 首批 API 文档 |

#### 下午（~3h）：补测试用例

| 时段 | 任务 | 预期产出 |
|------|------|---------|
| 0:00-0:30 | 确定测试缺失 API 的优先级列表 | 优先级排序 |
| 0:30-1:30 | 编写第一批测试用例（3-5 个 API） | 测试文件 test_xxx.py |
| 1:30-2:30 | 编写第二批测试用例（3-5 个 API） | 更多测试文件 |
| 2:30-3:00 | 运行测试确认通过 | 测试结果日志 |

#### 晚上（~1.5h）：整理 + 自查

| 时段 | 任务 | 预期产出 |
|------|------|---------|
| 0:00-0:30 | 检查所有 Docstring 格式一致性 | 自查清单 |
| 0:30-1:00 | 检查文档与代码的一致性，跑测试确认无回归 | 修正列表 |
| 1:00-1:30 | 整理工作产出清单 | 最终报告 |

---

## 五、重点阅读文件清单

### 必读（P0）

| 文件 | 原因 |
|------|------|
| `docs/TileLang-Ascend Programming Guide.md` | 理解硬件和编程模型，文档写作的基础 |
| `tilelang/language/ascend_tile.py` | 最大 API 文件（~60 个 API），Docstring 缺失最严重 |
| `tilelang/language/allocate.py` | alloc_L1/L0A/L0B/L0C/ub 5 个核心 API 完全没有文档 |
| `examples/gemm/example_gemm.py` | 理解完整 kernel 结构的最佳入口 |

### 重要（P1）

| 文件 | 原因 |
|------|------|
| `tilelang/language/__init__.py` | annotate_layout、use_swizzle 等缺少文档 |
| `tilelang/language/ascend.py` | Docstring 质量最好，**作为文档写作的参考模板** |
| `tilelang/language/copy_op.py` | T.copy 是核心 API，需确认文档状态 |
| `examples/developer_mode/` | 理解 Developer 模式的 API 用法 |
| `testing/python/` | 学习测试编写规范 |

### 参考（P2）

| 文件 | 原因 |
|------|------|
| `examples/flash_attention/` | 复杂算子中 API 的组合用法 |
| `examples/sparse_flash_attention/` | 高级 API 用法参考 |
| `tilelang/language/reduce_ascend.py` | 昇腾专用归约，待确认文档状态 |
| `tilelang/language/customize.py` | mma、atomic_add 等，待确认文档状态 |
| `examples/bench_test.sh` | 理解测试运行机制 |

---

## 六、文档规范

### 6.1 Docstring 模板

```python
def api_name(dst, src0, src1, ...):
    """一句话描述功能：``dst = f(src0, src1)``。

    更详细的功能说明（可选）。包括计算的数学公式、硬件行为等。

    Args:
        dst: 描述。内存 scope 要求。支持的 dtype。shape 约束。
        src0: 描述。
        src1: 描述。如果支持多种类型，逐一说明：
            - ``Buffer | BufferRegion``: 描述，shape 约束。
            - ``BufferLoad``: 描述。
            - ``PrimExpr | float``: 描述。
        ...

    Returns:
        返回值描述（通常是 tvm.tir.Call）。

    Raises:
        AssertionError: 什么条件下会触发。
        ValueError: 什么条件下会触发。

    Example::

        a_ub = T.alloc_ub((128, 256), "float16")
        b_ub = T.alloc_ub((128, 256), "float16")
        c_ub = T.alloc_ub((128, 256), "float16")
        T.tile.api_name(c_ub, a_ub, b_ub)

    Note:
        - 对应的 Ascend C 指令名称。
        - 特殊注意事项（如 buffer 大小对齐要求）。
        - 支持的芯片平台（A2/A3）。
    """
```

### 6.2 独立接口文档模板

```markdown
## T.tile.xxx

### 功能
简述该 API 的功能和计算语义。

### 函数签名
```python
T.tile.xxx(dst, src0, src1, ...)
```

### 参数说明

| 参数 | 类型 | 必填 | 说明 | 约束 |
|------|------|------|------|------|
| dst | Buffer \| BufferRegion | 是 | 输出缓冲 | UB scope，shape 与 src0 一致 |
| src0 | Buffer \| BufferRegion | 是 | 第一个输入 | UB scope |
| src1 | Buffer \| Scalar | 是 | 第二个输入 | Buffer 时 shape 同 src0；Scalar 时广播 |

### 支持的数据类型

| dtype | 约束 |
|-------|------|
| float16 | buffer 元素数必须是 16 的倍数 |
| float32 | buffer 元素数必须是 8 的倍数 |

### 支持场景

- ✅ 场景 1 描述
- ✅ 场景 2 描述
- ❌ 不支持的场景描述

### 使用示例

```python
# 完整可运行示例
```

### 对应 Ascend C 指令
`AscendC::Xxx`

### 参考
- Ascend C 官方文档链接
- 项目中的 example 路径
```

### 6.3 测试用例模板

```python
import torch
import tilelang
from tilelang import language as T


def test_api_name():
    """测试 T.tile.api_name 的基本功能和边界条件。"""

    # --- 基本功能测试 ---
    @tilelang.jit(out_idx=[-1])
    def kernel_fn(M, N, block_M, block_N, dtype="float16"):
        @T.prim_func
        def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
            C: T.Tensor((M, N), dtype),
        ):
            with T.Kernel(..., is_npu=True) as (cid, vid):
                a_ub = T.alloc_ub((block_M, block_N), dtype)
                b_ub = T.alloc_ub((block_M, block_N), dtype)
                c_ub = T.alloc_ub((block_M, block_N), dtype)
                with T.Scope("V"):
                    T.copy(A[...], a_ub)
                    T.copy(B[...], b_ub)
                    T.tile.api_name(c_ub, a_ub, b_ub)
                    T.copy(c_ub, C[...])
        return main

    # 构造输入
    a = torch.randn(M, N, dtype=torch.float16).npu()
    b = torch.randn(M, N, dtype=torch.float16).npu()

    # 运行 kernel
    func = kernel_fn(M, N, block_M, block_N)
    c = func(a, b)

    # Golden 对比
    expected = a + b  # 根据实际 API 修改
    torch.testing.assert_close(c, expected, rtol=1e-3, atol=1e-3)
    print("Test Passed!")


if __name__ == "__main__":
    test_api_name()
```

---

## 七、高效学习 Tips

1. **以 `ascend.py` 的 Docstring 为标杆**
   - `gemm_v0`、`set_flag`、`dump_tensor` 等函数的 Docstring 质量最高
   - 补其他文件时参考其格式和详细程度

2. **从 examples 反推参数约束**
   - 很多参数约束没有显式写在代码中，需从 `examples/` 的实际用法反推
   - 例：`T.tile.add` 的 buffer 大小对齐要求，从 example 中的 shape 选择推断

3. **用 assert 语句定位约束**
   - 代码中的 `assert` 直接暴露参数约束
   - 例：`assert kL0Size % 16 == 0` → 文档必须写明"kL0Size 必须是 16 的倍数"

4. **跳过 `@deprecated()` 标记的函数**
   - `brcb`、`wholereduce*`、`block_reduce*`、`bilinear_interpolation` 已废弃

5. **理解 `_pipe` 类型注解**
   ```python
   _pipe = Literal["fix", "mte1", "mte2", "mte3", "m", "v", "s"]
   ```
   这些是昇腾 NPU 的流水线阶段标识，文档中需逐一说明含义。

6. **pass_configs 是理解编译器自动化的关键**
   ```python
   TL_ASCEND_AUTO_SYNC      # 自动同步插入
   TL_ASCEND_MEMORY_PLANNING # 自动内存规划
   TL_ASCEND_AUTO_CV_COMBINE # 自动 CV 分离
   TL_ASCEND_AUTO_CV_SYNC    # 自动 CV 同步
   ```
   文档中需说明哪些 pass_configs 会影响 API 的使用方式。

---

## 八、产出检查清单

### Docstring 补全

- [ ] `allocate.py`：alloc_L1/L0A/L0B/L0C/ub 5 个 API 补全
- [ ] `ascend_tile.py`：所有非废弃 API 补全
- [ ] `__init__.py`：annotate_layout/use_swizzle/annotate_address 等补全
- [ ] `copy_op.py`：确认并补全
- [ ] `reduce_ascend.py`：确认并补全
- [ ] `customize.py`：确认并补全

### 独立接口文档

- [ ] 内存分配 API 文档
- [ ] 数据搬运 API 文档
- [ ] 矩阵计算 API 文档
- [ ] 向量计算 API 文档（按子类分组）
- [ ] 归约 API 文档
- [ ] 同步与控制 API 文档
- [ ] 核间通信 API 文档
- [ ] 调度原语文档

### 测试用例

- [ ] 每个核心 API 至少有 1 个基本功能测试
- [ ] 边界条件测试（shape 对齐、dtype 限制等）
- [ ] 所有测试通过 `bench_test.sh`

### 质量检查

- [ ] Docstring 格式一致性（Args/Returns/Raises/Example/Note）
- [ ] 参数类型标注与函数签名一致
- [ ] 数据范围描述经过验证
- [ ] 文档中的示例代码可以实际运行