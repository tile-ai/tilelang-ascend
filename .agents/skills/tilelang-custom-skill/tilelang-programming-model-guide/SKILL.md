---
name: tilelang-programming-model-guide
description: TileLang Ascend Developer/Expert 模式选择与 pass_configs 配置指南。当需要确定编程模式、配置 pass_configs、或在两种模式之间转换时触发。API 详情请参考 tilelang-api-best-practices skill。
---

# TileLang Ascend 编程模式与 pass_configs 指南


本 skill 帮助 agent 选择配置、保留已有控制方式并检查模式转换。默认值、API 语义和限制以
[公共语言参考](../../../../docs/language_ref/primitives.md#ascend-compilation-options-and-cv-scopes)
为准；可运行写法从文末的仓库示例选取，不在 skill 中维护另一份 API 合同。

---

## 1. 模式对比



| 维度 | Developer 模式 | Expert 模式 |
|------|---------------|-------------|
| **内存分配** | `T.alloc_shared` / `T.alloc_fragment` | `T.alloc_L1` / `T.alloc_ub` / `T.alloc_L0A/L0B/L0C` |
| **计算表达** | `T.Parallel` + 符号运算 | `T.tile.xxx` 扩展原语 |
| **作用域** | 编译器自动分离 Cube/Vector | 手动 `with T.Scope("C"/"V")` |
| **同步** | 编译器自动插入 | 手写 LOCK DSL，由预处理器生成底层 flag |
| **CV 交互** | 可将 workspace 与 vid 的处理交给编译器，见 §3.1.1 | 显式 GM `workspace` + 手动 `vid` 二分 |
| **pass_configs** | 按 §2.2 配置同步与内存规划，沿用默认 C/V 划分 | 手写 scope 可沿用默认 C/V 划分；全手动时显式关闭 |
| **适用场景** | 大多数算子，跨平台兼容 | 极致性能优化，需要底层控制 |
| **示例目录** | `examples/developer_mode/` | `examples/flash_attention/fa_opt/flash_attn_bhsd_expert_*.py` |

**混合模式**：Developer 主体 + 少量 Ascend 专属 `T.tile.xxx`。仍使用 Developer 的
pass_configs，由编译器生成 resource scope 和同步；不要因为用了 `T.tile.*` 就关闭 CombineCV。

---

## 2. pass_configs 详解（核心）

配置默认值与 scope 兼容规则以 [公共语言参考](../../../../docs/language_ref/primitives.md#ascend-compilation-options-and-cv-scopes) 为准。

### 2.1 先判断执行侧，再检查依赖

- 使用默认配置，不生成冗余的 `TL_ASCEND_AUTO_CV_COMBINE: True`。
- 遇到已有显式 scope 时，检查其中的操作归属；不要仅因手写 scope 就关闭 CombineCV。
- 只有设计要求关闭自动划分时才显式设置 `False`，并检查所有硬件操作都已放入正确 scope。
- 自动划分执行侧不代表数据依赖已同步；分别检查核内依赖和跨 C/V 的 workspace 交互。

### 2.2 按场景选择 pass_configs

沿用用户已选模式；新算子尚未确定模式时先确认。不要仅因为 API 名称包含 `T.tile.*`、
kernel 含有 GEMM，或已经手写 scope，就替用户切换整套配置。

**Developer/Hybrid 纯 Vector 或纯 Cube 算子**：
```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}
```

**存在需要自动同步的 C/V workspace 交互时**，先对照同类示例确认该交互可被识别，再追加：
```python
pass_configs[tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC] = True
```

检查生成代码中的通知与等待是否覆盖实际读写和 buffer 复用；不能仅凭开关开启就删除
现有跨核同步。单纯含有 GEMM 不意味着存在跨核依赖。

**已有手写 scope 或手动同步**：保留其控制方式，逐项检查配置。仅当设计明确要求关闭
所有自动处理时使用下面的全手动配置，不把它当成手写 scope 的必需配置：
```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
}
```

### 2.3 C/V 输入契约

检查生成代码时，确认 Cube/Vector 操作均位于对应 scope，纯 Vector kernel 也不例外。
遇到 GM 标量写入、外部调用、原始代码或无明确执行侧的屏障，先按
[公共 scope 规则](../../../../docs/language_ref/primitives.md#ascend-compilation-options-and-cv-scopes)
检查显式归属要求；不要用自动划分代替 GM 标量写入的显式归属，也不能关闭验证绕过。
若需要诊断自动划分，参见 [编译器的执行侧分析](../../../../docs/ascend/compiler_managed_vector_mask.md#resource-scope-contract)；
不要在其他 pass 中重复推导执行核。

### 2.4 Vector mask reuse 安全开关

怀疑相邻 Vector 操作复用了错误的 mask 状态时，在最小复现中追加以下配置进行对照。
不要用单项 dict 替换已有同步和内存配置：

```python
pass_configs[tilelang.PassConfigKey.TL_ASCEND_VECTOR_MASK_REUSE] = False
```

若关闭后结果恢复，保留两种配置的生成代码与输入，继续定位状态传递问题；不要据此认定
dtype、布局或同步都已正确。适用平台和开关语义见
[mask 复用与保守配置](../../../../docs/ascend/compiler_managed_vector_mask.md#scope-and-safe-fallback)。

---

## 3. 模式转换规则（Expert → Developer）

### 3.1 转换步骤

1. **配置 pass_configs**：按 §2.2 选择；恢复默认 C/V 划分，开启 AUTO_SYNC、MEMORY_PLANNING，
   有核间依赖时检查 AUTO_CV_SYNC 是否覆盖对应交互
2. **内存分配**：`T.alloc_L1` → `T.alloc_shared`，`T.alloc_L0C` → `T.alloc_fragment`，`T.alloc_ub` → `T.alloc_shared`
3. **检查作用域**：合法手写 scope 可以保留；只在归属可推导且移除有助于简化时删除，保留不透明调用所需的 scope
4. **检查同步**：在生成代码中确认自动同步覆盖对应依赖后，才移除该部分手动同步；跨核交互单独检查
5. **计算转换**（可选）：`T.tile.exp(dst, src)` → `for i,j in T.Parallel(...): dst[i,j] = T.exp(src[i,j])`
6. **检查内存规划**：确认显式地址没有承载必要的共享或布局约定，再移除 `T.annotate_address`；转换后运行正确性测试

### 3.1.1 可选改造：由编译器管理 workspace / vid

简化 Developer 模式的 Cube/Vector 交互时，可选用不显式传 GM `workspace`、不手动二分 `vid` 的写法。
这与默认开启 CombineCV 是不同选择，不要因此自动修改现有 kernel 的 `threads`。前提链：

```
threads=2  ──►  vid 消除  ──►  workspace 消除
```

四步改造：
1. **加 `threads=2`**：`T.Kernel(block_num, is_npu=True) as (cid, vid)` → `T.Kernel(block_num, threads=2, is_npu=True) as (cid)`（编译器自动并行 2 个 V 核，这是消 vid 的前提）。
2. **删 `workspace_idx`**：`@tilelang.jit(out_idx=[N], workspace_idx=[...], ...)` → `@tilelang.jit(out_idx=[N], ...)`，并删除 kernel 签名里的 `workspace_*` 参数。
3. **去 vid 偏移**：`v_block` 不再 `// 2`，循环恢复整程 `range(BI)`，删除全部 `vid * ...` 索引偏移。
4. **合并源代码中的搬运**：将两次经 `workspace` 的搬运表达为一次片上 buffer 间的 `T.copy`；编译器可生成 GM 中转，不能将这种写法解释为硬件片上直连。

> 采用此改造时，先对照 `examples/developer_mode/sparse_flash_attn_developer_vid_reduce.py`。
> [mode-examples.md §6](references/mode-examples.md#6-cv-融合--推荐写法消除-workspace--vidthreads2) 辅助解释映射；完整代码以仓库可执行示例为准。


### 3.2 转换对照表

| Expert 写法 | Developer 写法 |
|-------------|---------------|
| `T.alloc_L1(shape, dtype)` | `T.alloc_shared(shape, dtype)` |
| `T.alloc_ub(shape, dtype)` | `T.alloc_shared(shape, dtype)` |
| `T.alloc_L0A/L0B(shape, dtype)` | 删除（`gemm_v0` 内部处理） |
| `T.alloc_L0C(shape, dtype)` | `T.alloc_fragment(shape, dtype)` |
| `with T.Scope("C"): ...` | 可保留；归属可推导时可省略 |
| 核内 barrier / flag | 确认自动同步覆盖对应依赖后移除 |
| `T.set_cross_flag/T.wait_cross_flag(...)` | 单独确认跨核依赖与 buffer 复用均被覆盖后移除 |
| `T.tile.exp(dst, src)` | `for i,j in T.Parallel(...): dst[i,j] = T.exp(src[i,j])` 或保留 |
| `T.annotate_address({...})` | 确认不再需要显式地址约定后移除 |
| `@jit(..., workspace_idx=[...])` + 签名 `workspace_*` 参数 | 采用 §3.1.1 的编译器管理 workspace 写法时移除 |
| `T.Kernel(..., is_npu=True) as (cid, vid)` | 采用 §3.1.1 时改为 `T.Kernel(..., threads=2, is_npu=True) as cid` |
| `T.copy(buf, ws[cid,...])` + `T.copy(ws[cid,vid*..], buf2)` 两跳 | `T.copy(buf, buf2)`，由编译器安排中转 |

---

## 4. 示例代码与代码对比

| 模式 | 目录 | 说明 |
|------|------|------|
| Developer | `examples/developer_mode/` | GEMM、elementwise 等 |
| Developer（编译器管理 workspace/vid） | `examples/developer_mode/sparse_flash_attn_developer_vid_reduce.py` 与 `sparse_flash_attn_developer.py` | 对照显式与编译器管理的写法 |
| Expert | `examples/gemm/example_gemm_intrinsic.py`、`examples/flash_attention/fa_opt/flash_attn_bhsd_expert_*.py` | 极致性能优化 |
| 混合（核间流水线） | `examples/flash_attention/flash_attn_bhsd_cc_sync.py`、`examples/flash_attention/fa_opt/flash_attn_bhsd_auto_pipeline_*.py` | FA 核间流水线 |
| 纯 Vector | `examples/elementwise/`、`examples/softmax/` | 无 Cube 操作 |
| CV 融合 | `examples/dequantize_gemm/`、`examples/quant_batch_matmul/` | Vector 计算 + Cube GEMM |

**完整代码对比**（Developer vs Expert）：
- → [mode-examples.md](references/mode-examples.md)
- 包含 GEMM、Flash Attention、Softmax、CV 融合（消除 workspace/vid 推荐写法 §6 / workspace+vid 兜底写法 §7） 等示例
