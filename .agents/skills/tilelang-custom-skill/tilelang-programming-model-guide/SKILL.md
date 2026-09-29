---
name: tilelang-programming-model-guide
description: TileLang Ascend Developer/Expert 模式选择与 pass_configs 配置指南。当需要确定编程模式、配置 pass_configs、或在两种模式之间转换时触发。API 详情请参考 tilelang-api-best-practices skill。
---

# TileLang Ascend 编程模式与 pass_configs 指南


 **API 用法详情**（内存分配、计算原语、同步原语等）请参考 **tilelang-api-best-practices** skill，本文档不再重复。

---

## 1. 模式对比



| 维度 | Developer 模式 | Expert 模式 |
|------|---------------|-------------|
| **内存分配** | `T.alloc_shared` / `T.alloc_fragment` | `T.alloc_L1` / `T.alloc_ub` / `T.alloc_L0A/L0B/L0C` |
| **计算表达** | `T.Parallel` + 符号运算 | `T.tile.xxx` 扩展原语 |
| **作用域** | 编译器自动分离 Cube/Vector | 手动 `with T.Scope("C"/"V")` |
| **同步** | 编译器自动插入 | 手写 LOCK DSL，由预处理器生成底层 flag |
| **CV 交互** | 默认消除 workspace+vid（`threads=2` + 片上直连，见 §3.1.1） | 显式 GM `workspace` + 手动 `vid` 二分 |
| **pass_configs** | 按 §2.2 开启相关自动 passes；无显式 scope 时 CombineCV 必开 | 全部关闭或不设 |
| **适用场景** | 大多数算子，跨平台兼容 | 极致性能优化，需要底层控制 |
| **示例目录** | `examples/developer_mode/` | `examples/flash_attention/fa_opt/flash_attn_bhsd_expert_*.py` |

**混合模式**：Developer 主体 + 少量 Ascend 专属 `T.tile.xxx`。仍使用 Developer 的
pass_configs，由编译器生成 resource scope 和同步；不要因为用了 `T.tile.*` 就关闭 CombineCV。

---

## 2. pass_configs 详解（核心）



### 2.1 四个 Ascend 专用开关

```python
import tilelang

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,        # ① 自动核内同步
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,   # ② 自动内存规划
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,   # ③ 自动CV分离
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,      # ④ 自动核间同步
}
```

#### ① TL_ASCEND_AUTO_SYNC（自动核内同步）

- **底层 key**：`"tl.ascend_auto_sync"`，默认 False
- **功能**：自动在数据搬运和计算之间插入 `T.barrier_all()` 等同步指令
- **开启时**：无需手写 `T.barrier_all()`、`T.set_flag`/`T.wait_flag`
- **关闭时**：必须手动插入所有同步点

#### ② TL_ASCEND_MEMORY_PLANNING（自动内存规划）

- **底层 key**：`"tl.ascend_memory_planning"`，默认 False
- **功能**：自动分析 buffer 生命周期，实现片上内存复用
- **开启时**：自动复用 buffer 空间，减少片上内存占用
- **关闭时**：需手动通过 `T.annotate_address` 规划内存地址


#### ③ TL_ASCEND_AUTO_CV_COMBINE（自动 CV 分离）

- **底层 key**：`"tl.ascend_auto_cv_combine"`，默认 False
- **功能**：自动将 kernel 中的 Cube 操作和 Vector 操作分离到不同的执行核
- **开启时**：无需手写 `with T.Scope("C")` / `with T.Scope("V")`，编译器根据 buffer 类型和所用原语自动识别
- **关闭时**：必须手动用 `T.Scope` 标注每段代码的执行域

> 默认选择一种 ownership 写法：自动模式由 CombineCV 生成 scope，手动模式全部显式标注。
> CombineCV 可以保留已有的同类显式 scope，但不要无必要地混写，更不能做 C/V 冲突嵌套。

#### ④ TL_ASCEND_AUTO_CV_SYNC（自动核间同步）

- **底层 key**：`"tl.ascend_auto_cross_core_sync"`，默认 False
- **功能**：自动在 Cube Scope 和 Vector Scope 之间插入 `T.set_cross_flag`/`T.wait_cross_flag`
- **开启时**：无需手写核间同步
- **关闭时**：仅当 kernel 存在 C/V 依赖时，必须手动管理对应核间同步

### 2.2 按场景选择 pass_configs

| 场景 | AUTO_SYNC | MEMORY_PLANNING | AUTO_CV_COMBINE | AUTO_CV_SYNC | 手动 Scope |
|------|-----------|-----------------|-----------------|--------------|------------|
| **Developer/Hybrid 纯 Vector**（elementwise, softmax） | ✅ | ✅ | ✅ | ❌ | ❌ |
| **Developer GEMM**（完全自动） | ✅ | ✅ | ✅ | ✅ | ❌ |
| **Developer Flash Attention**（核间流水线） | ✅ | ✅ | ✅ | ✅ | ❌ |
| **Developer CV 融合**（Vector计算+Cube GEMM） | ✅ | ✅ | ✅ | ✅ | ❌ |
| **Expert / 手动 ownership** | 按设计，通常 ❌ | 按设计，通常 ❌ | ❌ | 按设计，通常 ❌ | ✅，所有 resource-specific work |

> **Developer Flash Attention / Developer CV 融合**：默认消除 workspace+vid（`threads=2` + 片上直连），写法见 §3.1.1 与 [mode-examples.md §6](references/mode-examples.md#6-cv-融合--推荐写法消除-workspace--vidthreads2)。

**纯 Vector 算子**（来自 Programming Guide §2.2）：
```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
}
```

**Developer GEMM / Developer CV 融合**（推荐配置）：
```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,  # 自动分离 Cube/Vector
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,        # 自动核内同步
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,  # 自动内存规划
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,     # 自动核间同步
}
```

**Expert 全手动**：
```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
}
```

### 2.3 C/V 输入契约

所有 resource-specific Ascend hardware work 都必须有明确 owner，纯 Vector kernel 也不例外：

- `TL_ASCEND_AUTO_CV_COMBINE=True`：CombineCV 用共用分类器生成 C/V scope；适合不手写
  resource ownership 的 Developer / Hybrid kernel。
- `TL_ASCEND_AUTO_CV_COMBINE=False`：作者必须把所有资源相关工作放入显式
  `T.Scope("C")` 或 `T.Scope("V")`；适合 Expert / 手动 ownership。
- outer 只保留 resource-independent 或两侧共同执行的 control。无法分类的 opaque extern 和
  raw source 必须放在正确的显式 scope。
- V→V / C→C 同类嵌套允许；C→V / V→C 冲突嵌套拒绝。

因此，“关闭 CombineCV 但保留无 scope 的纯 Vector kernel”不是受支持的第三种模式。
完整 compiler contract 见 `docs/ascend/compiler_managed_vector_mask.md` 的
Resource-scope contract。

### 2.4 Vector mask reuse 安全开关

`TL_ASCEND_VECTOR_MASK_REUSE` 默认 `True`，仅用于 A2/A3 AscendC/auto 的
compiler-managed Vector mask reuse。设为 `False` 时，每条已选择的 raw terminal 都保守地重建
完整 required mask；Instruction Selection 和严格 C/V verifier 仍然启用。

把这一项追加到 §2.2 中当前模式已经合法的 `pass_configs`，不要用单项 dict 替换原配置：

```python
pass_configs[tilelang.PassConfigKey.TL_ASCEND_VECTOR_MASK_REUSE] = False
```

这个开关用于隔离或临时规避 mask-state contract 问题，不能绕过不支持的 dtype / ABI，也不能
替代正确的 resource scope。详细语义见 `docs/ascend/compiler_managed_vector_mask.md`。

---

## 3. 模式转换规则（Expert → Developer）

### 3.1 转换步骤

1. **开启 pass_configs**：按 §2.2 选择；CV kernel 开四项，纯 Vector Developer kernel 开
   AUTO_SYNC、MEMORY_PLANNING、AUTO_CV_COMBINE
2. **内存分配**：`T.alloc_L1` → `T.alloc_shared`，`T.alloc_L0C` → `T.alloc_fragment`，`T.alloc_ub` → `T.alloc_shared`
3. **删除作用域**：移除 `with T.Scope("C")` / `with T.Scope("V")`
4. **删除同步**：移除 `T.barrier_all()`、`T.set_flag`/`T.wait_flag`、`T.set_cross_flag`/`T.wait_cross_flag`
5. **计算转换**（可选）：`T.tile.exp(dst, src)` → `for i,j in T.Parallel(...): dst[i,j] = T.exp(src[i,j])`
6. **删除手动内存规划**：移除 `T.annotate_address`

### 3.1.1 Developer 模式 CV 交互：优先消除 workspace / vid

Developer 模式下 Cube↔Vector 交互**默认不写 GM `workspace`、不手动二分 `vid`**，交给编译器自动处理。前提链（按序，不可跳级）：

```
threads=2  ──►  vid 消除  ──►  workspace 消除
```

四步改造：
1. **加 `threads=2`**：`T.Kernel(block_num, is_npu=True) as (cid, vid)` → `T.Kernel(block_num, threads=2, is_npu=True) as (cid)`（编译器自动并行 2 个 V 核，这是消 vid 的前提）。
2. **删 `workspace_idx`**：`@tilelang.jit(out_idx=[N], workspace_idx=[...], ...)` → `@tilelang.jit(out_idx=[N], ...)`，并删除 kernel 签名里的 `workspace_*` 参数。
3. **去 vid 偏移**：`v_block` 不再 `// 2`，循环恢复整程 `range(BI)`，删除全部 `vid * ...` 索引偏移。
4. **片上直连**：原「片上 buffer ↔ `workspace[cid,...]` ↔ 另一片上 buffer」两跳 GM 往返，合并为片上 `T.copy` 一跳；中转/同步交给四个 pass。

> 完整模板、映射表、代码骨架、自检清单与回退条件见 [mode-examples.md §6](references/mode-examples.md#6-cv-融合--推荐写法消除-workspace--vidthreads2)。
> **复杂同步/多版本流水场景**可回退保留 workspace+vid 写法（见 [mode-examples.md §7](references/mode-examples.md#7-cv-融合--workspace--vid-写法复杂场景兜底)）。


### 3.2 转换对照表

| Expert 写法 | Developer 写法 |
|-------------|---------------|
| `T.alloc_L1(shape, dtype)` | `T.alloc_shared(shape, dtype)` |
| `T.alloc_ub(shape, dtype)` | `T.alloc_shared(shape, dtype)` |
| `T.alloc_L0A/L0B(shape, dtype)` | 删除（`gemm_v0` 内部处理） |
| `T.alloc_L0C(shape, dtype)` | `T.alloc_fragment(shape, dtype)` |
| `with T.Scope("C"): ...` | 直接写代码（编译器自动分离） |
| `T.barrier_all()` | 删除（编译器自动插入） |
| `T.set_flag/T.wait_flag(...)` | 删除 |
| `T.set_cross_flag/T.wait_cross_flag(...)` | 删除 |
| `T.tile.exp(dst, src)` | `for i,j in T.Parallel(...): dst[i,j] = T.exp(src[i,j])` 或保留 |
| `T.annotate_address({...})` | 删除（开启 MEMORY_PLANNING） |
| `@jit(..., workspace_idx=[...])` + 签名 `workspace_*` 参数 | 删除（CV 交互改片上直连，见 §3.1.1） |
| `T.Kernel(..., is_npu=True) as (cid, vid)` | `T.Kernel(..., threads=2, is_npu=True) as (cid)`（消 vid 前提） |
| `T.copy(buf, ws[cid,...])` + `T.copy(ws[cid,vid*..], buf2)` 两跳 | `T.copy(buf, buf2)` 片上一跳直连 |

---

## 4. 示例代码与代码对比

| 模式 | 目录 | 说明 |
|------|------|------|
| Developer | `examples/developer_mode/` | GEMM、elementwise 等 |
| Developer（消除 workspace/vid） | `examples/developer_mode/sparse_flash_attn_developer_vid_reduce.py`（新）vs `sparse_flash_attn_developer.py`（旧） | `threads=2` + 片上直连，逐行对照消除范式 |
| Expert | `examples/gemm/example_gemm_intrinsic.py`、`examples/flash_attention/fa_opt/flash_attn_bhsd_expert_*.py` | 极致性能优化 |
| 混合（核间流水线） | `examples/flash_attention/flash_attn_bhsd_cc_sync.py`、`examples/flash_attention/fa_opt/flash_attn_bhsd_auto_pipeline_*.py` | FA 核间流水线 |
| 纯 Vector | `examples/elementwise/`、`examples/softmax/` | 无 Cube 操作 |
| CV 融合 | `examples/dequantize_gemm/`、`examples/quant_batch_matmul/` | Vector 计算 + Cube GEMM |

**完整代码对比**（Developer vs Expert）：
- → [mode-examples.md](references/mode-examples.md)
- 包含 GEMM、Flash Attention、Softmax、CV 融合（消除 workspace/vid 推荐写法 §6 / workspace+vid 兜底写法 §7） 等示例
