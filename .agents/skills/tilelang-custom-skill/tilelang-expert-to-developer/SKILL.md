---
name: tilelang-mode-guide
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
| **同步** | 编译器自动插入 | 手动 `T.barrier_all` / `T.set_flag` / `T.wait_flag` |
| **CV 交互** | 默认消除 workspace+vid（`threads=2` + 片上直连，见 §3.1.1） | 显式 GM `workspace` + 手动 `vid` 二分 |
| **pass_configs** | **全部开启** | **全部关闭或不设** |
| **适用场景** | 大多数算子，跨平台兼容 | 极致性能优化，需要底层控制 |
| **示例目录** | `examples/developer_mode/` | `examples/flash_attention/fa_opt/flash_attn_bhsd_expert_*.py` |

**混合模式**：Developer 主体 + 少量 Expert / Ascend 专属 `T.tile.xxx`。使用 Developer 的 pass_configs，不写 `T.Scope` 和手动同步。大多数实际算子使用混合模式。

---

## 2. pass_configs 详解（核心）



### 2.1 五个 Ascend 专用开关

```python
import tilelang

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,        # ① 自动核内同步
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,   # ② 自动内存规划
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,   # ③ 自动CV分离
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,      # ④ 自动核间同步
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC_VS: True,      # ⑤ 自动核内同步（VS 补位版）
}
```

#### ① TL_ASCEND_AUTO_SYNC（自动核内同步）

- **底层 key**：`"tl.ascend_auto_sync"`，默认 False
- **功能**：自动插入部分跨流水线同步（MTE2→V、V→MTE3 等）及 `PIPE_ALL` barrier，但**不覆盖** V→V 同流水线与 S↔其他 流水线场景
- **与 CCE 的关系**：① 单独使用时依赖 CCE 编译器默认 `--cce-auto-sync=on` 掩盖其盲区（CCE 会补齐①未覆盖的同步），结果正确但 CCE 同步策略偏保守、性能较差；一旦设 `TL_CCE_AUTO_SYNC=off`，① 的盲区暴露，需 ⑤ 补位
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

> 注意：避免在开启 AUTO_CV_COMBINE 同时手写 `T.Scope`，可能会导致编译器无法正确处理代码

#### ④ TL_ASCEND_AUTO_CV_SYNC（自动核间同步）

- **底层 key**：`"tl.ascend_auto_cross_core_sync"`，默认 False
- **功能**：自动在 Cube Scope 和 Vector Scope 之间插入 `T.set_cross_flag`/`T.wait_cross_flag`
- **开启时**：无需手写核间同步
- **关闭时**：必须手动管理核间同步

#### ⑤ TL_ASCEND_AUTO_SYNC_VS（自动核内同步 - VS 补位版）

- **底层 key**：`"tl.ascend_auto_sync_vs"`，默认 False
- **功能**：精准补位 ① 未覆盖的同步场景，仅跟踪 `PIPE_V` / `PIPE_S` / `PIPE_MTE2` / `PIPE_MTE3` 四条流水线，在 V→V 同流水线插 `PipeBarrier_V`、在 S↔其他 流水线之间插 `EventPair`（不含 S→S）
- **与 ① 的关系**：**互补协同，非互斥**。Pipeline 中 ① 先执行、⑤ 后执行补位；二者可同时开启。纯 V/S 算子可只开 ⑤（配套 `TL_CCE_AUTO_SYNC=off`）；涉及 MTE2→V / V→MTE3 等更复杂跨流水线依赖时建议 ①⑤ 同开
- **开启时**：无需手写 `T.barrier_all()`、`T.set_flag`/`T.wait_flag`（仅限 V/S 流水线场景）
- **关闭时**：必须手动插入所有同步点
- **适用场景**：纯 Vector 算子（elementwise、softmax 等），不涉及 Cube/Vector 核间协作

> **配套环境变量 `TL_CCE_AUTO_SYNC`**：开启 ⑤ 时，需同时设置环境变量 `TL_CCE_AUTO_SYNC=off` 关闭 CCE 编译器自带的自动同步，避免 CCE 保守同步与 TileLang pass 重复插入同步指令。默认 `on`（CCE 编译器自动插同步）。详见 §2.3。

### 2.2 按场景选择 pass_configs

| 场景 | AUTO_SYNC | MEMORY_PLANNING | AUTO_CV_COMBINE | AUTO_CV_SYNC | 手动 Scope |
|------|-----------|-----------------|-----------------|--------------|------------|
| **纯 Vector 算子**（elementwise, softmax） | ✅ | ✅ | ❌ | ❌ | ❌ |
| **Developer GEMM**（完全自动） | ✅ | ✅ | ✅ | ✅ | ❌ |
| **Developer Flash Attention**（核间流水线） | ✅ | ✅ | ✅ | ✅ | ❌ |
| **Developer CV 融合**（Vector计算+Cube GEMM） | ✅ | ✅ | ✅ | ✅ | ❌ |
| **混合模式 CV 融合** | ✅ | ✅ | ❌ | ❌ | ✅ |

> **Developer Flash Attention / Developer CV 融合**：默认消除 workspace+vid（`threads=2` + 片上直连），写法见 §3.1.1 与 [mode-examples.md §6](references/mode-examples.md#6-cv-融合--推荐写法消除-workspace--vidthreads2)。
| **Expert 极致性能** | ❌ | ❌ | ❌ | ❌ | ✅ |

**纯 Vector 算子**（来自 Programming Guide §2.2）：
```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
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

### 2.3 配套环境变量

除上述 `pass_configs` 开关外，Ascend 编译流程还受以下环境变量控制：

#### TL_CCE_AUTO_SYNC（CCE 编译器自动同步）

- **位置**：`tilelang/jit/adapter/libgen.py`
- **默认值**：`on`（CCE 编译器自动插同步）
- **取值**：`on` / `off`（大小写不敏感）
- **功能**：控制 CCE 编译器（`bisheng`）的 `--cce-auto-sync` 开关。设为 `off` 时，向编译命令追加 `--cce-auto-sync=off`，关闭 CCE 编译器自带的自动同步插入
- **与 ⑤ `TL_ASCEND_AUTO_SYNC_VS` 的关系**：开启 `TL_ASCEND_AUTO_SYNC_VS` 时**必须**同时设 `TL_CCE_AUTO_SYNC=off`，否则 CCE 编译器与 TileLang pass 会重复插入同步指令，导致运行时死锁或性能劣化
- **使用方式**：

```bash
# 启用 VS pass 时必须配套设置
export TL_CCE_AUTO_SYNC=off
```

```python
# Python 中也可设置
import os
os.environ["TL_CCE_AUTO_SYNC"] = "off"

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC_VS: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}
```

> 注意：单独使用 ① `TL_ASCEND_AUTO_SYNC` 时通常**无需**设置 `TL_CCE_AUTO_SYNC=off`——① 不覆盖 V→V 与 S↔其他 场景，其盲区恰好由 CCE 默认 `--cce-auto-sync=on` 掩盖（结果正确但 CCE 同步偏保守、性能较差）。一旦设 `TL_CCE_AUTO_SYNC=off` 关闭 CCE 自动同步，① 的盲区暴露，**必须**同时开启 ⑤ `TL_ASCEND_AUTO_SYNC_VS` 补位，否则会出现同步缺失。

---

## 3. 模式转换规则（Expert → Developer）

### 3.1 转换步骤

1. **开启 pass_configs**：添加完整 4 个 True 开关
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
