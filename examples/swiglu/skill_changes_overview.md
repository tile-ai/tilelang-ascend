# Skill 文件改动说明（A 部分：已推到 fork 的改动）

## 1. 概述

| 项 | 值 |
|---|---|
| 来源 | fork 仓库 `3053203568-del/tilelang-ascend:ascendc_pto` 相对 upstream `tile-ai/tilelang-ascend:ascendc_pto` |
| 涉及文件数 | 2 |
| 净新增行数 | +128（+26 / +102） |
| 改动主题 | chunk/split 类算子（如 SwiGLU `silu(x0)*x1`）的单输入 kernel 优化模式 |
| 改动性质 | 新增章节 / 子模式，无删除、无修改既有内容 |

## 2. 涉及文件清单

| 文件 | 行数 | 所属 Skill | 所属章节 |
|------|------|-----------|---------|
| `.agents/skills/tilelang-op-develop/references/coding-conventions.md` | +26 | tilelang-op-develop | §2 数据搬运索引（之后插入） |
| `.agents/skills/tilelang-perf-optimization/references/optimization-guide.md` | +102 | tilelang-perf-optimization | §2.12 之后（作为子模式插入） |

## 3. 改动动机

针对 TileLang-Ascend 算子开发中一类高频性能反模式：

```python
# 反模式：N 次 host 内存拷贝
x0, x1 = input.chunk(2, dim=-1)
x0 = x0.contiguous()    # 拷贝 1：遍历整个子张量
x1 = x1.contiguous()    # 拷贝 2：同上
output = kernel(x0, x1)
```

每次 `.contiguous()` 都是一次完整的 host 内存遍历拷贝。N 个子张量 = N 次拷贝，大 shape 下 host 拷贝耗时可接近甚至超过 kernel 本身。

**核心解法**：传单个完整输入 tensor 给 kernel，kernel 内部通过列/行偏移（如 `X[row, half_k + col]`）读取各子张量数据，将 N 次 host 拷贝降为 0 次（dim=-1 快路径）或 1 次 permute+contiguous（dim≠-1 慢路径）。

## 4. 详细改动

### 4.1 `coding-conventions.md`（+26 行）

**commit**：`c6ed07da` Enhance coding conventions with split index mode details

**插入位置**：§2 "数据搬运索引" 章节中 "易错点（仅回退写法）" 段落之后，§2.1 "T.copy 多维切片的硬件限制" 之前。

**新增内容摘要**：新增"✅ 单输入 split 索引模式"段落，给出 SwiGLU 类算子的 kernel 端标准写法：

```python
@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def kernel(block_M, block_N, K, dtype="float16"):
    half_k = K // 2
    M = T.symbolic("M")

    @T.prim_func
    def main(
        X: T.Tensor((M, K), dtype),       # 完整输入
        Y: T.Tensor((M, half_k), dtype),  # 输出
    ):
        with T.Kernel(...) as (cid, vid):
            x0_ub = T.alloc_ub((rows, block_N), dtype)
            x1_ub = T.alloc_ub((rows, block_N), dtype)

            # x0 = X[:, :half_k]，x1 = X[:, half_k:]
            T.copy(X[row, col], x0_ub)
            T.copy(X[row, half_k + col], x1_ub)
            # ... silu(x0) * x1 ...
```

并说明 Host 适配层：dim=-1 时仅 reshape（零拷贝）；dim≠-1 时 permute+contiguous（1 次拷贝）。

### 4.2 `optimization-guide.md`（+102 行）

**commit**：`581f3ea7` Enhance optimization guide with chunk/split strategies

**插入位置**：§2.12 之后，"减少 transpose 优化" 之前。用 `<!-- END 2.12 子模式 -->` 注释标记结尾，便于整段回退删除。

**新增内容结构**（5 个子章节）：

| 子章节 | 内容 |
|--------|------|
| 适用场景 | `torch.chunk` / `torch.split` + `.contiguous()` 反模式识别 |
| 性能问题 | N 个子张量 = N 次 host 拷贝的成本分析 |
| 核心思路 | 单输入 kernel + 列/行偏移读取 |
| Host 端改造 | 反模式 → 正模式代码对照（4 行 → 1 行） |
| Kernel 端改造 | 双输入 kernel → 单输入 + 列偏移 kernel 代码对照，附 `half_k + col` 符号偏移说明 |
| Host 端适配层 | dim=-1 快路径（reshape 零拷贝）vs dim≠-1 慢路径（1 次 permute+contiguous）完整代码 |
| 适用条件 | 3 项判定条件表格（element-wise / 末维等分 / T.copy 符号列偏移支持） |
| 检查清单 | 6 项 checkbox 验证优化完整性 |

## 5. 关联 commit

fork 上 10 个 commit 中，**仅以下 2 个对 A 部分有净贡献**：

| commit | message | 净改动 |
|--------|---------|--------|
| `c6ed07da` | Enhance coding conventions with split index mode details | coding-conventions.md +26 |
| `581f3ea7` | Enhance optimization guide with chunk/split strategies | optimization-guide.md +102 |

**关于其余 8 个 commit 的说明**：

fork 的 10 个 commit 中有 7 个改了 `SKILL.md` / `checklist.md`，但呈现"先加后删"的反复修改模式，最终 fork 相对 upstream 在这两个文件上净差异为 0。例如：
- `d9f5fbad` 给 `tilelang-op-design/SKILL.md` +1，随后 `927bbc83` -1 删除
- `c4caa155` 给 `tilelang-op-develop/SKILL.md` +2，随后 `2e2af267`/`9efa8495` 删除
- `1979bbaf` 给 `checklist.md` +1，随后 `34589c5c` -1 删除

第 10 个 commit `e168176b`（Add SwiGLU activation kernel）改的是 `examples/swiglu/swiglu.py`，不属于 skills 范围。

## 6. 影响范围

| Skill | 受影响的工作流阶段 | 行为变化 |
|-------|------------------|---------|
| `tilelang-op-develop` | 代码生成 | 生成 chunk/split 类算子 kernel 时遵循单输入 + 列偏移模式，避免 host 端 chunk + contiguous |
| `tilelang-perf-optimization` | 性能优化 / 反模式排查 | 识别 `chunk()/split() + .contiguous() × N` 反模式，按子模式方案给出优化建议 |

## 7. 交叉引用

两份文档互相引用：
- `coding-conventions.md` 引用 `optimization-guide.md §2.12 子模式` 作为完整模式参考
- `optimization-guide.md` 是完整模式定义，`coding-conventions.md` 是其 kernel 端速查版

## 8. 回退方法

如需回退 A 部分（删除这两个章节）：

```bash
# coding-conventions.md：删除 "✅ 单输入 split 索引模式" 段落
# optimization-guide.md：删除从 "---" 到 "<!-- END 2.12 子模式 -->" 之间的内容
git checkout origin/ascendc_pto -- \
  .agents/skills/tilelang-op-develop/references/coding-conventions.md \
  .agents/skills/tilelang-perf-optimization/references/optimization-guide.md
```
