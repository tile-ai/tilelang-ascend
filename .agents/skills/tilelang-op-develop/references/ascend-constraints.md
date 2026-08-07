# 技术约束清单（生成代码时必须遵守）

本文件定义 `tilelang-op-develop` 生成代码前/时必须遵守的强制规则。SKILL.md §3 代码生成
流程建立在这些约束之上；生成代码后会在上库前检查清单（[checklist.md](checklist.md) §0）
中逐项复核。

## 目录

- [1. 算子 kernel 划分原则（强制规则）](#1-算子-kernel-划分原则强制规则)
- [2. Host 侧 Buffer 操作约束（生成代码时必须遵守）](#2-host-侧-buffer-操作约束生成代码时必须遵守)

---

## 1. 算子 kernel 划分原则（强制规则）⭐

多 kernel 或 host 多路径方案必须保证支持域完整：保留覆盖全部声明输入域的通用
fallback，再按可判定的 dtype、shape、对齐性或输入拓扑增加有限快路径。具体 shape 或
perm 特化本身不是违规；没有 fallback、未命中即不支持才是设计错误。

生成前记录每条路径的适用谓词、fallback、语义等价依据和验证用例。若 design.md 没有
完整覆盖支持域，返回 `[DESIGN_ERROR]`；不要擅自缩小支持范围。

Edit 前必须从 design.md 读取 `[DISPATCH-COVERAGE]` 并在实现日志复核：

```text
[DISPATCH-COVERAGE-AUDIT]
supported_domain: <...>
generic_fallback: <...>
uncovered_input: none/<反例>
specialization_cases: <每条路径至少一个>
result: pass/fail
```

找出任何未覆盖输入时立即返回 `[DESIGN_ERROR]`，不得靠删除 case、缩小输入域或新增
无 fallback 的枚举分支继续。

---

## 2. Host 侧 Buffer 操作约束（生成代码时必须遵守）⭐

> **⚠️ 核心原则：算子的主要操作必须全部在 kernel 内实现，host 侧禁止触发 aclnn 调用**
>
> 算子的所有核心计算逻辑（包括数据搬运、数学运算、归约、归一化等）必须在 `@tilelang.jit` 装饰的 kernel 函数内部完成。**kernel 外部（host 侧 Python 代码）对 NPU 侧张量数据严禁以下行为**（约束范围覆盖 kernel 调用前的输入预处理和 kernel 调用后的输出后处理）：
>
> | # | 禁止行为 | 说明 | 典型反例 |
> |---|---------|------|---------|
> | 1 | 修改张量数据指针 | 禁止把输入/输出 tensor 重新绑定到另一个 tensor（改变 `data_ptr`）后传入 kernel | `x = y`（y 是另一个 tensor）后再传入 kernel |
> | 2 | 修改张量真实排布 | 禁止任何会触发真实数据拷贝/重排的操作 | `x.reshape(...).contiguous()`、`x.transpose(...).contiguous()`、`x.permute(...).contiguous()`、**`x_perm.reshape(-1)`（x_perm 非 contiguous 时等价于 `.contiguous()`）** |
> | 3 | 修改 buffer 真实内容 | 禁止在 host 侧直接改写 tensor 数据 | `x[:] = ...`、`x.add_(1)`、`torch.mul(x, 2, out=x)` |
> | 4 | 用新 buffer 作弊 | 禁止「创建新 buffer → host 侧处理 → 替换原 tensor」绕过限制 | `x = x.add(1)`（host 算完用新 tensor 顶替原输入） |
> | 5 | **隐式触发 aclnn 调用** | cann-bench 评测环境可能裁剪 aclnn 编译产物，以下操作在 NPU tensor 上会触发 aclnn 调用导致运行时失败：`torch.nn.functional.pad`/`cat`/`interpolate`、`torch.cat`/`stack`、`.to(dtype)` dtype 转换、`.clone()`、对非 contiguous 张量的 `reshape`（含**输出侧切片+reshape**） | `y = y[:, :, :, :S]; y.reshape(shape)`（切片后非 contiguous，reshape 隐式 `.contiguous()` → `aclnnCopy`）；`x = torch.nn.functional.pad(x, (0, pad_size))`（→ `aclnnPad`） |
>
> **允许**的 host 侧操作：经证明只改 metadata、共享原 storage 的 view 操作，以及
> 数据准备、kernel 调用和结果验证。
>
> **判定准则**：`is_contiguous()` 为 True 是 reshape 常见的零拷贝充分条件，不是必要
> 条件。非 contiguous 输入需证明目标 shape 与 stride 兼容且操作前后共享 storage；
> 不能证明时改用 stride-aware kernel。`permute`/`transpose` 本身通常只是 metadata view。

生成代码后逐项记录 `[HOST-METADATA-AUDIT]`。对每个 host tensor 操作写明输入/输出
stride、是否共享 storage/data pointer、是否触发 aclnn/物理拷贝；任一结论为 unknown
都不得交付。具体检查项见 [checklist.md §0](checklist.md#0-前置检查必须最先确认)。
