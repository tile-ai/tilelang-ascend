# Checklist

生成代码后逐项检查：

## 目录

- [0. 前置检查](#0-前置检查必须最先确认)
- [1. 功能验证](#1-功能验证)
- [2. Golden 与精度验证](#2-golden-与精度验证)
- [3. 上库前收尾检查](#3-上库前收尾检查)
- [4. 融合算子专项检查](#4-融合算子专项检查)
- [5. 融合算子常见错误排查](#5-融合算子常见错误排查)

---

## 0. 前置检查（必须最先确认）

| # | 检查项 | 说明 |
|---|--------|------|
| 0 | **算子主要操作全部在 kernel 内实现** | 算子的所有核心计算逻辑（数据搬运、数学运算、归约、归一化等）必须在 `@tilelang.jit` 装饰的 kernel 函数内部完成。host 侧对 NPU 张量严禁：①改数据指针（把输入/输出 tensor 重新绑定到别的 tensor）；②`.contiguous()` 等会触发真实数据拷贝/重排的操作（只改 stride 的 `reshape`/`view`/`transpose`/`permute`/`expand` 允许）；③直接改写 buffer 内容（`x[:]=`、in-place `_()`、`out=`）；④用新 buffer 作弊（host 侧处理完用新 tensor 顶替原输入）。**不满足时必须立即修改后再继续后续检查** |
| 0a | **host 侧无隐式 aclnn 调用** | host 侧代码（含 kernel 调用前的输入预处理和 kernel 调用后的输出后处理）不得出现会触发 aclnn 的操作：`torch.nn.functional.pad`/`cat`/`interpolate`、`torch.cat`/`stack`、`.to(dtype)` dtype 转换、`.clone()`/`.copy_()`、对非 contiguous 张量的 `reshape`。**重点检查输出侧**：若 kernel 输出为 padded shape，host 侧不得用切片（如 `y[:,:,:,:S]`）+ `reshape` 裁剪——切片后 tensor 非 contiguous，reshape 会隐式 `.contiguous()` → `aclnnCopy`。应改为让 kernel 直接输出到与原始 shape 一致的 buffer（`T.copy` + `pad_value` 处理尾块），host 侧仅做纯 view 的 `reshape` |
| 0b | **T.copy 4D+ 切片列方向 strided 检查** | 检查 kernel 内所有 `T.copy` 调用的切片表达式：如果切片的列维不是 buffer 最内有效维，且其后还有 shape>1 的维度，会产生静默错位。必须改成 JIT stride + 合法二维/逐连续段搬运，或在输入本来 contiguous 且 reshape 确认为零拷贝时做轴组归一化；**禁止对 permute 后的非 contiguous NPU tensor 做 host reshape**。详见 coding-conventions.md §2.1/§6 |
| 0c | **测试路径无 aclnn** | 测试输入、随机数、特殊值注入、dtype 转换和 golden 物理重排均在 CPU 完成；仅 H2D→kernel→D2H。`aclnnInplaceRandom` 属测试准备错误，不得归因于 kernel |
| 0d | **数据搬运成本通过** | 每条结构/dtype 路径已核算 GM pass、DMA transaction、GM 标量访问和地址 div/mod；大张量无逐元素 strided GM，连续 suffix record 已聚合搬运 |
| 0e | **性能哨兵通过** | 用户关键/最大 case 和每条最坏路径均实际执行且未超时；不得因标记 large/L1 而 skip 后宣称完成 |

## 1. 功能验证

| # | 检查项 |
|---|--------|
| 1 | `out_idx` 与函数签名中的输出参数位置一致 |
| 2 | V 核并行化（按模式）：Developer 默认 `threads=2` + 单 `cid` 轴、无 `vid` 偏移、无 workspace（细节参 `vector-parallelism.md` / mode-examples.md §6）；回退写法才校验 `block_M // VEC_NUM` 在 buffer 分配和索引中一致使用 |
| 3 | 所有 `T.alloc_ub` 的 shape 乘积不超 UB 容量 |
| 4 | Expert 模式有 `T.Scope("V")` 和 `T.barrier_all()` |
| 5 | Developer 模式有对应的 `pass_configs` |
| 6 | **kernel 与测试分文件**：`{op}.py` 只含 `@tilelang.jit` kernel（无 golden、无测试、无 `__main__`）；`test_{op}.py` 顶部 `from {op} import {op}`，含 golden + `test_{op}_l0()`（落地 DESIGN.md §9.2 全部 L0 用例，规则 shape）+ main；L1/L2/Boundary 先留桩（由 tilelang-op-test-design 场景 B 填充） |
| 7 | 含 GEMM：`gemm_v0` 第一次调用有 `init=True`（细节参 SKILL §子目录索引 `gemm-cv-fusion.md`） |
| 8 | 含 GEMM：block size 满足分形限制（细节参 SKILL §子目录索引 `gemm-cv-fusion.md`） |

## 2. Golden 与精度验证

| # | 检查项 | 说明 |
|---|--------|------|
| 9 | **Golden 实现一致** | 迁移算子必须使用原算子的 golden 实现 |
| 10 | **输出形状匹配** | 检查是否需要 transpose 来匹配原算子输出 shape |
| 10a | **特殊值位置敏感验证** | 支持 NaN/Inf 时，至少一个阻塞用例使用“有限值 + 稀疏特殊值”的确定性混合输入；先严格比较 NaN/正 Inf/负 Inf mask，再比较有限值。全 NaN/全 Inf 及 `allclose(equal_nan=True)` 不得作为唯一门禁 |

## 3. 上库前收尾检查

| # | 检查项 | 方法 |
|---|--------|------|
| 11 | **tilelang.disable_cache()** | 放在 `__main__` 下方或 `main()` 内部，避免编译缓存影响测试。**禁止**放在文件顶部全局调用、或用 `cache.clear_cache()`（会影响其他算子） |
| 12 | **注释转英文** | 人工检查所有注释，移除调试期临时中文注释 |
| 13 | **`# type: ignore`** | `T.Tensor` 参数定义后追加，避免 Pylance 报错 |
| 14 | **分层异常处理** | L0/L1 用例用 `try/except` 包裹：通过打 `[PRECISION_PASS]`、失败打 `[PRECISION_FAIL]` 并记 `ok=False`（不中断本层其余用例）。L2/Boundary 失败打 `[BOUNDARY_WARN]` 后继续，**不影响退出码**。不要裸 `assert` 直接崩 |
| 15 | **分层标记输出** | 每个用例按层打标记：L0/L1 → `[PRECISION_PASS]`/`[PRECISION_FAIL]`；L2/Boundary → `[BOUNDARY_PASS]`/`[BOUNDARY_WARN]`，含 shape/dtype，避免看似卡住 |
| 16 | **最终输出 + 退出码** | `test_{op}.py` 的 main 在 L0/L1 全过时最后一行 `print("Test Passed!")`（或 `"Kernel Output Match!"`）并 `sys.exit(0)`（据退出码 + 该行判定）；L0/L1 任一失败 `sys.exit(1)`。L2/Boundary 的 `[BOUNDARY_WARN]` 不改变退出码 |
| 17 | **--level 参数** | `argparse` 提供 `--level {l0,l1,l2,boundary,all}`（默认 `l0`）：精度收敛跑 `l0`、扩展后跑 `all`；main 按 level 分发各层函数 |
| 18 | **代码格式检查** | `ruff check examples/{op}/{op}.py examples/{op}/test_{op}.py` + `ruff format --check examples/{op}/{op}.py examples/{op}/test_{op}.py` 通过 |

## 4. 融合算子专项检查

| # | 检查项 | 说明 |
|---|--------|------|
| 19 | **CV 交互（按模式）** | Developer：无 `workspace_idx`、`threads=2`、单 `cid` 轴、无 `vid` 偏移；Expert/混合/回退：`workspace_idx` 与函数签名一致 |
| 20 | **AUTO_CV_COMBINE / AUTO_CV_SYNC 配置** | Developer 模式需开启 |
| 21 | **Cube ↔ Vector 数据流正确** | Developer：片上 `T.copy` 直连完整；回退：Cube → workspace → Vector 搬运路径完整 |
| 22 | **核分离方式与 pass_configs 匹配** | Developer 模式无需显式 T.Scope |

## 5. 融合算子常见错误排查

> Developer 模式（推荐）默认消除 workspace/vid，下列 workspace 相关项仅针对回退写法（Expert/混合或复杂场景）。Developer 模式应另外校验：`threads=2`、单 `cid` 轴、无 `vid` 残留偏移、无 `workspace_idx`、Cube↔Vector 片上直连。

| 错误类型 | 排查方向 |
|---------|---------|
| Developer 模式 CV 交互异常 | 确认 `threads=2`、无 `vid` 偏移、无 workspace、四个 pass_configs 全开（见 mode-examples.md §6） |
| workspace 未正确搬运（回退写法） | 检查 Cube 输出 T.copy 和 Vector 输入 T.copy 的索引 |
| 核间同步缺失 | 检查 AUTO_CV_SYNC 是否开启，或手动同步是否正确 |
| workspace shape 不匹配（回退写法） | 检查 block_num 计算是否正确 |
| 核分离方式错误 | Developer + 自动同步模式应无显式 T.Scope("C"/"V") |
| 精度误差超过 1% | 优先检查内存层级 API 选择和 pass_configs 配置 |
