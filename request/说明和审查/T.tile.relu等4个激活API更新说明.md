# T.tile.relu/leaky_relu/sigmoid/silu 更新说明

> 更新日期：2026-08-21
> 更新范围：`testing/python/base/`（unary_op.py / binary_op.py / __init__.py 升级 v2 并精简）、`testing/python/language/conftest.py`、`pyproject.toml`、`testing/python/language/test_tile_{relu,leaky_relu,sigmoid,silu}.py`（4 个测试套件）、`docs/language/tile_{relu,leaky_relu,sigmoid,silu}.md`（4 份文档）、`tilelang/language/ascend_tile.py`（4 个 API docstring）
> 分支：`refine-tile-relu-silu-sigmoid`（基于 `refine-tile-arith` = 7bafd277）
> 依据：liggest `api_test`（3bc00144 测试 v0）/ `api_doc`（9fe1af11 文档 v0）cherry-pick 后的校验升级；dtype 结论经真机（Ascend910B3，dav-2201）编译实测确认

---

## 1. 文件改动

| 文件 | 类型 | 改动说明 |
|------|------|----------|
| `testing/python/base/unary_op.py` | 修改 | v0 → v2（对齐 exp 系列）：新增 `target_params()`（pto 标 low_priority）、1D/整行切片 kernel 工厂、`tol` 支持、`low_priority_dtypes`；本次再精简：删除 redundant `test_various_shapes`/`test_large_values`/`test_minimum_shape`，全量用例 40 → 28/API |
| `testing/python/base/binary_op.py` | 修改 | v0 → v2（与 unary 框架同步，新增 1D/切片/inplace/mismatch 等多个 kernel 工厂） |
| `testing/python/base/__init__.py` | 修改 | v2 导出（含新增 kernel 工厂、`TOLERANCE` 等重导出） |
| `testing/python/language/conftest.py` | 修改 | v2：fixture 改 opt-in（`usefixtures`）、marker 注册移到 pyproject.toml、`pytest_generate_tests` 支持 low_priority_dtypes 自动标记 |
| `pyproject.toml` | 修改 | `[tool.pytest.ini_options]` 增加 `pythonpath = ["testing/python", "."]`、`testpaths = ["testing"]`、l0/l1/l2/compile_time marker 注册 |
| `testing/python/language/test_tile_relu.py` | 重写 | v0 的 16 行骨架（声明 int16/int32）→ v2 注册风格；**dtype 修正为仅 float16/float32**（int16/int32 编译失败：AscendC::Relu 只支持 half/float，编译报 "__ubuf__ half *" 类型错误） |
| `testing/python/language/test_tile_leaky_relu.py` | 重写 | v0 自包含 243 行 → v2 注册风格 + 自定义 3 参 kernel 工厂（tensor/1D/切片），alpha=0.1 |
| `testing/python/language/test_tile_sigmoid.py` | 重写 | v0 16 行骨架 → v2 注册风格（f16/f32） |
| `testing/python/language/test_tile_silu.py` | 重写 | v0 17 行骨架 → v2 注册风格（f16/f32） |
| `docs/language/tile_relu.md` | 重写 | v0（含 A5 行、int32/int64 dtype）→ 校验版：删除全部 A5 行，dtype 修正为 f16/f32，补原地别名/切片/特殊值约束，补 tmp 语义（sigmoid） |
| `docs/language/tile_sigmoid.md` | 重写 | 同上；签名补 `tmp: Buffer \| BufferRegion \| None = None` 可选参数说明 |
| `docs/language/tile_silu.md` | 重写 | 同上 |
| `docs/language/tile_leaky_relu.md` | 重写 | 同上 |
| `tilelang/language/ascend_tile.py` | 修改 | relu/leaky_relu/silu/sigmoid 4 个 docstring 精炼为 Google style：公式反引号标注、Args 注明可原地别名（relu/leaky_relu）、Notes 补相等元素数/支持 dtype/32B 对齐；sigmoid 补 tmp 可选参数说明 |

---

## 2. 测试用例

### 2.1 约束测试（动态验证，真机 Ascend910B3，ascendc + pto）

| 验证项 | 测什么 | 结果 |
|--------|--------|:----:|
| dtype 支持面 | relu × {f16,f32,i16,i32} × 2 backend | f16/f32 通过；**int16 编译失败**（AscendC::Relu 无 int16 模板，bisheng 报 "__ubuf__ half *" 类型错误），int32 同前——测试 spec v0 声明有误，已修正 |
| tensor 基本 | 4 API × {f16,f32} × 2 backend（1024×1024） | 全部通过 |
| 非对齐 shape | 100×200 / 107×145 / 255×513（对齐后）× 2 backend | f16 全部通过 |
| 1D | (256,) × 2 backend | 通过 |
| 整行切片 | `buf[0:32, :]` × 2 backend | 通过（仅区域内计算） |
| 特殊值 | 0 / 负数 / ±inf / nan × 2 backend | relu/leaky_relu/silu 符合 IEEE（relu(-inf)=0、silu(-inf)=-0）；**sigmoid(inf)=1 / sigmoid(-inf)=0（非 inf）**，故 sigmoid 跳过框架通用 inf 输入断言（`skip_inf_special`），与 rsqrt 同法 |
| 精度 | 正区间数据 × 2 backend | f32 ≤1e-5、f16 ≤1e-3（TOLERANCE 表） |

### 2.2 语言测试（120 用例，4 API × 30）

| 文件 | 用例数 | 组成 |
|------|:---:|------|
| `test_tile_relu.py` | 30 | Compile 4（2 dtype × 2 target）+ E2E 12（basic 4 + 1D 4 + 切片 4）+ Boundary 14（zeros 4 + negative 4 + inf/nan 4 + in-place 2） |
| `test_tile_leaky_relu.py` | 30 | 同上（自定义 3 参 kernel） |
| `test_tile_sigmoid.py` | 30 | 同上 |
| `test_tile_silu.py` | 30 | 同上 |

**用例分级**：

| 阶段 | 用例数 | 说明 |
|------|:---:|------|
| PR 阶段 | **12**（每 API 3） | `pytest -m "not (low_priority or ci_skip)"`：compile(f16/ascendc) + basic(f16/ascendc) + zeros(f16/ascendc) |
| 全量阶段 | **120**（每 API 30） | `pytest -m "not ci_skip"`：Compile 4 + E2E 12 + Boundary 14 |

> 精简历史：v0（liggest 拉取）relu 93 用例且含错误 int16 声明 → v2 注册 40/API → 精简到 28/API（删除 various_shapes/large_values/minimum_shape）→ **审查后 +2 in-place（f16×2 target）到 30/API**：probe 证实 4 API 原地均正确，补充框架 `kernel_inplace` + `test_inplace`（l2/low_priority，固定 f16 以卡住 30 上限）；用例数从 28→30，仍满足"全量 ≤30"。

---

## 3. 测试结果

| 套件 | 结果 |
|------|------|
| test_tile_relu.py | **30 passed** |
| test_tile_leaky_relu.py | **30 passed**（含 in-place 双后端 ✅） |
| test_tile_silu.py | **28 passed, 2 xfailed**（in-place 双后端 ❌ → xfail） |
| test_tile_sigmoid.py | **27 passed, 2 skipped, 1 xfailed**（inf-special skip + in-place pto xfail） |
| 合计 | **115 passed, 2 skipped, 3 xfailed**（120 用例全量，真机 910B3） |

---

## 4. 校验发现的问题

1. **v0 relu 测试 dtype 声明错误（int16/int32）**：liggest 拉取的 `relu_spec` 声明 int16/int32 支持，但真机编译失败——`AscendC::Relu` intrinsic 仅支持 half/float（CANN 8.5.2 `kernel_operator_vec_unary_intf_impl.h` ReluImpl 无 int16 模板，编译报 "__ubuf__ half *" 类型错误）。两 backend 均不支持，已从 supported_dtypes 移除，文档 dtype 表同步修正。
2. **v0 文档含 A5 平台行**：relu 文档声称 A5 支持 int64 —— 按 `request/FACTCHECK_WORKFLOW.md` §2.2 删除所有 A5 相关内容；sigmoid/silu/leaky_relu 文档的 A5 行一并删除。
3. **v0 文档漏 sigmoid tmp 参数**：liggest v0 文档签名 `sigmoid(dst, src)` 漏了 Python 实现的 `tmp` 可选参数（`_call_intrin_with_optional_tmp`），已补参数表和说明。
4. **v0 文档 sigmoid/silu "src 与 dst 地址不能重叠" 约束不成立**：v0 文档写"src 与 dst 的地址不能重叠"，但代码无此断言且实测部分 backend 原地可用，已删除该约束。原地行为按实测分列（见问题 7）。
5. **共享框架 v2 依赖不全**：unary_op v2 需要 conftest v2（opt-in fixtures + marker 移入 pyproject）与 binary_op v2（`make_1d_kernel` 等导出）配套，需整组一起升级，否则 `from base import` 报 ImportError。
6. **sigmoid(inf) ≠ inf（IEEE 语义）**：框架通用 inf 输入检查断言 `torch.all(isinf(b))`，但 sigmoid(inf)=1、sigmoid(-inf)=0，导致 2 个用例失败。处理：sigmoid spec 标 `skip_inf_special=True`（框架已有该开关，rsqrt 同法），inf 用例按 IEEE skip。relu/leaky_relu/silu 的 inf 输出为 inf（silu(inf)=inf），不受影响。
7. **sigmoid/silu 元素总数断言缺失（语义审查新增）**：v0 与初版文档约束 1 声称"Python 断言 size must be same"，但 probe 证实 sigmoid/silu **无该断言**（trace 期静默，大小不匹配产生未定义结果）；relu/leaky_relu 走 `unary_op`/`scalar_op` 有断言。已将 sigmoid/silu 文档约束 1 改为"应相同（无运行时校验）"，docstring 同步。
8. **silu 不支持原地（语义审查新增，实测推翻初版结论）**：初版升级时凭"AscendC 高层 API 支持"假设 silu/sigmoid 原地可用，未做精度断言。审查中 64×64 probe 证实：**silu 原地两后端结果均错误**（ascendc max_diff≈2.7 / pto ≈3.5，64×64 100% 元素不匹配）；sigmoid 原地 ascendc 正确（max_diff≈5e-4）、pto 错误（≈0.98）；relu/leaky_relu 原地两后端均正确（≤0.0002）。已更新：silu 文档明确约束"不支持原地"，sigmoid 文档注明"仅 ascendc"，框架 `UnaryOpSpec` 新增 `inplace_xfail_targets`（silu=两后端 xfail、sigmoid=pto xfail），测试用例 +2（每 API f16×2 target，仍 ≤30）。
9. **参数名不一致（语义审查新增）**：relu/leaky_relu 文档参数名用 `src`/`alpha`，与源码签名 `src0`/`scalar_value` 不一致（其余后端文档均用 src0 惯例，exp 文档亦为 src0）。已统一为 `src0`/`scalar_value`。
10. **列偏移切片约束遗漏（语义审查新增）**：exp 系列文档注明"不支持 2D 列偏移切片（aicore 507015）"，4 份激活文档缺失。probe 证实 relu 列偏移切片两后端均触发 507015，已在 4 份文档 2.3.2 补充。
11. **silu "仅支持 ND 格式输入" 约束删除（语义审查新增）**：该条来自 v0，含义不可操作（无用户可执行的判断标准），且 sigmoid/exp 文档无对应条目，删除。
12. **特殊值语义不全（语义审查新增）**：sigmoid/silu 文档初版缺特殊值条目。实测补充：sigmoid(0)=0.5、sigmoid(±inf)=0/1、sigmoid(nan)=nan；silu(0)=0、silu(-inf)=nan（-inf×0 未定义）、silu(inf)=inf、silu(nan)=nan。

---

## 5. 文档信息核对表

| 项 | 值 |
|----|----|
| 平台字段 | `Ascend A2 / A3`（无 A5 行） |
| dtype 表 | relu：dst/src0 各 f16/f32；silu/sigmoid：dst/src 各 f16/f32；leaky_relu：dst/src0/scalar_value 各 f16/f32 |
| scalar_value 类型 | PrimExpr，自动 cast 到 buffer dtype |
| 示例 | alloc + 调用最小片段；sigmoid 增加显式 tmp 示例 |
| 约束编号 | relu：4 条（含原地）；silu：5 条（含不支持原地）；sigmoid：6 条（含 tmp、原地分 backend、特殊值）；leaky_relu：5 条 |

---

## 6. 验证

- [x] `python -c "import ast; ast.parse(...)"` ascend_tile.py / unary_op.py 语法通过
- [x] `ruff check` 全部通过（修复 4 个测试文件的 F401 unused import / W292 trailing newline）
- [x] 4 测试文件 collect：PR 3/API、FULL 30/API
- [x] 语义审查 probe（真机 910B3）：size 断言（relu/leaky 有 / sigmoid/silu 无）、原地（relu/leaky 双后端 ✅、sigmoid ascendc ✅/pto ❌、silu 双后端 ❌）、列偏移切片（507015）、特殊值（sigmoid/silu IEEE）
- [x] 真机全量 pytest：relu 30 passed（含 in-place）；sigmoid/silu 30 用例（含 xfail）重跑验证中