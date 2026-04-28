---
name: tilelang-ascend-tile-api
description: TileLang-Ascend 新增 Ascend 专属 T.tile.xxx 小 API 的端到端开发流程。用户要求新增、封装、暴露、实现或测试 ascend_tile.py 中的 T.tile API / Ascend tile primitive 时必须使用本 skill，尤其适用于需要同时打通 Python 前端、C++ lowering/codegen、Ascend C helper、文档和 CI 测试的任务。
---

# TileLang Ascend Tile API 开发流程

当任务是新增一个面向用户的 `T.tile.xxx` 小 API 时使用本 skill，尤其是从 `tilelang/language/ascend_tile.py` 暴露、并由 Ascend C 代码生成支撑的 API。

这里的目标不只是增加一个 Python 函数名，而是交付一个真正可用的 API：它能编译、能 lowering、能生成合法的 Ascend C、有清晰稳定的语义边界，并且有合适的 CI 测试覆盖。

## 第一轮调研

开始实现前，先读清楚当前仓库形态，再决定方案：

1. 阅读 `AGENTS.md`。
2. 阅读 `.agents/skills/tilelang-custom-skill/tilelang-api-best-practices/SKILL.md`。
3. 阅读 `.agents/skills/tilelang-custom-skill/tilelang-api-best-practices/references/api-compute.md`。
4. 如果涉及编程模式或 `pass_configs`，阅读 `.agents/skills/tilelang-custom-skill/tilelang-expert-to-developer/SKILL.md`。
5. 查看 `tilelang/language/ascend_tile.py` 中最相近的现有 API。
6. 查看 `testing/python/language/` 中最相近的测试。
7. 查看 `src/op/ascend.{h,cc}`、`src/target/codegen_ascend.cc`、`src/tl_templates/ascend/common.h` 中相近的 lowering、codegen 和 helper 实现。

不要凭记忆推断 API 签名。已有本地模式优先于看起来更聪明的新抽象。

## 范围决策

编辑前，先明确并说明 API 边界：

- 用户 API 名称，例如 `T.tile.foo(dst, src, ...)`。
- 它属于纯 tile 计算、数据搬运、类 reduction 行为，还是带副作用的写回。
- 支持的 buffer scope，通常是 GM、UB/shared、L1、L0，或其中子集。
- 支持的 dtype 和 rank。
- 是否接受 `Buffer`、`BufferLoad`、`BufferRegion`。
- 不支持的参数组合和语义。
- 应该支持 Developer `pass_configs`、Expert 模式，还是两者都支持。

当语义无法和主仓 / GPU 的全局 API 对齐时，优先新增 Ascend 专属的 `T.tile.xxx` API。除非用户明确要求、且语义确实一致，否则不要新增或修改全局 `T.xxx` API。

## 实现路径

### 1. Python 前端

在 `tilelang/language/ascend_tile.py` 中新增用户入口。

优先复用该文件和 `tilelang/language/copy.py` 里的本地 helper 模式：

- 如果现有 API 会解析 let-bound value，新 API 也应保持一致。
- 只在语义明确时接受 `Buffer`、`BufferLoad` 或 `BufferRegion`。
- 当 C++ lowering 需要 region、rank 或 extent 信息时，把前端输入转换为 `tl.region`。
- 对不支持的 scope 或参数组合尽早报错，并给出清晰错误信息。
- 当操作需要 C++ lowering 时，发射命名清晰的 op，通常是 `tl.ascend_<api_name>`。

除非操作本质上就是普通 copy，否则不要为了省事复用 `tl.ascend_copy`。有副作用、会改变硬件模式、或语义不同的操作，通常应该有显式 op 名称。

### 2. C++ Operator 和 Lowering

如果前端发射新的 `tl.ascend_*` op，需要补齐对应 C++ operator 路径：

- 在 `src/op/ascend.h` 中声明。
- 在 `src/op/ascend.cc` 中实现参数解析和 lowering。
- 当它需要 tile-op lowering 时，用 `TIR_REGISTER_TL_OP` 注册。
- 保留 region 信息，直到 lowering 可以计算访问指针、extent、stride、mask 或合法 shape。
- 即使 Python 已经做过校验，C++ 侧也要重新检查关键约束。

lowering 结果应变成后端可识别的 `call_extern` 或已有 codegen pattern。降低后的调用名要稳定且有描述性，例如 `tl::ascend::<helper_name><...>`。

### 3. Ascend C Helper

可复用的 Ascend C 片段优先放在 `src/tl_templates/ascend/common.h`，除非仓内已有更合适的模板位置。

尽量把硬件状态切换封装在 helper 函数内部。比如 helper 开启了某种模式，它也应该在返回前恢复或关闭该模式。

如果需要兼容不同 CANN 版本：

- 先在本地或目标 Ascend C 头文件中搜索真实存在的版本宏。
- 只有确认环境或项目约定中存在后，再使用 `CANN_MAJOR` 之类的官方宏。
- 增加一个小的兼容 helper，而不是在 codegen 各处散落 `#if` 分支。
- 只有当注释能避免后续误解时，才补充 fallback 行为说明。

### 4. Codegen 和 Pipeline 集成

把 lowering 后的调用接入后端：

- 更新 `src/target/codegen_ascend.cc`，让它打印对应 helper 调用。
- 只有当参数顺序和指针打印逻辑完全匹配时，才复用 `CopyCodegen` 等现有 helper。
- 如果操作读写 GM，按需更新调度或 pipeline 元数据：
  - `src/transform/common/operation_config.h`
  - `src/transform/ascend_combinecv.cc`
  - `src/transform/cross_core_pipeline.cc`
- pipeline 分析中，应把带副作用的 GM 写回当作写操作处理。

默认不要添加 PTO 支持。只有当任务明确要求，或已有 PTO 路径能以很小、低风险的改动接入时，才考虑补充。

## 测试放置

不要自动创建新的独立测试文件。先选择和 API 行为最匹配、范围最窄的现有测试文件。

测试位置建议：

- 纯 elementwise `T.tile.xxx` 数学 API：优先放在 `testing/python/language/test_tilelang_ascend_language_elementwise.py`。
- compare/select API：优先放在已有 compare/select 主题测试文件中。
- cast/copy 类行为：优先放在已有 cast/copy 主题测试文件中。
- parallel lowering 行为：只有当 API 主要测试 `T.Parallel` 或 auto-copy 行为时，才放入 parallel 相关测试文件。
- 新的带副作用写回、新的 memory pipeline 类别，或没有合适主题归属的 API：创建聚焦的独立测试文件。

独立测试文件是允许的，但应该少用，并且要能清楚说明原因。记住 `examples/bench_test.sh` 会运行 `testing/python/`，所以每个新文件都会进入 CI 范围。

## CI 测试风格

默认编写面向 CI 的正确性测试，而不是开发阶段的 TDD 检查测试。

推荐的测试形态：

- 有 NPU 环境时，编译并运行 kernel。
- 当 `torch.npu` 不可用时，用 `pytest.mark.skipif` 跳过。
- 用 PyTorch 或简单 reference 计算期望结果，并通过 `torch.testing.assert_close` 对比。
- 覆盖能保护 API 契约的最小 dtype/rank 集合。
- 当 API 支持 Developer 或混合模式时，使用对应 pass configs：

```python
PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}
```

除非生成源码字符串、helper 名称或负例本身就是公开契约，否则不要把这些断言长期留在 CI 中。这类检查适合开发阶段，但实现稳定后容易变成噪声。

对于带副作用写回的 API，测试应在运行 kernel 前显式初始化目标 buffer。例如 accumulation API 应该先把 GM 清零，再检查累加后的值。

## 文档更新

对于面向用户的 API，更新未来用户和 agent 真正会看的文档：

- `docs/language_ref/tilelibrary.md`：简短语言参考。
- `docs/TileLang-Ascend Programming Guide.md`：详细使用指南。
- `.agents/skills/tilelang-custom-skill/tilelang-api-best-practices/references/api-compute.md`：agent 面向的 API 用法说明。
- `.agents/skills/tilelang-custom-skill/tilelang-expert-to-developer/SKILL.md`：仅当编程模式建议发生变化时更新。
- 不要为了内部 helper 去更新宽泛文档。

如果旧文档里有相似但语义不同的全局 API，增加简短提醒，而不是静默改写可能属于 GPU / 主仓教程的示例。

## 验证清单

运行本地能支持的最强验证：

```bash
python -m py_compile <changed-python-files>
conda run -n tilelang_dev ruff check <changed-python-files>
conda run -n tilelang_dev ruff format --check <changed-python-files>
git diff --check
```

如果修改了 C++ 文件，也运行：

```bash
conda run -n tilelang_dev clang-format --dry-run --Werror <changed-cpp-files>
```

如果当前环境可运行，执行目标 pytest：

```bash
pytest -q <selected-test-file-or-test-node>
```

如果本地缺少 pytest、TVM、CANN 或 NPU runtime，需要明确说明哪一层没有验证，以及是否需要服务器侧运行。

## 可选 Agent 拆分

只有当用户明确授权并行 agent 工作时，才拆分任务。

适合独立拆分的工作：

- `tilelang/language/ascend_tile.py` 中的前端 API 和 IR 形态。
- `src/op/ascend.{h,cc}` 中的 C++ op 和 lowering。
- `src/tl_templates/ascend/common.h` 中的 Ascend C helper。
- `src/target` 和 `src/transform` 中的 codegen 与 pipeline 元数据。
- 测试和文档。
- 集成验证。

给每个 agent 分配不重叠的写入范围，并明确提醒它不要 revert 其他 agent 的修改。

## 最终回复

完成后总结：

- 新增用户 API 以及精确支持边界。
- 修改过的前端、lowering、codegen/helper 和测试文件。
- 测试放置位置以及选择原因。
- 哪些检查已通过，哪些无法在本地运行。
- 已知不支持的语义。
