# PR 中文说明（可直接粘贴）

## 变更摘要

本 PR 包含两部分变更：

1. 按 `.agents/skills/tilelang-op-test-design/references/precision-standard.md` 修正 153 个精度相关 Python 文件中的精度判定函数，覆盖 FP16、BF16、FP32/HIFP32、FP8、整数及 NaN/Inf 规则。
2. 将性能评估中明显低于 ACLNN 基线的 23 个算子实现从 `examples/` 迁移至 `examples_experiment/`，并同步迁移存在的精度测试文件及其源码路径。

本 PR 不修改 TileLang 编译器核心逻辑，不改变低性能算子的算法实现。相关低性能评估报告：

https://gitcode.com/2402_86846624/tilelang-ascend-low-perf-report

本 PR 替代已关闭的 #1686。

## 变更类型

- [ ] `[Bugfix]` Bug 修复
- [ ] `[Feature]` 新功能/新算子
- [ ] `[Refactor]` 重构
- [ ] `[CI]` CI 或构建
- [ ] `[Skills]` Skill/开发流程
- [x] `[Testing]` 测试
- [x] `[Example]` 示例/Benchmark

## 1. 算子合入标准

- [ ] 新增算子默认场景编译、运行和精度验证（本 PR 不新增算子）。
- [ ] AscendC/PTO 双后端完整验证（尚未覆盖全部算子，需目标 CI 确认）。
- [ ] 未影响现有接口和行为（已完成静态路径检查，完整回归由 CI 确认）。
- [x] 已将历史低性能实现迁移到 `examples_experiment/`。
- [ ] 性能达到 AscendC 50% 以上（本 PR 归档低性能实现，不声称达标）。
- [ ] 已完成全部性能反模式审查（未在本 PR 中逐项完成）。
- [ ] 已提供全部 simulator 截图（未提供）。

性能报告：

https://gitcode.com/2402_86846624/tilelang-ascend-low-perf-report

## 2. 框架及接口代码合入标准

- [x] 本 PR 不修改框架或公共接口。
- [x] 已新增逐文件精度 checker 测试，共 55 个文件。
- [x] 已执行 Python 语法和路径检查。
- [ ] 所有算子均完成完整 NPU 端到端多轮验证（尚未完成）。
- [ ] 已登记 `ci/operator_test_manifest.yaml`（待维护者确认是否需要登记）。
- [ ] 已确认 legacy runner 与 pytest 不重复执行（待目标 CI 确认）。
- [ ] 已更新全部相关 API/硬件能力文档（本 PR 仅新增变更说明文档）。

## 3. 测试结果

静态检查：

```bash
python -m compileall -q examples examples_experiment
python scripts/validate_precision_all.py
```

CANN 9.1 环境下运行 55 个精度测试：

```bash
python -m pytest -q \
  $(find examples examples_experiment \
    -type f -name 'test_precision_*.py' | sort)
```

结果：

```text
53 passed, 2 skipped, 0 failed
```

两个 skipped 对应自动调优示例，它们没有独立精度 checker，属于预期跳过。

## 4. 变更明细

- 精度相关实现/测试文件：153 个。
- 新增逐文件精度测试：55 个，均包含 `load_checker` 和 `test_precision_checker`。
- 低性能算子实现迁移：23 个。
- 迁移后无效精度测试源码路径：0 个。
- 迁移实现文件本身不引入算法代码变化；组合 diff 中部分文件同时包含精度修改，因此可能显示为“修改后重命名”。
- 详细的文件—函数映射见 `PRECISION_MOVE_CHANGE_DETAILS.md`。

## 5. 已知限制和风险

- 新增 55 个测试主要验证精度判定函数和零误差基线，不替代所有算子的完整 NPU kernel 测试。
- GitHub Actions、manifest 收集范围及双后端结果需以目标仓库 CI 为准。
- 低性能算子迁移可能影响外部硬编码路径；对应测试路径已同步更新。

## 6. 提交规范

- [ ] 所有历史 commit 均包含模板要求的类型标签（已有历史提交未全部带标签）。
- [x] 已检查 `git diff`、`git status`，未跟踪的本地统计报告未纳入 PR。

## 提交前确认

本 PR 仅勾选已实际完成的 `[Testing]` 和 `[Example]` 项目；其余未完成项目已在上文说明原因，最终 CI 结果由目标仓库检查确认。

