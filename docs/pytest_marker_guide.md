# Pytest 标签使用指南

本文档介绍 TileLang-Ascend 测试框架中自定义 pytest 标签（marker）的使用方法，用于控制测试用例在 CI 流水线中的执行策略。

---

## 标签定义

标签注册于 `pyproject.toml`：

```toml
[tool.pytest.ini_options]
markers = [
    "low_priority: marks tests as low priority (only run in full test and scheduled tasks)",
    "ci_skip: marks tests to be skipped in all CI test scenarios",
]
```

### 标签作用对照

| 标签 | 作用 | PR 事件 | 全量测试 / 定时任务 |
|------|------|---------|---------------------|
| `low_priority` | 低优先级测试用例 | **跳过** | 执行 |
| `ci_skip` | 跳过测试用例 | **跳过** | **跳过** |

### CI 事件与标签过滤策略

CI 流水线（`.github/workflows/ci_cd.yml`）根据触发事件自动应用不同的 marker 过滤：

| 事件 | marker 表达式 | 含义 |
|------|---------------|------|
| `push` / `schedule` / `workflow_dispatch` | `not ci_skip` | 跳过 ci_skip，保留 low_priority |
| `pull_request` | `not (low_priority or ci_skip)` | 跳过 low_priority 和 ci_skip |

**说明：** `low_priority` 标签的用例，只有每天定时任务（`schedule`）及全量测试（`push` / `workflow_dispatch`）才会触发执行，通常的 PR 提交（`pull_request`）不会触发，从而在 PR 阶段减少不必要的测试耗时。`ci_skip` 标签的用例在所有 CI 场景均跳过，适用于存在已知问题或环境限制暂不执行的用例。

---

## 使用方法

### 1. 整体标注

对整个测试函数打标签，**所有参数组合**均继承该标签。

```python
@pytest.mark.low_priority
@pytest.mark.parametrize("target", ["ascendc", "pto"])
@pytest.mark.parametrize("shape", [1024])
def test_generate_arithmetic_progression(target, shape):
    N = shape
    block_size = 64
    run_test_generate_arithmetic_progression(N, block_size, target)
```

效果：`test_generate_arithmetic_progression[ascendc-1024]` 和 `test_generate_arithmetic_progression[pto-1024]` 均被标记为 `low_priority`。

### 2. 单参数标签

对 `@pytest.mark.parametrize` 中的**单个参数值**打标签。该参数值参与的所有组合均继承标签。

```python
@pytest.mark.parametrize(
    "target",
    ["ascendc", pytest.param("pto", marks=pytest.mark.low_priority)],
)
def test_vid_reduction_gm_ub_gm_identity(setup_random_seed, target):
    ...
```

效果：仅 `target=pto` 的用例被标记为 `low_priority`，`target=ascendc` 不受影响。

### 3. 笛卡尔积标签（多参数分别标注，OR 叠加）

当**多个 parametrize 分别对参数值打标签**时，标签按 OR 逻辑叠加：任一参数命中即生效。

```python
@pytest.mark.parametrize(
    "dtype",
    [
        "int16",
        "int32",
        pytest.param("uint16", marks=pytest.mark.low_priority),
        pytest.param("uint32", marks=pytest.mark.low_priority),
    ],
)
@pytest.mark.parametrize(
    "target",
    [
        "ascendc",
        pytest.param("pto", marks=pytest.mark.low_priority),
    ],
)
@pytest.mark.parametrize("shape", [(1024, 1024)])
def test_bitwise_lshift(dtype, target, shape):
    M, N = shape
    max_shift = 16 if dtype in ["int16", "uint16"] else 32
    scalarvalue = random.randint(1, max_shift)
    run_test_bitwise_lshift(M, N, 128, 256, scalarvalue=scalarvalue, dtype=dtype, target=target)
```

效果：

| dtype | target | 是否 low_priority | 命中原因 |
|-------|--------|------------------|----------|
| int16 | ascendc | 否 | — |
| int16 | pto | **是** | target=pto |
| int32 | ascendc | 否 | — |
| int32 | pto | **是** | target=pto |
| uint16 | ascendc | **是** | dtype=uint16 |
| uint16 | pto | **是** | dtype=uint16 + target=pto |
| uint32 | ascendc | **是** | dtype=uint32 |
| uint32 | pto | **是** | dtype=uint32 + target=pto |

> **OR 语义说明**：多个 parametrize 上的标签独立传播，任一命中即标记。无法实现"仅当 dtype=uint16 **且** target=pto 时标记"的 AND 语义。

### 4. 组合标签（精确标记特定参数组合）

需要精确标记特定参数组合（AND 语义）时，将多个参数合并为单个 parametrize，用 `pytest.param` 对元组打标签：

```python
transpose_dtype_target_params = [
    ("int16", "ascendc"),
    ("int16", "pto"),
    ("uint16", "ascendc"),
    ("uint16", "pto"),
    ("float16", "ascendc"),
    ("float16", "pto"),
    ("int32", "ascendc"),
    pytest.param("int32", "pto", marks=pytest.mark.low_priority),
    ("uint32", "ascendc"),
    ("uint32", "pto"),
    ("float", "ascendc"),
    ("float", "pto"),
]


@pytest.mark.parametrize("dtype,target", transpose_dtype_target_params)
@pytest.mark.parametrize("shape", [(16, 16)])
def test_transpose(dtype, target, shape):
    M, N = shape
    run_test_transpose(M, N, 16, 16, dtype, target)
```

效果：仅 `test_transpose[int32-pto-16-16]` 被标记为 `low_priority`，其余 11 个组合不受影响。

### 5. ci_skip 标签

`ci_skip` 用法与 `low_priority` 完全一致，区别在于**所有 CI 场景均跳过**：

```python
# 整体跳过
@pytest.mark.ci_skip
@pytest.mark.parametrize("target", ["ascendc", "pto"])
def test_unstable_feature(target):
    ...

# 特定参数组合跳过
@pytest.mark.parametrize(
    "dtype,target",
    [
        ("float", "ascendc"),
        ("float", "pto"),
        pytest.param("float16", "pto", marks=pytest.mark.ci_skip),
    ],
)
def test_known_issue(dtype, target):
    ...
```

---

## 标签选择指南

| 场景 | 推荐标签 | 示例 |
|------|----------|------|
| 测试用例耗时较长，PR 阶段无需验证 | `low_priority` | pto 后端的 uint16/uint32 组合 |
| 测试用例不稳定，存在已知 bug 待修复 | `ci_skip` | 特定 dtype+target 组合下结果异常 |
| 新增接口仅全量测试验证 | `low_priority` | 新 API 的非核心 dtype 覆盖 |
| 测试用例因环境/硬件限制无法运行 | `ci_skip` | 需要特定硬件版本才支持的特性 |

## 本地验证

```bash
# 查看已注册的标签
pytest --markers

# 模拟 PR 场景（跳过 low_priority 和 ci_skip）
pytest -m "not (low_priority or ci_skip)" testing/python/ -v

# 模拟全量测试场景（仅跳过 ci_skip）
pytest -m "not ci_skip" testing/python/ -v

# 仅运行 low_priority 用例
pytest -m "low_priority" testing/python/ -v

# 查看哪些用例被标记（不执行）
pytest --collect-only -m "low_priority" testing/python/ -q
```
