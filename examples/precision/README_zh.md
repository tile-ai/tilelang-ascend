# tilelang_precision_checker 使用说明

> 位于 `testing/python/precision/tilelang_precision_checker.py`

---

## 一、快速开始

```python
from tilelang_precision_checker import precision_check, PrecisionLevel, OpCategory

# 最简用法：默认 L0 等级 + 浮点计算类
precision_check(npu_output, torch_reference.cpu())

# 指定等级
precision_check(npu_output, torch_reference.cpu(), level=PrecisionLevel.L1)

# 静默模式，自行判断
report = precision_check(npu_output, torch_reference.cpu(), verbose=False)
if report.passed:
    print("通过")
else:
    print(report.summary())
```

---

## 二、核心概念

### 2.1 精度等级 (PrecisionLevel)

| 等级 | 含义 | MARE ≤ | MERE ≤ | RMSE ≤ |
|------|------|--------|--------|--------|
| `L0` | 常规算子，非敏感业务 | 10 | 2.0 | 2.0 |
| `L1` | 重要算子，多模态/LLM/推荐 | 5 | 1.5 | 1.5 |
| `L2` | 关键算子，业务核心 | 2 | 1.2 | 1.2 |

### 2.2 算子分类 (OpCategory)

| 分类 | 含义 | 校验方式 |
|------|------|---------|
| `NON_COMPUTE` | 非计算类（搬移/Cast） | 逐比特比对 |
| `INTEGER_COMPUTE` | 整数计算类 | 逐比特或 AE=0 |
| `QUANTIZE_COMPUTE` | 量化计算类 | AE ≤ 1 |
| `FLOAT_COMPUTE` | **浮点计算类（默认）** | MARE/MERE/RMSE + 小值域 + INF/NAN |

### 2.3 误差指标

| 指标 | 公式 | 含义 |
|------|------|------|
| **MARE** | `max(|NPU - golden| / (|golden| + 1e-7))` | 最大相对误差（最差点） |
| **MERE** | `mean(|NPU - golden| / (|golden| + 1e-7))` | 平均相对误差 |
| **RMSE** | `sqrt(mean((NPU - golden)^2))` | 均方根误差（整体离散程度） |
| **SmallVal errors** | `|golden|<阈值 且 |diff|>误差阈` 的元素数 | 小值域错误计数 |
| **Ratio** | `NPU指标 / 阈值` | Ratio ≤ 1.0 即通过 |

---

## 三、API 参考

### 3.1 `precision_check()`

```python
def precision_check(
    actual: torch.Tensor,                           # NPU 算子输出
    golden: torch.Tensor,                           # torch 参考输出 (需在 CPU 上)
    level: PrecisionLevel = PrecisionLevel.L0,      # 精度等级
    category: OpCategory = OpCategory.FLOAT_COMPUTE, # 算子分类
    verbose: bool = True,                           # 是否打印报告
) -> PrecisionReport:
```

**内部执行流程**：

```
actual, golden
    │
    ├─ Shape/Device 断言
    ├─ 按 category 分支:
    │   ├─ NON_COMPUTE     → 逐比特对比
    │   ├─ INTEGER_COMPUTE → 逐比特 或 AE=0
    │   ├─ QUANTIZE        → AE ≤ 1
    │   └─ FLOAT_COMPUTE   → 以下流程:
    │
    ├─ ① INF/NAN 位置一致性检查 _check_inf_nan()
    │     不一致 → 打印前 5 个不匹配位置 → 返回 FAIL
    │
    ├─ ② 计算原始 MARE/MERE/RMSE _compute_errors()
    │
    ├─ ③ 小值域检查 _check_small_value()
    │     统计 |golden| < 阈值 且 |diff| > 误差阈 的元素数
    │     Ratio = NPU_errs / max(Ref_errs, 1) ≤ 2 → 通过
    │
    ├─ ④ 排除小值域元素 → 重新计算 MARE/MERE/RMSE
    │     (小值域下 golden→0，相对误差会爆炸)
    │
    ├─ ⑤ 逐指标与 _THRESHOLDS[level] 比较
    │     全部通过才 PASS
    │
    └─ 返回 PrecisionReport
```

**返回对象 `PrecisionReport`**：

| 属性 | 类型 | 说明 |
|------|------|------|
| `passed` | `bool` | 是否通过 |
| `level` | `PrecisionLevel` | 精度等级 |
| `category` | `OpCategory` | 算子分类 |
| `mare` | `float` | MARE（排除小值域后） |
| `mere` | `float` | MERE（排除小值域后） |
| `rmse` | `float` | RMSE（排除小值域后） |
| `mare_ratio` | `float \| None` | MARE / 阈值 |
| `mere_ratio` | `float \| None` | MERE / 阈值 |
| `rmse_ratio` | `float \| None` | RMSE / 阈值 |
| `small_value_error_count_npu` | `int` | NPU 小值域错误数 |
| `small_value_error_count_ref` | `int` | 参考小值域错误数 |
| `small_value_passed` | `bool` | 小值域是否通过 |
| `inf_nan_match` | `bool` | INF/NAN 位置是否一致 |
| `num_elements` | `int` | 输出元素总数 |
| `message` | `str` | 描述信息 |
| `summary()` | `() -> str` | 生成可读报告字符串 |

### 3.2 `multi_seed_precision_check()`

单次 `precision_check` 失败后的**复检流程**（Bootstrap 统计判定）。

```python
def multi_seed_precision_check(
    kernel_fn,                          # tilelang kernel 函数（已编译）
    *input_args,                        # kernel 的输入张量
    level: PrecisionLevel = L0,         # 精度等级
    category: OpCategory = FLOAT_COMPUTE,
    num_seeds: int = 1000,              # 随机种子数量
    ref_fn=None,                        # 参考函数
    **input_kwargs,                     # kernel 额外参数
) -> Dict[str, Any]:
```

**判定规则**：

```
换 N 个随机种子各跑一次，收集 MARE Ratio
         ↓
Bootstrap 重采样 2000 次，求 Ratio 中位数的 95% 置信区间
         ↓
┌─ N < 200  ─→ 小样本熔断，直接 FAIL
├─ CI_lower > 1.0 → 确认精度异常 ❌
└─ CI_lower ≤ 1.0 → 偶发误报，通过 ✅
```

**返回字典**：

| 键 | 类型 | 说明 |
|---|------|------|
| `passed` | `bool` | 是否通过 |
| `median_ratio` | `float` | MARE Ratio 中位数 |
| `ci_lower` | `float` | 95% 置信区间下界 |
| `ci_upper` | `float` | 95% 置信区间上界 |
| `num_samples` | `int` | 样本数 |
| `message` | `str` | 判定描述 |

**注意**：当前实现假设 kernel 是纯函数式（输入→输出）。Flash Attention 等带 workspace 的算子需自行适配。

---

## 四、分类使用示例

### 4.1 浮点计算类（默认，最常见）

```python
# 大多数 tilelang 算子都用这个
report = precision_check(
    npu_out,
    torch_ref.cpu(),
    level=PrecisionLevel.L0,          # 开发阶段 L0
    category=OpCategory.FLOAT_COMPUTE, # 可省略，设为默认
)
# 输出:
# === Precision Check [✅ PASS] ===
#   Level:     L0
#   Category:  FLOAT_COMPUTE
#   Elements:  65536
#   MARE:      9.999e-02  (ratio=0.0100)
#   MERE:      6.075e-04  (ratio=0.0003)
#   RMSE:      4.005e-05  (ratio=0.0000)
```

### 4.2 非计算类（搬移/Cast）

```python
report = precision_check(
    npu_moved_data,
    torch_expected_data,
    category=OpCategory.NON_COMPUTE,
)
# → 逐比特比对，必须完全一致
```

### 4.3 整数计算类

```python
report = precision_check(
    npu_int_result,
    torch_int_result,
    category=OpCategory.INTEGER_COMPUTE,
)
# → 逐比特 或 最大绝对误差 = 0
```

### 4.4 量化计算类

```python
report = precision_check(
    npu_quant_result,
    torch_quant_result,
    category=OpCategory.QUANTIZE_COMPUTE,
)
# → 整型输出: 绝对误差 ≤ 1
# → 浮点输出: 同 FLOAT_COMPUTE 流程
```

### 4.5 多种子复检

```python
result = multi_seed_precision_check(
    my_kernel_func,        # 已编译的 kernel
    tensor_a, tensor_b,    # 输入
    level=PrecisionLevel.L2,
    num_seeds=1000,
    ref_fn=torch_reference_func,
)

if result["passed"]:
    print("统计复检通过")
else:
    print(f"确认精度异常: CI_lower={result['ci_lower']:.4f} > 1.0")
# 输出:
# === Multi-Seed Retest (n=1000) ===
#   Median ratio: 0.6534
#   95% CI:       [0.6201, 0.6893]
#   Verdict:      ✅ PASS  (PASS)
```

---

## 五、内部参数说明

### 5.1 小值域阈值

这些阈值定义在 `_SMALL_VALUE_THRESHOLDS` 中，无需用户修改：

| 数据类型 | 判定阈值 `|golden| <` | 误差阈值 `|diff| >` |
|---------|---------------------|-------------------|
| float16 | 2^-11 ≈ 0.00049 | 2^-16 ≈ 0.000015 |
| bfloat16 | 2^-8 ≈ 0.0039 | 2^-16 ≈ 0.000015 |
| float32 | 2^-14 ≈ 0.000061 | 2^-30 ≈ 9.3e-10 |
| float64 | 2^-14 ≈ 0.000061 | 2^-30 ≈ 9.3e-10 |

当 `|golden|` 小于判定阈值时，相对误差会因为分母接近 0 而爆炸，此时的 MARE/MERE/RMSE 不可信。脚本会自动排除这些"小值域"元素，改用 ErrorCount 方式单独判断。

### 5.2 精度等级阈值

定义在 `_THRESHOLDS` 中，无需用户修改：

| 等级 | MARE_th | MERE_th | RMSE_th |
|------|---------|---------|---------|
| L0 | 10.0 | 2.0 | 2.0 |
| L1 | 5.0 | 1.5 | 1.5 |
| L2 | 2.0 | 1.2 | 1.2 |

---

## 六、常见问题

### Q1: golden 必须在 CPU 上吗？

是的。`precision_check` 会断言 `golden` 在 CPU 上（或与 `actual` 同设备）。因为后续计算（误差、排序、median）需要 CPU 支持。NPU 上的 tensor 请先 `.cpu()`。

```python
# 正确
precision_check(npu_out, torch_ref.cpu())

# 错误：golden 在 NPU 上
precision_check(npu_out, torch_ref_on_npu)  # AssertionError
```

### Q2: 单标杆 returned FAIL，但 MARE/MERE/RMSE 很小？

很可能是**小值域误判**。查看报告中 `SmallVal` 一行——如果 `Ref_errors=0`，说明 torch（作为 golden）在小值域没有错误，而 NPU 有少量 FP16 舍入差异，导致 Ratio 爆炸。

解决方法：改用**双标杆**模式（参见 `ascend_precision_standard_guide.md` §5.3），让 torch 参考也与 FP64 真值比较，使小值域错误计数公平。或者直接跳过小值域检查，只看 MARE/MERE/RMSE。

### Q3: multi_seed_precision_check 报错怎么办？

该函数假设 kernel 是纯函数式调用（`kernel_fn(*args)` 直接返回输出）。Flash Attention 等带 workspace 的算子不兼容此入口，需要自行实现 Bootstrap 循环（参见 `examples/flash_attention/test_precision_flash_attn_dual.py` 中的复检代码）。

### Q4: 如何确认双标杆需要精度对齐？

对照 `ascend_precision_standard_guide.md` §5.3 的判定方法：
1. 分析 NPU kernel 数据流，找到精度瓶颈（FP16 中间存储）
2. 检查 torch 参考在同一位置是否用了更高精度
3. 如果是 → 在 torch 参考中对应位置插入 `.half().float()` 截断

---

## 七、文件结构

```
testing/python/precision/
├── tilelang_precision_checker.py   ← 本脚本
└── README.md                       ← 本说明
```

相关文档：
- `docs/ascend_precision_standard_guide.md` — 精度标准详细指南
- `examples/flash_attention/README.md` — Flash Attention 精度测试案例
