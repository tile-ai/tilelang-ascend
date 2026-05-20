# TileLang-Ascend 精度标准体系

**参考来源**：cannbot-skills ops-precision-standard

---

## 一、精度标准定义

### 1.1 基本概念

| 概念 | 定义 | 说明 |
|------|------|------|
| **atol** | Absolute tolerance | 绝对误差容忍度 |
| **rtol** | Relative tolerance | 相对误差容忍度 |
| **精度验证** | `abs(actual - expected) ≤ atol + rtol * abs(expected)` | PyTorch 默认验证公式 |

---

### 1.2 计算类型精度标准

| 计算类型 | dtype | atol | rtol | 说明 |
|---------|-------|------|------|------|
| **整数运算** | int8 | 0 | 0 | 精确匹配 |
| | int16 | 0 | 0 | 精确匹配 |
| | int32 | 0 | 0 | 精确匹配 |
| | uint8 | 0 | 0 | 精确匹配 |
| **浮点运算** | float16 | 1e-3 | 1e-3 | FP16 精度（半精度） |
| | float32 | 1e-5 | 1e-5 | FP32 精度（单精度） |
| | bfloat16 | 1e-2 | 5e-3 | BF16 精度（脑浮点） |
| **量化运算** | qint8 | 1e-2 | 1e-2 | 量化精度（8位量化） |
| | qint32 | 1e-5 | 1e-5 | 量化精度（32位量化） |

---

### 1.3 算子类别精度标准

#### GEMM 类（纯 Cube + Single）

| dtype | atol | rtol | 特殊场景 |
|-------|------|------|---------|
| float16 | 1e-3 | 1e-3 | 大矩阵（M/N/K > 1024）可放宽到 5e-3 |
| float32 | 1e-5 | 1e-5 | 无特殊要求 |
| bfloat16 | 1e-2 | 5e-3 | 无特殊要求 |

**说明**：
- GEMM 计算精度受累加次数影响
- 大矩阵累加次数多，精度误差累积
- 建议根据矩阵大小调整精度标准

---

#### Softmax 类（纯 Vector + Multi）

| dtype | atol | rtol | 特殊场景 |
|-------|------|------|---------|
| float16 | 1e-3 | 1e-3 | 无特殊要求 |
| float32 | 1e-4 | 1e-4 | 要求更严格（多步计算） |
| bfloat16 | 1e-2 | 5e-3 | 无特殊要求 |

**说明**：
- Softmax 有 4 个计算步骤（max → exp → sum → div）
- FP32 要求更严格（1e-4）
- exp 计算可能导致精度损失

---

#### Normalization 类（纯 Vector + Multi）

| dtype | atol | rtol | 特殊场景 |
|-------|------|------|---------|
| float16 | 1e-3 | 1e-3 | eps < 1e-5 时可能不稳定 |
| float32 | 1e-4 | 1e-4 | 无特殊要求 |
| bfloat16 | 1e-2 | 5e-3 | 无特殊要求 |

**说明**：
- Normalization 计算 mean/var/sqrt，多步计算
- eps 参数影响精度（过小可能导致除零）
- FP32 要求更严格（1e-4）

---

#### Activation 类（纯 Vector + Single）

| dtype | atol | rtol | 特殊场景 |
|-------|------|------|---------|
| float16 | 1e-3 | 1e-3 | sigmoid/gelu 可放宽到 5e-3 |
| float32 | 1e-5 | 1e-5 | 无特殊要求 |
| bfloat16 | 1e-2 | 5e-3 | 无特殊要求 |

**说明**：
- Activation 类计算简单（单步）
- sigmoid/gelu 涉及指数计算，精度可能略低
- FP16 可放宽到 5e-3

---

#### Reduction 类（纯 Vector + Single）

| dtype | atol | rtol | 特殊场景 |
|-------|------|------|---------|
| float16 | 1e-3 | 1e-3 | 大规模归约可放宽到 5e-3 |
| float32 | 1e-5 | 1e-5 | 无特殊要求 |
| bfloat16 | 1e-2 | 5e-3 | 无特殊要求 |

**说明**：
- Reduction 类涉及累加（sum）或比较（max）
- 大规模归约累加次数多，精度误差累积
- FP16 可放宽到 5e-3

---

#### Fusion 类（混合 + Fusion）

| dtype | atol | rtol | 特殊场景 |
|-------|------|------|---------|
| float16 | 1e-3 | 1e-3 | FlashAttention 可放宽到 5e-3 |
| float32 | 1e-4 | 1e-4 | 无特殊要求 |

**说明**：
- Fusion 类涉及多个算子组合
- 精度误差累积更明显
- FlashAttention 可放宽到 5e-3

---

## 二、特殊场景精度标准

### 2.1 特殊值处理

| 特殊值 | 处理方法 | 精度标准 |
|--------|---------|---------|
| **INF（无穷）** | 验证 `isinf(actual) == isinf(expected)` | 不验证数值误差 |
| **NAN（非数）** | 验证 `isnan(actual) == isnan(expected)` | 不验证数值误差 |
| **零值（±0）** | 验证 `abs(actual) < eps` | atol = 1e-7 |
| **极小值（< 1e-5）** | 仅验证 atol，忽略 rtol | atol = 1e-7 |

**示例**：
```python
def verify_special_values(actual, expected, dtype):
    if torch.isinf(expected):
        assert torch.isinf(actual), "INF mismatch"
    elif torch.isnan(expected):
        assert torch.isnan(actual), "NAN mismatch"
    elif abs(expected) < 1e-5:
        assert abs(actual - expected) < 1e-7, "Small value mismatch"
    else:
        atol, rtol = get_precision(dtype)
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
```

---

### 2.2 小值域精度标准

| 值域范围 | dtype | atol | rtol | 说明 |
|---------|-------|------|------|------|
| **< 1e-5** | float16 | 1e-7 | 0 | 仅验证绝对误差 |
| | float32 | 1e-7 | 0 | 仅验证绝对误差 |
| **1e-5 ~ 1e-3** | float16 | 1e-5 | 1e-3 | 小值域精度 |
| | float32 | 1e-6 | 1e-5 | 小值域精度 |
| **> 1e-3** | float16 | 1e-3 | 1e-3 | 正常精度 |
| | float32 | 1e-5 | 1e-5 | 正常精度 |

**说明**：
- 小值域（< 1e-5）仅验证绝对误差
- 避免 rtol 导致的小值验证失败

---

### 2.3 量化精度标准

| 量化类型 | atol | rtol | 说明 |
|---------|------|------|------|
| **对称量化** | 1e-2 | 1e-2 | int8 量化精度 |
| **非对称量化** | 5e-3 | 5e-3 | uint8 量化精度（更严格） |
| **混合精度量化** | 1e-3 | 1e-3 | 高精度量化 |

**说明**：
- 量化精度受量化范围影响
- 非对称量化精度更高（范围更小）

---

## 三、精度标准应用方法

### 3.1 在测试设计中的应用

**步骤 1：提取精度标准**
- 从 design.md §8 提取精度标准（如有）
- 或根据算子类别和 dtype 自动选择精度标准

**步骤 2：生成测试代码**
- 在测试代码中使用 `torch.testing.assert_close`
- 设置正确的 atol 和 rtol

**示例代码**：
```python
def test_silu_float16():
    dtype = torch.float16
    atol, rtol = 1e-3, 1e-3  # Activation 类 + FP16
    
    x = torch.randn(128, 256, dtype=dtype, device="npu")
    y_actual = silu_kernel(x)
    y_expected = torch.nn.functional.silu(x)
    
    torch.testing.assert_close(y_actual, y_expected, atol=atol, rtol=rtol)
```

---

### 3.2 在用户交互中的应用

**场景 A：design.md 输入**

Phase 3 用户交互中询问精度标准：
```
询问精度标准（如 §8 未定义）：
[1] 使用默认精度标准（根据算子类别和 dtype 自动选择）
[2] 自定义精度标准（atol=xxx, rtol=xxx）
[3] 不验证精度（仅验证功能）
```

---

### 3.3 精度标准查询函数

```python
def get_precision(op_category, dtype, special_case=None):
    """
    根据算子类别、dtype 和特殊场景获取精度标准
    
    参数：
        op_category: str - 算子类别（GEMM/Softmax/Activation等）
        dtype: str - 数据类型（float16/float32/bfloat16）
        special_case: str - 特殊场景（large_matrix/small_value等）
    
    返回：
        (atol, rtol) - 精度标准
    """
    precision_table = {
        "GEMM": {
            "float16": (1e-3, 1e-3),
            "float32": (1e-5, 1e-5),
            "bfloat16": (1e-2, 5e-3),
        },
        "Softmax": {
            "float16": (1e-3, 1e-3),
            "float32": (1e-4, 1e-4),
            "bfloat16": (1e-2, 5e-3),
        },
        "Activation": {
            "float16": (1e-3, 1e-3),
            "float32": (1e-5, 1e-5),
            "bfloat16": (1e-2, 5e-3),
        },
    }
    
    atol, rtol = precision_table[op_category][dtype]
    
    if special_case == "large_matrix":
        atol = atol * 5  # 大矩阵放宽精度
    elif special_case == "small_value":
        rtol = 0  # 小值仅验证绝对误差
    
    return atol, rtol
```

---

## 四、与 cannbot 精度标准对比

| 维度 | cannbot ops-precision-standard | TileLang precision-standard |
|------|-------------------------------|------------------------------|
| **覆盖范围** | 整数、浮点、量化、混合精度 | 整数、浮点、量化 |
| **特殊场景** | INF/NAN、小值域、量化范围 | INF/NAN、小值域、大规模累加 |
| **算子类别** | 未区分算子类别 | 区分算子类别（GEMM/Softmax等） |
| **动态调整** | 支持 | 支持（large_matrix/small_value） |

---

## 五、总结

### 5.1 核心要点

1. **精度标准体系完整**：覆盖整数、浮点、量化运算
2. **算子类别区分**：不同算子类别有不同精度标准
3. **特殊场景处理**：INF/NAN、小值域、大规模累加
4. **动态调整机制**：根据特殊场景调整精度标准

### 5.2 应用建议

- **Phase 3 用户交互**：询问精度标准（如未定义）
- **Phase 4 测试配置**：根据算子类别和 dtype 选择精度标准
- **测试代码生成**：使用正确的 atol 和 rtol
- **特殊值验证**：单独处理 INF/NAN/零值/极小值