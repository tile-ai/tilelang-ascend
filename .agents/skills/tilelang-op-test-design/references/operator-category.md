# 算子类别划分依据详解

## 一、划分依据的三个维度

算子类别划分不是凭直觉，而是基于以下三个科学维度：

### 1.1 计算类型（硬件特性）

**依据**：算子主要使用哪类硬件单元

昇腾 NPU 有两类主要计算单元：
- **Cube 核**：矩阵乘单元（用于 GEMM）
- **Vector 核**：向量单元（用于 element-wise、reduction）

---

#### 纯 Cube 类算子

**定义**：仅使用 Cube 核（矩阵乘单元）的算子

**数学公式特征**：
- 仅含 `matmul` / `@` / 矩阵乘
- 无其他运算（或仅有简单后处理）

**典型算子**：
- MatMul
- BatchMatMul
- GroupedGEMM（纯矩阵乘）

**硬件需求**：
- 需要 L1（矩阵输入） + L0A/L0B（矩阵寄存器） + L0C（累加器）

**测试重点**：
- 矩阵维度组合（M/N/K）
- block size 组合（影响性能）
- 矩阵形状对齐（M%block_M==0）

---

#### 纯 Vector 类算子

**定义**：仅使用 Vector 核（向量单元）的算子

**数学公式特征**：
- 无 matmul
- 仅含 element-wise 运算（add/mul/sigmoid等）
- 或仅含归约运算（sum/max/min）

**典型算子**：
- Activation（SiLU, GELU, ReLU）
- Normalization（Softmax, LayerNorm）
- Reduction（ReduceSum, ReduceMax）

**硬件需求**：
- 仅需 UB（向量缓冲区）

**测试重点**：
- dtype 组合（FP16/FP32/BF16）
- shape 组合（含不规则 shape）
- 归约维度（如适用）

---

#### 混合（CV 融合）类算子

**定义**：同时使用 Cube 和 Vector 核的算子

**数学公式特征**：
- 含 matmul + element-wise 后处理
- 例如：GEMM + softmax + GEMM

**典型算子**：
- FlashAttention（matmul → softmax → matmul）
- 其他融合算子

**硬件需求**：
- 需要 L1/L0A/L0B/L0C（Cube）
- 需要 UB（Vector）
- 需要 workspace（核间数据传递）

**测试重点**：
- workspace 配置
- Developer vs Expert 模式对比
- 核间协作验证

---

### 1.2 复杂度级别（计算步骤）

**依据**：算子有多少个计算步骤

---

#### 单步（Single）复杂度

**定义**：只有 1 个主要计算步骤

**数学公式特征**：
- 公式简单，只有 1 个运算
- 无中间步骤

**典型算子**：
- Add: `C = A + B`
- Mul: `C = A * B`
- ReLU: `y = max(0, x)`

**测试特点**：
- 配置简单
- dtype 组合少
- 快速验证

---

#### 多步（Multi）复杂度

**定义**：有 2~5 个计算步骤

**数学公式特征**：
- 公式有多个运算步骤
- 有中间变量

**典型算子**：
- Softmax: `max → exp → sum → div`（4步）
- LayerNorm: `mean → var → normalize → scale`（4步）
- RMSNorm: `var → sqrt → normalize`（3步）

**测试特点**：
- 配置详细
- dtype 组合多（FP16/FP32/BF16）
- 精度按 dtype 不同
- 需要验证中间步骤正确性

---

#### 融合（Fusion）复杂度

**定义**：多个算子组合，有核间协作

**数学公式特征**：
- 多个算子组合
- 例如：GEMM + softmax + GEMM

**典型算子**：
- FlashAttention: `GEMM(Q,K) → softmax → GEMM(attn,V)`
- 其他融合算子

**测试特点**：
- 配置复杂
- 需要 Developer vs Expert 模式对比
- 需要验证 workspace 配置
- 需要验证核间协作

---

### 1.3 数学公式特征

**依据**：数学公式中的关键运算

---

#### GEMM 特征

**数学公式**：
```
C = A @ B
C[i,j] = sum_k A[i,k] * B[k,j]
```

**关键运算**：
- `matmul` / `@` / 矩阵乘

**测试策略**：
- 三维参数（M/N/K）组合多
- block size 组合重要
- 矩阵形状对齐检查

---

#### Softmax 特征

**数学公式**：
```
softmax(x_i) = exp(x_i - max) / sum_j(exp(x_j - max))
```

**关键运算**：
- `max`（找最大值）
- `exp`（指数）
- `sum`（归约）
- `div`（除法）

**测试策略**：
- dtype 组合多（FP16/FP32/BF16）
- 精度按 dtype 不同（FP16=1e-3, FP32=1e-4）
- 归约维度验证

---

#### Normalization 特征

**数学公式**：
```
LayerNorm: y = (x - mean) / sqrt(var + eps) * gamma + beta
RMSNorm: y = x / sqrt(mean(x^2) + eps)
```

**关键运算**：
- `mean`（均值）
- `var`（方差）
- `sqrt`（开方）
- `normalize`（归一化）

**测试策略**：
- eps 参数重要
- dtype 组合多
- 精度验证严格

---

#### Activation 特征

**数学公式**：
```
SiLU: y = x * sigmoid(x)
ReLU: y = max(0, x)
GELU: y = x * Φ(x)
```

**关键运算**：
- `sigmoid` / `relu` / `gelu`

**测试策略**：
- 简单配置
- dtype 组合适中
- 特殊值验证（0值、负值）

---

#### Reduction 特征

**数学公式**：
```
ReduceSum: sum(x, dim=-1)
ReduceMax: max(x, dim=-1)
```

**关键运算**：
- `sum(dim)` / `max(dim)` / `min(dim)`

**测试策略**：
- 归约维度重要
- dtype 组合适中
- shape 验证（归约后形状）

---

## 二、综合分类示例

### 2.1 GEMM 类（纯 Cube + Single）

**算子**：MatMul, BatchMatMul

**综合判断**：
- 计算类型：纯 Cube
- 复杂度：Single
- 数学特征：matmul

**测试策略**：
```python
{
    "dtype_count": 2,  # FP16, FP32
    "shape_count": 5,  # M/N/K 组合多
    "block_count": 3,  # block size 重要
    "三维参数": True,
    "精度标准": {
        "float16": (1e-3, 1e-3),
        "float32": (1e-5, 1e-5),
    }
}
```

---

### 2.2 Softmax 类（纯 Vector + Multi）

**算子**：Softmax, LogSoftmax, OnlineSoftmax

**综合判断**：
- 计算类型：纯 Vector
- 复杂度：Multi（4步）
- 数学特征：exp+sum+div

**测试策略**：
```python
{
    "dtype_count": 3,  # FP16, FP32, BF16
    "shape_count": 4,
    "block_count": 2,
    "精度按 dtype": True,  # 不同 dtype 精度不同
    "精度标准": {
        "float16": (1e-3, 1e-3),
        "float32": (1e-4, 1e-4),
        "bfloat16": (1e-2, 5e-3),
    }
}
```

---

### 2.3 Normalization 类（纯 Vector + Multi）

**算子**：LayerNorm, RMSNorm, GroupNorm

**综合判断**：
- 计算类型：纯 Vector
- 复杂度：Multi（3-4步）
- 数学特征：mean+var+sqrt

**测试策略**：
```python
{
    "dtype_count": 3,  # FP16, FP32, BF16
    "shape_count": 4,
    "block_count": 2,
    "eps参数": True,  # eps 重要
    "精度标准": {
        "float16": (1e-3, 1e-3),
        "float32": (1e-4, 1e-4),
        "bfloat16": (1e-2, 5e-3),
    }
}
```

---

### 2.4 Activation 类（纯 Vector + Single）

**算子**：SiLU, GELU, ReLU, Sigmoid

**综合判断**：
- 计算类型：纯 Vector
- 复杂度：Single（1步）
- 数学特征：sigmoid/relu/gelu

**测试策略**：
```python
{
    "dtype_count": 2,  # FP16, FP32
    "shape_count": 5,
    "block_count": 3,
    "特殊值": True,  # 0值、负值、极值
    "精度标准": {
        "float16": (1e-3, 1e-3),
        "float32": (1e-5, 1e-5),
    }
}
```

---

### 2.5 Fusion 类（混合 + Fusion）

**算子**：FlashAttention, 其他融合算子

**综合判断**：
- 计算类型：混合（Cube + Vector）
- 复杂度：Fusion（多算子组合）
- 数学特征：matmul+softmax+matmul

**测试策略**：
```python
{
    "dtype_count": 2,  # FP16, FP32
    "shape_count": 3,
    "block_count": 2,
    "Developer_vs_Expert": True,  # 需对比两种模式
    "workspace配置": True,  # workspace 重要
    "精度标准": {
        "float16": (1e-3, 1e-3),
    }
}
```

---

## 三、AI 判断流程示例

### 3.1 MatMul 算子判断

**AI 阅读信息**：
```
数学公式：C = A @ B
算法描述：矩阵乘法，分块计算
输入：A(M,K), B(K,N)
输出：C(M,N)
```

**AI 理解逻辑**：
1. 公式中只有 `@` → 矩阵乘运算
2. 无其他运算 → 纯 Cube 计算
3. 只有 1 步 → Single 复杂度
4. 典型 GEMM → GEMM 类

**AI 判断**：
```
计算类型: 纯 Cube
复杂度: Single
数学特征: matmul
综合类别: GEMM（纯矩阵乘）
```

---

### 3.2 Softmax 算子判断

**AI 阅读信息**：
```
数学公式：softmax(x_i) = exp(x_i) / sum_j(exp(x_j))
算法描述：
  1. 计算 max
  2. 计算 exp(x - max)
  3. 计算 sum(exp)
  4. 计算 div
```

**AI 理解逻辑**：
1. 公式中有 exp、sum、div，无 matmul → 纯 Vector
2. 有 4 步 → Multi 复杂度
3. 典型 softmax 公式 → Softmax 类

**AI 判断**：
```
计算类型: 纯 Vector
复杂度: Multi
数学特征: exp+sum+div
综合类别: Softmax（多步归一化）
```

---

### 3.3 FlashAttention 算子判断

**AI 阅读信息**：
```
数学公式：Attention = softmax(Q @ K^T / sqrt(d)) @ V
算法描述：
  1. GEMM(Q, K^T) → scores
  2. softmax(scores) → attn_weights
  3. GEMM(attn_weights, V) → output
```

**AI 理解逻辑**：
1. 公式中有两次 matmul + softmax → Cube + Vector 混合
2. 有 3 个算子组合 → Fusion 复杂度
3. 典型融合算子 → Fusion 类

**AI 判断**：
```
计算类型: 混合（CV 融合）
复杂度: Fusion
数学特征: matmul+softmax+matmul
综合类别: Fusion（融合算子）
```

---

## 四、总结

### 划分依据总结

| 维度 | 依据 | 判断方法 |
|------|------|---------|
| **计算类型** | 算子使用哪类硬件单元 | AI 分析数学公式中的关键运算（matmul/exp/sum等） |
| **复杂度级别** | 计算步骤数 | AI 分析算法描述或数学公式中的步骤数 |
| **数学公式特征** | 公式中的关键运算 | AI 识别关键运算类型 |

### 判断原则

**核心原则**：AI 自己理解算子实现逻辑后判断，不是脚本匹配

**判断流程**：
1. 阅读数学公式和算法描述
2. 理解计算逻辑和硬件需求
3. 综合判断计算类型、复杂度、数学特征
4. 给出最终类别判断

**禁止**：使用 Python 负脚本机械匹配公式字符串