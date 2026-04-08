---
name: tilelang-op-generate
description: "基于设计文档生成 TileLang-Ascend 算子实现代码与测试。从 design.md 中提取关键信息，结合 examples/ 中的参考实现生成可运行代码。触发：实现算子、写 kernel、生成代码、算子编码、根据设计文档实现。"
---

# TileLang-Ascend 算子代码生成

基于设计文档（`design.md`）和已有示例，生成可运行的算子实现与测试。

---

## 1. 从 design.md 中提取的信息（只取这些）

design.md 可能很长，**只提取以下字段，忽略其余内容**：

| 提取字段 | 所在章节 | 用途 |
|---------|---------|------|
| 数学公式 | §1 概述 | 理解计算逻辑 |
| 算法步骤分解 | §1 算法描述 | 确定计算顺序 |
| API 映射表 | §3 API 映射设计 | **核心**：每步用哪个 TileLang API |
| 伪代码 | §3 计算伪代码 | **核心**：代码骨架 |
| 输入输出 shape 和 dtype | §4 数据规格 | 函数签名和测试数据 |
| block 大小 | §5 Tiling 策略 | 分块参数 |
| pass_configs | §7 同步策略 | JIT 配置 |
| Golden 函数 | §8 验证方案 | 测试对比基准 |
| 测试用例表 | §8 验证方案 | 测试配置 |
| 精度标准 | §8 验证方案 | atol / rtol |

**明确忽略的内容**（这些容易误导）：
- 模式选型的分析推理过程
- 内存预算的计算过程和多轮优化迭代
- 风险点与注意事项（过于笼统）
- 交付清单（仅是文件列表）
- 任何标注为"待确认"的内容

---

## 2. 参考示例（优先级高于 design.md 伪代码）

**当 design.md 伪代码与 examples/ 中同类实现有冲突时，以 examples/ 为准。**

生成代码前，必须查阅 `examples/` 中的同类算子：

| 算子类型 | 参考示例 |
|---------|---------|
| 逐元素运算（add/mul/sigmoid/relu） | `examples/vec_add/`、`examples/activation/` |
| 归约运算（reduce_sum/max/min） | `examples/reduce/` |
| 归一化（softmax/layernorm/rmsnorm） | `examples/softmax/`、`examples/layer_norm/`、`examples/rms_norm/` |
| GEMM | `examples/gemm/` |
| 融合算子 | `examples/flash_attention/` |

查阅示例时关注：
1. **Kernel 结构**：`T.Kernel` 参数、`cid`/`vid` 用法
2. **Buffer 分配方式**：`T.alloc_ub` 的 shape 和 dtype
3. **tmp buffer 写法**：归约操作的 tmp 大小和 dtype
4. **同步方式**：`T.barrier_all()` 的位置
5. **数据搬运**：`T.copy` 的索引写法

---

## 3. 代码生成流程

### 步骤 1：读取设计文档

读取 `design.md`，按 §1 的表格提取字段。

### 步骤 2：查找参考示例

在 `examples/` 中找到最相似的算子实现，**完整阅读其代码**。

### 步骤 3：生成实现代码

基于 design.md 的 API 映射 + 参考示例的代码风格，生成 `example_{op}.py`。

文件结构：
```python
import tilelang
from tilelang import DataType, language as T
import torch

tilelang.cache.clear_cache()

# ========== 算子实现 ==========
@tilelang.jit(out_idx=[...], pass_configs={...})
def op_name(M, N, block_M, block_N, dtype="float"):
    # 分块计算
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)
    VEC_NUM = 2

    @T.prim_func
    def main(Input: T.Tensor((M, N), dtype), Output: T.Tensor((M, N), dtype)):
        with T.Kernel(..., is_npu=True) as (cid, vid):
            # buffer 分配
            # 数据搬入
            # 计算
            # 数据搬出
            pass

    return main

# ========== 测试 ==========
if __name__ == "__main__":
    torch.manual_seed(0)
    test_configs = [...]  # 来自 design.md §8

    for config in test_configs:
        # 1. 创建 kernel
        # 2. 生成输入数据
        # 3. 执行 kernel
        # 4. golden 对比
        # 5. 精度检查
        pass

    print("All tests passed!")
```

### 步骤 4：运行验证

```bash
python examples/{op}/example_{op}.py
```

如果报错，按以下顺序排查：
1. **编译错误** → 检查 buffer 大小、API 参数、对齐
2. **运行错误** → 检查索引越界、同步缺失
3. **精度错误** → 检查计算公式、数据类型、容差设置

---

## 4. 关键编码规范

### Buffer 分配

```python
# VEC_NUM = 2，每个 vector 核处理 block_M // VEC_NUM 行
a_ub = T.alloc_ub([block_M // VEC_NUM, block_N], dtype)

# 归约 tmp buffer（uint8 类型）
tmp_size = 3 * DataType(dtype).bits // 8 * block_M // VEC_NUM * block_N
tmp_ub = T.alloc_ub([tmp_size], "uint8")
```

### 数据搬运索引

```python
# 标准索引模式
row_start = bx * block_M + vid * block_M // VEC_NUM
T.copy(A[row_start, by * block_N], a_ub)
T.copy(a_ub, B[row_start, by * block_N])
```

### 同步

```python
# Expert 模式：手��同步
with T.Scope("V"):
    T.copy(A[...], a_ub)
    T.barrier_all()
    T.tile.exp(a_ub, a_ub)
    T.barrier_all()
    T.copy(a_ub, B[...])

# Developer 模式 + 自动同步：无需手动 barrier
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}
```

### 广播

```python
# 归约结果 [M, 1] 广播到 [M, N]
max_ub = T.alloc_ub([block_M // VEC_NUM, 1], dtype)
max_2d_ub = T.alloc_ub([block_M // VEC_NUM, block_N], dtype)
T.tile.broadcast(max_2d_ub, max_ub, tmp_ub)  # 需要 tmp buffer
```

### 测试模板

```python
# golden 对比
ref_output = torch.nn.functional.softmax(input_data, dim=-1)  # 或手写 golden
torch.testing.assert_close(output.cpu(), ref_output.cpu(), rtol=rtol, atol=atol)
```

---

## 5. Checklist

生成代码后逐项检查：

| # | 检查项 |
|---|--------|
| 1 | `out_idx` 与函数签名中的输出参数位置一致 |
| 2 | `block_M // VEC_NUM` 在 buffer 分配和索引中一致使用 |
| 3 | 所有 `T.alloc_ub` 的 shape 乘积不超 UB 容量 |
| 4 | 归约 tmp buffer 使用 `"uint8"` dtype |
| 5 | Expert 模式有 `T.Scope("V")` 和 `T.barrier_all()` |
| 6 | Developer 模式有对应的 `pass_configs` |
| 7 | 测试包含至少 2 个配置（小规模 + 典型规模） |
| 8 | golden 函数使用 PyTorch 标准实现 |
