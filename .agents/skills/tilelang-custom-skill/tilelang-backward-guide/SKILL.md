---
name: tilelang-backward-guide
description: "TileLang-Ascend 反向梯度计算算子开发指南。涵盖真值构造方法、双重验证策略、精度检查标准、常见陷阱与实现模式。触发关键词：bwd、backward、grad、反向、梯度、dQ/dK/dV、backward kernel、gradient kernel、计算梯度、反向算子、精度不对。"
---

# TileLang-Ascend 反向梯度计算算子开发指南

## 1. 核心概念

反向计算算子的输出是**梯度**（gradient），不是前向计算结果。它接收的是上游传来的梯度 `dY`（即 `∂L/∂Y`）和前向保存的中间数据 `X`，输出的是对输入的梯度 `dX`（即 `∂L/∂X`）。

**与普通算子的本质区别**：

- 普通算子输出可直接用标准 PyTorch 函数作为 golden
- 反向算子必须通过 **autograd** 对前向计算结果反向传播，或**手动推导**梯度公式

---

## 2. 真值（Ground Truth）构造方法

真值构造是反向算子开发中最关键的环节，**出错会直接导致 kernel 验证无效**。禁止直接编写 gradient 公式来验证另一个 gradient 公式。

### 方法 A：PyTorch autograd（首选 ⭐）

通过可微前向计算 + `.backward()` 自动求导，不依赖手写梯度公式。

```python
def compute_ground_truth(inputs, dY):
    """通过 autograd 自动求导生成真实梯度"""
    X = inputs.clone().detach().float().requires_grad_(True)
    Y = forward_reference(X)  # 标准 PyTorch 前向
    Y.backward(dY.float())
    return X.grad.clone()
```

**关键原则**：
1. **`.detach()` 断开原计算图**，避免干扰
2. **`.float()` 提升到 float32**，保证真值精度足够
3. **`.requires_grad_(True)`** 让 autograd 能对 X 求导
4. **`dY` 可独立生成**（`torch.randn`），不需要从前向推导

### 方法 B：独立 CPU autograd（用于交叉验证 ⭐）

将数据 `.cpu().float()` 后在 CPU 上重新计算，排除 NPU autograd 潜在问题。

```python
X_cpu = X.cpu().float().clone().detach().requires_grad_(True)
Y_cpu = forward_reference(X_cpu)
Y_cpu.backward(dY.cpu().float())
grad_cpu = X_cpu.grad.clone()
```

**两种方法结果应完全一致**。如果偏差较大（> 1e-6），说明前向实现或 autograd 链路有问题。

### 方法 C：NPU 原生算子 autograd

当 NPU 有对应的前向原生算子（如 `torch_npu.npu_gelu`）时，直接在前向结果上 `.backward()`。

```python
x_ref = x.clone().detach().requires_grad_(True)
y_ref = torch_npu.npu_gelu(x_ref)
y_ref.backward(dy)
dx_ref = x_ref.grad.clone()
```

### 方法 D：手写数学公式

仅当 autograd 不可用时使用。必须从数学公式推导，**禁止凭直觉编写**。

```python
# 示例：silu 的梯度公式
# forward: y = x * sigmoid(x)
# backward: dx = dy * (sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x)))
sig = torch.sigmoid(x)
dx_ref = dy * (sig + x * sig * (1 - sig))
```

---

## 3. 双重验证策略 ⭐

反向算子真值容易隐蔽出错，**必须**采用双重验证。

```
真值构造
  ├── 方法 1：autograd 自动求导（主方案）
  ├── 方法 2：CPU 独立计算或独立推导（验证方案）
  └── 两个方法的结果必须一致（误差 < 1e-6）→ 真值可信
```

**双重验证检查**：

```python
# 交叉验证：autograd vs CPU autograd
assert torch.allclose(grad_autograd, grad_cpu, atol=1e-6, rtol=1e-6), \
    "真值不一致，请检查前向实现"
```

**常见双重验证不一致原因**：
- 前向实现有 bug（如 mask 方向反、reshape 顺序错）
- `.float()` / `requires_grad` 操作打断了计算图
- `einops.rearrange` 维度变换与预期不符
- 多卡/多线程时 `torch.manual_seed()` 未正确设置

---

## 4. 精度检查

### 检查函数标准模板

```python
def check(name, a, b, atol=1e-2, rtol=1e-2):
    """比较 kernel 输出和真值，输出详细误差信息"""
    a_cpu = a.cpu().float()
    b_cpu = b.cpu().float() if b.is_floating_point() else b.float()
    ok = torch.allclose(a_cpu, b_cpu, atol=atol, rtol=rtol)
    if not ok:
        diff = (a_cpu - b_cpu).abs()
        print(f"  {name}: max={diff.max():.6e}, mean={diff.mean():.6e}, "
              f"ratio={(diff > atol).float().mean():.2%}")
    return ok
```

### 容差选择指南

| 场景 | atol | rtol | 说明 |
|------|------|------|------|
| 简单 element-wise（gelu_grad） | 1e-3 | 1e-3 | 单步运算，精度高 |
| 归约类（rms_norm_grad） | 1e-2 | 1e-2 | 归约累积误差增大 |
| 混合融合（linear_attn_bwd） | 5e-2 | 5e-2 | 多步 GEMM + mask，float16 累积 |

### 误差等级判断

| max_err 数量级 | 问题类型 | 处理 |
|---------------|---------|------|
| < 1e-4 | 正常数值精度误差 | 无需处理 |
| ~1e-2 | float16 累积或归约误差 | 调宽容差或排查 |
| > 0.1 | 公式/实现有 bug | **必须修复** |

---

## 5. 常见陷阱

### 5.1 真值构造陷阱

| 陷阱 | 症状 | 修复 |
|------|------|------|
| `q.grad = None` 后再调用 `ref_program(q)` 做 backward | 不清楚 `q` 的计算图是否重建 | 用 `q.clone().detach().requires_grad_(True)` 创建独立张量 |
| `.float()` 创建的张量不参与 autograd | 梯度为 None | 确保 `requires_grad` 传播；或在 `.float()` 前先 `.requires_grad_(True)` |
| `retain_graph=True` 导致内存累积 | OOM | 不需要时设为 `False`，或每次 `backward` 后 `zero_grad()` |
| 使用同一张量的 `.grad` 做多次比较 | 被累积污染 | 每次用 `t.clone().detach()` 保存 `.grad` 值 |

### 5.2 前向 scale 因子陷阱

正向有 `Q * scale` 时，反向需要处理 scale 对梯度的影响：

```
前向: Y = f(Q * scale, K, V)
后向: dQ = ∂L/∂(Q*scale) * scale   ← 需要额外乘 scale
       dK = ∂L/∂K                  ← 不需要额外乘 scale（∂f/∂K 内部已含 scale）
       dV = ∂L/∂V                  ← 同上
```

**验证方式**：用 autograd 真值对比，如果 dK/dV 正确但 dQ 错误（或反），则是 scale 处理错误。

### 5.3 设备与 dtype 陷阱

| 陷阱 | 症状 | 修复 |
|------|------|------|
| `a.cpu()` 对比 `b.float()` 但 b 在 NPU | 隐式传输，结果不对 | 明确 `.cpu()` 再对比 |
| gradient 是 float16 而 kernel 输出是 float32 | `allclose` 返回 False | 统一 cast 到 float32 再对比 |
| `torch.manual_seed()` 在 NPU 上效果不确定 | 每次结果不同 | 在 CPU 上生成数据再 `.to(device)` |

### 5.4 反向迭代顺序陷阱

很多融合算子的反向需要**逆序**遍历前向的分块（如 chunked attention 的 dK/dV 从最后一个 chunk 向前遍历）：

```python
# 正向: for i in range(NT): ...
# 反向: for i in range(NT):
#           start = NT - 1 - i  # 逆序
#           或 for i in reversed(range(NT))
```

**验证方式**：用 NT=2（恰好 2 个 chunk）测试，如果通过而 NT=1 也通过但 NT=8 不通过，很可能是迭代顺序问题。

---

## 6. 反向 Kernel 实现模式

### 模式 1：简单 Element-wise 梯度

```python
# 示例：silu backward = dy * derivative(x)
y = forward(x)
dy_grad = dy * derivative(x)   # 从 x 重新计算导数
```

### 模式 2：需要保存前向中间结果的梯度

```python
# 示例：rms_norm backward 需要 forward 时的 rms 和统计量
# forward: save rms, x_normalized
# backward: dx = (dy - x_norm * dot(dy, x_norm) / N) / rms
```

### 模式 3：需要 workspace 传递的融合算子梯度

```python
@tilelang.jit(out_idx=[...], workspace_idx=[...], pass_configs={...})
def bwd_kernel(...):
    @T.prim_func
    def main(Q, K, V, dO, dQ, dK, dV, ws_s, ws_h, ws_dh, ws_s2, ws_h2, ws_dh2):
        with T.Kernel(total_blocks, is_npu=True) as (cid, vid):
            with T.Scope("C"):
                # Cube 核：计算部分梯度，写入 workspace
                T.gemm_v0(A, B, partial_grad, ...)
                T.copy(partial_grad, ws_buffer[...])
            with T.Scope("V"):
                # Vector 核：从 workspace 读取，后处理，累加到输出
                T.copy(ws_buffer[...], grad_ub)
                # RMW 模式累加
                T.copy(dQ[...], dq_tmp)
                dq_tmp += grad_ub
                T.copy(dq_tmp, dQ[...])
```

**关键点**：
- Cube 核计算部分积（partial gradient），写入 workspace
- Vector 核从 workspace 读取，做 mask/scale 后处理，RMW 到输出
- `out_idx` 指向 `dQ, dK, dV`，`workspace_idx` 指向中间缓冲

### 模式 4：手动掩码 + 累积的融合梯度

```python
# 示例：chunked attention 中 tril/triu 掩码
with T.Scope("V"):
    T.copy(ws_s[...], ds_ub)
    for r, c in T.Parallel(...):
        if r < c:          # 上三角清零（对应数学中的 tril 条件）
            ds_ub[r, c] = 0
    T.copy(ds_ub, ws_s2[...])
```

---

## 7. 开发与调试流程

```
1. 确定前向公式
   └── 写出标准 PyTorch 前向实现

2. 构造真值（双重验证）
   ├── 方法 A：autograd 自动求导
   ├── 方法 B：CPU 独立 autograd
   └── 确认 A ≈ B（误差 < 1e-6）

3. 实现反向 kernel
   └── 基于真值推导每个阶段的计算公式

4. 分段验证（小 → 大配置）
   ├── 最小配置（B=1, S=small, H=1, D=small）
   ├── 典型配置
   └── 多 block 配置

5. 精度调试
   ├── max_err > 0.1     → 公式/实现错误，用 T.printf 打印中间值
   ├── max_err ~ 1e-2    → float16 累积，检查 cast 和 accum_dtype
   └── max_err < 1e-4    → 正常精度误差
```

---

## 8. 快速自检清单

实现反向 kernel 后，逐项确认：

- [ ] 真值通过**双重验证**（autograd vs CPU 结果一致，误差 < 1e-6）
- [ ] 真值用 `.detach().float().requires_grad_(True)` 构造，不依赖原有 `.grad`
- [ ] 前向有 scale 因子时，检查了每个梯度是否需要乘 scale
- [ ] 容差设置与算子复杂度匹配（GEMM 融合 > 归约 > element-wise）
- [ ] `.cpu()` 在对比前统一执行，dtype 统一为 float32
- [ ] 最小配置（1 batch, 1 head, small dim）通过后再测试大配置
- [ ] 单 chunk + 多 chunk 分别测试（验证迭代顺序）

---

## 9. 参考示例

| 算子类型 | 示例文件 | 真值方法 | 特点 |
|---------|---------|---------|------|
| Element-wise 梯度 | `examples/activation/gelu_grad.py` | NPU 原生算子 autograd | 单步计算 |
| 归约梯度 | `examples/normalization/rms_norm.py` | 前向 + autograd | 需要 forward 中间结果 |
| 融合注意力梯度 | `examples/linear_attention/example_linear_attn_bwd_ascend.py` | CPU autograd | workspace 传递、逆向遍历 |
| 转置 GEMM 梯度 | `examples/grouped_gemm/example_grouped_gemm_bwd.py` | 手写参考 | 分组转置 |
| 排序/置换梯度 | `examples/moe_token_permute/moe_token_permute_grad.py` | NPU 前向 + autograd | gather-reduce 模式 |
