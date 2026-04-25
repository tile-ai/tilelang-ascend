# causal_conv1d 算子设计文档

## 1. 概述

### 1.1 算子名称

`causal_conv1d` - 因果 1D 卷积算子，用于线性注意力机制（如 Mamba、RWKV 等）中的隐藏状态递推计算。支持 Prefill（FN VARLEN）和 Decode（UPDATE）两种模式。

### 1.2 数学公式

该算子实现因果卷积的滑动窗口计算：

```
对于每个 token t:
  1. hist = [x[t-width+1], x[t-width+2], ..., x[t-1]]  (历史 tokens)
  2. acc = bias + weight[0] * hist[0] + weight[1] * hist[1] + ... + weight[width-2] * hist[width-2] + weight[width-1] * x[t]
  3. 若 activation: y[t] = acc / (1 + exp(-acc))  (silu/swish)
     否则: y[t] = acc
  4. hist 更新: 移动窗口，添加 x[t]
```

其中：
- `width` 为卷积核大小（Prefill 支持 3/4/5/6，Decode 支持 3/4）
- `hist` 维持 `width-1` 个历史 token
- 支持可选 bias 和 activation（silu/swish）
- 核心计算为 **多步 element-wise mul + add**

### 1.3 计算特征分析

| 维度 | 分析结果 |
|------|---------|
| **计算类型** | Vector element-wise（无 GEMM） |
| **复杂度级别** | 中等（mul → add → mul → add ...，历史 buffer 管理） |
| **动态 shape** | Prefill: B、T 为符号维度，支持变长（cu_seqlens）；Decode: batch、seqlen 可变 |
| **核间协作** | 纯 Vector 核计算，无 Cube 核 |

### 1.4 典型配置示例

| 参数 | Prefill | Decode | 说明 |
|------|---------|--------|------|
| batch_size | 1 | 1 | batch size |
| seqlen | 2048 | 1 (或 >1 投机解码) | sequence length |
| dim | 2048 | 2048 | hidden dimension |
| width | 4 | 4 | 卷积核大小 |
| state_len | 3 | 3 | 状态长度 (= width-1) |
| block_M | 64 | - | Prefill seqlen 分块 |
| block_D | 512 | 512 | dim 分块 |

---

## 2. 编程模式选型

### 2.1 选型结论：**Developer 模式**

### 2.2 选型理由

| 因素 | 分析 |
|------|------|
| **无 GEMM 计算** | 纯 element-wise 操作，`T.tile.mul/add/exp` 即可 |
| **简单数据流** | GM → UB → element-wise → GM，无复杂层级 |
| **自动内存规划** | Prefill 关闭 `TL_ASCEND_MEMORY_PLANNING`，Decode 开启 |
| **历史 buffer 管理** | 使用 UB buffer 维护历史 tokens，编译器自动分配 |

Developer 模式的优势：
- 使用 `T.alloc_ub` 自动映射到 Unified Buffer
- 使用 `T.tile.*` 原语进行 element-wise 计算
- 编译器自动管理数据搬运和同步

### 2.3 pass_configs 配置

```python
# Prefill 模式
pass_configs_fn = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,  # 自动 CV 分离
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,        # 自动插入 barrier
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False, # 关闭内存规划
}

# Decode 模式
pass_configs_decode = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,  # 自动 CV 分离
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,        # 自动插入 barrier
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,  # 开启内存规划
}
```

---

## 3. API 映射设计

### 3.1 核心计算步骤 → TileLang API 映射

| 计算步骤 | PyTorch 参考 | TileLang API |
|---------|-------------|--------------|
| 加载 weight | `weight[0:width, d_offset]` | `T.copy(weight[i, d_offset], w*_ub)` |
| 加载 bias | `bias[d_offset]` | `T.copy(bias[d_offset], bias_ub)` 或 `T.tile.fill(bias_ub, 0.0)` |
| 加载历史 token | `x[hist_idx, d_offset]` | `T.copy(x[hist_global_idx, d_offset], hist*_ub)` |
| 加载当前 token | `x[t, d_offset]` | `T.copy(x[seq_start + t, d_offset], x_cur_ub)` |
| 初始化累加器 | `acc = bias` | `T.copy(bias_ub, acc_ub)` |
| **weight[i] * hist[i]** | `weight[i] * hist[i]` | `T.tile.mul(tmp_ub, w*_ub, hist*_ub)` |
| **累加** | `acc += tmp` | `T.tile.add(acc_ub, acc_ub, tmp_ub)` |
| **weight[width-1] * x[t]** | `weight[width-1] * x[t]` | `T.tile.mul(tmp_ub, w_last_ub, x_cur_ub)` → `T.tile.add` |
| **silu activation** | `acc / (1 + exp(-acc))` | `T.tile.sub(denom_ub, zero_ub, acc_ub)` → `T.tile.exp` → `T.tile.add` → `T.tile.div` |
| 存储输出 | `y[t] = out` | `T.copy(out_ub, y[seq_start + t, d_offset])` |
| 历史窗口移动 | `hist = hist[1:] + [x[t]]` | `T.copy(hist1_ub, hist0_ub)` ... `T.copy(x_cur_ub, hist_last_ub)` |
| 加载初始状态 | `hist = conv_state[ci, :]` | `T.copy(conv_state[ci, h, d_offset], hist*_ub)` |
| 存储最终状态 | `conv_state[ci, :] = last hist` | `T.copy(state_tmp_ub, conv_state[ci, pos, d_offset])` |

### 3.2 关键 API 说明

- **`T.copy(src, dst)`**：自动推断层级间数据搬运（GM → UB 或 UB → GM）
- **`T.tile.mul/add/sub/div(dst, src1, src2)`**：Vector 核 element-wise 原语
- **`T.tile.exp(dst, src)`**：Vector 核指数运算
- **`T.tile.fill(dst, value)`**：Vector 核填充原语
- **`T.alloc_ub(shape, dtype)`**：Unified Buffer 分配

---

## 4. 数据规格与内存规划

### 4.1 Prefill 模式输入张量

| 张量 | Shape | Dtype | 说明 |
|------|-------|-------|------|
| x | `[total_len, dim]` | float16 | packed layout，所有序列拼接 |
| weight | `[width, dim]` | float16 | 卷积核，kernel format only |
| bias | `[dim]` | float16 | 可选偏置 |
| conv_state | `[num_cache_lines, state_len, dim]` | float16 | 缓存状态，kernel format only |
| cu_seqlens | `[batch_size + 1]` | int32 | 变长序列边界 |
| cache_indices | `[batch_size]` | int32 | 缓存索引（可选） |
| initial_state_mode | `[batch_size]` | int32 | 初始状态标志（可选） |

### 4.2 Prefill 模式输出张量

| 张量 | Shape | Dtype | 说明 |
|------|-------|-------|------|
| y | `[total_len, dim]` | float16 | 卷积输出 |
| conv_state | `[num_cache_lines, state_len, dim]` | float16 | 更新后状态（最后 width-1 tokens） |

### 4.3 Decode 模式输入张量

| 张量 | Shape | Dtype | 说明 |
|------|-------|-------|------|
| x | `[batch, seqlen, dim]` 或 `[batch, dim]` 或 `[dim]` | float16 | 输入 tokens |
| weight | `[width, dim]` | float16 | 卷积核，kernel format only |
| bias | `[dim]` | float16 | 可选偏置 |
| conv_state | `[num_cache_lines, state_len, dim]` | float16 | 缓存状态，kernel format only |
| cache_indices | `[batch]` | int32 | 缓存索引（可选） |
| num_accepted_tokens | `[batch]` | int32 | 投机解码实际接受 token 数（可选） |

### 4.4 Decode 模式输出张量

| 张量 | Shape | Dtype | 说明 |
|------|-------|-------|------|
| y | `[batch, seqlen, dim]` 或 `[batch, dim]` | float16 | 卷积输出 |
| conv_state | `[num_cache_lines, state_len, dim]` | float16 | 更新后状态 |

### 4.5 内存层级规划（Developer 模式）

```
GM (全局内存)
  │
  ├─ T.copy → UB (T.alloc_ub)
  │   ├─ w0_ub ~ w5_ub:  [block_D]     — weight 分量（width 个）
  │   ├─ bias_ub:        [block_D]     — 偏置
  │   ├─ hist0_ub ~ hist4_ub: [block_D] — 历史 tokens（width-1 个）
  │   ├─ x_cur_ub:       [block_D]     — 当前 token
  │   ├─ acc_ub:         [block_D]     — 累加器
  │   ├─ tmp_ub:         [block_D]     — 临时 buffer
  │   ├─ out_ub:         [block_D]     — 输出 buffer
  │   ├─ (activation)
  │   │   ├─ zero_ub:    [block_D]     — 零值
  │   │   └─ denom_ub:   [block_D]     — 分母 buffer
  │   └─ state_tmp_ub:   [block_D]     — 状态更新临时 buffer
```

### 4.6 UB 容量估算（典型配置: dim=2048, block_D=512）

```
Prefill 模式 (width=4):
  w0~w3_ub:      4 × 512 × 2B = 4KB
  bias_ub:       512 × 2B = 1KB
  hist0~hist2_ub: 3 × 512 × 2B = 3KB
  x_cur_ub:      512 × 2B = 1KB
  acc_ub:        512 × 2B = 1KB
  tmp_ub:        512 × 2B = 1KB
  out_ub:        512 × 2B = 1KB
  (activation):  2 × 512 × 2B = 2KB
  state_tmp_ub:  512 × 2B = 1KB
  总计 ≈ 14KB（远小于 UB 容量上限）

Decode 模式 (width=4):
  w0~w3_ub:      4 × block_D × 2B
  bias_ub:       block_D × 2B
  hist0~hist2_ub: 3 × block_D × 2B
  (per token) x_cur/acc/tmp/out: 4 × block_D × 2B
  总计 ≈ 8 × block_D × 2B
```

---

## 5. Tiling 策略

### 5.1 Prefill Block 划分

| 维度 | 策略 | 说明 |
|------|------|------|
| **Grid** | `batch_size * dim_num * seqlen_num` | 三维分块 |
| **batch 分块** | 每个 batch item 独立处理 | |
| **dim 分块** | `dim_num = ceildiv(dim, block_D)` | block_D = 512 或 256 |
| **seqlen 分块** | `seqlen_num = ceildiv(total_len, block_M)` | block_M = 64 |

### 5.2 Decode Block 划分

| 维度 | 策略 | 说明 |
|------|------|------|
| **Grid** | `dim_num` | 仅按 dim 分块 |
| **dim 分块** | `dim_num = ceildiv(dim, block_D)` | block_D = dim（单块避免越界） |

### 5.3 Tile Shape 设计（典型配置: dim=2048）

| Buffer | Shape | 说明 |
|--------|-------|------|
| w*_ub | `[512]` | block_D |
| hist*_ub | `[512]` | block_D |
| x_cur_ub | `[512]` | block_D |
| acc_ub | `[512]` | block_D |

---

## 6. 循环与调度结构

### 6.1 Prefill Kernel 结构设计

```python
@tilelang.jit(out_idx=[-1], pass_configs=pass_configs_fn)
def causal_conv1d_fn_kernel(batch_size, width, has_bias, has_activation, 
                            has_cache_indices, has_initial_state_mode,
                            block_M=64, block_D=512):
    @T.prim_func
    def main(x, weight, bias, conv_state, cu_seqlens, cache_indices, 
             initial_state_mode, y):
        with T.Kernel(grid_size, is_npu=True) as (cid, vid):
            # 计算 batch_id, seq_block, dim_block
            batch_id = cid // (dim_num * seqlen_num)
            seq_block = (cid % (dim_num * seqlen_num)) // dim_num
            dim_block = (cid % (dim_num * seqlen_num)) % dim_num
            d_offset = dim_block * block_D
            
            # 获取序列范围
            seq_start = cu_seqlens[batch_id]
            seq_end = cu_seqlens[batch_id + 1]
            seqlen = seq_end - seq_start
            
            # 分配 UB buffer
            hist0_ub = T.alloc_ub((block_D,), dtype)  # width-1 个历史 buffer
            w0_ub = T.alloc_ub((block_D,), dtype)      # width 个 weight buffer
            bias_ub = T.alloc_ub((block_D,), dtype)
            x_cur_ub = T.alloc_ub((block_D,), dtype)
            acc_ub = T.alloc_ub((block_D,), dtype)
            tmp_ub = T.alloc_ub((block_D,), dtype)
            out_ub = T.alloc_ub((block_D,), dtype)
            
            # 加载 weight
            T.copy(weight[0, d_offset], w0_ub)
            T.copy(weight[1, d_offset], w1_ub)
            ...
            
            # 加载初始状态或历史 tokens
            if has_initial_state_mode and seq_block == 0:
                T.copy(conv_state[ci, h, d_offset], hist*_ub)
            else:
                T.copy(x[hist_global_idx, d_offset], hist*_ub)
            
            # 主循环：遍历 block 内 tokens
            for t_idx in T.serial(num_tokens):
                t = t_block_start + t_idx
                T.copy(x[seq_start + t, d_offset], x_cur_ub)
                
                # 累加计算
                T.copy(bias_ub, acc_ub)
                T.tile.mul(tmp_ub, w0_ub, hist0_ub)
                T.tile.add(acc_ub, acc_ub, tmp_ub)
                ...
                
                # activation (可选)
                if has_activation:
                    T.tile.sub/exp/add/div(out_ub, ...)
                else:
                    T.copy(acc_ub, out_ub)
                
                T.copy(out_ub, y[seq_start + t, d_offset])
                
                # 历史窗口移动
                T.copy(hist1_ub, hist0_ub)
                ...
                T.copy(x_cur_ub, hist_last_ub)
            
            # 最后一个 block：更新 conv_state
            if seq_block == seqlen_num - 1:
                T.copy(x[last_hist_idx, d_offset], state_tmp_ub)
                T.copy(state_tmp_ub, conv_state[ci, pos, d_offset])
    
    return main
```

### 6.2 Decode Kernel 结构设计

```python
@tilelang.jit(out_idx=[-1], pass_configs=pass_configs_decode)
def causal_conv1d_decode_kernel(batch, seqlen, dim, state_len, width,
                                has_bias, has_activation, 
                                has_cache_indices, has_num_accepted_tokens,
                                block_D=512):
    @T.prim_func
    def main(x, weight, bias, conv_state, cache_indices, 
             num_accepted_tokens, y):
        with T.Kernel(dim_num, is_npu=True) as (cid, vid):
            d_offset = cid * block_D
            
            # 加载 weight
            T.copy(weight[0, d_offset], w0)
            ...
            
            for b_idx in T.serial(batch):
                ci = cache_indices[b_idx]
                
                # 计算状态偏移（投机解码支持）
                accepted = num_accepted_tokens[b_idx] if has_num_accepted_tokens else seqlen
                state_token_offset = clamp(accepted - 1, 0, state_len - hist_len)
                
                # 加载历史
                T.copy(conv_state[ci, state_token_offset + h, d_offset], hist*_ub)
                
                for t_idx in T.serial(seqlen):
                    T.copy(x[b_idx, t_idx, d_offset], x_cur)
                    
                    # 卷积计算
                    T.tile.mul/add(...)
                    
                    # activation (可选)
                    if has_activation:
                        T.tile.sub/exp/add/div(out, ...)
                    
                    T.copy(out, y[b_idx, t_idx, d_offset])
                    
                    # 历史移动（使用中间 buffer 确保顺序）
                    T.copy(hist1_ub, tmp_hist0)
                    T.copy(hist2_ub, tmp_hist1)
                    T.copy(tmp_hist0, hist0_ub)
                    T.copy(tmp_hist1, hist1_ub)
                    T.copy(x_cur, hist2_ub)
                
                # 状态更新
                T.copy(conv_state[ci, state_token_offset + 1, d_offset], tmp_state1)
                T.copy(conv_state[ci, state_token_offset + 2, d_offset], tmp_state2)
                T.copy(tmp_state1, conv_state[ci, 0, d_offset])
                T.copy(tmp_state2, conv_state[ci, 1, d_offset])
                
                for write_t in T.serial(seqlen):
                    T.copy(x[b_idx, write_t, d_offset], write_x)
                    T.copy(write_x, conv_state[ci, 2 + write_t, d_offset])
    
    return main
```

### 6.3 调度选择

| 循环/操作 | 调度类型 | 理由 |
|----------|---------|------|
| token 循环 | `T.serial(num_tokens)` | token 间有历史依赖，必须串行 |
| batch 循环 (Decode) | `T.serial(batch)` | 每个 batch item 独立处理 |
| element-wise | `T.tile.*` | Vector 核自动向量化 |

---

## 7. 同步策略

### 7.1 自动同步（pass_configs）

由于使用 `TL_ASCEND_AUTO_SYNC: True`，编译器自动在以下位置插入同步：

| 同步点 | 位置 |
|--------|------|
| 数据加载后 | `T.copy` → element-wise 之间 |
| element-wise 后 | `T.tile.*` → `T.copy` 之间 |

### 7.2 核间同步（CV 分离）

由于使用 `TL_ASCEND_AUTO_CV_COMBINE: True`，编译器自动处理 Vector 核内的操作合并。

---

## 8. 验证方案

### 8.1 Golden 函数（PyTorch 参考实现）

```python
def causal_conv1d_fn_ref(x, weight, conv_states, bias=None, activation="silu",
                         cache_indices=None, query_start_loc=None, 
                         initial_state_mode=None):
    batch_size = cache_indices.size(0)
    width = weight.shape[0]
    dim = x.shape[1]
    
    y = torch.zeros_like(x)
    
    for b in range(batch_size):
        seq_start = query_start_loc[b].item()
        seq_end = query_start_loc[b + 1].item()
        seqlen = seq_end - seq_start
        ci = cache_indices[b].item() if cache_indices is not None else b
        
        # 加载初始状态
        history = []
        if initial_state_mode[b].item() != 0:
            for h in range(width - 1):
                history.append(conv_states[ci, h, :].clone())
        else:
            for _ in range(width - 1):
                history.append(torch.zeros(dim))
        
        for t in range(seqlen):
            x_t = x[seq_start + t, :]
            
            # 卷积计算
            acc = bias.clone() if bias is not None else torch.zeros(dim)
            for w_idx in range(width - 1):
                acc = acc + weight[w_idx, :] * history[w_idx]
            acc = acc + weight[width - 1, :] * x_t
            
            # activation
            if activation in ["silu", "swish"]:
                out = acc / (1.0 + torch.exp(-acc))
            else:
                out = acc
            
            y[seq_start + t, :] = out
            
            # 历史移动
            for h in range(width - 2):
                history[h] = history[h + 1]
            history[width - 2] = x_t.clone()
        
        # 更新状态
        for pos in range(width - 1):
            last_idx = seqlen - (width - 1) + pos
            if last_idx >= 0:
                conv_states[ci, pos, :] = x[seq_start + last_idx, :]
    
    return y
```

### 8.2 测试配置

| 模式 | 参数配置 | 测试组合 |
|------|---------|---------|
| **Prefill** | seqlen=2048, dim=2048, width=4 | has_bias=True/False, has_activation=True/False |
| **Prefill** | seqlen=2048, dim=2048, width=3/5/6 | 基础功能验证 |
| **Decode** | batch=1, seqlen=1, dim=2048, width=4 | has_bias=True/False, has_activation=True/False |
| **Decode** | batch=1, seqlen=4 (投机解码), dim=2048, width=3/4 | 投机解码验证 |

### 8.3 精度容忍度

| 配置 | rtol | atol |
|------|------|------|
| dim ≤ 512 | 1e-2 | 1e-2 |
| dim > 512 | 1e-2 | 1e-2 |

---

## 9. 特殊功能说明

### 9.1 投机解码支持 (Decode 模式)

当 `seqlen > 1` 且 `num_accepted_tokens` 提供时：
- 状态偏移: `state_token_offset = num_accepted_tokens[b] - 1`
- 用途: 投机解码被拒绝时，从实际接受的位置读取历史
- 状态更新: 移动旧状态 + 写入新 tokens

```python
# 状态偏移计算
accepted = num_accepted_tokens[b_idx]
max_offset = state_len - hist_len
state_token_offset = clamp(accepted - 1, 0, max_offset)

# 从偏移位置读取历史
T.copy(conv_state[ci, state_token_offset + h, d_offset], hist*_ub)

# 状态更新：先移动，再写入
T.copy(conv_state[ci, state_token_offset + 1], tmp_state1)
T.copy(conv_state[ci, state_token_offset + 2], tmp_state2)
T.copy(tmp_state1, conv_state[ci, 0])
T.copy(tmp_state2, conv_state[ci, 1])

for write_t in range(seqlen):
    T.copy(x[b_idx, write_t], conv_state[ci, 2 + write_t])
```

### 9.2 变长序列支持 (Prefill 模式)

- 输入为 packed layout: `x = [total_len, dim]`
- 使用 `cu_seqlens` 定位每个序列的起始和结束
- 支持不同的序列长度在同一 batch 中

### 9.3 初始状态模式 (Prefill 模式)

- `initial_state_mode[b] != 0` 表示使用预加载的 conv_state 作为初始历史
- 否则使用零值或从前序 tokens 加载历史

---

## 10. 当前限制

| 限制 | Prefill | Decode | 说明 |
|------|---------|--------|------|
| **width 支持** | 3, 4, 5, 6 | 3, 4 | 卷积核大小限制 |
| **weight layout** | kernel format only | kernel format only | `(width, dim)` |
| **conv_state layout** | kernel format only | kernel format only | `(num_cache_lines, state_len, dim)` |
| **dtype** | float16 → float32 计算 | float16 | Prefill 使用 float32 提高精度 |

---

## 11. 交付清单

| 交付物 | 路径 | 状态 |
|--------|------|------|
| 设计文档 | `examples/causal_conv1d/design.md` | 本文档 |
| 算子实现 v2 | `examples/causal_conv1d/causal_conv1d_v2.py` | 已完成 |

---

## 附录 A：Kernel Cache 管理

### A.1 设计背景

不同配置组合（batch, width, has_bias, has_activation 等）需要不同的 kernel，使用缓存避免重复编译。

```python
_kernel_cache_fn = {}
_kernel_cache_decode = {}

def get_fn_kernel(batch_size, width, has_bias, has_activation, 
                  has_cache_indices, has_initial_state_mode, 
                  block_M=64, block_D=512):
    key = (batch_size, width, has_bias, has_activation, 
           has_cache_indices, has_initial_state_mode, block_M, block_D)
    if key not in _kernel_cache_fn:
        _kernel_cache_fn[key] = causal_conv1d_fn_kernel(...)
    return _kernel_cache_fn[key]
```

---

## 附录 B：Pass Config 差异说明

### B.1 Prefill vs Decode

| Pass | Prefill | Decode | 说明 |
|------|---------|--------|------|
| TL_ASCEND_MEMORY_PLANNING | False | True | Prefill 手动管理，Decode 自动规划 |

Prefill 关闭内存规划的原因：
- 需要显式控制 UB buffer 的生命周期（跨 token 循环）
- 历史 buffer 在整个 block 内持续使用，不能被编译器回收

Decode 开启内存规划的原因：
- seqlen 通常较小（1 或投机解码的几个 tokens）
- 可以利用编译器自动优化内存使用

---