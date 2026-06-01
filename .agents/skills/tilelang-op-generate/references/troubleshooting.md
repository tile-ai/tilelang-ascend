# 算子疑难解答

## 目录

- [编译时错误](#编译时错误)
- [运行时错误](#运行时错误)
- [测试设计原则](#测试设计原则)
- [调试技巧](#调试技巧)
- [常见模式问题](#常见模式问题)
- [性能调优清单](#性能调优清单)

---

## 编译时错误

### 1. 内存分配失败

**错误信息**:
```
TVMError: Memory allocation failed for: buffer_name required: XXXX, new memory available: YYYY
```

**原因**: UB空间不足，所有buffer总大小超过限制

**解决方案**:
1. 减小分块大小：
   ```python
   # 原始
   block_M, block_N = 128, 256
   
   # 修改为更小的值
   block_M, block_N = 64, 128
   ```

2. 开启自动内存规划以复用buffer：
   ```python
   pass_configs = {
       tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
   }
   @tilelang.jit(out_idx=[1], pass_configs=pass_configs)
   ```

3. 减少中间buffer数量，尽可能复用

### 2. 维度不匹配

**错误信息**:
```
error: Source and Dest dimension must match.
```

**原因**: broadcast操作的源和目标shape不符合要求

**解决方案**:
确保源buffer的shape为 `[M, 1]` 或 `[1, N]`，目标buffer为 `[M, N]`：

```python
# 正确
max_ub = T.alloc_ub([block_M // VEC_NUM, 1], dtype)      # [M, 1]
max_2d_ub = T.alloc_ub([block_M // VEC_NUM, block_N], dtype)  # [M, N]
T.tile.broadcast(max_2d_ub, max_ub)

# 错误：源buffer是1D
max_ub = T.alloc_ub([block_M // VEC_NUM], dtype)  # [M] - 错误
```

### 3. API参数错误

**错误信息**:
```
error: max() takes 3 positional arguments but 4 were given
```

**原因**: API调用参数不正确

**解决方案**:
查看API文档确认正确的参数签名：

```python
# 错误
T.tile.max(dst, src0, src1, src2)  # 参数过多

# 正确
T.tile.max(dst, src0, src1)  # dst = max(src0, src1)
```

### 4. GEMM 除零编译错误

**错误信息**:
```
InternalError: Check failed: pb->value != 0 (0 vs. 0) : Divide by zero
 --> ...py:65:18  bx = cid // n_num
```

**原因**: `n_num = N // block_N = 0`（当 `block_N > N`），导致 `cid // 0`。

**解决方案**: 在调用 GEMM 前确保 M, N ≥ block size。如果 `M < block_M` 或 `N < block_N`，zero-padding 矩阵到 block 倍数再调用 GEMM，完成后裁剪。

### 5. Autotune supply_prog IndexError

**错误信息**:
```
An error occurred while testing config {...}
```

**原因**: `supply_prog(params)` 中 `params` 仅含输入 tensor 描述符（不含输出），`params[2]` 访问越界。

**解决方案**: 从 `params[0].shape` 和 `params[1].shape` 提取维度：
```python
def supply_prog(params):
    M_val, K_val = int(params[0].shape[0]), int(params[0].shape[1])
    _, N_val = int(params[1].shape[0]), int(params[1].shape[1])
    return [torch.randn(M_val, K_val).half().npu(), torch.randn(K_val, N_val).half().npu()]
```

### 6. Autotune get_configs 参数格式错误

**错误信息**:
```
TypeError: get_configs() missing 1 required positional argument: 'K'
```

**原因**: autotuner 调用 `get_configs` 时传参为 `(key_args_tuple, key_kwargs_tuple)`，即 `((M,N,K), ())`。直接声明 `get_configs(M, N, K)` 会收到 tuple 而非 3 个 int。

**解决方案**: 签名为 `get_configs(key_args, _key_kwargs=None)`，从 `key_args` 解包 M, N, K。调用时传递 callable 引用（`configs=get_configs`），而非调用结果（`configs=get_configs()`）。

### 7. L0C 溢出 Segfault

**现象**: autotune 编译通过但 benchmark 时进程直接 crash（Segfault），无 Python 异常。

**Segfault类似问题排查建议**: Segment fault 需要通过 gdb 等工具定位具体 crash 位置和调用栈，再结合 kernel 配置、访存范围、片上内存使用量等因素判断根因。
可能原因之一：当 `block_M * block_N * sizeof(accum_dtype) > L0C_capacity` 时，可能导致片上 buffer 使用超过硬件限制。例如 A2/A3 设备 L0C 为 128KB，float32 accum 元素数不应超过 32768；

**解决方案**: autotune 的 `get_configs` 中过滤超大 block：
```python
block_M = [bs for bs in [64, 128] if bs <= M]  # 排除 256
```



## 运行时错误

### 1. 结果不正确

**可能原因**:
1. 缺少同步
2. 公式实现错误
3. 数据类型问题

**解决方案**:

1. 根据编程模式处理同步。Expert 手动模式需要显式同步；Developer / 混合模式开启 `TL_ASCEND_AUTO_SYNC` 后不要额外插入手动 barrier。
   ```python
   # Expert 手动模式
   with T.Scope("V"):
       T.tile.exp(a_ub, a_ub)
       T.barrier_all()  # 必需
       T.tile.add(a_ub, a_ub, 1.0)
       T.barrier_all()  # 必需

   # Developer / 混合模式
   pass_configs = {
       tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
       tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
   }
   T.tile.exp(a_ub, a_ub)
   ```

2. 用小数据验证公式：
   ```python
   # 使用小shape测试
   M, N = 4, 8
   a = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])
   ```

3. 检查数据类型是否匹配

### 2. 精度问题

**现象**: 输出与参考实现有差异

**可能原因**:
1. **Golden 实现不一致** ⭐（最常见）
2. float16 精度较低，累积误差
3. 输出形状不匹配

**解决方案**:

#### 1. Golden 实现不一致（迁移算子时最常见）

**症状**：测试失败，精度误差很大（如 98% 的元素不匹配）

**排查步骤**：
1. 检查是否使用了原算子的 golden 实现
2. 对比原算子代码，确保算法逻辑一致
3. 检查输出形状是否需要 transpose

**示例**：
```python
# 原算子输出 (N, M)，你的 kernel 输出 (M, N)
# 需要 transpose 来匹配
result = kernel(A, B, workspace, C)  # (M, N)
expected = ref_program(A, B)  # (N, M)

torch.testing.assert_close(result.cpu().transpose(0, 1), expected)
```

**详细参考**：[tilelang-op-generate SKILL.md §8 Checklist #9-#10](../SKILL.md)（Golden 一致性 / 输出形状匹配）

#### 2. float16 精度问题

使用 float32 进行计算或调整容差：
```python
torch.testing.assert_close(b.cpu(), ref_b.cpu(), rtol=1e-2, atol=1e-2)
```

#### 3. 输出形状不匹配

检查原算子输出 shape，可能需要 transpose 或 reshape。

### 3. 性能问题

**现象**: kernel执行速度慢

**可能原因**:
1. 分块大小不合理
2. 过多同步
3. 内存访问模式不佳

**解决方案**:

1. 调整分块参数：
   ```python
   # 测试不同配置
   configs = [
       (32, 64),
       (64, 128),
       (128, 256),
   ]
   for block_M, block_N in configs:
       # 测试性能
   ```

2. 合并连续的同类操作，减少同步次数

3. 确保数据访问是连续的

### 4. Autotune 全部配置 benchmark 失败

**现象**: autotune 编译全部/部分完成，但所有编译成功的配置都在 benchmark 阶段报 `An error occurred`，最终 `RuntimeError: No configuration successfully compiled and passed benchmarking/validation.`

**排查顺序**:
1. 检查 `supply_prog` 是否正确提取维度（见编译错误 §5）
2. 检查 `ref_prog` 签名是否正确，输入输出 shape 是否匹配 GEMM
3. 检查 `get_configs` 是否过滤了 `block > dimension` 的配置（见编译错误 §4, §6）

## 测试设计原则

### GEMM 类算子测试覆盖

GEMM 类算子（含 im2col+GEMM 卷积）必须覆盖以下 4 类场景：

| 序号 | 类型 | M | N | K | 验证点 |
|------|------|---|---|---|--------|
| 1 | 完美对齐 | block 整数倍 | block 整数倍 | block 整数倍 | 零 padding 路径 |
| 2 | 单维 padding | < block | block 整数倍 | block 整数倍 | M padding+裁剪 |
| 3 | 单维 padding | block 整数倍 | < block | < block | N/K padding+裁剪 |
| 4 | 全维 padding | < block | < block | < block | 组合 padding+裁剪 |
| 5 | 多 block | 数倍 block | 数倍 block | — | 多 block 并行 |
| 6 | stride/padding | — | — | — | im2col 边界条件 |

### 检查原有逻辑正确性

生成新代码或修改现有实现后，**必须先用原有默认参数跑通**，确认 baseline 无回归：
```bash
python examples/{op}/example_{op}.py  # 默认参数测试
```
确认通过后，再用 `--b/--c/...` 扩维参数测试新场景。

## 调试技巧


### 1. 打印中间值

在kernel中添加：
```python
T.printf("value = %f\n", buffer[0])
```

### 2. 查看生成的代码

```python
func = my_op(...)
print(func.get_kernel_source())
```

### 3. 分步验证

1. 验证数据拷贝：
   ```python
   # 只做拷贝，不做计算
   T.copy(A[...], a_ub)
   T.copy(a_ub, B[...])
   ```

2. 逐步添加计算，每步验证

### 4. 小规模测试

```python
# 从最小规模开始
test_configs = [
    (4, 8, 4, 8),       # 最小
    (64, 64, 32, 32),   # 小
    (256, 256, 64, 64), # 中
]
```

## 常见模式问题

### 1. 如何处理动态shape?

使用 `T.dyn` 或 `T.dynamic`：
```python
# 方法1: 通过buffer.shape获取
N = T.dyn['N']  # 从buffer shape推断

# 方法2: 直接声明
N = T.dynamic('N', 'int32')
```

### 2. 如何实现带参数的算子?

使用函数参数传递：
```python
def my_op(M, N, block_M, param1=0.1, dtype="float"):
    @T.prim_func
    def main(...):
        # 使用param1
        T.tile.add(a_ub, a_ub, param1)
```

### 3. 如何处理非2D数据?

调整索引和分块策略：
```python
# 1D数据
@T.prim_func
def main(A: T.Tensor((N,), dtype), B: T.Tensor((N,), dtype)):
    # 使用1D索引

# 3D数据
@T.prim_func
def main(A: T.Tensor((B, M, N), dtype), ...):
    # 增加 batch 维度的循环
```

### 4. 如何优化内存使用?

1. 开启自动内存规划
2. 复用中间buffer：
   ```python
   # 使用同一个buffer存储中间结果
   temp_ub = T.alloc_ub([M, N], dtype)
   
   # 第一阶段
   T.tile.exp(temp_ub, a_ub)
   
   # 第二阶段（复用temp_ub）
   T.tile.add(temp_ub, temp_ub, 1.0)
   ```

3. 避免不必要的buffer分配

## 性能调优清单

- [ ] 分块大小是否合理？(block_M: 32-128, block_N: 64-256)
- [ ] 是否开启自动内存规划？
- [ ] 是否减少不必要的同步？
- [ ] 数据访问是否连续？
- [ ] 是否复用了中间buffer？
- [ ] 是否使用了合适的数据类型？
