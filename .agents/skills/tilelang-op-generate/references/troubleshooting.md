# 算子疑难解答

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

## 运行时错误

### 1. 结果不正确

**可能原因**:
1. 缺少同步
2. 公式实现错误
3. 数据类型问题

**解决方案**:

1. 确保在T.tile操作间添加同步：
   ```python
   with T.Scope("V"):
       T.tile.exp(a_ub, a_ub)
       T.barrier_all()  # 必需
       T.tile.add(a_ub, a_ub, 1.0)
       T.barrier_all()  # 必需
   ```

2. 用小数据验证公式：
   ```python
   # 使用小shape测试
   M, N = 4, 8
   a = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])
   ```

3. 检查数据类型是否匹配

### 2. 精度问题

**现象**: 输出与参考实现有微小差异

**原因**: float16精度较低，累积误差

**解决方案**:
1. 使用float32进行计算
2. 调整测试容差：
   ```python
   torch.testing.assert_close(b.cpu(), ref_b.cpu(), rtol=1e-2, atol=1e-2)
   ```

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