# 生成规范与检查清单

## 代码结构规范

### 文件组织

1. **文件头部**：`import tilelang`, `import tilelang.language as T`, `import torch`
2. **缓存控制**：`tilelang.cache.clear_cache()` 或 `tilelang.disable_cache()`
3. **pass_configs 定义**（Developer 模式必须）
4. **JIT 装饰器**：`@tilelang.jit(out_idx=..., workspace_idx=..., pass_configs=...)`
5. **参数函数**：外层函数接收形状参数，返回 `@T.prim_func`
6. **Tensor 声明**：所有输入/输出/workspace 均为 `T.Tensor`
7. **Kernel 上下文**：`with T.Kernel(block_num, is_npu=True) as (cid, vid)`
8. **实例化**：`func = my_op(M, N, K, ...)`
9. **调用**：`result = func(tensor1, tensor2, ...)`
10. **验证**：`torch.testing.assert_close(result, ref, rtol=1e-2, atol=1e-2)`

### @tilelang.jit 参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `out_idx` | 输出 tensor 在参数列表中的索引 | `[-1]`（最后一个）, `[2]`（第 3 个）, `[3]` |
| `workspace_idx` | 自动分配的 workspace tensor 索引 | `[4, 5, 6]`, `4`（单个） |
| `pass_configs` | 编译 pass 配置 | `{...}` |
| `target` | 编译目标 | `"pto"`（默认） |
| `execution_backend` | 执行后端 | `"cython"` |

---

## Developer 模式检查清单

- [ ] pass_configs 已正确设置
  - 纯 Vector：`AUTO_SYNC` + `MEMORY_PLANNING`
  - 含 Cube：4 个全部开启
- [ ] 内存分配仅使用 `T.alloc_shared` / `T.alloc_fragment`
- [ ] **无** `T.Scope("C")` / `T.Scope("V")`
- [ ] **无** `T.barrier_all()` / `T.set_flag` / `T.wait_flag`
- [ ] **无** `T.set_cross_flag` / `T.wait_cross_flag`
- [ ] **无** `T.annotate_address`
- [ ] Cube→Vector 数据中转使用 workspace tensor
- [ ] `workspace_idx` 在 `@tilelang.jit` 中正确声明
- [ ] `out_idx` 正确指向输出 tensor
- [ ] Element-wise 运算使用 `T.Parallel` + 符号运算
- [ ] 归约操作正确传入 tmp_ub 缓冲区

---

## Expert 模式检查清单

- [ ] pass_configs 全部设为 False 或不设置
- [ ] 内存分配使用显式 API（`T.alloc_L1/ub/L0A/L0B/L0C`）
- [ ] 显式声明 `with T.Scope("C"):` 和 `with T.Scope("V"):`
- [ ] 所有管线 `set_flag` / `wait_flag` 正确配对
- [ ] 核间 `set_cross_flag` / `wait_cross_flag` 正确配对
- [ ] Flag 初始化（`init_flag`）和清理（`clear_flag`）成对出现
- [ ] 双缓冲 slot 索引使用 `% 2` 交替
- [ ] 搬运与计算的依赖关系通过 flag 正确表达
- [ ] 不可跨级访问内存（GM→L1→L0，不可 GM→L0）
- [ ] workspace 核间信号 ID 不冲突

---

## 混合模式检查清单

- [ ] pass_configs 使用 Developer 配置（4 个全开）
- [ ] 内存分配使用 Developer API（`alloc_shared` / `alloc_fragment`）
- [ ] **无** `T.Scope` / 手动同步
- [ ] Expert 扩展 API 仅用于 Developer 模式不支持的操作
- [ ] `T.tile.fill` / `T.reduce_*` 使用正确

---

## 通用检查清单

- [ ] 数据类型正确：GEMM 输入 `float16`，累加器 `float`（float32）
- [ ] 分块参数合理：`block_M` / `block_N` 通常为 16 的倍数
- [ ] `T.ceildiv` 处理非整除情况
- [ ] `vid` 用于分割 Vector 工作：每个 vid 处理 `block_M // 2` 行
- [ ] `tmp_ub` 大小足够：`[3 * DataType(accum_dtype).bits // 8 * rows * cols]`，dtype 为 `"uint8"`
- [ ] 边界条件处理（tail block、非整除维度）
- [ ] 参考验证函数使用 `float32` 精度计算 reference
- [ ] `torch.testing.assert_close` 使用 `rtol=1e-2, atol=1e-2`

---

## 常见错误与解决

| 错误现象 | 可能原因 | 解决方案 |
|---------|---------|---------|
| 编译报错 scope 未定义 | Expert 模式漏写 `T.Scope` | 添加 `with T.Scope("C"/"V")` |
| 运行时死锁 | Flag 不配对 | 检查 `set_flag`/`wait_flag` 成对，init/clear 成对 |
| 精度不达标 | 累加器用了 `float16` | 累加器使用 `float`（float32） |
| 数据搬运报错 | 跨级访问 | 检查内存层级路径（GM→L1→L0） |
| Developer 模式同步错误 | pass_configs 未全开 | 4 个开关全部设为 `True` |
| workspace 数据错误 | `workspace_idx` 未声明 | 在 `@tilelang.jit` 中添加 `workspace_idx` |
| Vector 计算结果为 0 | 未初始化 buffer | 使用 `T.tile.fill` 初始化 |
| GEMM 结果错误 | `init` 参数不对 | 首次迭代 `init=True`，后续 `init=False` |
| Reduce 输出异常 | `tmp_ub` 太小 | 确保 tmp 大小为 `3 * dtype_bytes * rows * cols` |

---

## 模式转换速查

### Expert → Developer

```
T.alloc_L1(shape, dtype)         → T.alloc_shared(shape, dtype)
T.alloc_ub(shape, dtype)         → T.alloc_shared(shape, dtype)
T.alloc_L0C(shape, dtype)       → T.alloc_fragment(shape, dtype)
with T.Scope("C"):              → 删除
with T.Scope("V"):              → 删除
T.barrier_all()                  → 删除
T.set_flag(...)                  → 删除
T.wait_flag(...)                 → 删除
T.set_cross_flag(...)            → 删除
T.wait_cross_flag(...)           → 删除
T.tile.add(c, a, b)             → for i,j in T.Parallel(...): c[i,j] = a[i,j] + b[i,j]
T.tile.exp(dst, src)            → for i,j in T.Parallel(...): dst[i,j] = T.exp(src[i,j])
T.mma(A, B, C, init)            → T.gemm_v0(A, B, C, init=init)
逐行 for h_i in range(M): ...  → for h_i, j in T.Parallel(M, N): ...（自动广播）
T.annotate_address(...)          → 删除
```

### Developer → Expert

```
T.alloc_shared → T.alloc_L1（被 GEMM 使用）或 T.alloc_ub（被 Vector 使用）
T.alloc_fragment → T.alloc_L0C
自动同步 → 手动添加 T.set_flag / T.wait_flag
自动 CV 分离 → 手动添加 with T.Scope("C"/"V")
T.Parallel + 符号 → T.tile.* 操作
pass_configs 全部 True → 全部 False 或删除
```
