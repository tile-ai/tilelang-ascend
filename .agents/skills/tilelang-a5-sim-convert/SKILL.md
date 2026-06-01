---
name: tilelang-a5-sim-convert
description: "将 tilelang example 脚本转换为可在 A5 camodel 仿真器上直接运行的版本。输入脚本路径，输出一个新的 *_sim.py 文件，不覆盖原始文件。触发：仿真运行、camodel、A5 仿真、sim 模式、转换脚本为仿真、不需要 NPU 跑 kernel、simulate A5。"
---

# TileLang A5 Camodel 仿真脚本转换

将任意 tilelang DSL 脚本转换为 A5 camodel 仿真可运行的独立脚本。

## 模板结构（260 行，只改两处）

模板文件：`.agents/skills/tilelang-a5-sim-convert/scripts/run_a5_sim_template.py`

```
行 1-24    import 语句            ← 不动
行 25-96   环境自动设置            ← 不动（_find_ascend_home, _source_cann, _find_sim_lib, setup）
行 99-133  加载 camodel 运行时    ← 不动（load_runtime, dev_malloc）
行 136-166 kernel 定义            ← ★ 第 1 处要改
行 169-260 main() 编译+运行+验证   ← 部分要改（详见下方）
```

## 工作流程

收到脚本路径后，按以下步骤执行：

### Step 1: 运行解析脚本获取 kernel 信息

```bash
cd <tilelang-ascend-root>
python .agents/skills/tilelang-a5-sim-convert/scripts/parse_example.py <target_script>
```

输出 JSON，包含 `kernel_name`、`buffers`（shape/dtype 列表）。

### Step 2: 读取模板 + 原始脚本

- 读取 `.agents/skills/tilelang-a5-sim-convert/scripts/run_a5_sim_template.py`
- Read 目标脚本，找到 kernel 定义部分（`@T.prim_func` 或 `@tilelang.jit` 装饰的函数体）

### Step 3: 生成 *_sim.py

输出路径：`<原路径>/<原名>_sim.py`（**绝不覆盖原始文件**）。

---

## 改动清单

### 改动 1：kernel 定义（模板 136-166 行）

| 原始脚本 | 仿真脚本 |
|---------|---------|
| `@tilelang.jit(out_idx=[-1])` | 删掉 |
| `def matmul(M, N, K, ...):` | `def make_kernel():` |
| `T.Tensor((M, K), dtype)` | `T.Tensor((1024, 256), "float16")` ← 用 Step1 解析出的具体数值 |
| `T.alloc_L0C(..., "float16")` | `T.alloc_L0C(..., "float")` ← A5 pto-isa 要求 float32 |
| `func = matmul(...)` 触发编译 | 删掉，编译在 main() 里统一处理 |

**生成的代码结构**：

```python
def make_kernel():
    import tilelang.language as T
    @T.prim_func
    def main(
        A: T.Tensor((1024, 256), "float16"),   # ← 具体数值
        B: T.Tensor((256, 1024), "float16"),
        C: T.Tensor((1024, 1024), "float16"),
    ):
        # ... kernel 逻辑（和原始脚本一模一样）...
    return main
```

### 改动 2：数据准备（模板 214-233 行）

**a) 维度变量**（第 215 行）

根据 Step1 的 `buffers` 设置：

```python
# 原始模板（gemm 专用）
M, N, K = 1024, 512, 256

# 通用写法：从 buffers 提取
# buffers[0].shape = [M, K]  →  M = shape[0], K = shape[1]
# buffers[1].shape = [K, N]  →  N = shape[1]
# buffers[2].shape = [M, N]
```

如果不是矩阵（比如 1D/3D tensor），按实际 shape 处理。

**b) 数据 dtype（第 216-218 行）**

```python
# float16 → np.float16
# float32 → np.float32
# int32   → np.int32
```

**c) 数据填充（第 219-224 行）**

float16 必须用小值防止溢出（>65504 就变成 inf）：

```python
# 模式：np.float16((i % 100 + 1) * (j % 100 + 1) * 0.0001)
# 根据实际 tensor 维度调整循环层数
```

**d) 设备内存分配（第 227-229 行）**

```python
# float16：每个元素 2 字节 → size * 2
# float32：每个元素 4 字节 → size * 4
itemsize = 2 if dtype == "float16" else 4
d_A = dev_malloc(rt, total_elements * itemsize)
```

**e) 参考计算（第 225 行）**

```python
# gemm：       h_Ref = h_A.astype(np.float32) @ h_B.astype(np.float32)
# elementwise：h_Ref = (h_A.astype(np.float32) + h_B.astype(np.float32))
# 其他：根据原始脚本的逻辑写对等的 numpy 计算
```

### 改动 3：`call` 函数签名（模板 210 行）

```python
# 原始模板（3 输入 + stream = 4 个参数）
kl.call.argtypes = [ctypes.c_void_p] * 4

# 实际参数数量 = kernel 的 buffer 数量 + 1（stream）
# buffers 有 3 个 → argtypes = [ctypes.c_void_p] * 4
# buffers 有 2 个 → argtypes = [ctypes.c_void_p] * 3
# buffers 有 4 个 → argtypes = [ctypes.c_void_p] * 5
```

`call()` 的调用（第 237 行）也要对应：

```python
# 3 个 buffer：kl.call(d_A, d_B, d_C, stream)
# 2 个 buffer：kl.call(d_in, d_out, stream)
# 4 个 buffer：kl.call(d_A, d_B, d_C, d_D, stream)
```

### 改动 4：H2D 和 D2H 拷贝

```python
# H2D（CPU → 设备），第 230-231 行
rt.rtMemcpy(d_A, size, h_A.ctypes.data, size, 1)  # 最后一个参数 1 = host→device
                                                    # 每个输入 buffer 都要拷一次

# D2H（设备 → CPU），第 239 行
rt.rtMemcpy(h_C.ctypes.data, size, d_C, size, 2)  # 最后一个参数 2 = device→host
                                                    # 只有输出 buffer 需要拷
```

### 不需要改的部分

**以下代码在任何转换中都保持原样：**
- `import` 语句（12-18 行）
- `_find_ascend_home()`、`_source_cann()`、`_find_sim_lib()`、`setup()`（25-96 行）
- `load_runtime()`、`dev_malloc()`（99-133 行）
- `tilelang.lower()` + `LibraryGenerator` 编译流程（189-211 行）
- `rtStreamCreate` / `rtStreamSynchronize` / `rtStreamDestroy`（232-233、238、254 行）
- 验证逻辑框架（242-251 行）
- 清理代码（253-256 行）

## 测试数据生成规则

- float16：`np.float16((i % 100 + 1) * (j % 100 + 1) * 0.0001)`，防溢出（max ≈ 65504）
- float32：直接用 `np.float32(...)`，范围宽不溢出
- 参考输出：统一用 float32 计算，保证精度
