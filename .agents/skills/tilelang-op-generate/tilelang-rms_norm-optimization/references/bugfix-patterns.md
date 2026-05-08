# 问题修复模式

## 编译期错误

### 1. Unified Buffer (UB) 未对齐

**错误**: `The UB address accessed by the VEC instruction is not aligned`

**原因**: `block_M` 较小或者 `VEC_NUM` 导致 `ROWS` 不是 8 或 16 的倍数（取决于数据类型）。

**修复**:
- 强制对齐：确保 `ROWS` 是 16 的倍数（FP32 下 16 * 4 字节 = 64 字节，满足 32 字节对齐）。
- 规约缓冲区对齐：`sum_sq_col` 等列向量虽然只有 1 列，但在 UB 布局中，每一行必须满足对齐要求。
- 计算逻辑修正：去掉会导致非对齐访问的标量偏移。

### 2. Gather-free 方案编译失败

**错误**: `v_thread` 未定义变量等编译错误

**原因**: 使用 even/odd 列分离（`x[..., 0::2]`）的 gather-free 方案在 TileLang Ascend 后端可能触发代码生成问题。在 NPU（昇腾）后端使用 T.Parallel 或 T.Kernel 时，TileLang 的底层编译器尝试将逻辑映射到 GPU 风格的线程（Thread Index），但在 NPU 的指令集上下文中找不到对应的变量定义。

**结论**: 当前阶段优先使用 gather-mask 方案，等编译器成熟后再尝试 gather-free。


## 运行时错误

### 1. 数据覆盖导致结果错误

**现象**: 运行报错 `Multiple writes to overlapping buffer regions detected`

**原因**: 重叠写入冲突。dy_i3 被 T.copy 写入一次（从 Global Memory 读取），随后又在 T.tile.axpy 中作为输出目标被写入（计算结果）。

**修复**: 引入一个专门的输出 Buffer dx_o3。这样，流水线中的数据流向就是：GM -> dy_i3 -> 计算 -> dx_o3 -> GM。每个 Buffer 在流水线周期内只负责单一的写入源。