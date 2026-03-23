# 调试工具

## 概述

TileLang 提供设备端调试接口 `T.printf` 和 `T.dump_tensor`，用于在 NPU 上运行 kernel 时打印信息和转储 tensor 数据。

> 注意：这两个是设备端调试工具。对于主机端，直接使用 Python `print` 即可。

## T.printf(format_str, *args)

格式化打印输出，类似 C 语言的 printf。

**参数**：
- `format_str`：格式字符串
- `*args`：可变参数，Buffer 会自动转换为 access pointer

**格式说明符**：
- `%d` / `%i`：十进制整数
- `%f`：浮点数
- `%x`：十六进制整数（可用于输出地址）
- `%s`：字符串
- `%p`：指针地址（建议使用 `%x`）

**示例**：
```python
T.printf("fmt %s %d\n", "string", 0x123)
T.printf("A_L1:\n")
```

## T.dump_tensor(tensor, desc, dump_size, shape_info=())

转储指定 Tensor 的内容。

**参数**：
- `tensor`：要转储的张量（支持 ub_buffer、l1_buffer、l0c_buffer、global_buffer）
- `desc`：用户自定义附加信息（uint32，如行号）
- `dump_size`：转储的元素数量
- `shape_info`：shape 信息元组（可选，用于格式化输出）

**示例**：

```python
# 基础用法
T.printf("A_L1:\n")
T.dump_tensor(A_L1, 111, 64)       # l1_buffer

T.printf("a_ub:\n")
T.dump_tensor(a_ub, 444, 64)       # ub_buffer

T.printf("C_L0C:\n")
T.dump_tensor(C_L0C, 333, 64)      # l0c_buffer

T.printf("A_GLOBAL:\n")
T.dump_tensor(a_global, 555, 64)   # global_buffer

# 带 shape_info 的格式化输出
T.dump_tensor(A_L1, 111, 64, (8, 8))
T.dump_tensor(B_L1, 222, 64, (8, 9))
```

**输出信息**包含：
- CANN 版本、时间戳
- Kernel 类型、算子详情
- 内存信息、数据类型
- 位置信息（UB/L1/L0C/GM）

```
opType=AddCustom, DumpHead: AIV-0, CoreType=AIV, block dim=8, ...
CANN Version: XX.XX, TimeStamp: XXXXXXXXXXXXXXXXX
DumpTensor: desc=111, addr=0, data_type=float16, position=UB, dump_size=32
```

## 查看生成的 AscendC 代码

```python
func = tile_add(M, N, block_M, block_N)
print(f"{func.get_kernel_source()}")
```

## 最佳实践

1. **desc 参数**使用有意义的数字（如行号），方便区分多处 dump
2. **shape_info** 参数有助于格式化显示多维数据
3. **调试完成后移除** T.printf 和 T.dump_tensor，避免影响性能
4. **dump_size** 不宜过大，避免输出过多数据
