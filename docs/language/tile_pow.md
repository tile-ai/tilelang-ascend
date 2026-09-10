# T.tile.pow

## 1. 功能说明

对两个操作数逐元素做幂运算：`dst[i] = src0[i] ** src1[i]`

以 `src0` 为底、`src1` 为指数的逐元素求幂计算。本 API 仅支持 tensor-tensor 形式，不支持 scalar 指数。

> **已知缺口**：Ascend C 原生 `Power` 接口支持 src0/src1 传入标量（扩展为逐元素广播），当前 `T.tile.pow` 尚未暴露该能力（`src1` 仅接受 Buffer/BufferRegion，传标量会在前端报错）。固定标量指数的场景可先用 `T.tile.fill` 构造常量 buffer（见示例 2）。

## 2. 函数原型

### 2.1 函数定义

```python
def pow(
    dst: Buffer | BufferRegion,
    src0: Buffer | BufferRegion,
    src1: Buffer | BufferRegion,
    *,
    tmp: Buffer | BufferRegion | None = None,
)
```

### 2.2 参数说明

| 参数名 | 输入/输出 | 描述 | 类型 | 必填/可选 |
|--------|----------|------|------|----------|
| dst | 输出 | 存放幂运算结果 | 张量（tensor） | 必填 |
| src0 | 输入 | 底数 | 张量（tensor） | 必填 |
| src1 | 输入 | 指数 | 张量（tensor） | 必填 |
| tmp | - | 可选的显式 UB 临时缓冲区（A2/A3 Ascend C 后端使用） | 张量（tensor） | 可选 |

> **类型说明**：
> - **tensor**：通过 `T.alloc_ub`、`T.alloc_shared` 等分配的缓冲区（Buffer），或其切片（BufferRegion）
> - **tmp 语义**：仅 Ascend C 后端消费；其标量 dtype 只决定字节几何（`extent × sizeof(dtype)`），不表示 workspace 数据类型，lowering 会在同一字节存储上建立目标所需 view。PTO 后端忽略该参数

### 2.3 参数规格

#### 2.3.1 DataType 支持

| 平台 | dst | src0 | src1 |
|------|:---:|:----:|:----:|
| Ascend A2 / A3（Ascend C） | float16, float32, int32 | float16, float32, int32 | 同 dst |
| Ascend A2 / A3（PTO） | float16, float32 | float16, float32 | 同 dst |

- 三者的 dtype 必须一致；int16 等其余 dtype 在两端都会编译报错（Ascend C 报 "Pow only supports float16/float32/int32"）

#### 2.3.2 Shape 支持

- 支持 1D 和 2D
- 支持整行切片（如 `buf[0:32, :]`）；仅计算切片区域内元素，区域外内容未定义
- 不支持 2D 列偏移切片（如 `buf[:, 8:40]`）：真机实测两端均触发 aicore 异常（507015）

#### 2.3.3 tmp 临时缓冲区（Ascend C）

- 省略 `tmp` 时由编译器自动分配：float16 为 `max(2*S, 1152)` 字节，float32/int32 为 `max(2*S, 768)` 字节（S 为源 tensor 字节数，来自框架的保守启发式）
- 显式 `tmp` 需为一维、静态、连续的 UB（起始字节地址 32B 对齐），容量由调用者保证

### 2.4 约束条件

1. dst、src0、src1 的 dtype 必须一致（编译期检查）；不支持 scalar 指数，固定指数需先用 `T.tile.fill` 填成 buffer
2. 三者的元素总数应一致；**无大小校验**：Ascend C 后端大小不一致仍会静默运行（结果未定义），PTO 后端因 tile 形状不匹配编译失败
3. 原地运算（dst 与 src0 为同一 buffer）两后端均支持；dst 与 src1 为同一 buffer 时仅 Ascend C 后端支持（PTO 后端结果错误，真机实测，未设独立用例）
4. **PTO 后端会原地修改 src0**（内部先对 src0 求对数再参与运算），调用后 src0 不可复用；Ascend C 后端运算过程中不改写 src0/src1 的取值（输入保持原值，原地别名时 dst 的写入按常规覆盖目标缓冲区）
5. 特殊值语义（真机实测）：Ascend C 遵循 IEEE（`0^0=1`、`(-2)^3=-8`、`1^nan=1`）；PTO 基于 log/exp 路径，`0^0`、负底数、`1^nan` 均产生 nan
6. 精度（真机实测）：Ascend C 内部按 float32 计算，float32 结果与 torch 逐位一致、float16 误差 ≤ 1 ulp，int32 对不可精确表示的幂可能偏差 ±1；PTO float32 精确、float16 为近似实现（最大相对误差约 3.6e-3）
7. 操作数地址需 32 字节对齐（硬件约束）

## 3. 示例代码

**示例 1：逐元素幂运算**

```python
base_ub = T.alloc_ub((128,), "float16")
exp_ub = T.alloc_ub((128,), "float16")
dst_ub = T.alloc_ub((128,), "float16")
T.tile.pow(dst_ub, base_ub, exp_ub)  # dst[i] = base[i] ^ exp[i]
```

**示例 2：平方运算（固定指数需先 fill 指数 buffer）**

```python
values = T.alloc_ub((256,), "float32")
squared = T.alloc_ub((256,), "float32")
exp_two = T.alloc_ub((256,), "float32")
T.tile.fill(exp_two, 2.0)  # 指数 buffer 填充为 2.0
T.tile.pow(squared, values, exp_two)  # dst[i] = values[i] ^ 2
```