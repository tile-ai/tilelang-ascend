# SIMT 架构概念参考

> SIMT vs SIMD 的核心差异、混合编程和选型指南已在 `npu-arch-guide.md` 中覆盖，此处不再重复。

---

## SIMT 线程层次结构

### 概述

AscendC SIMT 编程模型采用三级线程层次：

| 层级 | 概念 | 对应 CUDA | 设置方式 |
|------|------|----------|---------|
| 启动核数 | Block 数量 | Grid 中的 Block 数 | tiling 侧 `SetBlockDim` |
| 启动线程数 | 单核线程数 | 单 Block 线程数 | kernel 侧 `constexpr` 常量 |
| Warp | 32 线程分组 | Warp | 硬件自动分组 |

### 核数（Block Dim）

- 通过 tiling 侧 `SetBlockDim` 接口设置
- 对应 CUDA 中的 Block 概念
- 切分公式：`总核数 = ceil(输出元素总数 / 单核最少处理元素数)`

### 线程数

- 单核最大支持 **2048** 线程
- **必须在编译期确定**（使用 `constexpr` 或字面量）
- 禁止从 tiling 数据动态获取
- `LAUNCH_BOUND(N)` 和 `Simt::Dim3(N)` 必须使用同一个编译期常量
- 默认值: 1024，建议范围根据数据量和计算复杂度调整

### Warp 调度

- 内部按 **32 线程** 分组形成 warp
- Warp 是硬件调度的最小单位
- 每个 AIV 核包含 4 个 Warp Scheduler，调度器编号为 `warp_id % 4`
- 同一时刻一个 AIV 核只执行一个线程块任务
- 线程块内多个 Warp 被依次调度执行

---

## Warp 调度机制

### 基本概念

- 每 **32 个线程** 组成一个 Warp
- Warp 是硬件调度的最小单位
- Warp 内每个线程称为 Lane，编号 0~31

### 硬件调度

- 每个 AIV 核包含 **4 个 Warp Scheduler**，调度器编号为 `warp_id % 4`
- 同一时刻一个 AIV 核只执行一个线程块任务
- 线程块内多个 Warp 被依次调度到 AI Core 执行

### SIMT 执行语义

- Warp 内所有线程执行**相同指令**（SIMT 语义）
- 分支发散（Branch Divergence）时串行执行各分支路径
  - 若 warp 内部分线程走 if 分支，部分走 else 分支
  - 硬件先执行 if 路径（else 线程等待），再执行 else 路径（if 线程等待）
  - 导致流水线气泡，降低有效利用率

### 分支发散优化

- 减少同一 warp 内的分支分化
- 将运行时分支提升到编译期（`if constexpr` / 模板参数）
- 同一 warp 内线程尽量走相同执行路径

---

## SIMT 内存空间

### 概述

AscendC SIMT 编程模型包含三类内存空间：

| 内存类型 | 说明 | 访问范围 | 地址空间修饰符 |
|---------|------|---------|--------------|
| 寄存器和栈 | 每个线程独立 | 线程私有 | 无（默认） |
| Unified Buffer (UB) | 本地内存 | 核内共享 | `__ubuf__` |
| Global Memory (GM) | 全局内存 | 所有线程可访问 | `__gm__` |

### 寄存器和栈

- 每个线程独享一组寄存器
- 寄存器数量受线程数影响（参见 `launch_bounds_registers.md`）
- 超出寄存器容量的变量会栈溢出

### Unified Buffer (UB)

- 核内共享的本地内存
- 总量 256KB，按分区使用（详见 `ub_partition.md`）
- SIMT 算子不能使用全部 UB 空间，需为 DCache 预留 ≥32KB
- 可用 UB = 256KB - 8KB(预留) - 32KB(DCache) = **216KB**
- 在 SIMT VF 中通过 `__ubuf__` 指针访问

### Global Memory (GM)

- 所有核所有线程均可访问
- SIMT VF 支持直接读写 GM，无需显式 Load/Store
- 通过 `__gm__` 指针访问
- 在 kernel 入口通过 `GM_ADDR` 参数传入

### 与 SIMD 的差异

| 维度 | SIMD | SIMT |
|------|------|------|
| 数据搬运 | 需显式 Load/Store | 支持直接读写 GM 和 UB |
| GM 访问 | 不支持直接到寄存器 | 支持直接访问 |

---

## UB 内存分区

### 总量

Unified Buffer 总量 **256KB**，从低地址到高地址依次分区：

| 区域 | 大小 | 说明 |
|------|------|------|
| 静态内存 | 编译期确定 | `__ubuf__` 数组声明，混合编程中用于 SIMD/SIMT 共享数据 |
| 动态内存 | tiling 侧 `SetLocalMemory` 设置 | TBuf/LocalTensor 申请，混合编程中通过 `<<<>>>` 的 dynUBufSize 指定 |
| 预留空间 | 固定 8KB | 编译器预留，不可使用 |
| Data Cache | = 256KB - 静态 - 动态 - 8KB | SIMT 专有 DCache |

### 关键约束

- **SIMT 算子不能使用全部 UB 空间**，需为 DCache 预留 ≥32KB
- 若 DCache < 32KB，编译校验报错
- 可用 UB = 256KB - 8KB - 32KB = **216KB**

### Tiling 侧设置

定义 `DCACHE_SIZE` 为 128KB，使用 `SetLocalMemorySize` 设置参数为 `ubsize - DCACHE_SIZE`：

```cpp
constexpr uint64_t DCACHE_SIZE = 128 * 1024;
uint64_t ubsize = 256 * 1024;
context->SetLocalMemorySize(ubsize - DCACHE_SIZE);
```

### 静态内存 vs 动态内存

| 类型 | 声明方式 | 适用场景 |
|------|---------|---------|
| 静态内存 | `__ubuf__ T buffer[SIZE];` | 编译期已知大小的共享 buffer |
| 动态内存 | `TBuf<QuePosition::VECCALC>` + `pipe_->InitBuffer()` | 运行时按需分配 |

---

## 数据类型速查

### 按位宽分类

| 位宽 | 数据类型 |
|------|---------|
| b8 | bool, int8_t, uint8_t, hifloat8_t, fp8_e5m2_t, fp8_e4m3fn_t |
| b16 | int16_t, uint16_t, **half**, **bfloat16_t** |
| b32 | int32_t, uint32_t, **float**, complex32 |
| b64 | int64_t, uint64_t, double, complex64 |

### SIMT VF 函数参数支持类型

#### 标量类型

bool, int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t, float, half, bfloat16_t

#### 指针类型

- `__gm__ T*` — Global Memory 指针
- `__ubuf__ T*` — Unified Buffer 指针

#### 返回值

必须是 `void`

### 特殊值宏

| 宏名 | 描述 | 头文件 |
|------|------|--------|
| ASCRT_INF_BF16 | bfloat16 正无穷 | asc_bf16.h |
| ASCRT_MAX_NORMAL_BF16 | bfloat16 最大值 | asc_bf16.h |
| ASCRT_NAN_BF16 | bfloat16 NaN | asc_bf16.h |
| ASCRT_INF_F16 | half 正无穷 | asc_fp16.h |
| ASCRT_MAX_NORMAL_F16 | half 最大值 | asc_fp16.h |
| ASCRT_NAN_F16 | half NaN | asc_fp16.h |
| ASCRT_INF_F32 | float 正无穷 | asc_simt.h |
| ASCRT_MAX_NORMAL_F32 | float 最大值 | asc_simt.h |

---

## 函数调用层级

### 层级结构

```
核函数 (__global__ __aicore__)
  ├── __aicore__ 函数
  ├── SIMD VF (__simd_vf__) ← 通过 asc_vf_call 调用
  │     └── __simd_callee__ 子函数
  └── SIMT VF (__simt_vf__) ← 通过 VF_CALL 调用
        └── __simt_callee__ 子函数
```

### 各层级说明

#### 核函数

```cpp
__global__ __aicore__ void {op_name}(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
```

- 算子入口，由框架调用
- 负责 tiling 数据解析和场景分发
- 通过 `REGISTER_TILING_DEFAULT` + `GET_TILING_DATA_WITH_STRUCT` 获取 tiling

#### __aicore__ 函数

```cpp
__aicore__ inline void Process(GM_ADDR x, GM_ADDR y, const TilingData* tilingData)
```

- 核内主逻辑函数
- 负责 UB buffer 初始化、GM 地址转换
- 调用 `Simt::VF_CALL` 启动 SIMT VF

#### SIMT VF 函数

```cpp
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void OpComputeSimt(...)
```

- 线程级并行计算函数
- 每个线程独立执行，以 stride 间隔遍历数据
- 可调用 `__simt_callee__` 子函数和 `constexpr` 函数

#### __simt_callee__ 子函数

```cpp
__simt_callee__ inline void HelperFunc(...)
```

- SIMT VF 内部的辅助函数
- 必须带 `__simt_callee__` 修饰符
- 可被 `__simt_vf__` 函数调用

### 调用约束

- `__simt_vf__` 内只能调用 `__simt_callee__` 函数和 `constexpr` 函数
- `__simd_vf__` 内只能调用 `__simd_callee__` 函数和 `constexpr` 函数
- 不可跨模式调用（SIMT VF 不能调用 SIMD callee，反之亦然）

---

## LAUNCH_BOUND 与寄存器数量

### 寄存器数量映射

`__launch_bounds__(N)` 限定每个 VF Block 使用的最大线程数，线程数直接影响每线程可用寄存器数：

| 线程数范围 | 每线程可用寄存器数 |
|-----------|-----------------|
| 1025~2048 | 16 |
| 513~1024 | 32 |
| 257~512 | 64 |
| 1~256 | 127 |

### 使用原则

- 寄存器用于存储线程局部变量
- 超出寄存器容量则栈溢出，影响性能
- 建议 `__launch_bounds__(N)` 的 N 与实际启动线程数一致
- `LAUNCH_BOUND(N)` 和 `Simt::Dim3(N)` 必须使用同一个编译期常量

### 声明语法

```cpp
__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void YourKernel(...);
```

- `LAUNCH_BOUND(thread_num)` 可选，默认 1024
- 参数必须是 `constexpr` 常量或字面量

### 线程数选择参考

| 算子类型 | 建议线程数 | 原因 |
|---------|-----------|------|
| 搬运类算子 | 2048 / 1024 | 内存带宽受限，更多线程隐藏延迟 |
| 计算类算子 | 512 / 1024 | 寄存器压力较大，需平衡并行度和寄存器 |

### 寄存器压力与线程数调优

寄存器压力随 VF 复杂度增加而增大：

| VF 复杂度 | uint32_t 索引线程数 | uint64_t 索引线程数 |
|-----------|-------------------|-------------------|
| 极低（NONE/1D） | 1024 | 512 |
| 中低（2D） | 1024 | 512 |
| 中等（3D） | 1024 | 512 |
| 较高（4D） | 1024 | 512 |
| 最高（ND运行时循环） | 256 | 128 |