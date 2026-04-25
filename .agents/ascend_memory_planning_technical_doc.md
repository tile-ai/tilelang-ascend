# Ascend Memory Planning Pass 技术说明文档

**源码位置**：`src/transform/ascend_memory_planning.cc`

**核心功能**：为昇腾 NPU 进行专用内存规划，通过活跃度分析和线性扫描算法实现内存复用，输出 address_map/size_map 函数属性。

---

## 1. Pass 概览

| 特性 | 说明 |
|-----|-----|
| 内存域 | shared, shared.dyn, wmma.matrix_a/b/accumulator |
| 分配策略 | 顺序分配（默认）/ 线性扫描+复用（自动规划） |
| 核心算法 | Liveness（GEN/KILL）+ Linear Scan Allocation |
| 输出格式 | 函数属性 address_map/size_map |
| 对齐要求 | 32 字节对齐 |

---

## 2. Pass 开发模板

### 2.1 类结构模板

```
主类：AscendMemoryPlanning : public IRMutatorWithAnalyzer
  └─ static PrimFunc Substitute(PrimFunc f, PassContext ctx)

内部类：AscendMemoryPlanner : public StmtExprVisitor
  ├─ 构造时：VisitStmt + PlanMemory
  ├─ LivenessAnalysis()：活跃度分析
  └─ PlanMemoryForScope / PlanMemoryForScopeLinear

内部类：LinearScanAllocator
  └─ allocate(intervals)：线性扫描分配
```

### 2.2 标准处理流程骨架

```
Substitute 主入口：
  1. 读取 Pass 配置 tl.ascend_memory_planning
  2. 获取外部 address_map / buffer_shapess
  3. AscendMemoryPlanner 构造（遍历 IR + 规划）
  4. 写入 address_map / size_map 到函数属性
  5. 返回修改后的 PrimFunc

AscendMemoryPlanner 构造流程：
  1. SetPreAllocBuffer() → 处理预分配
  2. SetTmpBuffers() → 处理临时缓冲区
  3. VisitStmt 遍历 IR → 收集 buffer 信息
  4. PlanMemory() → 活跃度分析 + 分配
```

### 2.3 Pass 注册模板

```
TVM_REGISTER_GLOBAL("tl.transform.AscendMemoryPlanning")
    .set_body_typed(AscendMemoryPlanning);
```

---

## 3. 处理流程骨架

```
Substitute 入口
    │
    ├─► AscendMemoryPlanner 构造
    │      ├─ SetPreAllocBuffer()
    │      ├─ SetTmpBuffers()
    │      └─ VisitStmt 遍历 IR
    │             ├─ VisitStmt_(Allocate)：记录 scope/size
    │             ├─ VisitStmt_(BufferStore/Load)：跟踪访问
    │             └─ VisitNewScope：记录作用域层级
    │
    ├─► PlanMemory()
    │      ├─ LivenessAnalysis()
    │      │      ├─ 反向遍历 → KILL 点
    │      │      ├─ 正向遍历 → GEN 点
    │      │      └─ ReorderKillPoints()
    │      │
    │      └─ 按 scope 分组规划
    │             ├─ memory_auto_plan=false → 顺序分配
    │             └─ memory_auto_plan=true  → 线性扫描+复用
    │
    └─► 写入 address_map / size_map
```

---

## 4. 数据结构

### 4.1 核心结构体

| 结构体 | 关键字段 | 用途 |
|-------|---------|-----|
| **LiveInterval** | buffer, start, end, size | 活跃区间（GEN→KILL）|
| **Allocation** | buffer, offset, size, is_reused | 分配结果 |
| **StmtEntry** | stmt, scope_pair_offset, touched | 语句条目 |
| **EventEntry** | gen, kill | GEN/KILL 集合 |

### 4.2 核心数据成员

```cpp
// AscendMemoryPlanner
unordered_map<VarNode*, AllocEntry> alloc_info_;        // 分配信息
unordered_map<VarNode*, int64_t> address_map_;           // 输出：地址
unordered_map<VarNode*, size_t> buffer_sizes_;           // 输出：大小
unordered_map<VarNode*, string> buffer_scopes_;          // 内存域
unordered_map<VarNode*, size_t> first_use_;              // 首次使用位置
unordered_map<Object*, EventEntry> event_map_;           // GEN/KILL
unordered_map<string, int64_t> pre_alloc_buffer_;        // 预分配
vector<StmtEntry> linear_seq_;                           // 语句序列

// LinearScanAllocator
size_t memory_limit_;                                    // 内存限制
size_t next_new_offset_;                                 // 下一个新分配位置
vector<pair<size_t, size_t>> free_blocks_;               // 空闲块列表
```

---

## 5. 算法逻辑

### 5.0 内存信息

所采用的策略是先把所有内存域的空间全部申请，再做复用规划。

| 内存域 | 限制（字节）| 典型缓冲区 | 来源操作 |
|-------|-----------|----------|---------|
| shared | 196352 | L1 缓冲区 | alloc_shared, copy_gm_to_l1 |
| shared.dyn | 524032 | 动态共享 | alloc_shared.dyn |
| wmma.matrix_a | 65536 | L0A | alloc_L0A, copy_l1_to_l0a |
| wmma.matrix_b | 65536 | L0B | alloc_L0B, copy_l1_to_l0b |
| wmma.accumulator | 131072 | L0C | alloc_fragment, mma 输出 |

### 5.1 分配策略选择

| 配置 | 策略 | 特点 |
|-----|-----|-----|
| tl.ascend_memory_planning=false | PlanMemoryForScopeLinear | 顺序分配，无复用 |
| tl.ascend_memory_planning=true | PlanMemoryForScope | 线性扫描，支持复用 |

### 5.2 活跃度分析（LivenessAnalysis）

| 阶段 | 遍历方向 | 输出 | 关键操作 |
|-----|---------|-----|---------|
| 1 | 反向 | KILL 集合 | 记录每个 buffer 最后使用位置 |
| 2 | 正向 | GEN 集合 | 记录 first_use_ 匹配位置 |
| 3 | 调整 | reordered KILL | 若 KILL 层级 > GEN 层级，调整到同层级最后 |

**ReorderKillPoints 原理**：避免跨作用域提前释放，将 KILL 点调整到同层级的最后一条语句。

### 5.3 线性扫描分配决策

| 步骤 | 条件 | 操作 |
|-----|-----|-----|
| 释放过期 | `active.top().end < interval.start` | 移入 free_blocks，mergeFreeBlocks() |
| 预分配 | `buffer in pre_alloc_buffer_` | 直接使用，CheckConflict() |
| 新分配 | `free_blocks 无合适块` | `next_new_offset_ += align(size, 32)` |
| 复用 | `findReusableBlock(size)` | 分割空闲块，is_reused=true |
| 失败 | 所有尝试失败 | LOG(FATAL) |

**分配优先级**：预分配 > 新内存 > 复用空闲块

---

## 6. IR 示例

### 6.1 三种分配模式

| 模式 | 活跃区间关系 | 内存策略 | 示例 |
|-----|------------|---------|-----|
| 顺序分配 | 全部重叠，无释放时机 | 紧邻排列，32字节对齐 | A,B,C 连续放置 |
| 复用分配 | 不重叠，新buffer在旧buffer释放后 | 复用已释放空间 | C 复用 A 的地址 |
| 预分配 | 外部指定地址 | 保留地址，检查冲突 | address_map={D:5000} |

---

### 6.2 顺序分配

**场景**：三个缓冲区都在整个函数内使用，生命周期完全重叠，无法复用。

```
输入 IR（简化）：
  Allocate(bufA, shared, 1024)
  Allocate(bufB, shared, 2048)
  Allocate(bufC, shared, 512)
  // 三者都在同一作用域，从头用到尾

分配过程：
  bufA: offset=0, size=1024
    └─ 结束地址 = 0 + 1024 = 1024（已32对齐 ✓）

  bufB: offset=1024, size=2048
    └─ 起始地址=1024（bufA结束处，已对齐，无需padding）
    └─ 结束地址 = 1024 + 2048 = 3072

  bufC: offset=3072, size=512
    └─ 起始地址=3072（已对齐 ✓）
    └─ 结束地址 = 3584

输出 address_map：
  {bufA: 0, bufB: 1024, bufC: 3072}

内存布局可视化：
  地址:    0        1024      3072      3584
           ├─────────┼──────────┼─────────┤
           │ bufA    │ bufB     │ bufC    │
           │ 1024字节 │ 2048字节 │ 512字节  │
           └─────────┴──────────┴─────────┘
  总内存占用：3584 字节
```

---

### 6.3 复用分配

**场景**：bufA 先释放，bufC 可以复用 bufA 的空间。

```
输入 IR（语句线性化后的索引）：
  索引 0-10：  BufferStore(bufA, ...)  ← bufA 活跃 [0, 10]
  索引 5-15：  BufferStore(bufB, ...)  ← bufB 活跃 [5, 15]，与A重叠
  索引 12-20： BufferStore(bufC, ...)  ← bufC 活跃 [12, 20]

活跃度分析结果：
  bufA: GEN=0,  KILL=10, size=1024
  bufB: GEN=5,  KILL=15, size=2048
  bufC: GEN=12, KILL=20, size=512

关键判断：
  bufC.GEN=12 > bufA.KILL=10  → bufA 已释放，bufC 可复用 bufA 的空间
  bufC.GEN=12 < bufB.KILL=15  → bufB 还活跃，bufC 不能复用 bufB

分配过程：
  bufA: offset=0, size=1024, is_reused=false
  bufB: offset=1024, size=2048, is_reused=false
  bufC: offset=0, size=512, is_reused=true（复用bufA）

输出 address_map：
  {bufA: 0, bufB: 1024, bufC: 0}

内存布局可视化（时间轴）：
  时间 0-10（bufA 活跃）:
    地址: 0        1024      3072
          ├─────────┼──────────┤
          │ bufA    │ bufB     │
          │ 1024    │ 2048     │
          └─────────┴──────────┘

  时间 11-20（bufA 已释放，bufC 复用）:
    地址: 0        1024      3072
          ├─────────┼──────────┤
          │ bufC    │ bufB     │
          │ 512     │ 2048     │ ← bufC 放在 bufA 原位置
          └─────────┴──────────┘

  总内存占用：3072 字节（比顺序分配节省 512 字节）
```

---

### 6.4 预分配

**场景**：外部已指定缓冲区地址（如手工优化或前置Pass结果）。

```
输入：函数属性 address_map = {bufD: 5000}
  
处理流程：
  1. SetPreAllocBuffer() 检测到 bufD 有预分配地址
  2. CheckConflict() 检查与其他分配是否地址重叠
  3. 保留预分配地址，跳过重新规划

输出 address_map：
  {bufD: 5000}  // 保持不变

冲突检测示例：
  若有其他分配 offset=4800, size=400 → 区间 [4800, 5200]
  bufD 预分配在 5000, size=1024 → 区间 [5000, 6024]
  重叠区间 [5000, 5200] → CheckConflict() 报错 LOG(FATAL)
```

---

### 6.5 对齐规则说明

**32字节对齐**：所有分配的起始地址必须是 32 的倍数。

```
AlignUp(addr, 32) = ((addr + 31) / 32) * 32

需要额外对齐的情况：
  bufA size=1000 → 结束地址=1000
  1000 % 32 = 8（未对齐）
  bufB 需要 AlignUp(1000, 32) = 1024

不需要额外对齐的情况：
  bufA size=1024 → 结束地址=1024
  1024 % 32 = 0（已对齐）
  bufB 直接从 1024 开始
```

---

## 7. 边界处理

| 条件 | 处理方式 |
|-----|---------|
| 重复缓冲区名 | LOG(FATAL) 要求唯一名称 |
| 非整数 extent | ICHECK 要求 IntImmNode |
| 预分配冲突 | CheckConflict() 检测重叠 |
| 内存不足 | LOG(FATAL) 报告失败 |
| 对齐要求 | AlignUp(value, 32) 强制对齐 |

---

## 8. API 与方法索引

### 8.1 按功能分类

| 分类 | 方法 | 功能 |
|-----|-----|-----|
| **入口** | Substitute | 主入口 |
| **构造** | AscendMemoryPlanner 构造 | 遍历+规划 |
| | SetPreAllocBuffer | 处理预分配 |
| | SetTmpBuffers | 处理临时缓冲区 |
| **遍历** | VisitStmt_(Allocate) | 记录 scope/size |
| | VisitStmt_(BufferStore) | 跟踪写访问 |
| | VisitNewScope | 记录作用域 |
| | TrackBufferTouch | 记录 touched |
| **分析** | LivenessAnalysis | GEN/KILL 分析 |
| | ReorderKillPoints | 调整 KILL 位置 |
| | FindEventIndex | 查找事件索引 |
| **规划** | PlanMemory | 主规划流程 |
| | PlanMemoryForScope | 自动规划（复用）|
| | PlanMemoryForScopeLinear | 顺序规划 |
| **分配** | LinearScanAllocator::allocate | 线性扫描 |
| | findReusableBlock | 查找可复用块 |
| | mergeFreeBlocks | 合并空闲块 |
| | CheckConflict | 预分配冲突检测 |
| **辅助** | CalculateBufferSize | 计算大小 |
| | IsNPUSharedMemory | 判断内存域 |
| **输出** | GetAddressMap | 返回 address_map |
| | GetBufferSizes | 返回 size_map |

---

## 10. 验证方案

### 10.1 测试场景分类表

| 类别 | 场景 | 活跃区间关系 | 预期行为 |
|-----|-----|------------|---------|
| 顺序分配 | 不开优化or全重叠 | 所有buffer GEN=0, KILL=end | 紧邻排列，32字节对齐 |
| 复用分配 | 不重叠 | 新buffer GEN > 旧buffer KILL | 复用已释放空间 |
| 预分配 | 外部指定 | address_map 已提供 | 保留地址，跳过规划 |
| 冲突检测 | 预分配重叠 | 预分配地址与其他分配重叠 | LOG(FATAL) |
| 对齐验证 | 未对齐size | buffer size 非32倍数 | 自动对齐到32字节边界 |

### 10.2 测试用例设计

#### 用例1: 顺序分配

**测试意图**：验证多个全重叠 buffer 正确顺序分配，32字节对齐。

**输入 IR**：
```
Allocate(bufA, shared, 1024)
Allocate(bufB, shared, 2048)
Allocate(bufC, shared, 512)
// 三者都在整个函数内活跃
```

**预期输出 address_map**：
```
{bufA: 0, bufB: 1024, bufC: 3072}
```

**内存布局**：
```
地址: 0     1024    3072    3584
      |bufA |bufB   |bufC   |
```

**验证点**：紧邻排列，32字节对齐

---

#### 用例2: 复用分配

**测试意图**：验证不重叠活跃区间正确复用已释放空间。

**输入 IR（活跃区间）**：
```
bufA: GEN=0,  KILL=10, size=1024
bufB: GEN=5,  KILL=15, size=2048
bufC: GEN=12, KILL=20, size=512
```

**预期输出 address_map**：
```
{bufA: 0, bufB: 1024, bufC: 0}  // bufC 复用 bufA
```

**内存布局**：
```
时间0-10:  |bufA|bufB|
时间11-20: |bufC|bufB|  // bufC 复用 bufA 位置
```

**验证点**：bufC.GEN(12) > bufA.KILL(10) → 复用生效

---

#### 用例3: 预分配

**测试意图**：验证外部预分配地址保留，跳过规划。

**输入配置**：
```
函数属性 address_map = {bufD: 5000}
```

**预期输出 address_map**：
```
{bufD: 5000}  // 保持不变
```

**验证点**：预分配地址保留，跳过规划

---

#### 用例4: 冲突检测

**测试意图**：验证预分配地址与其他分配重叠时正确报错。

**输入配置**：
```
预分配 bufD: 地址 5000, size=1024
其他分配 bufE: 地址 4800, size=400 → 区间 [4800, 5200]
```

**预期行为**：
```
LOG(FATAL) - 地址重叠检测失败
```

**验证点**：冲突检测生效

---

#### 用例5: 对齐验证

**测试意图**：验证 size 非32倍数时自动对齐到32字节边界。

**输入 IR**：
```
Allocate(bufA, shared, 1000)  // size 非32倍数
Allocate(bufB, shared, 512)
```

**预期输出 address_map**：
```
{bufA: 0, bufB: 1024}  // bufB 从1024开始（对齐后）
```

**内存布局**：
```
bufA: [0, 1000] → 对齐后结束地址 1024
bufB: [1024, 1536]
```

**验证点**：32字节对齐生效

### 10.3 验证命令

**IR dump 验证（查看 address_map/size_map）**
```python
import tilelang
import tilelang.language as T

tilelang.cache.clear_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

@tilelang.jit(out_idx=[2], pass_configs=pass_configs)
def test_memory(M=128, N=128, K=128):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), 'float16'),
        B: T.Tensor((K, N), 'float16'),
        C: T.Tensor((M, N), 'float16'),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_l1 = T.alloc_L1((M, K), 'float16')
            b_l1 = T.alloc_L1((K, N), 'float16')
            c_l0c = T.alloc_L0C((M, N), 'float')
            T.copy(A, a_l1)
            T.copy(B, b_l1)
            T.gemm_v0(a_l1, b_l1, c_l0c)
            T.copy(c_l0c, C)
    return main

func = test_memory()
mod = func.ir_module
print('address_map:', mod['main'].attrs.get('address_map'))
print('size_map:', mod['main'].attrs.get('size_map'))
```

---

## 附录 A：内存限制常量

```
ASCEND_SHARED_MEM_SIZE           = 196352   // shared
ASCEND_SHARED_DYN_MEM_SIZE       = 524032   // shared.dyn
ASCEND_WMMA_MATRIX_A_MEM_SIZE    = 65536    // wmma.matrix_a
ASCEND_WMMA_MATRIX_B_MEM_SIZE    = 65536    // wmma.matrix_b
ASCEND_WMMA_ACCUMULATOR_MEM_SIZE = 131072   // wmma.accumulator
```

---

## 附录 B：配置与依赖

**配置选项**：
- Pass 配置：`tl.ascend_memory_planning`（Bool，默认 false）
- 函数属性输入：`address_map`（预分配），`buffer_shapess`（临时缓冲区）
- 函数属性输出：`address_map`，`size_map`

**依赖**：`arith/ir_mutator_with_analyzer.h`, `tir/transforms/ir_utils.h`, TVM TIR

**注册**：`TVM_REGISTER_GLOBAL("tl.transform.AscendMemoryPlanning")`