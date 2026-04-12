# IR 变换示例模板

本文件提供典型 Pass 的 IR 变换示例，使用 **伪 IR (Pseudo IR)** 格式演示。

---

## 一、伪 IR 格式说明

### 什么是伪 IR？

伪 IR 是一种简化的 IR 表示格式，用于直观展示 Pass 的变换效果，而非完整的 TVM TIR 结构。

### 伪 IR vs 真实 TIR

| 对比维度 | 伪 IR | 真实 TIR |
|---------|-------|---------|
| 目的 | 简化展示，便于理解 | 完整表示，用于编译 |
| 结构 | 保留关键元素，省略细节 | 包含完整属性和类型 |
| 类型标注 | 简化或省略 | 严格标注 (int32, float16 等) |
| 变量表示 | 简化名称 (i, j, A) | TVM Var 对象 |

### 真实 TIR 结构示例（仅供参考）

```
# 真实 TIR 结构 (TVM IR)

PrimFunc {
  name: "kernel"
  params: [
    Var("A", Pointer(float32, "global")),
    Var("B", Pointer(float16, "shared")),
    Var("C", Pointer(float32, "global"))
  ]
  attrs: {
    "target": Target("cuda"),
    "global_symbol": "kernel"
  }
  body: Stmt {
    SeqStmt {
      Allocate {
        buffer_var: Var("B", Pointer(float16, "shared"))
        extents: [1024]
        condition: True
        body: Stmt {
          For {
            loop_var: Var("i", int32)
            min: IntImm(0)
            extent: IntImm(1024)
            kind: ForKind::Serial
            body: Stmt {
              Store {
                buffer_var: Var("B")
                index: Var("i")
                value: Cast(float16, Load("A", Var("i")))
              }
            }
          }
          For {
            loop_var: Var("j", int32)
            min: IntImm(0)
            extent: IntImm(1024)
            kind: ForKind::Parallel
            body: Stmt {
              Store {
                buffer_var: Var("C")
                index: Var("j")
                value: Add(Load("B", Var("j")), IntImm(1.0))
              }
            }
          }
        }
      }
    }
  }
}
```

### 伪 IR 格式约定

```
# 伪 IR 格式约定

# Buffer 声明
Buffer A[scope, dtype, shape]
  - scope: global / shared / L1 / UB / L0A / L0B / L0C
  - dtype: float16 / float32 / int32 (可省略)
  - shape: [N, M] (可简化)

# 循环结构
For var in [min, max] { body }
Parallel For var in [min, max] { body }  # 并行循环

# 内存操作
Load(buffer, index)      # 读取
Store(buffer, index, value)  # 写入
  - 简化写法: A[i] = B[i] + 1.0

# 同步指令
Sync[scope]              # 同步指令

# Allocate
Allocate buffer[scope, dtype, size]

# 向量指令 (Ascend)
VectorOp { dst, src1, src2, ... }
  - VectorAdd, VectorMul, VectorCopy 等

# 矩阵指令 (Ascend Cube)
CubeOp { dst, src1, src2, ... }
  - CubeGemm 等

# 变化标注
← 新增/修改            # 用箭头标注新增或修改的内容
# [备注]               # 用井号备注说明
```

---

## 二、Ascend 平台 Pass IR 示例

### AscendSyncInsert

**Pass 目标：** 通过循环展开分析所有可能的同步需求，在依赖点插入 PipeBarrier 或 EventPair 同步指令。

**核心原理：**
1. **循环展开**：每个 For 展开为 iter1 + iter2 两个迭代（用 iteration_start/end 标记）
2. **依赖检测**：遍历所有语句，分析内存访问，检测 WAW/RAW/WAR 依赖
3. **同步类型选择**（GetRequiredSyncType）：
   - 切片操作 → `PipeBarrier_ALL`（全局同步）
   - 同一 pipeline 内 → `PipeBarrier_<pipeline>`（如 PipeBarrier_MTE2, PipeBarrier_FIX）
   - 跨 pipeline → `EventPair_<src>_<dst>`（共26种组合，见下表）

**EventPair 映射表（基于 src/transform/common/operation_config.h:264-300）：**
```
PIPE_MTE2 → PIPE_MTE1: EventPair_MTE2_MTE1
PIPE_MTE1 → PIPE_MTE2: EventPair_MTE1_MTE2
PIPE_MTE1 → PIPE_M: EventPair_MTE1_M
PIPE_M → PIPE_MTE1: EventPair_M_MTE1
PIPE_MTE2 → PIPE_V: EventPair_MTE2_V
PIPE_V → PIPE_MTE2: EventPair_V_MTE2
PIPE_MTE3 → PIPE_V: EventPair_MTE3_V
PIPE_V → PIPE_MTE3: EventPair_V_MTE3
PIPE_M → PIPE_V: EventPair_M_V
PIPE_V → PIPE_M: EventPair_V_M
PIPE_V → PIPE_V: EventPair_V_V
PIPE_MTE3 → PIPE_MTE1: EventPair_MTE3_MTE1
PIPE_MTE1 → PIPE_MTE3: EventPair_MTE1_MTE3
PIPE_MTE1 → PIPE_V: EventPair_MTE1_V
PIPE_MTE2 → PIPE_M: EventPair_MTE2_M
PIPE_M → PIPE_MTE2: EventPair_M_MTE2
PIPE_V → PIPE_MTE1: EventPair_V_MTE1
PIPE_M → PIPE_FIX: EventPair_M_FIX
PIPE_FIX → PIPE_M: EventPair_FIX_M
PIPE_MTE3 → PIPE_MTE2: EventPair_MTE3_MTE2
PIPE_MTE2 → PIPE_MTE3: EventPair_MTE2_MTE3
PIPE_S → PIPE_V: EventPair_S_V
PIPE_V → PIPE_S: EventPair_V_S
PIPE_S → PIPE_MTE2: EventPair_S_MTE2
PIPE_MTE2 → PIPE_S: EventPair_MTE2_S
PIPE_S → PIPE_MTE3: EventPair_S_MTE3
PIPE_MTE3 → PIPE_S: EventPair_MTE3_S
PIPE_MTE2 → PIPE_FIX: EventPair_MTE2_FIX
PIPE_FIX → PIPE_MTE2: EventPair_FIX_MTE2
PIPE_FIX → PIPE_S: EventPair_FIX_S
PIPE_M → PIPE_S: EventPair_M_S
PIPE_FIX → PIPE_MTE3: EventPair_FIX_MTE3
```
4. **合并重建**：将 iter1/iter2 的同步合并，重建 For 循环

**输入伪 IR：**
```
# 变换前：循环内存在跨 pipeline 依赖
For i in [0, N] {
  copy_gm_to_l1(A[i])         # PIPE_MTE2: GM → L1 (写入 A)
  gemm(A_l1, B_l0a, C_l0c)    # PIPE_M: 矩阵计算 (读取 A)
}
# 依赖：MTE2 写 A → M 读 A，需要跨 pipeline 同步 EventPair_MTE2_M
```

**Step 1: 循环展开**
```
iteration_start[iter1]
  copy_gm_to_l1(A[0])         # pipeline=PIPE_MTE2
  gemm(A_l1, B_l0a, C_l0c)    # pipeline=PIPE_M
iteration_end[iter1]

iteration_start[iter2]
  copy_gm_to_l1(A[1])
  gemm(A_l1, B_l0a, C_l0c)
iteration_end[iter2]
```

**Step 2: 分析依赖并插入同步 (VisitStmt_(EvaluateNode))**

```
# 核心逻辑伪代码
VisitStmt_(EvaluateNode op):
  1. current_accesses = AnalyzeStmtAccesses(op)   # 分析当前语句内存访问
  
  2. for each access in current_accesses:
     a. if access.is_sliced:                      # 切片操作
          sync_requirements.add("PipeBarrier_ALL")
     
     b. related_buffers = FindRelatedBuffers(access.buffer_name)  # 查地址重叠的buffer
     
     c. for each buffer in related_buffers:
          if buffer in current_access_history_:   # 有历史访问
            latest = history[buffer]
            if HasDataDependency(latest, access): # 检测 WAW/RAW/WAR
              sync_type = GetRequiredSyncType(latest, access)
              sync_requirements.add(sync_type)
  
  3. optimized_syncs = OptimizeSyncRequirements(sync_requirements)  # 合并冗余
  
  4. InsertSynchronization(optimized_syncs)      # 插入同步语句
  
  5. UpdateSyncStatesAfterSync(optimized_syncs)  # 更新同步状态
  
  6. 返回 SeqStmt([sync_stmts..., op])           # 同步 + 原语句
```

**Step 3: 合并重建循环**
```
For i in [0, N] {
  copy_gm_to_l1(A[i])
  EventPair_MTE2_M            ← 合并后的跨 pipeline 同步
  gemm(A_l1, B_l0a, C_l0c)
}
```

**关键函数说明：**
| 函数 | 作用 |
|------|------|
| AnalyzeStmtAccesses | 解析语句中的 buffer 读/写操作，返回 BufferAccess 列表 |
| FindRelatedBuffers | 查找地址空间重叠的所有 buffer（基于 address_map） |
| HasDataDependency | 检测 WAW（写-写）/ RAW（读-写）/ WAR（写-读）依赖 |
| GetRequiredSyncType | 根据 pipeline 类型选择 PipeBarrier 或 EventPair |
| OptimizeSyncRequirements | 合并重复同步，移除冗余同步 |
| InsertSynchronization | 生成同步语句并插入到语句序列前 |

**变换要点：**
- 循环展开后，每个 EvaluateNode 都会被访问并分析内存访问
- 基于 address_map 检测地址重叠的 buffer（内存复用场景）
- 根据 pipeline 组合选择正确的同步类型

---

### AscendMemoryPlanning

**Pass 目标：** 通过生命周期分析和线性扫描分配，规划 buffer 地址，优化内存复用。

**核心类：** `AscendMemoryPlanner` (继承 `StmtExprVisitor`) + `LinearScanAllocator`

**核心实现逻辑：**

```
# 两阶段处理流程

Phase 1: AscendMemoryPlanner (StmtExprVisitor)
  → 收集 buffer 信息
  
  VisitStmt_(AllocateNode):
    1. 记录 buffer scope (shared.dyn/wmma.matrix_a/wmma.matrix_b/wmma.accumulator)
    2. 计算 buffer size
    3. 记录到 buffer_sizes_[buf]
  
  VisitStmt_(BufferStoreNode/BufferLoadNode/EvaluateNode):
    1. TrackBufferTouch(buf) - 记录 buffer 被访问
    2. 记录到 linear_seq_ 语句序列
    3. 记录 first_use_[buf] 首次使用位置
  
Phase 2: PlanMemory()
  → 生命周期分析 + 地址分配
  
  LivenessAnalysis():
    1. 逆序遍历 linear_seq_，找出每个 buffer 的最后使用点 (KILL)
    2. 正序遍历，找出每个 buffer 的首次使用点 (GEN)
    3. 构建 event_map_[stmt] = {gen: [...], kill: [...]}
  
  PlanMemoryForScope(scope, buffers):
    1. 为每个 buffer 构建生命区间 [start, end]
    2. 按开始时间排序
    3. LinearScanAllocator.allocate(intervals) - 线性扫描分配
    4. 输出 address_map_[buf] = offset
```

**关键数据结构：**
| 结构 | 作用 |
|------|------|
| `linear_seq_` | 按执行顺序记录语句，用于生命周期分析 |
| `event_map_` | 记录每个语句点的 GEN/KILL 集合 |
| `buffer_sizes_` | 记录每个 buffer 的大小 |
| `address_map_` | 输出：buffer → 偏移地址映射 |
| `LiveInterval` | 生命区间 {buffer, start, end, size} |

**输入伪 IR：**
```
Allocate A[shared.dyn, 1024]
Allocate B[shared.dyn, 512]
Allocate C[shared.dyn, 2048]

Stmt1: 使用 A
Stmt2: 使用 B
Stmt3: 使用 A  # A 生命周期结束
Stmt4: 使用 C  # C 可复用 A 的地址
Stmt5: 使用 B  # B 生命周期结束
```

**输出伪 IR：**
```
# 生命周期: A[0,2], B[1,4], C[3,4]
# 地址分配: A→0, B→1024, C→0 (复用 A)

address_map: {A: 0, B: 1024, C: 0}
size_map: {A: 1024, B: 512, C: 2048}
total_memory: max(1024+512, 2048) = 2048 bytes
```

**变换要点：**
- 生命周期不重叠的 buffer 可复用地址
- 线性扫描算法按起始时间排序，贪心分配
- 输出 address_map 和 size_map 属性供 AscendSyncInsert 使用

---

### CombineCV

**Pass 目标：** 分离 Cube 和 Vector 操作，将混合代码拆分为两块独立代码块，分别发送给 Cube 核和 Vector 核执行。

**核心类：** `CombineCV` (继承 `IRMutatorWithAnalyzer`) + `CVCombineEmitter` (继承 `StmtMutator`)

**核心实现逻辑：**

```
# 两阶段处理流程

Phase 1: CombineCV.VisitStmt_(BlockRealizeNode)
  → 找到 tilelang_root，准备分离
  
  if (block.name_hint == "tilelang_root"):
    1. 创建两个 CVCombineEmitter:
       - cubeStmt(is_aiv=false) - 过滤 Cube 操作
       - vecStmt(is_aiv=true)    - 过滤 Vector 操作
    
    2. 分别处理 body:
       cube_code = cubeStmt(body)
       vec_code = vecStmt(body)
    
    3. 包装为 AttrStmt:
       AttrStmt[resource_scope=0] { cube_code }  # Cube 核
       AttrStmt[resource_scope=1] { vec_code }   # Vector 核

Phase 2: CVCombineEmitter.VisitStmt_(EvaluateNode)
  → 过滤操作，保留或丢弃
  
  1. 获取 api_name (如 "gemm", "copy", "relu" 等)
  
  2. 判断 1 - 根据 API 名称:
     if is_aiv_ (Vector):       # Vector emitter
       if api_name == "vec":    → 保留 (switch=on)
       if api_name == "cube":   → 丢弃 (switch=off)
     else (Cube):               # Cube emitter
       if api_name == "cube":   → 保留 (switch=on)
       if api_name == "vec":    → 丢弃 (switch=off)
  
  3. 判断 2 - 根据 Buffer Scope:
     checkBufferScope(var):
       if scope == "wmma.*":    → Cube
       if scope == "shared":    → Vector
       返回 1(保留)/0(丢弃)/-1(未知)
  
  4. 返回结果:
     if switch=on:  → StmtMutator::VisitStmt_(op)  # 保留
     if switch=off: → Evaluate(0)                  # 丢弃

CVCombineEmitter.VisitStmt_(BufferStoreNode):
  → 根据 buffer scope 过滤写入操作
  
  if is_aiv_:
    if scope == "shared":  → 保留
    else:                  → 丢弃
  else:
    if scope == "shared":  → 丢弃
    else:                  → 保留
```

**关键函数说明：**
| 函数 | 作用 |
|------|------|
| `isSubstringInMap` | 检查 API 名称是否匹配已知的 Cube/Vector 操作 |
| `checkBufferScope` | 根据 buffer scope 判断操作属于哪个核 |
| `IsRetainedInBothScopes` | 某些 API（如 printf）需要在两个核上都保留 |

**输入伪 IR：**
```
tilelang_root {
  For i in [0, 64] {
    Copy(global, L0A[i])           # Cube
    CubeGemm(L0A[i], L0B[i], L0C[i]) # Cube
    VectorReLU(L0C[i], UB[i])      # Vector
    Copy(UB[i], global)            # Vector
  }
}
```

**输出伪 IR：**
```
tilelang_root {
  AttrStmt[resource_scope=0] {
    For i in [0, 64] {
      Copy(global, L0A[i])
      CubeGemm(L0A[i], L0B[i], L0C[i])
    }
  }
  
  AttrStmt[resource_scope=1] {
    For i in [0, 64] {
      VectorReLU(L0C[i], UB[i])
      Copy(UB[i], global)
    }
  }
}
```

**变换要点：**
- 两个 Emitter 实例分别过滤，生成两套独立代码
- 根据 API 名称 + Buffer Scope 双重判断操作归属
- resource_scope 标记用于后端代码生成区分核类型

---

### CrossCorePipeline

**Pass 目标：** 检测跨核流水线，将单循环拆分为多 stage 循环，实现 Cube-Vector 流水线并行。

**核心类：** `CrossCorePipeline` (继承 `IRMutatorWithAnalyzer`) + `CrossCoreDetector` + `LoopAnalyzer` + `LoopRewriter`

**核心实现逻辑：**

```
# 三阶段处理流程

Phase 1: CrossCoreDetector (StmtVisitor)
  → 检测跨核流水线循环
  
  VisitStmt_(ForNode):
    if annotation["num_stages"] 存在:
      1. 创建 PipelineInfo
      2. 遍历 body 中的 EvaluateNode
  
  VisitStmt_(EvaluateNode):
    1. 获取操作的 buffer scope
    2. checkBufferScope(var) → CUBE_SCOPE / VEC_SCOPE
    3. if scene == CUBE_SCOPE 后出现 VEC_SCOPE:
         → is_cross_core = true

Phase 2: LoopAnalyzer (StmtVisitor)
  → 分析循环内的 Cube/Vector 操作
  
  VisitStmt_(EvaluateNode):
    1. 判断操作属于 Cube 还是 Vector
    2. 记录 workspace_writes_C (Cube 写入)
    3. 记录 workspace_writes_V (Vector 写入)
    4. 分析操作依赖关系

Phase 3: LoopRewriter (StmtMutator)
  → 重写循环为多 stage
  
  Rewrite():
    1. 获取 num_stages, cross_interval
    2. 创建 outer_loop_var (stage 循环)
    3. 为每个 stage 创建副本:
       - stage 0: Cube 操作
       - stage 1: Vector 操作
       - ...
    4. 调整 buffer 访问的索引
    5. 插入跨核同步 (set_flag / wait_flag)
```

**关键函数说明：**
| 函数 | 作用 |
|------|------|
| `DetectCrossCorePipelines` | 检测是否存在跨核流水线 |
| `checkBufferScope` | 根据 buffer scope 判断 Cube/Vector |
| `Analyze` | 分析 Cube/Vector 操作分布 |
| `Rewrite` | 重写循环为多 stage 流水线 |

**输入伪 IR：**
```
For i in [0, N] annotations={num_stages: 2} {
  # Stage 0: Cube
  Copy(global, L0A[i])
  CubeGemm(L0A, L0B, L0C)
  
  # Stage 1: Vector (跨核依赖 L0C)
  VectorReLU(L0C, UB)
  Copy(UB, global)
}
```

**输出伪 IR：**
```
For stage in [0, N+num_stages] {
  # Stage 偏移执行，形成流水线
  
  # Cube stage
  if stage < N:
    Copy(global, L0A[stage])
    CubeGemm(L0A, L0B, L0C[stage])
    set_flag(Cube_done[stage % num_stages])
  
  # Vector stage (延迟 1 个 stage)
  if stage >= 1 && stage < N+1:
    wait_flag(Cube_done[(stage-1) % num_stages])
    VectorReLU(L0C[stage-1], UB)
    Copy(UB, global)
}
```

**变换要点：**
- 检测 num_stages 注解的循环是否有跨核操作
- 将单循环拆分为多 stage，形成流水线
- 使用 set_flag/wait_flag 替代阻塞同步
- stage 之间偏移执行，重叠 Cube 和 Vector 计算

---

### AscendLowerParallelToVector

**Pass 目标：** 将 Parallel 循环 lowering 为 Ascend Vector 指令。

**输入伪 IR：**
```
# 变换前：Parallel 循环（抽象表示）
Parallel For i in [0, 256] {
  C[i] = A[i] + B[i]              # 元素级加法
}
```

**输出伪 IR：**
```
# 变换后：Ascend Vector 指令
VectorAdd {                        ← Vector 核指令
  dst: C
  src1: A
  src2: B
  len: 256
  dtype: float16
}
```

**变换要点：**
- Parallel 循环 → Vector 指令
- 元素级操作合并为向量指令
- 指定 dtype、len 等参数

---

### InferAllocScope

**Pass 目标：** 根据 buffer 的使用方式推断正确的 scope（L0A/L0B/L0C/shared.dyn/shared）。

**核心类：** `ScopeCorrector` (继承 `StmtExprMutator`) + `BufferUseCollector` (继承 `StmtExprVisitor`)

**核心实现逻辑：**

```
# 三阶段处理流程

Phase 1: BufferUseCollector (StmtExprVisitor)
  → 收集 buffer 使用信息
  
  BuildHandleAllocMapping():
    → 建立 handle → Allocate 映射
  
  VisitExpr_(CallNode):
    1. 分析每个 call 的 buffer 参数
    2. if IsGEMMFunction(func_name):
         → used_in_cube = true
         → 记录 gemm_position (0=A, 1=B, 2=C)
    3. if IsVectorFunction(func_name):
         → used_in_vector = true
    4. 记录到 buffer_use_info_[handle]

Phase 2: InferCorrectScopes()
  → 根据使用信息推断 scope
  
  for each buffer:
    if original_scope == "local.fragment":
      if gemm_positions.count(0) > 0:
        → corrected_scope = "wmma.matrix_a"   # L0A
      elif gemm_positions.count(1) > 0:
        → corrected_scope = "wmma.matrix_b"   # L0B
      elif gemm_positions.count(2) > 0:
        → corrected_scope = "wmma.accumulator" # L0C
    
    elif original_scope == "shared.dyn":
      if used_in_vector && !used_in_cube:
        → corrected_scope = "shared"          # Vector 专用
      elif used_in_cube:
        → corrected_scope = "shared.dyn"      # Cube 可用

Phase 3: ScopeCorrector (StmtExprMutator)
  → 应用 scope 修正
  
  VisitStmt_(BlockNode):
    1. 遍历 alloc_buffers
    2. 替换 buffer.data 为新 scope 的 Var
    3. 更新 buffer_replacements_
  
  VisitExpr_(VarNode):
    1. 查找 var_replacements_
    2. 返回修正后的 Var
  
  InjectDefaultLayoutMap():
    1. 为 shared.dyn buffer 注入默认 zN Layout
    2. 避免用户未指定 layout 时后端无法计算地址
```

**关键函数说明：**
| 函数 | 作用 |
|------|------|
| `IsGEMMFunction` | 判断是否为矩阵乘操作（gemm/mma/matmul） |
| `IsVectorFunction` | 判断是否为 Vector 操作 |
| `DetermineGEMMPosition` | 确定 buffer 在 gemm 中的位置（A/B/C） |
| `InferCorrectScopes` | 根据使用信息推断正确的 scope |
| `InjectDefaultLayoutMap` | 为 L1 buffer 注入默认 layout |

**输入伪 IR：**
```
Allocate A[local.fragment, float16]   # generic scope
Allocate B[local.fragment, float16]
Allocate C[local.fragment, float32]

# 使用:
Gemm(A, B, C)  # A 是第0个参数, B 是第1个, C 是第2个
```

**输出伪 IR：**
```
Allocate A[wmma.matrix_a, float16]    # L0A
Allocate B[wmma.matrix_b, float16]    # L0B
Allocate C[wmma.accumulator, float32] # L0C
```

**变换要点：**
- 分析 buffer 在 gemm 中的位置推断 L0A/L0B/L0C
- 分析 Cube/Vector 使用情况推断 shared.dyn/shared
- 为 L1 buffer 注入默认 zN Layout 用于后端地址计算

---

### AscendLowerOpaqueBlock

**Pass 目标：** 将 Block IR lowering 为可执行的底层 IR，移除调度抽象。

**核心类：** `OpaqueBlockLower` (继承 `StmtExprMutator`)

**核心实现逻辑：**

```
# 核心变换流程

VisitStmt_(BlockRealizeNode):
  1. 检查 iter_values 为空（opaque block）
  
  2. 处理 predicate:
     if !is_one(predicate):
       body = IfThenElse(predicate, body)
  
  3. 处理 annotations:
     HandleAnnotations() → 提取 pragma 属性
  
  4. 处理 alloc_buffers（逆序）:
     for each buffer in alloc_buffers:
       a. body = DeclBuffer(buffer, body)
       b. body = Allocate(buffer.data, shape, body)
       c. 注入 storage_align annotation
  
  5. 插入 AttrStmt（pragma 转换）
  
  6. 返回变换后的 body

VisitStmt_(ForNode):
  1. 处理 unit loop（extent=1）:
     → 直接展开为 body
  
  2. 处理 ThreadBinding:
     → 转换为 AttrStmt[thread_extent]
  
  3. 处理普通循环:
     → 保留 For，处理 annotations

VisitExpr_(VarNode):
  → 替换 unit loop 变量为常量
```

**关键数据结构：**
| 结构 | 作用 |
|------|------|
| `unit_loop_vars_` | 记录 unit loop 变量到常量的映射 |
| `storage_align_` | 记录 buffer 的对齐要求 |
| `local_var_init_map_` | 记录局部变量初始化 |

**输入伪 IR：**
```
BlockRealize {
  block: Block {
    alloc_buffers: [A, B, C]
    body: For i in [0, N] {
      A[i] = B[i] + C[i]
    }
  }
}
```

**输出伪 IR：**
```
Allocate A[...] {
  Allocate B[...] {
    Allocate C[...] {
      For i in [0, N] {
        A[i] = B[i] + C[i]
      }
    }
  }
}
```

**变换要点：**
- Block → Allocate + DeclBuffer 嵌套
- predicate → IfThenElse
- unit loop → 展开
- ThreadBinding → AttrStmt
- 移除调度抽象，生成可执行代码

---

### AscendStorageRewrite

**Pass 目标：** 分析内存访问模式，优化存储共享和重写。

**核心类：** `LinearAccessPatternFinder` (继承 `StmtExprVisitor`) + `InplaceOpVerifier` + `StoragePlanRewriter` (继承 `StmtExprMutator`)

**核心实现逻辑：**

```
# 三阶段处理流程

Phase 1: AllocateCollector (StmtExprVisitor)
  → 收集 shared.dyn 和 shared 分配
  
  VisitStmt_(AllocateNode):
    if IsDynamicSharedMemory(buffer_var):
      → dyn_shmem_allocs_[buf] = op
    elif IsStaticSharedMemory(buffer_var):
      → static_shmem_allocs_[buf] = op

Phase 2: LinearAccessPatternFinder (StmtExprVisitor)
  → 构建线性访问序列，分析生命周期
  
  VisitStmt_(AllocateNode):
    1. 记录 alloc_info_[buf]
    2. 记录 num_physical_dimensions
  
  VisitStmt_(BufferStoreNode/BufferLoadNode):
    1. TrackBufferTouch(buf)
    2. 记录到 linear_seq_[stmt].touched
  
  VisitStmt_(AttrStmtNode/ForNode/IfThenElseNode):
    → 创建 scope entry/exit 点

Phase 3: StoragePlanRewriter (StmtExprMutator)
  → 应用存储优化计划
  
  VisitStmt_(BufferStoreNode):
    → 重写 buffer 访问
  
  VisitExpr_(BufferLoadNode):
    → 重写 buffer 访问
  
  VisitExpr_(VarNode):
    → 替换为新的 buffer var
```

**关键数据结构：**
| 结构 | 作用 |
|------|------|
| `linear_seq_` | 线性访问序列，每个点记录 touched buffers |
| `alloc_info_` | 记录每个 buffer 的分配信息 |
| `scope_` | 当前作用域栈 |

**变换要点：**
- 构建线性访问序列用于生命周期分析
- 分析 shared.dyn 和 shared memory 使用
- 检测可以合并或共享的存储
- 重写 buffer 访问以实现存储优化

---

### Flatten2DBuffer

**Pass 目标：** 将 buffer 形状扁平化为 2D。

**输入伪 IR：**
```
# 变换前：多维 buffer
Buffer A[shared, float16, [64, 64, 64]]   # 3D → 需要扁平化
Buffer B[shared, float16, [1024]]         # 1D → 需要扁平化
Buffer C[shared, float16, [128, 256]]     # 2D → 无需变换
```

**输出伪 IR：**
```
# 变换后：统一为 2D
Buffer A[shared, float16, [4096, 64]]     ← [64*64, 64]
Buffer B[shared, float16, [1, 1024]]      ← [1, 1024]
Buffer C[shared, float16, [128, 256]]     ← 保持不变
```

**变换要点：**
- 1D [M] → 2D [1, M]
- ND [D1, D2, ..., Dn] → 2D [D1*D2*...*Dn-1, Dn]
- 适配 Ascend 硬件对 2D tensor 的优化

---

## 三、通用 Pass IR 示例

### LowerTileOp

**Pass 目标：** 将高级 Tile 操作 lowering 为底层 IR，应用 Layout 变换重映射 buffer 访问索引。

**核心类：** `LowerTileOpPass` (继承 `IRMutatorWithAnalyzer`) + `RemapBufferRewriter`

**Tile 操作 Lower 实现：** `src/op/ascend.cc`
- `AscendCopy::Lower()` (行61-500+) - 将 T.copy 操作转换为具体的内存拷贝 API
- 支持多种 scope 转换：GM↔L1、L1↔L0A/L0B、L0C↔GM、GM↔UB、UB↔GM
- 根据 src/dst scope 自动选择对应的 C++ 函数模板（如 `copy_gm_to_l1`, `copy_l1_to_l0a` 等）

**核心实现逻辑：**

```
# 两阶段处理流程

Phase 1: LowerTileOpPass (IRMutatorWithAnalyzer)
  → 解析并 Lower Tile 操作
  
  VisitStmt_(BlockNode):
    1. 记录 buffer_map_ (buffer data → buffer)
    2. 读取 layout_map 注解
    3. for each (buffer, layout):
       buffer_remap_[buffer] = makeBufferWithLayout(buffer, layout)
    4. 更新 alloc_buffers
  
  VisitStmt_(EvaluateNode):
    1. ParseOperator(stmt) → tile_op
    2. if tile_op != null:
         lowered = tile_op->Lower(LowerArgs{...})
         → 调用具体操作的 Lower 实现（如 AscendCopy::Lower）
    3. 返回 lowered stmt
  
  VisitExpr_(BufferLoadNode/BufferStoreNode):
    if buffer in buffer_remap_:
      new_indices = layout_map_[buffer]->Forward(indices)

Phase 2: RemapBufferRewriter
  → 重映射 buffer 引用，更新 padding 注解
```

**关键数据结构：**
| 结构 | 作用 |
|------|------|
| `buffer_remap_` | buffer → 新 buffer（带 layout 变换后的 shape） |
| `layout_map_` | buffer → Layout 变换规则 |
| `var_remap_` | Var → 新 Var（local.fragment → local） |

**输入伪 IR：**
```
Block {
  alloc_buffers: [A, B]
  annotations: {layout_map: {A: zN_Layout}}
  body: {
    # T.copy 操作
    Evaluate(Call(tl.ascend_copy, [Region(A), Region(B)]))
  }
}
```

**输出伪 IR：**
```
Block {
  alloc_buffers: [A_new, B]
  body: {
    # Lowered 为底层 API 调用
    Evaluate(Call(call_extern, [
      "tl::ascend::copy_l1_to_l0a<half, 128, 64>",
      src_ptr,
      dst_ptr,
      ...
    ]))
  }
}
```

**变换要点：**
- ParseOperator 解析 Tile 操作（T.copy, T.gemm 等）
- 调用具体操作的 Lower 方法生成底层调用（实现在 `src/op/ascend.cc`）
- 根据 layout_map 对 buffer 访问索引进行变换
- local.fragment → local 转换

**详细 Lower 实现：**
`src/op/ascend.cc` 中的 `AscendCopy::Lower()` 方法负责：
1. 根据 src/dst buffer 的 scope 判断拷贝类型（如 GM→L1、L1→L0A 等）
2. 生成对应的 C++ 函数调用（如 `tl::ascend::copy_gm_to_l1<half, M, N>`）
3. 处理 layout 变换，计算新的 buffer indices
4. 生成完整的 TIR Evaluate 语句

---

### VectorizeLoop

**Pass 目标：** 将标量循环向量化。

**输入伪 IR：**
```
# 变换前：标量循环
For i in [0, 128] {
  A[i] = B[i] * 2.0
}
```

**输出伪 IR：**
```
# 变换后：向量化
Vectorized {
  A[0:128] = B[0:128] * 2.0       ← 批量向量操作
}
```

**变换要点：**
- 循环展开为向量操作
- 提高并行度

---

### InjectSoftwarePipeline

**Pass 目标：** 注入软件流水线优化。

**输入伪 IR：**
```
# 变换前：普通循环
For iter in [0, N] {
  Load(A, global)                 # Stage 1: 加载
  Compute(B, A)                   # Stage 2: 计算
  Store(global, B)                # Stage 3: 存储
}
```

**输出伪 IR：**
```
# 变换后：软件流水线
Pipeline {
  prologue: {
    Load(A_0, global)             # 预加载第一轮数据
  }
  main_loop: For iter in [0, N-1] {
    Load(A_next, global)          ← 下一轮加载
    Compute(B_cur, A_cur)         ← 当前轮计算
    Store(global, B_prev)         ← 前一轮存储
    # 流水线: 加载-计算-存储并行执行
  }
  epilogue: {
    Compute(B_last, A_last)
    Store(global, B_last)
  }
}
```

**变换要点：**
- 循环拆分为 prologue/main_loop/epilogue
- 重叠执行不同 stage
- 提高指令级并行

---

### LayoutInference

**Pass 目标：** 推断 tensor layout。

**输入伪 IR：**
```
# 变换前：未指定 layout
Buffer A[fragment, float16, [M, K]]
Buffer B[fragment, float16, [K, N]]

# 用于矩阵乘: C = A * B
```

**输出伪 IR：**
```
<<<<<<< HEAD
# 变换后：推断 layout
=======
# 叇换后：推断 layout
>>>>>>> [Skill] Add pass skill pass-analyzer & pass-workflow-analyzer
Buffer A[fragment, float16, [M, K], layout="row_major"]    ← 行主序
Buffer B[fragment, float16, [K, N], layout="column_major"] ← 列主序
Buffer C[fragment, float32, [M, N], layout="row_major"]

# Layout 优化矩阵乘访问效率
```

**变换要点：**
- 分析操作类型（矩阵乘等）
- 推断最优 layout
- 添加 layout 属性

---

### Simplify

**Pass 目标：** 简化 IR 结构。

**输入伪 IR：**
```
# 变换前：冗余结构
For i in [0, 128] {
  For j in [0, 1] {               ← 冗余循环 (extent=1)
    A[i] = B[i] + (0 * C[i])      ← 冗余计算 (乘 0)
  }
}
```

**输出伪 IR：**
```
# 变换后：简化
For i in [0, 128] {
  A[i] = B[i]                     ← 消除冗余
}
```

**变换要点：**
- 消除 extent=1 的循环
- 消除常数计算
- 合并相邻操作

---

## 四、复杂 Pass 组合示例

### Ascend GEMM 完整编译链

**输入伪 IR：**
```
# 原始 Python DSL
@tilelang.jit
def gemm(M, N, K):
  A = T.alloc_shared([M, K], dtype="float16")
  B = T.alloc_shared([K, N], dtype="float16")
  C = T.alloc_shared([M, N], dtype="float32")
  
  T.copy(global_A, A)
  T.copy(global_B, B)
  T.gemm(A, B, C)
  T.copy(C, global_C)
```

**Pass 链变换结果：**
```
# Pass 链: InferAllocScope → AscendMemoryPlanning → AscendSyncInsert → AscendLowerParallelToVector

# 最终 IR (伪 IR 表示)
Allocate A[L0A, float16, addr=0, size=M*K]
Allocate B[L0B, float16, addr=M*K, size=K*N]
Allocate C[L0C, float32, addr=0, size=M*N]

# Cube 核矩阵乘
For tile in [0, num_tiles] {
  Copy { src: global_A[tile], dst: A[tile] }
  Copy { src: global_B[tile], dst: B[tile] }
  Sync[L1]                         ← AscendSyncInsert 插入
  CubeGemm { dst: C, src1: A, src2: B }
  Sync[CrossCore]                  ← 等待 Cube 完成
  Copy { src: C, dst: global_C }
}
```

**各 Pass 贡献：**
- InferAllocScope: 推断 A→L0A, B→L0B, C→L0C
- AscendMemoryPlanning: 规划地址，A/C 复用空间
- AscendSyncInsert: 插入 Sync[L1], Sync[CrossCore]
- AscendLowerParallelToVector: (此例无 Parallel 循环)

---

## 五、使用指南

### 如何编写 IR 示例

**步骤：**
```
1. 确定变换前 IR 结构（突出关键特征）
2. 确定变换后 IR 结构（标注变化点 ←）
3. 列出变换要点（不超过 3-5 点）
4. 保持格式一致，使用本文件的约定
```

**注意事项：**
- 首次使用伪 IR 时，添加说明："本示例为伪 IR 格式，真实 TIR 结构见 ir-examples.md 开头说明"
- 简化结构，不展示完整类型和属性
- 用箭头 ← 或井号 # 标注变化
- 变换要点简洁明了

### 参考模板

见 SKILL.md 中 "IR 示例编写规范" 部分。