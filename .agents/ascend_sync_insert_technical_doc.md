# Ascend Sync Insert Pass 技术说明文档

**源码位置**：`src/transform/ascend_sync_insert.cc`

**核心功能**：为昇腾 NPU 自动插入同步指令（PipeBarrier/EventPair），通过数据依赖分析和图可达性优化，确保多流水线间正确同步。

---

## 1. Pass 概览

| 特性 | 说明 |
|-----|-----|
| 循环展开重建 | 展开→插入→合并重建，区分迭代内/间依赖 |
| 同步依赖插入 | RAW/WAR/WAW + 物理地址区间重叠 |
| 同步优化 | SyncGraph 传递闭包，避免冗余同步 |
| 同步类型 | PipeBarrier（同流水线）/ EventPair（跨流水线）|

---

## 2. Pass 开发模板

### 2.1 类结构模板

```
继承：arith::IRMutatorWithAnalyzer
必须实现：
  ├─ static PrimFunc Substitute(PrimFunc, config, ctx, target, platform)  // 主入口
  ├─ VisitStmt_(EvaluateNode)   // 核心处理：分析→检测→插入
  ├─ VisitStmt_(AttrStmtNode)   // 处理 resource_scope/unrolled_loop
  ├─ VisitStmt_(SeqStmtNode)    // 递归处理语句序列
  └─ VisitStmt_(IfThenElseNode) // 条件分支前后全同步（可选）
```

### 2.2 标准处理流程骨架

```
Substitute 主入口：
  1. 检查 Pass 配置 → 未启用则返回原函数
  2. 预处理（如循环展开）
  3. 调用 VisitStmt 遍历 IR → 插入同步
  4. 后处理（如循环重建）
  5. 返回修改后的 PrimFunc

VisitStmt_(EvaluateNode) 核心：
  1. AnalyzeStmtAccesses → 分析缓冲区访问
  2. FindRelatedBuffers → 查找重叠缓冲区
  3. HasDataDependency → 检测数据依赖
  4. GetRequiredSyncType → 确定同步类型
  5. OptimizeSyncRequirements → 优化去冗余
  6. InsertSynchronization → 生成 IR
  7. UpdateSyncStatesAfterSync → 更新状态
```

### 2.3 Pass 注册模板

```
tvm::transform::Pass MySyncPass(Target target, std::string platform) {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return MySyncInsert::Substitute(f, "", ctx, target, platform);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.MySyncInsert", {});
}

TVM_REGISTER_GLOBAL("tl.transform.MySyncInsert").set_body_typed(MySyncInsert);
```

---

## 3. 处理流程骨架

```
Substitute 入口
    │
    ├─► 检查 tl.ascend_auto_sync 配置
    │
    ├─► PreprocessUnrollForLoops() ─► ForLoopUnroller
    │      └─ 保存 LoopInfo
    │      └─ 用 iteration_start/end 标记
    │
    ├─► VisitStmt 遍历展开后的 IR
    │      └─ VisitStmt_(EvaluateNode)：核心处理
    │
    ├─► MergeAndRebuildForLoops() ─► LoopRebuilder
    │      └─ 合并 iter1/iter2 同步
    │      └─ 重建原始循环
    │
    └─► 返回修改后的 PrimFunc
```

---

## 4. 数据结构速查表

### 4.1 核心结构体

| 结构体 | 关键字段 | 用途 |
|-------|---------|-----|
| **BufferAccess** | buffer_name, is_write, pipeline (MTE2/MTE1/MTE3/M/V/S/FIX), operation, sync_graph, pipe_barriers, physical_address, is_sliced | 访问分析核心 |
| **SyncGraph** | graph: map<string, set<string>> | 同步优化核心 |
| **LoopInfo** | loop_var, min, extent, kind, annotations, loop_id, depth | 循环重建 |
| SyncRequirement | sync_type, buffer_name | 同步需求 |
| BufferInfo | buffer_name, is_read, is_write, is_sliced | 缓冲区信息 |

### 4.2 SyncGraph 核心方法

| 方法 | 功能 |
|-----|-----|
| AddSync("EventPair_X_Y") | 提取边 X→Y |
| HasPath(src, dst) | DFS 判断可达 |
| Merge(other) | 合并图的边 |
| ComputeTransitiveClosure() | Floyd-Warshall 传递闭包 |

**用途示例**：已插入 EventPair_M_V → graph 有 M→V 边。若需 EventPair_M_MTE3 且存在 V→MTE3，则 M→MTE3 已传递满足，无需插入。

---

## 5. 算法逻辑

### 5.1 流水线类型

| 流水线 | 名称 | 典型操作 |
|-------|-----|---------|
| **PIPE_MTE2** | 内存加载 | copy_gm_to_l1/ub |
| **PIPE_MTE3** | 内存存储 | copy_ub_to_gm/copy_l0c_to_gm |
| **PIPE_M** | 矩阵计算 | gemm_v0/v1 |
| **PIPE_V** | 向量计算 | Add, Mul, Exp, Reduce, Sub, Div 等 |
| **PIPE_S** | 标量计算 | Scalar |
| **PIPE_FIX** | L0C搬运 | copy_l0c_to_gm/l1 |

**常见同步类型**（见 GetEventMapping）：
- MTE2→V, MTE3→V, V->MTE2, V->MTE3（内存与向量间）
- MTE2→M, MTE2→M, M->MTE2, M->MTE3（搬运与矩阵间）
- M→V（矩阵与向量间）
- V→V（向量内部）

### 5.2 同步依赖检测决策

| 条件 | 内存共享 | 依赖类型 | 结果 |
|-----|---------|---------|-----|
| 同名缓冲区 | true | - | 进入依赖判断 |
| 不同名 + 地址重叠 | true | - | 进入依赖判断 |
| 内存共享 + prev.write && curr.write | - | WAW | 需同步 |
| 内存共享 + prev.write && !curr.write | - | RAW | 需同步 |
| 内存共享 + !prev.write && curr.write | - | WAR | 需同步 |

**地址重叠公式**：`prev_addr < curr_end && curr_addr < prev_end`

### 5.3 同步类型决策

| 条件 | 同步类型 | IR 生成 |
|-----|---------|---------|
| 同一流水线（如 V→V）| PipeBarrier_<pipeline> | `Call tl.ascend_auto_barrier` |
| 不同流水线（如 MTE2→M）| EventPair_<src>_<dst> | SetFlag + WaitFlag |
| is_sliced=true | PipeBarrier_ALL（强制）| 全管道屏障 |
| IfThenElse前后 | PipeBarrier_ALL（保守）| 全管道屏障 |

**EventPair 命名规则**：从 GetEventMapping 查询，如 `PIPE_MTE2_PIPE_V` → `MTE2_V`

### 5.4 同步优化算法

```
输入：requirements = [(sync_type, buffer_name)]
输出：final_syncs = 需实际插入的同步列表

OptimizeSyncRequirements:
  1. all_syncs = unique(requirements.sync_type)
  2. for sync_type in all_syncs:
       buffer_graph = GetBufferSyncGraph(buffer_name)
       extended_graph = buffer_graph + other_syncs
       extended_graph.ComputeTransitiveClosure()
       if sync_type is EventPair_X_Y:
         if !extended_graph.HasPath(X, Y):
           final_syncs.append(sync_type)
  3. return final_syncs
```

**优化示例**：
```
已插入：EventPair_M_V → graph={M→V}
新需求：EventPair_V_MTE3, EventPair_M_MTE3
扩展图：{M→V, V→MTE3} → 传递闭包后 {M→V, V→MTE3, M→MTE3}
判断：HasPath(M, MTE3)=true → EventPair_M_MTE3 被优化掉
```

### 5.5 同步 IR 映射

最终实现的IR插入效果：

| sync_type | IR 语句 |
|----------|---------|
| PipeBarrier_ALL | `Evaluate(Call tl.ascend_auto_barrier, ["PIPE_ALL"])` |
| PipeBarrier_X | `Evaluate(Call tl.ascend_auto_barrier, ["PIPE_X"])` |
| EventPair_X_Y | `Evaluate(Call tl.ascend_auto_set_flag, ["X_Y", id])` + `Evaluate(Call tl.ascend_auto_wait_flag, ["X_Y", id])` |

**EventPair 生成**：提取 event_type → 分配 event_id（模8）→ SetFlag + WaitFlag

---

## 6. IR 变换示例

### 6.1 典型模式

| 场景 | 输入 IR 模式 | 输出 IR 模式 | 同步类型 |
|-----|------------|------------|---------|
| RAW 跨流水线 | `[PIPE_MTE2 write buf] → [PIPE_M read buf]` | `[write] → [EventPair_MTE2_M] → [read]` | EventPair |
| WAW 同流水线 | `[PIPE_V write buf] → [PIPE_V write buf]` | `[write] → [PipeBarrier_V] → [write]` | PipeBarrier |
| 地址重叠 | `[PIPE_MTE2 write buf_A] → [PIPE_V read buf_B]`（地址重叠）| `[write] → [EventPair_MTE2_V] → [read]` | EventPair |
| 同步优化 | `[EventPair_M_V] + [EventPair_V_MTE3] + 需求 M→MTE3` | `[EventPair_M_V] + [EventPair_V_MTE3]`（M→MTE3优化掉）| 传递满足 |

### 6.2 具体示例

**RAW 跨流水线（MTE2→M）**：
```
输入：
  copy_gm_to_l1(A, a_l1);  // PIPE_MTE2 写 a_l1
  mma(c_l0c, a_l1, b_l1);  // PIPE_M 读 a_l1

输出：
  copy_gm_to_l1(A, a_l1);
  [EventPair_MTE2_M]
  mma(c_l0c, a_l1, b_l1);
```

**WAW 同流水线（PIPE_V）**：
```
输入：
  ascend_add(A, data1, tmp);  // PIPE_V 写 A
  ascend_mul(A, data2, tmp);  // PIPE_V 写 A

输出：
  ascend_add(A, data1, tmp);
  [PipeBarrier_V]
  ascend_mul(A, data2, tmp);
```

**地址重叠（MTE2→V）**：
```
输入：
  copy_gm_to_ub(A_ub, data);  // PIPE_MTE2 写 A_ub
  ascend_add(B_ub, src, tmp); // PIPE_V 读 B_ub（与A_ub地址重叠）

输出：
  copy_gm_to_ub(A_ub, data);
  [EventPair_MTE2_V]
  ascend_add(B_ub, src, tmp);
```

---

## 7. 边界处理清单

| 条件 | 处理方式 |
|-----|---------|
| platform="A5" + PIPE_V | 跳过 PipeBarrier_V |
| Event 数量 > 8 | ID 模8循环，可能冲突 |
| resource_scope | 进入时清空访问历史 |
| IfThenElse | 前后插入 PipeBarrier_ALL |
| is_sliced=true | 强制 PipeBarrier_ALL |
| 嵌套循环 | depth 记录深度，先内后外重建 |
| LetStmt + 切片 | 前插入 PipeBarrier_ALL |

---

## 8. API 与方法索引

### 8.1 按功能分类

| 分类 | 方法 | 功能 |
|-----|-----|-----|
| **入口** | Substitute | 主入口，执行同步插入 |
| **预处理** | PreprocessUnrollForLoops | 预处理，展开循环 |
| **重建** | MergeAndRebuildForLoops | 循环重建阶段 |
| **分析** | AnalyzeStmtAccesses | 分析 EvaluateNode 的缓冲区访问 |
| | AnalyzeExprAccesses | 分析 PrimExpr 的缓冲区访问 |
| | FindRelatedBuffers | 查找地址重叠的缓冲区 |
| **检测** | HasDataDependency | 判断两 BufferAccess 是否有依赖 |
| | GetRequiredSyncType | 根据流水线确定同步类型 |
| **优化** | OptimizeSyncRequirements | 图可达性优化同步需求（核心）|
| **生成** | InsertSynchronization | 生成同步 IR 节点 |
| | CreatePipeBarrier | 创建 PipeBarrier 语句 |
| | CreateSetFlag/CreateWaitFlag | 创建 EventPair 语句 |
| **状态** | UpdateSyncStatesAfterSync | 同步插入后更新 sync_graph |
| | UpdateLatestAccessHistory | 更新访问历史记录 |
| **辅助** | GetPhysicalAddress | 获取缓冲区物理地址 |
| | GetBufferSize | 获取缓冲区大小 |
| | AllocateEventId | 分配事件 ID（模 8 循环）|

### 8.2 辅助类

| 类 | 功能 |
|---|-----|
| ForLoopUnroller | 循环展开为两次迭代 |
| LoopRebuilder | 合并同步并重建循环 |
| StmtFlattener | 展平嵌套 SeqStmt |
| ExprAccessAnalyzer | 提取表达式中的缓冲区访问 |

---

## 9. 类与方法详解

### 9.1 AscendSyncInsert 类

**继承**：`arith::IRMutatorWithAnalyzer`

**核心方法**：

| 方法 | 签名 | 功能 |
|-----|-----|-----|
| Substitute | `static PrimFunc(PrimFunc f, string config, PassContext ctx, Target target, string platform)` | 主入口 |
| PreprocessUnrollForLoops | `pair<Stmt, vector<LoopInfo>>(Stmt)` | 预处理展开 |
| MergeAndRebuildForLoops | `Stmt(Stmt, vector<LoopInfo>)` | 循环重建 |

**关键 VisitStmt_ 重写**：

| 方法 | 功能 |
|-----|-----|
| VisitStmt_(EvaluateNode) | 分析→检测→插入同步 |
| VisitStmt_(AttrStmtNode) | resource_scope 清空历史 |
| VisitStmt_(IfThenElseNode) | 分支前后全同步 |
| VisitStmt_(LetStmtNode) | 切片访问处理 |
| VisitStmt_(SeqStmtNode) | 递归处理序列 |

### 9.2 LoopRebuilder 类

**核心方法**：

| 方法 | 功能 |
|-----|-----|
| MergeIterations | 分离 iter1/iter2 |
| MergeStatementSequences | 合并同步，去重 |
| IsSyncStatement | 判断同步语句 |
| IsSameSyncOperation | 判断同步等价 |

---

## 10. 验证方案

### 10.1 测试场景分类表

| 类别 | 场景 | 流水线组合 | 依赖类型 | 预期同步 |
|-----|-----|-----------|---------|---------|
| 跨流水线 | RAW | MTE2→M, M→V | 读后写 | EventPair |
| 跨流水线 | WAR | V→MTE2, V→MTE3 | 写后读 | EventPair |
| 同流水线 | WAW | V→V, M→M | 写后写 | PipeBarrier |
| 地址重叠 | 重叠检测 | MTE2→V（不同buffer重叠地址）| - | EventPair |

### 10.2 测试用例设计

#### 用例1: RAW 跨流水线（MTE2→M）

**测试意图**：验证跨流水线 RAW 依赖正确插入 EventPair。

**输入 IR**：
```
copy_gm_to_l1(A, a_l1);  // PIPE_MTE2 写 a_l1
mma(c_l0c, a_l1, b_l1);  // PIPE_M 读 a_l1
```

**预期输出 IR**：
```
copy_gm_to_l1(A, a_l1);
[EventPair_MTE2_M]
mma(c_l0c, a_l1, b_l1);
```

**验证点**：
- 同步类型正确（跨流水线用 EventPair）
- 同步位置在写后读之间

---

#### 用例2: WAW 同流水线（V→V）

**测试意图**：验证同流水线 WAW 依赖正确插入 PipeBarrier。

**输入 IR**：
```
ascend_add(A, data1, tmp);  // PIPE_V 写 A
ascend_mul(A, data2, tmp);  // PIPE_V 写 A
```

**预期输出 IR**：
```
ascend_add(A, data1, tmp);
[PipeBarrier_V]
ascend_mul(A, data2, tmp);
```

**验证点**：
- 同步类型正确（同流水线用 PipeBarrier）
- 同步位置在两个写操作之间

---

#### 用例3: WAR 跨流水线（V→MTE3）

**测试意图**：验证跨流水线 WAR 依赖正确插入 EventPair。

**输入 IR**：
```
ascend_add(A_ub, src, tmp);  // PIPE_V 写 A_ub
copy_ub_to_gm(A_ub, Output); // PIPE_MTE3 读 A_ub
```

**预期输出 IR**：
```
ascend_add(A_ub, src, tmp);
[EventPair_V_MTE3]
copy_ub_to_gm(A_ub, Output);
```

**验证点**：
- 同步类型正确（跨流水线用 EventPair）
- 同步方向正确（V→MTE3）

---

#### 用例4: 地址重叠（不同buffer，物理地址重叠）

**测试意图**：验证地址重叠检测生效，跨流水线同步正确插入。

**输入 IR**：
```
copy_gm_to_ub(A_ub, data);    // PIPE_MTE2 写 A_ub，地址 [0, 1024]
ascend_add(B_ub, src, tmp);   // PIPE_V 读 B_ub，地址 [512, 1536]（重叠）
```

**预期输出 IR**：
```
copy_gm_to_ub(A_ub, data);
[EventPair_MTE2_V]
ascend_add(B_ub, src, tmp);
```

**验证点**：
- 地址重叠检测生效（[0,1024] 与 [512,1536] 重叠）
- 跨流水线同步正确插入

### 10.3 验证命令

**IR dump 验证（查看同步插入后的 IR）**
```python
import tilelang
import tilelang.language as T

tilelang.cache.clear_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}

@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def test_sync(M=256, N=256):
    @T.prim_func
    def main(A: T.Tensor((M, N), 'float'), B: T.Tensor((M, N), 'float')):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            ub1 = T.alloc_ub((M, N), 'float')
            ub2 = T.alloc_ub((M, N), 'float')
            T.copy(A, ub1)
            T.tile.add(ub2, ub1, ub1)
            T.copy(ub2, B)
    return main

func = test_sync()
print(func.ir_module['main'])
```

---

## 附录 A：循环展开原理

### 为什么需要展开？

| 问题 | 说明 |
|-----|-----|
| 无法区分迭代内/间依赖 | 直接处理循环难以区分 |
| 同步位置不确定 | 循环内→过度同步，循环外→同步不足 |

### 三阶段策略

```
1. 展开为两次迭代 → 用 iteration_start/end 标记
2. 对展开后的 IR 插入同步
3. 合并 iter1/iter2 的同步 → 重建循环
```

**合并策略**：以 iter1 结构为基准，对应位置同步取并集，末尾同步保留。

---

## 附录 B：依赖环境

| 依赖文件 | 用途 |
|---------|-----|
| `../op/ascend.h` | 同步原语定义 |
| `./common/operation_config.h` | 操作配置 |
| TVM TIR 库 | IR 基础设施 |

**配置选项**：
- Pass 配置：`tl.ascend_auto_sync`（Bool）
- 函数属性：`enable_auto_sync`

**注册**：`TVM_REGISTER_GLOBAL("tl.transform.AscendSyncInsert")`