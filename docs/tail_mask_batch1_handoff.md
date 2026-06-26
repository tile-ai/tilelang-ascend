# Vector 尾块二阶段方案 — Batch 1 实现交接文档

> 给测试环境 agent 的上下文交接。分支：`claude/brave-stonebraker-d3d0b2`
> 关键 commit：`18194b2`(batch1) → `cdc4def`(CallNode 修复) → `f0d8a48`(reduce 用 Pattern-AR)

---

## 1. 目标

把 AscendC 后端的尾块处理从「**前端显式 `pad_value`**」升级为「**前端无感知、后端自动传播 valid shape、V 核按有效区域计算**」。
范围：仅 AscendC 后端,`GM ↔ UB ↔ Vector Core`,不涉及 PTO。

动机:主仓 `T.copy` 不带 `pad_value` 语义,为合并回主仓需要去掉它;同时尾块正确性从「补 pad 值」改为「真实有效区域计算」,且**用户想让 vector 算子按尾块实际大小用 mask/stride 计算以减少计算量**(性能诉求,不只是正确性)。

---

## 2. 整体架构

```
T.copy / T.tile.* (前端,无 pad_value)
        │  LowerTileOp
        ▼
普通 tl.ascend_* + call_extern(copy_gm_to_ub<...>, ..., validRow, validCol, physRow, physCol)
        │  ★ AscendTailMaskPropagation (本方案新增, 插在 LowerTileOp 之后)
        ▼
尾块算子改写成内部 tl.ascend_tail_{unary,binary,scalar,reduce}
        │  CodeGenTileLangAscend
        ▼
tl::ascend::tail_* helper (common.h)
        │
        ▼
AscendC API (按 valid 区域计算)
```

**核心机制**:`copy_gm_to_ub` 在 lowering 时已算出 `validRow/validCol`(逻辑有效区)和 `physRow/physCol`(物理 tile 尺寸)。新 pass 把这个有效矩形按 UB 数据流传播,触达尾块 buffer 的 V 核算子改写成 `tail_*`,helper 在运行时只算有效区。

---

## 3. 关键设计决策(对话中确认过的)

1. **去 pad 与 tail-aware reduce 必须一起上**:去掉 pad 后 gap 是垃圾,`reduce` 必须按 valid 区算才正确(softmax 的 `reduce_max` 原来靠 `pad=-inf`)。
2. **per-lane 算子 gap 天然无害**:`copy_ub_to_gm` 只回写 `validRow×validCol`,gap 不回写;`add/exp/select` 等 per-lane 算子 gap 只停在 gap lane,不跨 lane。所以**正确性上**只有跨 lane 的 reduce 必须 tail-aware。但用户要**性能**(少算 gap),所以 elementwise 也做 tail-aware。
3. **省的是 repeat 次数,不是 mask 本身**:带 mask 的 repeat 与满 mask 的 repeat cycle 一样;省 cycle 靠发更少的 256B repeat(跳 gap 行 / 跳整列块)。
4. **mask+repeat 有 vl 上限**:AscendC normal-mode `mask ≤ vl`(fp32 64 / fp16 128)。`validCol ≤ vl` 才能「一行=一次 repeat、mask=validCol」;`validCol > vl` 退回逐行循环。helper 阶梯:满块→连续 `count`→窄列 `mask+repeat`→逐行兜底。
5. **分两批**:Batch 1 = unary/binary/scalar/reduce 改写 + broadcast/cast 仅传播。**Batch 2(未做)= compare/select 的 packed mask + broadcast 改写**。
6. **用 `IRMutatorWithAnalyzer` 绑定循环界**:让**可证明满块**的 copy 标 `kFull` → 完全不改写,使非尾块内核保持原路径(降低回归面)。是否真能证出,取决于 analyzer——见 §7 待验证。
7. **reduce 用 `Pattern::Reduce::AR`**(commit `f0d8a48`):最初用 `WholeReduce` + 逐行 `ReduceSum(dst,src,work,count)`,后者对 `validCol>vl` 算错。改成与现有 fp32 reduce 相同的 `Pattern-AR` 原语(已验证可靠)。

---

## 4. 文件改动地图

### 前端 / 去 pad
- `tilelang/language/copy.py` — `npu_copy_v2` 删 `pad_value` 参数。
- `src/op/ascend.h` / `src/op/ascend.cc` — `AscendCopy` 删 `padValue` 成员/解析;gm2ub lowering 仍 push `physRow/physCol`(供 pass 用),不再 push pad。
- `src/tl_templates/ascend/common.h` — `copy_gm_to_ub` 删 pad 参数与 gap 填充 `Duplicate`。
- `src/target/codegen_ascend.cc` — `CopyCodegen` 的 `copy_gm_to_ub` extra_args `4→3`。
- `src/transform/ascend_workspace_reduction.cc` — workspace 复用的 copy 发射不再 push pad。
- `examples/softmax/example_online_softmax.py` — 去掉 `pad_value=-inf`。

### 新 pass + 数据模型
- `src/transform/common/ascend_tail_mask.h`(新)— `TailMaskKind/TailMaskInfo` + `MakeCopyMask/IntersectMasks/MakeFullMask`。
- `src/transform/ascend_tail_mask_propagation.cc`(新)— pass 本体。
  - 从 `copy_gm_to_ub` 取 `validRow=args[4], validCol=args[5], physRow/physCol=args[6/7]`。
  - 传播:gm2ub(seed)、ub_to_ub(继承)、unary/binary/scalar(改写)、reduce(改写)、cast(仅传播)、broadcast(仅传播)。
  - 改写为 `tl.ascend_tail_*`,带 runtime `valid_row/valid_col/physical_col`。
  - 用 `arith::IRMutatorWithAnalyzer`。
- `tilelang/transform/__init__.py` — `AscendTailMaskPropagation()` wrapper。
- `tilelang/engine/phase.py` — 在 `LowerTileOp()` 之后、`AscendWorkspaceReduction()` 之前插入。

### 内部 op + codegen + helper
- `src/op/ascend.h` / `ascend.cc` — 注册 `ascend_tail_{unary,binary,scalar,reduce}`(变参)。
- `src/transform/common/operation_config.h` — 4 个新 op 的读写位 + `PIPE_V`(arg0 是 op-tag 字符串,buffer 下标从 1 开始)。
- `src/target/codegen_ascend.cc` / `.h` — `Tail{Unary,Binary,Scalar,Reduce}OpCodegen` + dispatch。
- `src/tl_templates/ascend/common.h` — `tail_{unary,binary,scalar,reduce_sum,reduce_max,reduce_min}` helper(含 `TailVec{Un,Bin,Scalar}Op` enum、`TailApply*` dispatch)。

### 测试 / 样例
- `testing/python/language/test_tilelang_ascend_language_tail_mask_codegen.py`(新)— codegen 断言(host-side,不跑核)。
- `examples/tail_mask/example_tail_add.py`、`example_tail_reduce.py`(新)。

### 内部 op 的 arg 布局(pass↔codegen↔operation_config 三方一致)
```
tail_unary : [tag, dst, src,      validRow, validCol, physCol]
tail_binary: [tag, dst, src0,src1,validRow, validCol, physCol]
tail_scalar: [tag, dst, src, scalar, validRow, validCol, physCol]
tail_reduce: [kind, out, src, tmp, dim, validRow, validCol, physCol, clear]
```
`tag` 短名("Add"/"Exp"/"Adds"),codegen 前缀成 `tl::ascend::TailVec*Op::<tag>`。`kind`="reduce_sum"/"reduce_max"/"reduce_min" → helper `tl::ascend::tail_<kind>`。

---

## 5. 当前状态(截至本文档)

| 项 | 状态 |
|---|---|
| host 编译(`install_ascend.sh`) | ✅ 通过(修了 `const Call*`→`const CallNode*`,commit `cdc4def`) |
| `example_tail_add.py`(float / float16 / 100×200 block64×128) | ✅ **全部通过** |
| `example_tail_reduce.py` | ⚠️ 旧版(count-form)算错;已改 Pattern-AR(`f0d8a48`),**待重验** |
| `example_online_softmax.py`(验收:M=34/N=130 双尾块) | ⏳ **待跑** |
| 满块内核 codegen 回归(应无 `tail_`) | ⏳ **待跑** |

**`example_tail_add` 通过 = 已验证**:前端去 pad、copy lowering、pass 传播、`tail_binary` 的三条路径(满块 `count` / 窄列 `mask+repeat`(BinaryRepeatParams)/ 宽列逐行)、整条编译链路。这一项把最大的不确定性消掉了。

---

## 6. 验证命令(测试机)

```bash
git pull
# common.h 是 device 模板,改它无需重编 host 扩展;改 .cc/.h 才需要:
bash install_ascend.sh --enable-incremental   # 仅当 pull 到 .cc/.h 变更时

# 数值
python examples/tail_mask/example_tail_add.py
python examples/tail_mask/example_tail_reduce.py
python examples/softmax/example_online_softmax.py

# codegen 断言(需 host build,不需 NPU 跑核)
python -m pytest testing/python/language/test_tilelang_ascend_language_tail_mask_codegen.py -v
#   注意:测试机无 pytest 时用 `python -m pytest`

# 回归:满块内核应无 tail_(确认非尾块路径不变)
python -c "import testing.python.language.test_tilelang_ascend_language_tail_mask_codegen as t; print(t._source(t._tail_add(128,128,64,64,'float')))" | grep -c tail_
#   期望 0;若非 0 = analyzer 没证出满块(结果仍对,只是多一层运行时判断)

# 现有算子回归
python -m pytest testing/python/language/test_tilelang_ascend_language_elementwise.py -v
python -m pytest testing/python/language/test_tilelang_ascend_language_reduce.py -v
```

看生成源码:任意尾块 kernel `func.get_kernel_source()`,grep `tail_` / `Duplicate` / `copy_gm_to_ub`。

---

## 7. 待验证 / 已知风险(给 agent 重点分析的点)

1. **`example_tail_reduce` 重验**:`f0d8a48` 后是否通过。若仍错,重点查 `tail_reduce_sum`(common.h ~L800)的 Pattern-AR 调用——尤其 **runtime `shape[]={validRow,validCol}` 是否被 AscendC `ReduceSum<T,Pattern::AR>` 正确接受**(现有代码只用编译期 M,N 填 shape,但 shape 是运行时数组指针,理论上 runtime 值可行)。
2. **`WholeReduce` 已从 reduce 路径移除**(改 Pattern-AR),所以 fp32 `WholeReduce.dstRepStride` 语义的旧风险**已不适用**。
3. **softmax(验收)**:会跑 `tail_unary`(exp,UnaryRepeatParams 路径,尚未被 add 验到)、`tail_binary`、`tail_reduce`(max+sum)、`broadcast`(仅传播)。**这是检验 UnaryRepeatParams mask 路径的第一个用例。**
4. **回归 / blast radius**:满块内核是否被误改成 `tail_`(§6 的 grep)。若 analyzer 证不出满块,所有 tiled 内核走 tail helper —— 结果正确但多运行时判断。若要严格保持满块 codegen 不变,需加 divisibility 分析(未做)。
5. **`clear=false` 的 tail reduce**:用 scalar `GetValue/SetValue` 备份+合并(镜像现有 reduce),**栈数组上限 256 行**,且 scalar/vector 同步未加显式 barrier(沿用现有代码做法)。softmax 用 `clear=true`,未被验证。
6. **`dim==0` 的 tail reduce**:用逐行 `Adds/Add/Max/Min` 累积,**未被任何样例验证**。
7. **非建模算子断链**:不在 {unary,binary,scalar,reduce,cast,broadcast,ub_to_ub} 集合内、又跨 lane 的算子(sort/gather/sigmoid/whole_reduce 变体…)读尾块再喂 reduce,会丢 mask → reduce 当成满块 → 错。需要时扩展 pass。
8. **同 pitch 假设**:`tail_binary`/未来 select 的 repeat plan 假设 dst/src0/src1 同 `physical_col`,不一致回退逐行(仅性能影响)。

---

## 8. Batch 2(未实现,按约定推迟)

- **compare / select 的尾块**:compare 输出 packed mask(uint8,1 bit/元素),列尾块带 stride 时 bit 布局 + `storage_col` 是最复杂处。`TailMaskInfo` 里已留 `kPackedCmp`/`storage_col` 占位。
- **broadcast 改写**:目前只传播 mask(`[M,1]->[M,N]` 行尾块带过去,列全有效;`[1,N]->[M,N]` 反之),计算仍走 full-tile(正确但不省算量)。要省 gap 行需新 helper。

---

## 9. 给 agent 的建议工作流

1. 先跑 §6 全部命令,记录 pass/fail + 任何 device 报错(`ERR99999` 等)。
2. `example_tail_reduce` 或 softmax 若数值错:打印 `get_kernel_source()`,定位是 `tail_reduce_*` 还是 `tail_unary/binary`,对照 §4 arg 布局 + common.h helper 检查。
3. 编译错(device JIT,跑核时报)大多在 `common.h` 的 AscendC API 细节(`UnaryRepeatParams` 字段名、mask 重载、`Pattern::AR` runtime shape)。把报错连同所在行贴出。
4. §6 的满块 grep 若非 0,属预期内的"次优但正确",记录即可,不阻塞。
5. 所有 helper 改动只动 `common.h`(device 模板)→ 无需重编 host;pass/codegen/op 改动(`src/**/*.cc/.h`)→ 需 `install_ascend.sh --enable-incremental`。
