# TileLang-Ascend 测试覆盖矩阵（强制契约）

**定位**：本文件把 SKILL.md 中散落的"各类型测试场景"收敛成一张**带 ID、带适用谓词、带最小数量**的强制清单。
它是 `tilelang-op-test-design` 生成用例时的**覆盖契约**，也是覆盖自检 checker（`scripts/coverage_check.py`）判定 PASS/MISS 的唯一依据。

**核心原则**：
1. 场景"该不该有"是**可判定的**——由每个维度的「适用谓词」决定，不再靠 agent 主观。
2. 场景"有没有"是**可机器校验的**——每条用例必须打覆盖标签（`tags=`），checker 反查命中集合。
3. 合理缺失必须**显式豁免**——通过 `COVERAGE_NA` 声明维度 ID + 理由，区别于"静默漏掉"。

---

## 一、覆盖维度总表

每个维度有：**ID**、**所属层**、**含义**、**最小数量**、**适用谓词**（何时必须覆盖）。
checker 对每个适用维度判定：`PASS`（实际数量 ≥ 最小数量）/ `MISS`（不足且未豁免）/ `N/A`（带理由豁免）。

| 维度 ID | 层 | 含义 | 最小数量 | 适用谓词 |
|---|---|---|---|---|
| `D-DTYPE-<dt>` | L0/L1 | 每个 proto 支持的 dtype 各至少一条（`<dt>` ∈ {fp16,fp32,bf16,int8,…}） | 每 dtype ≥1 | proto.yaml 声明的每个输入 dtype |
| `D-SHAPE-ALIGNED` | L0/L1 | block 整除的规则 shape | ≥1 | 总是 |
| `D-SHAPE-TAIL-1` | L1 | 尾块余数=1（最易暴露边界处理 bug） | ≥1 | 有 tiling/分块的算子（纯 Cube / 混合 / 多步 Vector） |
| `D-SHAPE-TAIL-MID` | L1 | 尾块余数=block 中间值 | ≥1 | 同上 |
| `D-SHAPE-PRIME` | L1 | 全质数 / 完全非对齐 shape | ≥1 | 同上 |
| `D-SHAPE-RANK-<r>` | L1 | 不同维数（2D/3D/4D/5D…）各覆盖 | 按支持范围每 rank ≥1 | 支持多维（rank 范围 >1）的算子 |
| `D-SHAPE-EDGE` | L1 | 退化 shape：1×N / N×1 / 单元素 | ≥1 | 总是 |
| `D-VALRANGE-S` | L1 | 对称小值域（如 [-1,1]） | ≥1 | 浮点算子 |
| `D-VALRANGE-M` | L1 | 对称中值域（如 [-10,10]） | ≥1 | 浮点算子 |
| `D-VALRANGE-L` | L1 | 对称大值域（接近 dtype 上限） | ≥1 | 浮点算子 |
| `D-VALRANGE-ASYM` | L1 | 非对称值域（如 [-5,10]） | ≥1 | 浮点算子 |
| `D-PARAM-<name>` | L1 | 关键属性的非默认取值（dim/axis/eps/scale/shift/base…） | 每关键 attr ≥1 | 含 attr 的算子（每个影响计算路径的 attr 一个 ID） |
| `D-SPECIAL-INF` | Boundary | 含 ±inf 输入 | ≥1 | 浮点算子 |
| `D-SPECIAL-NAN` | Boundary | 含 nan 输入 | ≥1 | 浮点算子 |
| `D-SPECIAL-ZERO` | Boundary | 全零 / ±0 输入 | ≥1 | 总是 |
| `D-SPECIAL-DBOUND` | Boundary | dtype 边界值（fp16≈±65504 / fp32≈±88 exp 边界等） | ≥1 | 浮点算子 |
| `D-EXC-DTYPE` | L2 | 不支持的 dtype（应被拒绝或报错） | ≥1 | 总是 |
| `D-EXC-SHAPE` | L2 | 非法 shape（维度不匹配 / 约束违反） | ≥1 | 总是 |

> **GEMM 类补充**：纯 Cube / 含 matmul 的算子，`D-SHAPE-*` 必须在 **M、N、K 三个轴**上分别出现非对齐（不能只非对齐 M）。checker 对 GEMM 类要求 `D-SHAPE-TAIL-*` / `D-SHAPE-PRIME` 至少各覆盖一次非对齐 K 轴的用例。

---

## 二、适用谓词的判定（按算子类别裁剪）

维度是否"适用"由算子类别（见 `operator-category.md`）决定，避免对不相关场景硬凑：

| 算子类别 | 强制维度 | 豁免维度（默认 N/A，可不写理由） |
|---|---|---|
| **Activation（纯 Vector + Single）** | D-DTYPE-*, D-SHAPE-ALIGNED/EDGE, D-VALRANGE-*, D-SPECIAL-* , D-EXC-* | D-SHAPE-TAIL-*/PRIME（逐元素无 tiling 边界，可选）、D-PARAM-*（无 attr 时） |
| **Reduction（纯 Vector + Single）** | 上 + D-PARAM-dim（归约轴）, D-SHAPE-TAIL-*/PRIME | — |
| **Softmax / Normalization（纯 Vector + Multi）** | D-DTYPE-*, D-SHAPE-ALIGNED/TAIL-*/PRIME/EDGE, D-VALRANGE-*, D-PARAM-（dim/eps）, D-SPECIAL-*, D-EXC-* | — |
| **GEMM（纯 Cube）** | D-DTYPE-*, D-SHAPE-ALIGNED/TAIL-*/PRIME（含 K 轴非对齐）, D-SHAPE-RANK（batched 时）, D-EXC-* | D-SPECIAL-INF/NAN（矩阵乘溢出语义弱，可 N/A）、D-VALRANGE-L（按需） |
| **Fusion（混合 CV）** | 全部适用维度（最严格） | 仅在算子语义明确不支持时豁免 |
| **量化 / 整数输出算子** | D-DTYPE-*, D-SHAPE-*, D-EXC-* | D-SPECIAL-INF/NAN（整数无浮点特殊值，N/A）、D-VALRANGE-* 视输入 dtype |
| **纯整数算子（gcd / unique 整数路径）** | D-DTYPE-*, D-SHAPE-*, D-SPECIAL-ZERO, D-EXC-* | D-SPECIAL-INF/NAN/DBOUND（N/A）、D-VALRANGE-*（N/A） |

判定流程：
```
1. 用 operator-category.md 判出综合类别（纯Cube/Vector/混合 × Single/Multi/Fusion × GEMM/Softmax/…）
2. 查上表得到「强制维度集」
3. 结合 proto.yaml 的 dtype / attr / shape 支持范围实例化每个 D-* 的最小数量
4. 不在强制集、且类别表标了可豁免的维度 → 默认 N/A
```

> **proto.yaml 来源**：由 Stage 1（analyst / `tilelang-op-design`）从 DESIGN.md **§9.3 精度表**（dtype 全集）+ §4/§1（attr/shape）派生并写入 `examples/{op}/proto.yaml`——**每个算子都产出**，故覆盖门禁始终有权威的 dtype/attr 来源（`D-DTYPE-*` / `D-PARAM-*` 恒可强制）。缺 proto 时 checker 优雅降级（仅按文件内 tags 校验，可能漏检缺失 dtype）。

---

## 三、确定性 Shape 生成规则（保证非对齐必出）

把"自然包含不规则 shape"改成**由 block 反推的强制公式**，从结构上杜绝"漏非对齐"。
给定主分块 `block=(bM, bN[, bK])`，L1 的 shape 集合**至少**包含：

```python
def gen_l1_shapes(bM, bN):
    k = 4  # 倍数，按规模需要调整
    return {
        "D-SHAPE-ALIGNED":  (bM * k,            bN * k),            # block 整除
        "D-SHAPE-TAIL-1":   (bM * k + 1,        bN * k),            # 余数 1
        "D-SHAPE-TAIL-MID": (bM * k + bM // 2,  bN * k + bN // 2),  # 中间余数
        "D-SHAPE-PRIME":    (nearest_prime(bM*k), nearest_prime(bN*k)),  # 完全非对齐
        "D-SHAPE-EDGE":     (1, bN * k),                            # 退化（另配 (bM*k, 1) / 单元素）
    }
```

规则：
- 这些 shape **默认必出**，不再用"是否需要不规则 shape"软问法；用户只能"加量"，不能减到 0。
- GEMM 类把上式扩到 K 轴：至少一条 `K = bK*k + 1` 或质数 K。
- 多维算子（D-SHAPE-RANK）：对支持的每个 rank 重复 ALIGNED + 一条非对齐。
- `nearest_prime` 取 ≤ 目标值的最近质数（避免超出支持范围）。

---

## 四、用例打标约定（机器可校验的关键）

每条用例在生成时**必须**带 `tags`（命中的维度 ID 列表）。两种等价落地形式，二选一：

**形式 A：用例元组带 tags（推荐，贴近 §9 模板）**
```python
L1_CASES = [
    # (shape,            dtype,      block,    tags)
    ((512, 512),        "float16",  (128,128), ["D-DTYPE-fp16","D-SHAPE-ALIGNED","D-VALRANGE-S"]),
    ((513, 512),        "float16",  (128,128), ["D-SHAPE-TAIL-1"]),
    ((509, 503),        "bfloat16", (128,128), ["D-DTYPE-bf16","D-SHAPE-PRIME"]),
    # ...
]
```

**形式 B：文件末尾汇总 manifest**
```python
COVERAGE_MANIFEST = {
    "D-DTYPE-fp16": 6, "D-DTYPE-fp32": 4, "D-DTYPE-bf16": 4,
    "D-SHAPE-ALIGNED": 3, "D-SHAPE-TAIL-1": 1, "D-SHAPE-TAIL-MID": 1,
    "D-SHAPE-PRIME": 2, "D-SHAPE-EDGE": 1,
    "D-VALRANGE-S": 2, "D-VALRANGE-M": 1, "D-VALRANGE-L": 1, "D-VALRANGE-ASYM": 1,
    "D-PARAM-dim": 2,
    "D-SPECIAL-INF": 1, "D-SPECIAL-NAN": 1, "D-SPECIAL-ZERO": 1, "D-SPECIAL-DBOUND": 1,
    "D-EXC-DTYPE": 1, "D-EXC-SHAPE": 1,
}
```

无标注 → checker 无法反查 → 直接判 **MISS**（不允许靠正则猜测）。

---

## 五、豁免机制（区分"不适用"与"漏掉"）

合理缺失必须显式声明，带理由：

```python
COVERAGE_NA = {
    "D-SPECIAL-INF": "纯整数算子（gcd），无浮点特殊值",
    "D-SPECIAL-NAN": "纯整数算子（gcd），无浮点特殊值",
    "D-SHAPE-TAIL-1": "逐元素激活算子无 tiling 边界，尾块与对齐路径等价",
}
```

checker 接受的状态只有三种：**命中**（manifest 有且达量）/ **带理由 N/A**（在 COVERAGE_NA 且类别表允许豁免）/ **MISS**（其余）。
类别表中标记为「强制」的维度即使写进 COVERAGE_NA 也判 MISS（防止用豁免绕过强制项）。

---

## 六、checker 判定与退出码

`scripts/coverage_check.py test_{op}.py [--proto proto.yaml]`：

```
1. 解析算子类别 + proto 支持的 dtype/attr → 计算「应覆盖维度集」与各自最小数量
2. 解析 test_{op}.py 的 tags / COVERAGE_MANIFEST → 计算「实际覆盖集」
3. 逐维度判定 PASS / N/A / MISS，打印覆盖矩阵
4. 任一强制维度 MISS → 退出码 1（视为自检失败，触发补齐）；全 PASS/N/A → 退出码 0
```

输出示例：
```
== Coverage Matrix: softmax ==
[PASS] D-DTYPE-fp16        need>=1 got 6
[PASS] D-SHAPE-PRIME       need>=1 got 2
[MISS] D-VALRANGE-L        need>=1 got 0      <-- 缺大值域
[N/A ] D-SPECIAL-INF       reason: 见 COVERAGE_NA
COVERAGE: 17 PASS / 1 MISS / 1 N/A  -> FAIL (exit 1)
```

---

## 七、与 cann-bench 的对应（为何这些维度）

本矩阵的维度直接对标 `cann-bench/tasks` 用例的设计轴，确保 orchestrator 产出的算子在覆盖面上不弱于基准：

| 本矩阵维度 | cann-bench cases 对应轴 |
|---|---|
| D-DTYPE-* | dtype 轮换（float16/float32/bfloat16） |
| D-SHAPE-ALIGNED / TAIL / PRIME | 对齐 vs 非对齐 vs 质数非对齐（`[1009,1021]`、`[363,367,373]`） |
| D-SHAPE-RANK | 2D~5D 维数覆盖 |
| D-VALRANGE-* | 对称小/中/大值域、非对称值域 |
| D-SPECIAL-INF/NAN/ZERO/DBOUND | inf / nan / 零值 / dtype 边界（`±65504`、`±88`） |
| D-PARAM-* | attr 组合（base/scale/shift、dim、eps…） |

> 注：cann-bench 的「S/M/L 大规模（最大 268M 元素）」与「baseline_kernels 性能对照」属于**规模/性能轴**，不在功能覆盖矩阵内，由 perf-tuner 阶段单独处理。
