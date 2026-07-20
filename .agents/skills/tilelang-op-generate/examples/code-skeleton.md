# 代码骨架模板

develop skill 生成**两个文件**（kernel 与 test 分离，便于阅读与各阶段职责隔离）：

- `examples/{op}/{op}.py` —— **纯 kernel**（`@tilelang.jit` + `pass_configs`），可被 import，**不含 golden、不含测试、不含 `__main__`**
- `examples/{op}/test_{op}.py` —— golden + 精度判定 + 分层测试套件（L0 + L1/L2/Boundary 桩）+ `main`（`--level` 入口）；从 `{op}.py` **import kernel**

运行入口是 test 文件：`python examples/{op}/test_{op}.py --level {l0|all}`。

---

## 一、`{op}.py`（kernel 文件）

```python
"""{算子简述}。"""

import tilelang
from tilelang import DataType, language as T

# ========== 算子实现 ==========
pass_configs = {...}


@tilelang.jit(out_idx=[...], pass_configs=pass_configs)
def {op}(M, N, block_M, block_N, dtype="float"):
    # 分块计算
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)
    VEC_NUM = 2

    @T.prim_func
    def main(Input: T.Tensor((M, N), dtype), Output: T.Tensor((M, N), dtype)):
        with T.Kernel(..., is_npu=True) as (cid, vid):
            # buffer 分配 / 数据搬入 / 计算 / 数据搬出
            pass

    return main
```

> kernel 文件保持纯净：只放 `@tilelang.jit` kernel 与 `pass_configs`。**不放 golden、不放测试、不放 `__main__`**——便于 test 文件 import，也便于 `precision_fix` / perf 调优时聚焦 diff。
> **命名约束**：kernel 函数名必须与文件名一致（都为 `{op}`），这样 `test_{op}.py` 才能 `from {op} import {op}` 正确导入被测 kernel。

---

## 二、`test_{op}.py`（测试文件）

```python
"""{算子} 分层测试：L0/L1/L2/Boundary + main(--level)。"""

import argparse
import os
import sys

import tilelang
import torch

# 从同目录 kernel 文件导入被测 kernel（把自身目录加进 sys.path）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from {op} import {op}   # noqa: E402


# ========== Golden 参考实现 ==========
def golden_{op}(input_data):
    # 根据算子数学公式实现（PyTorch 参考）
    ...


# ========== 精度标准（混合容差，详见 tilelang-op-test-design/references/precision-standard.md）==========
def get_precision(dtype):
    """返回 (atol, rtol, max_abs_error_limit, required_matched_ratio)。
    浮点：混合容差；整型：0 误差精确匹配。"""
    fp_table = {
        "float16":     (2**-14, 2**-9,  1e-1, 0.99),
        "bfloat16":    (2**-10, 2**-6,  1e0,  0.99),
        "float32":     (2**-16, 2**-10, 1e-2, 0.99),
        "hifloat32":   (2**-16, 2**-10, 1e-2, 0.99),
        "float8_e4m3": (2**-4,  2**-2,  1e0,  0.99),
        "float8_e5m2": (2**-3,  2**-1,  1e-1, 0.99),
    }
    if dtype in {"int8", "int16", "int32", "int64", "uint8"}:
        return (0.0, 0.0, 0.0, 1.0)
    return fp_table.get(dtype, (2**-14, 2**-9, 1e-1, 0.99))


def check_precision(actual, golden, dtype):
    """混合容差双门限判定：返回 (passed, matched_ratio, max_abs_error)。"""
    atol, rtol, max_abs_limit, required_ratio = get_precision(dtype)
    a, g = actual.detach().cpu(), golden.detach().cpu()
    if atol == 0.0 and rtol == 0.0:                      # 整型精确匹配
        mism = (a != g).sum().item()
        return mism == 0, 1.0 - mism / max(a.numel(), 1), (0.0 if mism == 0 else float("inf"))
    a, g = a.float(), g.float()
    m = torch.isfinite(g)                                # golden 有限值位置全比：actual 若为 inf/nan 计为不达标
    if m.sum().item() == 0:
        return True, 1.0, 0.0
    abs_err = (a[m] - g[m]).abs()                        # actual 为 inf/nan 处 abs_err=inf/nan → 判 False 且拉高 max_abs
    ratio = (abs_err <= (atol + rtol * g[m].abs())).float().mean().item()
    max_abs = abs_err.max().item()
    return (ratio >= required_ratio and max_abs <= max_abs_limit), ratio, max_abs


# ========== L0 测试：门槛（精度收敛，来自 DESIGN.md §9.2 L0 计划）==========
def test_{op}_l0():
    """L0 门槛测试：规则 shape（block 整除），用于精度收敛。"""
    test_configs = [...]  # (dtype, shape, block) —— 来自 DESIGN.md §9.2
    ok = True
    for dtype, shape, block in test_configs:
        try:
            # 1. 用 {op}(...) 创建 kernel  2. 造输入  3. 执行 → out  4. golden → ref  5. 比对
            passed, ratio, max_abs = check_precision(out, ref, dtype)
            tag = "PASS" if passed else "FAIL"
            print(f"[PRECISION_{tag}] l0 shape={shape} dtype={dtype} matched_ratio={ratio:.4f} max_abs={max_abs:.3e}")
            ok &= passed
        except Exception as e:
            print(f"[PRECISION_FAIL] l0 shape={shape} dtype={dtype}: {e}")
            ok = False
    return ok


# ========== L1/L2/Boundary：留桩，由 tilelang-op-test-design（场景 B）在 L0 通过后填充 ==========
def test_{op}_l1():
    """L1 功能测试（含不规则/尾块 shape）——留桩。"""
    print("[L1] not expanded yet — run tilelang-op-test-design (scenario B)")
    return True


def test_{op}_l2():
    """L2 异常测试（负向：非法输入应被拒绝）——留桩。"""
    print("[L2] not expanded yet — run tilelang-op-test-design (scenario B)")


def test_{op}_boundary():
    """Boundary 边界/特殊值测试（合法极值，按精度标准比对）——留桩。"""
    print("[BOUNDARY] not expanded yet — run tilelang-op-test-design (scenario B)")


# ========== 主函数：--level 分发 + 退出码 ==========
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="l0",
                        choices=["l0", "l1", "l2", "boundary", "all"])
    args = parser.parse_args()

    tilelang.disable_cache()  # 禁用编译缓存，避免旧产物干扰
    torch.manual_seed(0)

    blocking_ok = True  # 仅 L0/L1 计入阻塞判定
    if args.level in ("l0", "all"):
        blocking_ok &= test_{op}_l0()
    if args.level in ("l1", "all"):
        blocking_ok &= test_{op}_l1()
    if args.level in ("l2", "all"):
        test_{op}_l2()        # L2 负向：非法输入正确拒绝=PASS，静默接受=WARN，非阻塞
    if args.level in ("boundary", "all"):
        test_{op}_boundary()  # Boundary 比精度：精度不过=WARN，非阻塞

    if blocking_ok:
        print("Test Passed!")  # L0/L1 全过；据此（退出码 + 该行）判定
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
```

---

**分层运行约定**（与 tilelang-op-test-design 一致）：

| 层级 | 通过标记 | 失败标记 | 是否计入退出码 |
|------|---------|---------|--------------|
| L0 / L1 | `[PRECISION_PASS]` | `[PRECISION_FAIL]` | 是（任一失败 → exit 1） |
| L2 / Boundary | `[BOUNDARY_PASS]` | `[BOUNDARY_WARN]` | 否（仅记录，不影响退出码） |

- 精度收敛阶段跑 `python examples/{op}/test_{op}.py --level l0`；扩展后跑 `--level all`。
- L2（负向）：非法输入**正确抛异常 = `[BOUNDARY_PASS]`**，静默接受 = `[BOUNDARY_WARN]`；不比精度。
- Boundary（合法特殊值）：跑 kernel+golden **按精度标准比对**，精度不过 = `[BOUNDARY_WARN]`。
- L2/Boundary 用例都 `try/except` 包裹、非阻塞，失败后**继续**，不得中断、不得改退出码。

**融合算子注意事项**（写在 `{op}.py` kernel 内）：

**Developer 模式（推荐，默认消除 workspace/vid）**：
- 装饰器无 `workspace_idx`，函数签名无 workspace 参数
- `T.Kernel(block_num, threads=2, is_npu=True) as (cid)`（单轴 + `threads=2`）
- Cube↔Vector 用 `alloc_shared/alloc_fragment` 片上 `T.copy` 直连，无 GM 往返、无 `vid` 偏移
- 完整骨架/映射表见 [tilelang-expert-to-developer mode-examples.md §6](../../tilelang-custom-skill/tilelang-expert-to-developer/references/mode-examples.md#6-cv-融合--推荐写法消除-workspace--vidthreads2)

**回退（Expert/混合或复杂同步场景）**：
- 函数签名包含 workspace 参数，`workspace_idx` 指定索引位置
- Cube 核输出通过 `T.copy` 写入 workspace，Vector 核从 workspace 读取（见 mode-examples.md §7）
