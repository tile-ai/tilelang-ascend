# 代码骨架模板

生成 `example_{op}.py` 时的文件结构：

```python
import tilelang
from tilelang import DataType, language as T
import torch

# ========== 算子实现 ==========
@tilelang.jit(out_idx=[...], pass_configs={...})
def op_name(M, N, block_M, block_N, dtype="float"):
    # 分块计算
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)
    VEC_NUM = 2

    @T.prim_func
    def main(Input: T.Tensor((M, N), dtype), Output: T.Tensor((M, N), dtype)):
        with T.Kernel(..., is_npu=True) as (cid, vid):
            # buffer 分配
            # 数据搬入
            # 计算
            # 数据搬出
            pass

    return main

# ========== 测试（分层）==========
# 本 skill 只落地 L0（精度收敛用）。L1/L2/Boundary 先留桩，
# 由 tilelang-op-test-design（场景 B）在 L0 通过后填充桩体；
# main 分发器与 --level 接口保持稳定，扩展时不改动。
import argparse
import sys


def test_{op}_l0():
    """L0 门槛测试：规则 shape（block 整除），用于精度收敛。
    用例来自 DESIGN.md §9.2「L0 门槛测试计划」。"""
    test_configs = [...]  # (dtype, shape, block) —— 来自 DESIGN.md §9.2
    ok = True
    for dtype, shape, block in test_configs:
        try:
            # 1. 创建 kernel  2. 造输入  3. 执行  4. golden 对比
            torch.testing.assert_close(out.cpu(), ref.cpu(), rtol=rtol, atol=atol)
            print(f"[PRECISION_PASS] l0 shape={shape} dtype={dtype}")
        except Exception as e:
            print(f"[PRECISION_FAIL] l0 shape={shape} dtype={dtype}: {e}")
            ok = False
    return ok


def test_{op}_l1():
    """L1 功能测试（含不规则/尾块 shape）——留桩，由 tilelang-op-test-design 场景 B 填充。"""
    print("[L1] not expanded yet — run tilelang-op-test-design (scenario B)")
    return True


def test_{op}_l2():
    """L2 异常测试——留桩，由 tilelang-op-test-design 场景 B 填充。"""
    print("[L2] not expanded yet — run tilelang-op-test-design (scenario B)")


def test_{op}_boundary():
    """Boundary 边界/特殊值测试——留桩，由 tilelang-op-test-design 场景 B 填充。"""
    print("[BOUNDARY] not expanded yet — run tilelang-op-test-design (scenario B)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="l0",
                        choices=["l0", "l1", "l2", "boundary", "all"])
    args = parser.parse_args()

    tilelang.disable_cache()  # 在 main 内禁用编译缓存，避免旧产物干扰
    torch.manual_seed(0)

    blocking_ok = True  # 仅 L0/L1 计入阻塞判定
    if args.level in ("l0", "all"):
        blocking_ok &= test_{op}_l0()
    if args.level in ("l1", "all"):
        blocking_ok &= test_{op}_l1()
    if args.level in ("l2", "all"):
        test_{op}_l2()        # L2 失败只记录（[BOUNDARY_WARN]），不阻塞
    if args.level in ("boundary", "all"):
        test_{op}_boundary()  # Boundary 失败只记录，不阻塞

    if blocking_ok:
        print("Test Passed!")  # L0/L1 全过；bench_test.sh 据此判定
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
```

**分层运行约定**（与 tilelang-op-test-design 一致）：

| 层级 | 通过标记 | 失败标记 | 是否计入退出码 |
|------|---------|---------|--------------|
| L0 / L1 | `[PRECISION_PASS]` | `[PRECISION_FAIL]` | 是（任一失败 → exit 1） |
| L2 / Boundary | `[BOUNDARY_PASS]` | `[BOUNDARY_WARN]` | 否（仅记录，不影响退出码） |

- 精度收敛阶段跑 `--level l0`；扩展后跑 `--level all`。
- L2/Boundary 用例必须 `try/except` 包裹，失败打 `[BOUNDARY_WARN]` 后**继续**，不得中断、不得改退出码。

**融合算子注意事项**：

**Developer 模式（推荐，默认消除 workspace/vid）**：
- 装饰器无 `workspace_idx`，函数签名无 workspace 参数
- `T.Kernel(block_num, threads=2, is_npu=True) as (cid)`（单轴 + `threads=2`）
- Cube↔Vector 用 `alloc_shared/alloc_fragment` 片上 `T.copy` 直连，无 GM 往返、无 `vid` 偏移
- 完整骨架/映射表见 [tilelang-expert-to-developer mode-examples.md §6](../../tilelang-custom-skill/tilelang-expert-to-developer/references/mode-examples.md#6-cv-融合--推荐写法消除-workspace--vidthreads2)

**回退（Expert/混合或复杂同步场景）**：
- 函数签名包含 workspace 参数，`workspace_idx` 指定索引位置
- Cube 核输出通过 `T.copy` 写入 workspace，Vector 核从 workspace 读取（见 mode-examples.md §7）
