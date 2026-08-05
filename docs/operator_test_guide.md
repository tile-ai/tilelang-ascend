# 算子测试提交指南

本文档介绍如何为 `examples/`以及`examples_experiment/` 下的算子编写 Pytest 测试并提交,用于把算子从 `bench_test.sh` 的脚本执行方式迁移到 Pytest。

老流程判断算子对不对,是去 stdout 里 grep `Kernel Output Match!` 这句话 —— 算子把这句话打出来就算过,哪怕结果是错的。新流程用真的数值断言。

相关文档:

- [`pytest_marker_guide.md`](pytest_marker_guide.md) —— `low_priority` / `ci_skip` 标签怎么用
- [`coverage_guide.md`](coverage_guide.md) —— 覆盖率怎么统计、报告在哪

---

## 大致流程

```
① 从 ascendc_pto 中fork当前仓库
② 查登记表，拿到你的测试该叫什么名字
③ 照样板写测试
④ 本地跑通 + 格式检查
⑤ 提 PR，base 选 ascendc_pto
⑥ 等 CI（约 11 分钟）
```

**你只需要交一个文件:你写的那个测试。**登记表已经把 140 个算子的测试文件名全预留好了,**不用改它**，如果有新增算子需要修改，但是修改该文件会触发全量测试需要注意。

---

## 一、你可以提交什么

打开 `ci/operator_test_manifest.yaml`,里面按目录分组列了 140 个算子:

```yaml
  examples/xattention/xattention.py: examples/xattention/test_xattention.py
```

- **冒号左边**:算子文件
- **冒号右边**:你要新建的测试文件,完整路径,照抄,一个字都别改
- 如果有新增算子需要参照该文件进行修改，同时注意不要有重名的test文件

一条登记只有在测试文件真实存在时才生效。文件没建之前,那个算子照常由 `bench_test.sh` 跑,不受影响。

---

## 二、照样板写

样板:**`examples/batch_gemm/test_batch_gemm.py`**

打开它照着写。**算子原文件一个字都别改。**

样板里那 20 行同时示范了三件事,下面逐条说。

### 红线:加载算子的那句,必须写在测试函数里面

```python
# ✗ 写在模块顶层 —— pytest 收集阶段就会执行算子
EXAMPLE = _load_example()

def test_xxx():
    ...
```

```python
# ✓ 写在测试函数里 —— 真正跑这个测试时才执行
def test_xxx():
    example = _load_example()
```

pytest 分两步:先「收集」(把每个 `test_*.py` import 一遍,看看里面有哪些测试),再「运行」。

写模块顶层的话,算子在**收集阶段**就跑了。这时候它一旦失败,pytest 认为是「这个文件读不进来」,后果比测试失败严重得多:

```
顶层加载：  Interrupted: 1 error during collection   → 同一批里其它文件一个都没跑
函数内加载：1 failed, 3 passed                        → 只死自己那一个
```

写在函数里,收集阶段只读测试文件本身那几行 import,完全不碰 NPU。

**这一条比下面所有内容都重要。**

### 算子顶层有 `parse_args()` 的,要把 `sys.argv` 换掉

不少算子在顶层就解析命令行:

```python
args = parser.parse_args()      # 它去读 sys.argv
```

而 CI 跑的是 `pytest --forked -n 8 -m "..."`,`sys.argv` 里全是 pytest 的参数。算子的 parser 不认识 `--forked`,直接 `SystemExit`。

所以加载前把 `sys.argv` 临时换成算子自己的样子,加载完换回来 —— 样板里就是这么写的:

```python
original_argv = sys.argv
try:
    # batch_gemm.py parses arguments at import time. Hide Pytest arguments
    # while loading it without changing the original Example.
    sys.argv = [str(source)]
    spec.loader.exec_module(module)
finally:
    sys.argv = original_argv
```

`finally` 里那句一定要有:算子中途抛异常时也得把 `sys.argv` 还回去,否则后面的测试全遭殃。

要指定 shape 就把参数拼进去:

```python
sys.argv = [str(source), "--m", str(m), "--n", str(n), "--k", str(k)]
```

### 三种写法,按你的算子是什么结构挑一种

**A. 算子有现成函数可以调**(`main()` / `check_case()` / `batch_matmul()` 之类)

加载完直接调它,最省事,样板就是这一种:

```python
def test_batch_gemm_accuracy() -> None:
    example = _load_example()
    kernel = example.batch_matmul(batch, m, n, k, block_M=128, block_N=256, K_L1=64)
    ...
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
```

**B. 算子没有 `if __name__ == "__main__"`,顶层直接跑完并断言**

`exec_module` 执行完就等于算子跑完、精度也比完了,异常自然抛给 pytest。测试体就一行:

```python
def test_silu_run() -> None:
    _load_example()
```

**C. 算子的活儿写在 `if __name__ == "__main__":` 里**

这种文件被 import 时那段**根本不执行**(因为 `__name__` 不是 `__main__`),要用 `runpy` 指定名字来跑:

```python
import runpy
import sys
from pathlib import Path


def _run_example(*argv) -> None:
    source = Path(__file__).with_name("xattention.py")
    original_argv = sys.argv
    try:
        sys.argv = [str(source), *argv]
        runpy.run_path(str(source), run_name="__main__")
    finally:
        sys.argv = original_argv
```
---

## 三、提交前检查

### 1. 本地跑通

运行前请参考仓库的doc文件自行完成基本环境配置；

完成环境配置后

```bash
python3 -m pytest examples/<目录>/test_<算子>.py -v
```

**看到 `passed` 才算过:**

```
============================= test session starts ==============================
collected 1 item

examples/batch_gemm/test_batch_gemm.py::test_batch_gemm_accuracy PASSED  [100%]

============================== 1 passed in 12.36s ==============================
```

写了几个 case 就是 `N passed`,数字对上你写的个数。


### 2. 确认文件名和登记表一致

再核对一遍路径和文件名,和登记表冒号右边**完全一样**。

名字对不上不会当场报错,但那个测试**不会被 Pytest 接管** —— 它会被当成普通脚本由 `bench_test.sh` 收走执行,而 pytest 风格的文件直接 `python` 跑是不干活的,CI 那一格就会红。日志里会有一句 `operator test(s) matching no manifest entry`,把没认领的文件名列出来,照着改就行。

改完再跑一遍上面的「查」确认全绿,然后**把 ruff 改过的版本提交上去**。

### 3. 只提交 test 文件

**只允许新增 `test_*.py`,不允许修改仓库里的任何其他文件。**

```
你的算子文件本身                     ← 一个字都别改
ci/operator_test_manifest.yaml       ← 名字已经预留好了，不用改，如果有新增算子需要修改，但是修改该文件会触发全量测试需要注意
```

提之前 `git status` 看一眼,或者在 PR 的 **Files changed** 页面确认:**列表里应该只有你新建的 test 文件。**

**一次可以提交多个测试。**几个算子写完放一个 PR 里没问题,CI 也只跑你新加的那几个。

---

## 总结

```
从 ascendc_pto 切分支 → 查登记表拿文件名 → 照 test_batch_gemm.py 写
→ 本地跑通 + ruff 检查 → PR 提到 ascendc_pto
```

写测试时守住这四条:

1. **加载算子那句写在测试函数里**,不写模块顶层 —— 写顶层的话一个算子失败会让整批测试 `Interrupted`
2. 算子顶层有 `parse_args()` 的,**加载前后换掉 `sys.argv`**
3. **别用 `subprocess` 包装原脚本** —— 覆盖率会记成 0
4. **shape 挑小的** —— CI 是 8 路并行,照抄原文件的大 shape 容易 OOM

**只交测试文件,不改 manifest。**
