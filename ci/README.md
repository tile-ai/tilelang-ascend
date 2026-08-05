# 算子测试提交指南

我们在把 `examples/` 下的算子从老的 `bench_test.sh` 迁到 Pytest。

老流程判断算子对不对,是去 stdout 里 grep `Kernel Output Match!` 这句话 —— 算子把这句话打出来就算过,哪怕结果是错的。新流程用真的数值断言。

**这件事需要大家按照任务分配,写测试文件。**

---

## 大致流程

```
① 从 ascendc_pto 切分支
② 查登记表，拿到你的测试该叫什么名字
③ 照样板写测试
④ 本地跑通 + 格式检查
⑤ 提 PR，base 选 ascendc_pto
⑥ 等 CI（约 11 分钟）
```

**你只需要交一个文件:你写的那个测试。**登记表已经把 140 个算子的测试文件名全预留好了,**不用改它**。

---

## 一、你可以提交什么

打开 `ci/operator_test_manifest.yaml`,里面按目录分组列了 140 个算子:

```yaml
  examples/xattention/xattention.py: examples/xattention/test_xattention.py
```

- **冒号左边**:算子文件
- **冒号右边**:你要新建的测试文件,完整路径,照抄,一个字都别改

一条登记只有在测试文件真实存在时才生效。文件没建之前,那个算子照常由 `bench_test.sh` 跑,不受影响。

---

## 二、照样板写

样板:**`examples/batch_gemm/test_batch_gemm.py`**

打开它照着写。**算子原文件一个字都别改。**

### 一条要守住的:让算子代码在 pytest 进程里跑

别用 `subprocess` 另起一个进程去跑原脚本。那样测试也能过,精度也能测到,但有两个代价:

- **覆盖率会记成 0。**`--cov` 挂在 pytest 进程上,追不进子进程。实测同一个算子:进程内直调记到 3655 行 tilelang 代码,子进程写法记到 0 行。
- **失败时只剩一段 stdout**,看不到 `assert_close` 的断言现场(哪个元素差多少)。

### 原脚本不好 import 怎么办

没有 `if __name__ == "__main__"`、顶层直接跑、甚至顶层就 `parse_args()` 的,都能进程内跑 —— 临时换掉 `sys.argv` 再 `exec_module` 就行:

```python
import importlib.util
import sys
from pathlib import Path

import pytest


def _run_example(m: int, n: int, k: int) -> None:
    source = Path(__file__).with_name("example_gemm.py")
    spec = importlib.util.spec_from_file_location("_example_gemm_under_test", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load example module: {source}")

    module = importlib.util.module_from_spec(spec)
    original_argv = sys.argv
    try:
        # The example parses its arguments and runs the comparison at import
        # time, so the shape is chosen here and the module body is the test.
        sys.argv = [str(source), "--m", str(m), "--n", str(n), "--k", str(k)]
        spec.loader.exec_module(module)
    finally:
        sys.argv = original_argv


@pytest.mark.parametrize(
    ("m", "n", "k"),
    [(1024, 1024, 1024), (512, 512, 512)],
    ids=["default_1024x1024x1024", "small_512x512x512"],
)
def test_example_gemm_precision(m: int, n: int, k: int) -> None:
    _run_example(m, n, k)
```

原脚本里已经有现成函数可以调的(`main()` / `check_case()` 之类),直接 import 那个函数更省事:

```python
from example_quant_batch_matmul import check_case
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

### 3. 本地进行格式检查

CI 的格式检查只跑 **ruff**。本地装一个就能完全复现,不用起昇腾环境,几秒钟的事。

#### 安装检查工具

```bash
pip install -U ruff
python3 -m ruff --version          # 能打出版本号就装好了
```

#### 检查

**先 `cd` 到仓库根目录**,再跑。规则(行宽 140、双引号等)写在根目录的 `pyproject.toml` 里,ruff 会自己往上找 —— 在别的目录下跑读不到配置,结果和 CI 对不上。

```bash
cd /path/to/tilelang-ascend        # ← 必须先切到仓库根目录

python3 -m ruff format --check --diff examples/<目录>/test_<算子>.py
python3 -m ruff check examples/<目录>/test_<算子>.py
```

**两条都是这个输出、退出码都是 0,才算过:**

```
1 file already formatted
All checks passed!
```

#### 修改

没过不用手动改,让 ruff 自己来 —— 把 `--check` 去掉、给 `check` 加 `--fix`,它会**直接改你的文件**:

```bash
python3 -m ruff check --fix examples/<目录>/test_<算子>.py
python3 -m ruff format examples/<目录>/test_<算子>.py
```

```
Found 1 error (1 fixed, 0 remaining).
1 file reformatted
```

改完再跑一遍上面的「查」确认全绿,然后**把 ruff 改过的版本提交上去**。

### 4. 只提交 test 文件

**只允许新增 `test_*.py`,不允许修改仓库里的任何其他文件。**

```
你的算子文件本身                     ← 一个字都别改
ci/operator_test_manifest.yaml       ← 名字已经预留好了，不用改
examples/bench_test.sh
.github/workflows/ci_cd.yml
.gitignore
src/ 或 tilelang/ 下的任何文件
```

提之前 `git status` 看一眼,或者在 PR 的 **Files changed** 页面确认:**列表里应该只有你新建的 test 文件。**

碰了上面那些,CI 会从 11 分钟变 50 分钟,还会被打回。

**一次可以提交多个测试。**几个算子写完放一个 PR 里没问题,CI 也只跑你新加的那几个。

---

## 四、提到哪个分支

**base 选 `ascendc_pto`**,也就是仓库的默认分支,开 PR 时一般不用改。

四个下拉框核对成这样:

```
base repository:  tile-ai/tilelang-ascend        base:     ascendc_pto
head repository:  <你的用户名>/tilelang-ascend    compare:  <你自己的分支>
```

唯一容易错的是 **`base repository` 默认可能是你自己的 fork** —— 选错就变成给自己提 PR 了。

提错了不用重开,PR 页面标题下面点 **Edit** 可以改。

标题写 `[Test] Add <算子> Pytest test`。

---

## 五、提完之后关注什么

### CI 跑三项

| 检查项 | 干什么 | 多久 |
|---|---|---|
| `Check PR Status and Changes` | 看你改了哪些文件、决定跑多少 | 几秒 |
| `AscendC-PTO CI` | 代码格式(ruff) | 半分钟 |
| `Benchmark Tests (Ascend NPU)` | 编译 + 跑测试 | **约 11 分钟** |

最后那项内部是:

```
增量编译    约 9.5 分钟    任何 PR 都要，省不掉
你的测试    约 1 分钟      只跑你新加的这一个
```

一次加好几个测试也一样,只跑你加的那几个,不跑全量。

### 一直 queued 不动

在排队等昇腾机器,别的 PR 占着。**这是正常的,等就行**,见过排 70 多分钟的。

### 红了怎么办

点 **Details** 进去,展开红色那一步看报错。常见两种:

**① 日志里有 `operator test(s) matching no manifest entry`,列出了你的文件**

文件名或位置和登记表对不上。回「一、你可以提交什么」查表,照冒号右边那个路径改。

**② 跟你的代码没关系的**

比如 `Checkout code` 报 `curl 28: Connection timed out`、编译中途莫名断了、排队超时被取消。

这种在 PR 下面发一条评论,内容包含 `/re-test`,机器人会只把失败的那几项重跑一遍:

```
/re-test
```

---

## 总结

```
从 ascendc_pto 切分支 → 查登记表拿文件名 → 照 test_batch_gemm.py 写
→ 算子代码在 pytest 进程里跑，别开 subprocess
→ 本地跑通 + ruff 检查 → PR 提到 ascendc_pto
```

**只交测试文件,不改 manifest。**
