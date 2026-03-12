# TileLang-Ascend 贡献指南

- [入门](#入门)
- [开发指导](#开发指导)
  - [代码风格](#代码风格)
  - [Fork-Pull开发模式](#fork-pull开发模式)
  - [代码门禁异常处理](#代码门禁异常处理)
  - [ISSUE规范](#issue规范)
  - [提出PR](#提出pr)


## 入门

- 在 [GitHub](https://github.com/tile-ai/tilelang-ascend) 上 Fork TileLang-Ascend 存储库。
- 阅读 [README.md](https://github.com/tile-ai/tilelang-ascend/blob/npuir/README.md) 获取项目信息和构建开发环境。

## 开发指导

- [代码风格](#代码风格)
- [Fork-Pull开发模式](#fork-pull开发模式)
- [代码门禁异常处理](#代码门禁异常处理)
- [ISSUE规范](#issue规范)
- [提出PR](#提出pr)

### 代码风格

请遵循以下编码风格，以使得 TileLang-Ascend 易于开发、维护和审查。

- 编码指南

  请使用 TileLang-Ascend 社区统一的编码风格，python 建议的编码风格是 [PEP 8 编码样式](https://pep8.org/)，C++ 编码所建议的风格是 [LLVM 编码规范](https://llvm.org/docs/CodingStandards.html)。可以使用 [clang-tidy](https://github.com/llvm/llvm-project/blob/main/.clang-tidy)，[CppLint](https://github.com/cpplint/cpplint)，[CppCheck](http://cppcheck.sourceforge.net/)，[CMakeLint](https://github.com/cmake-lint/cmake-lint)，[CodeSpell](https://github.com/codespell-project/codespell)，[ShellCheck](https://github.com/koalaman/shellcheck) 和 [pylint](https://pylint.org/) 检查代码的格式，建议在您的 IDE 中安装这些插件。

- 单元测试指南

  请使用 TileLang-Ascend 社区统一的单元测试风格，python 建议的单元测试风格是 [pytest](http://www.pytest.org/en/latest/)，C++ 建议的单元测试风格是 [Googletest Primer](https://github.com/google/googletest/blob/master/docs/primer.md)。测试用例的设计意图应该通过它的注释名称来反映。NPUIR 相关测试组织与示例请参考 [testing/npuir/README.md](https://github.com/tile-ai/tilelang-ascend/blob/npuir/testing/npuir/README.md)。

- 重构指南

  我们鼓励开发人员对我们的代码进行重构来消除【代码坏味道】。重构的代码也应该遵循编码风格和测试风格的要求。当您收到警告时，您需要重构要合并的代码。

### Fork-Pull开发模式

1、Fork TileLang-Ascend 项目

在您向 TileLang-Ascend 项目提交自己的代码之前，请确保已经将 TileLang-Ascend 项目 Fork 到您自己的存储库。后续您将在自己 Fork 的项目上进行开发，并通过 Pull Request 的方式合并到 TileLang-Ascend 项目。这意味着 TileLang-Ascend 存储库和您自己的存储库之间存在并行开发，因此请注意保持存储库之间的一致性。

2、克隆远程仓库

使用 git 克隆您 fork 的 TileLang-Ascend 项目 & 添加上游仓库 upstream：

```shell
git clone https://github.com/{your_forked_repo}/tilelang-ascend.git && cd tilelang-ascend && git submodule update --init --recursive
git remote add upstream https://github.com/tile-ai/tilelang-ascend.git
```

3、本地环境开发代码

在开发您的代码之前，您需要根据 TileLang-Ascend 安装说明搭建开发环境（见仓库 README 与 [docs/get_started/Installation.md](https://github.com/tile-ai/tilelang-ascend/blob/npuir/docs/get_started/Installation.md)）。

为避免多个分支间的不一致问题，请创建新的本地开发分支进行新特性的开发：

```shell
git checkout -b {new_branch_name} origin/npuir
git fetch upstream
git rebase upstream/npuir
```

TileLang-Ascend 可能会根据需要创建版本分支或下游开发分支。当您创建完分支 & 同步上游目标分支更新后，就可以开始开发您的代码了。

4、代码更改自测

完成代码更改后，请检查您的更改是否可以通过测试：

在本地为您开发的代码编写测试用例（例如 `testing/npuir` 下相应目录），并在本地环境中验证您的测试脚本，确保您的更改可以通过测试。

5、代码推送到远程仓库

代码更新 & 测试完成后，推送您的 commit 到您的远程仓库。

```shell
git add .
git status
git commit -m "Your commit title"
git commit -s --amend
git push origin {your_new_branch_name}
```

6、向 TileLang-Ascend 主仓创建拉取请求

代码推送至您的远程仓库后，您需要在您的新分支和 TileLang-Ascend 上游目标分支之间新建 Pull Request。完成新建合并请求后，“Github Actions”将自动设置为您构建流水线测试。您的 Pull Request 请尽快合并到上游目标分支，以降低合并风险。


### 代码门禁异常处理

代码门禁异常主要包含以下几种情况，请根据相关提示信息解决门禁异常问题。

- 编译失败

  请根据提示信息，检查编译失败的原因，解决后重新编译即可。

- 静态检查失败

  请根据提示信息，查找出代码中的异常信息并解决。

- CI 流水线未通过

  请根据提示信息，查找出 CI 流水线未通过的测试用例并检查原因，解决后重新运行 CI 流水线。

### ISSUE规范

为项目做贡献的一个好的方法是在遇到问题时发送详细报告。我们总是非常感谢写得详细、彻底的错误报告，并会因此非常感谢您！

本仓库 Issue 标题须以 `[npuir]` 开头，用于标明属于 npuir 分支 / NPUIR 路线，便于与其它分支区分。例如：`[npuir] Developer 模式下 xxx 报错`。

在报告问题时，请参考以下格式：

- 您环境里使用的软件版本（TileLang-Ascend 分支与提交、python、CANN、os 等）？
- 这是一个错误报告还是功能请求？
- 您报告的是什么样的问题，添加对应的标签以便在问题仪表盘上突出显示？
- 发生了什么？
- 您预计会发生什么？
- 如何重现它？（尽可能精确）

不同类别的 ISSUE 填写模板以本仓库 Issues 创建页可选模板为准。

问题咨询：

- 如果您发现一个未解决的问题，而这个问题正是您要解决的，请对该问题发表评论，告诉其他人您将负责这个问题。
- 如果问题已经打开一段时间，请您在解决该问题前进行预检查。
- 如果您解决了自己报告的问题，在关闭该问题前还需要让其他人知道。

### 提出PR

- 在 [GitHub](https://github.com/tile-ai/tilelang-ascend) 上提出您的想法作为问题（Issue 标题须以 `[npuir]` 开头）。
- 如果要开发的新功能需要大量设计细节，您还应提交设计方案。
- 在问题讨论和设计方案审查达成共识后，再进行 fork 开发并提交 PR。

本仓库 MR/PR 标题规范：

- 须以 `[AscendNPU-IR][涉及的方面]` 开头，其后为简短描述。`[涉及的方面]` 用于概括改动领域（如：编译、算子、测试、文档、CI 等）。
- 若尚未完成、不欲合入，须在标题中增加 `[WIP]` 标识（例如：`[WIP][AscendNPU-IR][测试] 补充 slice+gemm 用例`）。

- PR 合入以 CI 通过为准；Maintainer 视需要做 code review。
- 在 PR 被充分讨论后，将根据讨论结果对 PR 进行合并、拒绝或放弃。

注意事项：

- 应避免任何不相关的更改。
- 确保您的提交历史是简洁有序的。
- 创建 PR 前请 rebase 上游仓库最新代码。
- 对于错误修复 PR，请确保链接所有相关 Issue 和 PR。

