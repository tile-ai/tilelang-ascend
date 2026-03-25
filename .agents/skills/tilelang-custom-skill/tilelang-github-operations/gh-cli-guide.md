# GitHub CLI 安装与配置指南

> 参考文档：
> - Linux: https://github.com/cli/cli/blob/trunk/docs/install_linux.md
> - macOS: https://github.com/cli/cli/blob/trunk/docs/install_macos.md
> - Windows: https://github.com/cli/cli/blob/trunk/docs/install_windows.md

## 安装

### Ubuntu/Debian

```bash
sudo mkdir -p -m 755 /etc/apt/keyrings
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null
sudo chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
sudo apt update && sudo apt install gh -y
```

### CentOS/RHEL/Fedora

```bash
sudo dnf install gh
# 或
sudo yum install gh
```

### macOS

```bash
brew install gh
```

### Windows

```powershell
winget install GitHub.cli
# 或
choco install gh
```

### 验证安装

```bash
gh --version
```

---

## 认证 GitHub

### 方式一：交互式登录（推荐）

```bash
gh auth login
```

按提示选择：
1. **GitHub.com** 或 GitHub Enterprise
2. **HTTPS** 协议
3. **Login with a web browser**

### 方式二：环境变量

设置环境变量后无需执行 `gh auth login`：

```bash
export GH_TOKEN="your_token_here"
```

### 验证认证状态

```bash
gh auth status
```

---

## Token 管理

- **查看/撤销授权**：https://github.com/settings/applications
- **登出**：`gh auth logout`

---

## 注意事项

- 国内网络连接 GitHub 不稳定，遇到问题请重试