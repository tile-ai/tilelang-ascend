# VSCode Joint Python + C++ Debug Configuration

## 目录

- [1. Setup Overview](#1-setup-overview)
- [2. Step 1: Check/Create .vscode Directory](#2-step-1-checkcreate-vscode-directory)
- [3. Step 2: Configure launch.json](#3-step-2-configure-launchjson)
- [4. Step 3: Configure tasks.json](#4-step-3-configure-tasksjson)
- [5. Step 4: Install Required VSCode Extensions](#5-step-4-install-required-vscode-extensions)

---

## 1. Setup Overview

VSCode needs two configurations:
1. **Python Debug Configuration**: Launches the Python script
2. **C++ GDB Attach Configuration**: Attaches GDB to the running Python process

This allows seamless debugging across Python and C++ code.

## 2. Step 1: Check/Create .vscode Directory

```bash
mkdir -p .vscode
```

## 3. Step 2: Configure launch.json

Create or merge `.vscode/launch.json` with the following configurations:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Step 1: Debug Python Example",
      "type": "debugpy",
      "request": "launch",
      "program": "${file}",
      "console": "integratedTerminal",
      "justMyCode": false,
      "preLaunchTask": "set env",
      "envFile": "${workspaceFolder}/.env"
    },
    {
      "name": "Step 2: Attach C++ (GDB)",
      "type": "cppdbg",
      "request": "attach",
      "processId": "${command:pickProcess}",
      "MIMode": "gdb",
      "setupCommands": [
        {
          "description": "Enable pretty-printing for gdb",
          "text": "-enable-pretty-printing",
          "ignoreFailures": true
        },
        {
          "description": "Set Disassembly Flavor to Intel",
          "text": "-gdb-set disassembly-flavor intel",
          "ignoreFailures": true
        },
        {
          "description": "Set breakpoint pending on",
          "text": "-gdb-set breakpoint pending on",
          "ignoreFailures": true
        }
      ]
    }
  ]
}
```

## 4. Step 3: Configure tasks.json

First, find the `set_env.sh` script:
- Priority 1: Project root directory (`./set_env.sh`)
- Priority 2: Shortest path found in the project

Create or merge `.vscode/tasks.json` with the following task:

```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "set env",
      "type": "shell",
      "command": "bash",
      "args": [
        "-c",
        "source <path_to_set_env.sh> && env > ${workspaceFolder}/.env"
      ],
      "problemMatcher": []
    }
  ]
}
```

Replace `<path_to_set_env.sh>` with the actual path to `set_env.sh`.

## 5. Step 4: Install Required VSCode Extensions

Ensure the following extensions are installed:
- Python: `ms-python.python`
- C/C++: `ms-vscode.cpptools`
