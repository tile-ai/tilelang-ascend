---
name: tilelang-debug-helper
description: How to add debugging capabilities to TileLang Ascend example operators. Use this skill whenever the user asks to debug a TileLang example, add GDB debugging code, create a debug version of an example, or mentions GDB, debugging, breakpoints, or VSCode debugging in the context of TileLang operators. This skill covers adding debug code to Python examples, configuring CMakeLists.txt for C++ debugging, and setting up VSCode for Python + C++ joint debugging.
---

# TileLang Debug Helper

This skill helps you add debugging capabilities to TileLang Ascend example operators so they can be debugged with GDB in VSCode.

## Overview

Debugging TileLang examples involves three main components:
1. **Python Debug Code**: Add PID printing and wait logic to Python examples
2. **C++ Debug Build**: Configure CMakeLists.txt to preserve debug symbols
3. **VSCode Configuration**: Set up joint Python + C++ debugging environment

## When to Use This Skill

Use this skill when:
- User asks to "debug" or "add debugging code" to a TileLang example
- User mentions GDB, breakpoints, or VSCode debugging
- User wants to step through C++ code in a TileLang operator
- User needs to inspect variables or execution flow in a TileLang kernel
- User wants to set up the complete debugging environment

---

## Part 1: Adding Debug Code to Python Examples

### Understanding the Task

When a user wants to debug a TileLang example, they need to:
1. Add code to print the process ID (PID)
2. Add code to wait for GDB attachment
3. This allows attaching a GDB debugger to the running Python process

### Step 1: Read the Original Example

First, read the example file that needs debugging. These are typically located in the `examples/` directory and end with `.py`.

### Step 2: Identify the Right Location

Find the best place to insert the debugging code. Look for:
- After imports and before the main test execution
- Before the function is called with `@tilelang.jit`
- After `torch.manual_seed()` if present
- Before the test loop starts

The goal is to pause execution before the actual kernel runs, so GDB can be attached.

### Step 3: Add Debug Code

Insert a small PID-printing + `input()` wait snippet before kernel execution. **Snippet, complete file example, and 3 common patterns (single test / multi-config / multi-function)** see [examples/python-debug-code.md](examples/python-debug-code.md).

### Step 4: Save the Debug Version

Save the modified file. Common naming conventions:
- Add `_debug` suffix: `sigmoid_debug.py`
- Or keep the original name if replacing it

---

## Part 2: Configuring CMakeLists.txt for C++ Debugging

To debug C++ code in TileLang, the project must be compiled with debug symbols (`-g`) and no optimizations (`-O0`).

**Quick steps**:
1. Locate `tilelang-ascend/CMakeLists.txt`
2. Find `add_library(tilelang_objs OBJECT ${TILE_LANG_SRCS})`
3. Add `target_compile_options(tilelang_objs PRIVATE -g -O0)` immediately after
4. Rebuild

**Complete CMake snippet + rebuild commands** see [examples/cmake-debug.md](examples/cmake-debug.md).

---

## Part 3: Configuring VSCode for Python + C++ Joint Debugging

VSCode needs two configurations:
1. **Python Debug Configuration** (launch.json) — launches the Python script
2. **C++ GDB Attach Configuration** (launch.json) — attaches GDB to the running Python process

Plus a tasks.json that sources `set_env.sh` and exports env to `.env` before launch.

> **Setting up VSCode for the first time** in a project: copy the full `launch.json` + `tasks.json` JSON and required extensions list from [examples/vscode-config.md](examples/vscode-config.md).

---

## Complete Debugging Workflow

### Step-by-Step Guide

1. **Configure CMakeLists.txt** (if not already done)
   - Add `target_compile_options(tilelang_objs PRIVATE -g -O0)`
   - Rebuild the project

2. **Configure VSCode** (if not already done)
   - Create `.vscode/launch.json` and `.vscode/tasks.json`
   - Install required extensions

3. **Add Debug Code to Python Example**
   - Insert PID printing and wait logic
   - Save the file

4. **Start Debugging**
   - Open the Python example in VSCode
   - Run "Step 1: Debug Python Example"
   - The script will print the PID and pause
   - Run "Step 2: Attach C++ (GDB)"
   - Select the Python process when prompted
   - Set breakpoints in C++ code
   - Press Enter in the Python console to continue execution
   - GDB will hit breakpoints in C++ code

### What Happens During Debugging

1. User runs the Python script with debug code
2. Script prints PID and pauses at `input()`
3. User attaches GDB to that PID in VSCode
4. User presses Enter in the Python console
5. Execution continues and GDB can hit breakpoints in C++ code
6. User can step through both Python and C++ code seamlessly

---

## Verification

### After Adding Debug Code to Python

1. Verify the file is syntactically correct
2. Confirm `import os` is present
3. Confirm the debug code is placed before kernel execution
4. The file should run and pause at the `input()` call

### After Modifying CMakeLists.txt

1. Verify the `target_compile_options` line is present
2. Rebuild the project successfully
3. Debug symbols are present in the compiled binaries

### After Configuring VSCode

1. Verify `.vscode/launch.json` exists and is valid JSON
2. Verify `.vscode/tasks.json` exists and is valid JSON
3. Verify `set_env.sh` path is correct in tasks.json
4. Verify `.env` file is generated when running "set env" task

---

## Troubleshooting

**When debugging session fails to start, breakpoints don't hit, or GDB can't find process** — see [references/troubleshooting.md](references/troubleshooting.md) covering Python debug code issues, C++ debugging issues (breakpoints / variable inspection), and VSCode configuration issues (launch / attach failures / process not found).

---

## Output

Always provide:
- The path to the created/modified debug file (if applicable)
- Brief instructions on how to use it (run, attach GDB, continue)
- Mention any additional setup needed if not already done
- Clear steps for the complete debugging workflow

---

## Prerequisites Summary

Before starting debugging, ensure:
- Linux operating system
- Git installed
- Python 3.10 or higher
- GDB installed (`sudo apt install gdb`)
- VSCode with Python and C/C++ extensions
- TileLang project compiled with debug information
- CMakeLists.txt configured with `-g -O0`
- VSCode configured with launch.json and tasks.json

---

## 子目录索引

- [examples/python-debug-code.md](examples/python-debug-code.md) — Python debug snippet / complete example / 3 common patterns
- [examples/cmake-debug.md](examples/cmake-debug.md) — CMakeLists.txt full snippet + rebuild commands
- [examples/vscode-config.md](examples/vscode-config.md) — launch.json + tasks.json full JSON + required extensions
- [references/troubleshooting.md](references/troubleshooting.md) — Common debug-session failures by category (Python / C++ / VSCode)
