# Troubleshooting

## 目录

- [1. Python Debug Code Issues](#1-python-debug-code-issues)
- [2. C++ Debugging Issues](#2-c-debugging-issues)
- [3. VSCode Configuration Issues](#3-vscode-configuration-issues)

---

## 1. Python Debug Code Issues

**Problem**: Script doesn't pause at `input()`
- **Solution**: Ensure debug code is placed before kernel execution, not inside the kernel function

**Problem**: `import os` not found
- **Solution**: Add `import os` at the top of the file

## 2. C++ Debugging Issues

**Problem**: Breakpoints don't work in C++ code
- **Solution**: 
  - Verify `target_compile_options(tilelang_objs PRIVATE -g -O0)` is in CMakeLists.txt
  - Rebuild the project after modifying CMakeLists.txt
  - Ensure GDB is installed: `sudo apt install gdb`

**Problem**: Can't inspect variables in GDB
- **Solution**: Ensure the project is compiled with `-g` flag

## 3. VSCode Configuration Issues

**Problem**: "Step 1: Debug Python Example" fails
- **Solution**:
  - Install Python extension: `ms-python.python`
  - Verify `set_env.sh` path is correct in tasks.json
  - Check that `.env` file is generated

**Problem**: "Step 2: Attach C++ (GDB)" fails
- **Solution**:
  - Install C/C++ extension: `ms-vscode.cpptools`
  - Ensure GDB is installed: `sudo apt install gdb`
  - Verify the Python process is running when attaching

**Problem**: Can't find the Python process
- **Solution**: 
  - Run the Python debug configuration first
  - Wait for the PID to be printed
  - Then attach GDB
