# Python Debug Code Examples

## 目录

- [1. Debug Snippet](#1-debug-snippet)
- [2. Complete Example](#2-complete-example)
- [3. Common Patterns](#3-common-patterns)

---

## 1. Debug Snippet

Insert the following code at the identified location:

```python
import os

# Debug: Print PID and wait for GDB attachment
print(f"PID: {os.getpid()}")
input("Press Enter after attaching GDB...")
```

**Important:**
- Make sure `import os` is at the top of the file (or add it if not present)
- Place this code BEFORE the kernel execution, not inside the kernel function
- The `input()` call will pause execution, giving time to attach GDB

## 2. Complete Example

Here's how a debug version should look:

```python
import os
import tilelang
import tilelang.language as T
import torch

tilelang.cache.clear_cache()

@tilelang.jit(out_idx=[1])
def sigmoid(M, N, block_M, block_N, dtype="float"):
    # ... kernel implementation ...
    pass

torch.manual_seed(0)

# Debug: Print PID and wait for GDB attachment
print(f"PID: {os.getpid()}")
input("Press Enter after attaching GDB...")

# Test execution
test_configs = [(256, 256, 64, 64)]
for M, N, block_M, block_N in test_configs:
    func = sigmoid(M, N, block_M, block_N)
    a = torch.randn(M, N).npu()
    b = func(a)
    # ... assertions ...
```

## 3. Common Patterns

### Pattern 1: Simple Example with Single Test

For examples with a simple structure:
- Add debug code after imports and setup
- Before the function call

### Pattern 2: Multiple Test Configurations

For examples with multiple test cases:
- Add debug code before the test loop
- This allows debugging any of the test cases

### Pattern 3: Examples with Multiple Functions

For examples with multiple kernel functions:
- Add debug code before the first function call
- User can set breakpoints in specific functions
