---
name: Bug report
about: Create a report to help us improve
title: '[Bug] '
labels: 'bug'
assignees: ''
---

## Bug Description
A clear and concise description of what the bug is.

## Environment

**Hardware:**
- NPU Device: [e.g. Ascend A2, A3]
- Other hardware info: [e.g. CPU, memory]

**Software:**
- OS: [e.g. Ubuntu 20.04, EulerOS]
- CANN version: [e.g. 8.3.RC1]
- torch-npu version: [e.g. 2.6.0.RC1]
- Python version: [e.g. 3.10]
- TileLang-Ascend version: [e.g. commit hash or tag]
- Installation method: [wheel package / build from source]

## Code to Reproduce

```python
# Minimal reproducible code
import tilelang
import torch

# Your code here
```

## Error Message

```
Paste the full error message here
```

## Expected Behavior
A clear and concise description of what you expected to happen.

## Actual Behavior
A clear and concise description of what actually happened.

## Additional Context

**Programming Mode:**
- [ ] Developer mode (automatic sync/buffer management)
- [ ] Expert mode (manual T.Scope/T.barrier_all)
- [ ] Mixed mode

**Backend:**
- [ ] Ascend C & PTO backend
- [ ] AscendNPU IR backend

**Pass Configs (if applicable):**
```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: ...,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: ...,
    # ...
}
```

**Operator Type:**
[e.g. GEMM, Flash Attention, Softmax, Convolution, etc.]

**Additional Information:**
- Have you checked the [Programming Guide](../docs/TileLang-Ascend%20Programming%20Guide.md)?
- Have you searched existing issues?
- Any other context or screenshots?