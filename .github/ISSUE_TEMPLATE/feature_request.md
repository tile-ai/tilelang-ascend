---
name: Feature request
about: Suggest an idea for this project
title: '[Feature] '
labels: 'enhancement'
assignees: ''
---

## Feature Description
A clear and concise description of the feature you'd like to request.

## Use Case
Describe the use case or scenario where this feature would be helpful.

## Proposed Solution
If you have a proposed solution or implementation idea, describe it here.

## Feature Type
- [ ] New operator/kernel (e.g., new GEMM variant, new attention mechanism)
- [ ] New API/primitive (e.g., new T.xxx operation)
- [ ] Compiler optimization (e.g., new pass, performance improvement)
- [ ] New backend support (e.g., new hardware target)
- [ ] Developer tooling (e.g., debugging, profiling)
- [ ] Documentation improvement
- [ ] Other: [specify]

## Operator Details (if applicable)

**Operator Type:**
[e.g. GEMM, Flash Attention, Softmax, LayerNorm, Convolution, etc.]

**Input/Output Specification:**
```
Input shapes and dtypes:
- A: [M, K], float16
- B: [K, N], float16

Output shapes and dtypes:
- C: [M, N], float16
```

**Performance Requirements:**
[e.g. target throughput, latency, memory bandwidth]

**Reference Implementation (if any):**
[e.g. PyTorch implementation, CUDA kernel, paper reference]

## Alternative Solutions
Describe any alternative solutions or features you've considered.

## Implementation Constraints (if known)

**Programming Mode Preference:**
- [ ] Developer mode (automatic)
- [ ] Expert mode (manual control)
- [ ] Both modes supported

**Memory Constraints:**
[e.g. L1 buffer size, UB buffer size constraints]

**Target Hardware:**
- [ ] Ascend A2
- [ ] Ascend A3
- [ ] Other Ascend variants

## Additional Context
Add any other context, references, or screenshots about the feature request here.

## Willingness to Contribute
- [ ] I'm willing to submit a PR for this feature
- [ ] I can help with testing
- [ ] I can provide reference implementations
- [ ] I need guidance to get started