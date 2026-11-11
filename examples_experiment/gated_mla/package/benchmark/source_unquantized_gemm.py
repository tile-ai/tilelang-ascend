# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Exact function AST extracted from vllm-project/vllm-ascend@e4d533d33211f0696bcdf65eda47245d52e428c5
# vllm_ascend/ops/linear.py; no full vLLM import or custom dispatcher emulation.
from __future__ import annotations

import torch


def unquantized_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.nn.functional.linear(x, weight, bias)
