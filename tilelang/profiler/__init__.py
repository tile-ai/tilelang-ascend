# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""The profiler and convert to torch utils"""

from __future__ import annotations

from typing import Callable, Any
from functools import partial
import torch
from contextlib import suppress
from dataclasses import dataclass
import tvm
from tilelang.utils.tensor import (
    get_tensor_supply,
    TensorSupplyType,
    torch_assert_close,
    adapt_torch2tvm,
)
from tilelang.engine.param import KernelParam
from tilelang.jit.adapter import BaseKernelAdapter
from tilelang.profiler.bench import do_bench


@dataclass
class Profiler:
    """A profiler class for benchmarking and validating kernel implementations.

    Attributes:
        params: List of kernel parameters defining the input/output specifications
        result_idx: Indices indicating which parameters are output tensors
        supply_type: Type of tensor supply to use (e.g., random, zeros, etc.)
        adapter: Optional kernel adapter for interfacing with different backends
    """

    params: list[KernelParam]
    result_idx: list[int]
    workspace_idx: list[int]
    supply_type: TensorSupplyType
    adapter: BaseKernelAdapter | None = None

    def __post_init__(self):
        """Initialize tensor supply after dataclass initialization"""
        self.result_idx = self._legalize_auto_memory_idx(self.result_idx, "result_idx")
        self.workspace_idx = self._legalize_auto_memory_idx(self.workspace_idx, "workspace_idx")
        self.supply = get_tensor_supply(self.supply_type)

    def _legalize_auto_memory_idx(self, memory_idx: list[int] | int | None = None, memory_name="auto_memory_idx") -> list[int]:
        params = self.params
        if memory_idx is None:
            memory_idx = []

        elif isinstance(memory_idx, int):
            if memory_idx >= len(params) or memory_idx < -len(params):
                raise ValueError(f"{memory_name} should be an integer between {-len(params)} and {len(params) - 1}")
            if memory_idx < 0:
                memory_idx = len(params) + memory_idx
            memory_idx = [memory_idx]

        elif isinstance(memory_idx, list):
            for i, idx in enumerate(memory_idx):
                if idx >= len(params) or idx < -len(params):
                    raise ValueError(f"{memory_name} should be an integer between {-len(params)} and {len(params) - 1}")
                if idx < 0:
                    memory_idx[i] = len(params) + idx

        else:
            raise ValueError(f"{memory_name} should be a list of integers")

        return memory_idx

    def with_default_adapter(self, adapter: BaseKernelAdapter) -> Profiler:
        self.adapter = adapter
        return self

    def _get_inputs(self, with_output=False, with_workspace=False):
        ins = []
        for i in range(len(self.params)):
            if (with_output or i not in self.result_idx) and (with_workspace or i not in self.workspace_idx):
                ins.append(self.supply(self.params[i]))
        return ins

    def _get_params(self, with_output=False, with_workspace=False):
        params = []
        for i in range(len(self.params)):
            if (with_output or i not in self.result_idx) and (with_workspace or i not in self.workspace_idx):
                params.append(self.params[i])
        return params

    def assert_allclose(
        self,
        reference_program: Callable,
        input_tensors: list[torch.Tensor] | None = None,
        atol: float = 1e-2,
        rtol: float = 1e-2,
        max_mismatched_ratio=0.01,
    ):
        """Validates kernel output against a reference implementation.

        Args:
            reference_program: Reference implementation to compare against
            input_tensors: Optional pre-generated input tensors
            atol: Absolute tolerance for comparison
            rtol: Relative tolerance for comparison
            max_mismatched_ratio: Maximum allowed ratio of mismatched elements
        """
        ins = self._get_inputs() if input_tensors is None else input_tensors
        ref_outs = reference_program(*ins)
        if hasattr(torch, "npu") and torch.npu.is_available():
            torch.npu.synchronize()
        elif torch.cuda.is_available():
            torch.cuda.synchronize()
        lib_outs = self.func(*ins)
        if hasattr(torch, "npu") and torch.npu.is_available():
            torch.npu.synchronize()
        elif torch.cuda.is_available():
            torch.cuda.synchronize()

        if isinstance(lib_outs, torch.Tensor):
            lib_outs = [lib_outs]
        elif lib_outs is None:
            lib_outs = []

        if isinstance(ref_outs, torch.Tensor):
            ref_outs = [ref_outs]
        elif ref_outs is None:
            ref_outs = []

        ref_tensors = ins + ref_outs
        lib_tensors = ins + lib_outs

        assert len(lib_tensors) == len(ref_tensors), "len(lib_tensors) not equals to len(ref_tensors) !"
        # torch.set_printoptions(edgeitems=torch.inf)
        for lhs, rhs in zip(lib_tensors, ref_tensors):
            # close_mask = torch.isclose(lhs, rhs, rtol=rtol, atol=atol)
            # total_elements = lhs.numel()
            # num_not_close = (~close_mask).sum().item()
            # percentage_not_close = (num_not_close / total_elements) * 100
            # print(f"{percentage_not_close:.2f}% of the elements are not close.")
            # print(f"Total elements: {total_elements}, Not close elements: {num_not_close}")
            torch_assert_close(
                lhs,
                rhs,
                rtol=rtol,
                atol=atol,
                max_mismatched_ratio=max_mismatched_ratio,
                base_name="tilelang",
                ref_name="ref",
            )

    def manual_assert_close(
        self,
        reference_program: Callable,
        input_tensors: list[torch.Tensor] | None = None,
        manual_check_prog: Callable = None,
    ):
        """Validates kernel output against a reference implementation.

        Args:
            reference_program: Reference implementation to compare against
            input_tensors: Optional pre-generated input tensors
            atol: Absolute tolerance for comparison
            rtol: Relative tolerance for comparison
            max_mismatched_ratio: Maximum allowed ratio of mismatched elements
        """
        ins = self._get_inputs() if input_tensors is None else input_tensors
        ref_outs = reference_program(*ins)
        if hasattr(torch, "npu") and torch.npu.is_available():
            torch.npu.synchronize()
        elif torch.cuda.is_available():
            torch.cuda.synchronize()
        lib_outs = self.func(*ins)
        if hasattr(torch, "npu") and torch.npu.is_available():
            torch.npu.synchronize()
        elif torch.cuda.is_available():
            torch.cuda.synchronize()

        if isinstance(lib_outs, torch.Tensor):
            lib_outs = [lib_outs]
        if isinstance(ref_outs, torch.Tensor):
            ref_outs = [ref_outs]
        elif ref_outs is None:
            ref_outs = []
        assert len(lib_outs) == len(ref_outs), f"{len(lib_outs)=} not equals to {len(ref_outs)=} !"
        torch.set_printoptions(edgeitems=torch.inf)
        manual_check_prog(lib_outs, ref_outs)

    def assert_consistent(self, repeat=10):
        """Check returned-output repeatability on the same inputs.

        Snapshot the first result so adapters may reuse output buffers. Later
        calls must return the same number and shapes of tensors, with values
        matching ``torch.allclose`` at its default tolerances. This is a
        repeatability check, not a proof that a kernel is free of races.

        Args:
            repeat: Number of calls to compare with the initial result.
        """
        ins = self._get_inputs()
        ref_outs = self.func(*ins)
        if isinstance(ref_outs, torch.Tensor):
            ref_outs = [ref_outs]
        ref_outs = [out.detach().clone() for out in ref_outs]

        for _ in range(repeat):
            lib_outs = self.func(*ins)
            if isinstance(lib_outs, torch.Tensor):
                lib_outs = [lib_outs]
            assert len(lib_outs) == len(ref_outs), f"Number of output tensors changed: {len(lib_outs)} != {len(ref_outs)}"
            for lhs, rhs in zip(lib_outs, ref_outs):
                assert lhs.shape == rhs.shape, f"Output tensor shape changed: {lhs.shape} != {rhs.shape}"
                assert torch.allclose(lhs, rhs), [
                    "result is not consistent",
                    lhs,
                    rhs,
                ]

    def run_once(self, func: Callable | None = None):
        ins = self._get_inputs()
        if not func:
            func = self.__call__
        return func(*ins)

    def determine_profiler(self, func: Callable | None = None):
        """Determines which profiler backend to use based on function type.

        Args:
            func: Function to be profiled
            profiler: Explicitly specified profiler type or "auto" for automatic detection

        Returns:
            str: The determined profiler type ("torch" or "tvm")
        """
        if isinstance(func, tvm.runtime.Module):
            return "tvm"
        else:
            return "torch"

    def do_bench(
        self,
        func: Callable | None = None,
        warmup: int = 25,
        rep: int = 100,
        n_warmup: int = 1,
        n_repeat: int = 1,
        input_tensors: list[torch.Tensor] = None,
    ) -> float:
        """Benchmarks the execution time of a given function.

        Args:
            func: Function to benchmark (uses adapter if None)
            warmup: Warmup time in milliseconds
            rep: Number of repetitions for timing
            n_warmup: Number of warmup iterations
            n_repeat: Number of timing iterations
            profiler: Which profiling backend to use
            input_tensors: Optional pre-generated input tensors

        Returns:
            float: Average execution time in milliseconds
        """
        profiler = self.determine_profiler(func)
        if profiler == "torch":
            if func is None:
                assert self.adapter is not None, "benchmarking function should be provided"
                func = self.adapter
            ins = self._get_inputs() if input_tensors is None else input_tensors
            bench_func = partial(func, *ins)
            return do_bench(
                bench_func,
                warmup=warmup,
                rep=rep,
                _n_warmup=n_warmup,
                _n_repeat=n_repeat,
            )
        elif profiler == "tvm":
            assert func is not None, "func should not be None"
            assert isinstance(func, tvm.runtime.Module), f"func should be a TVM module, but got {type(func)}"

            ins = self._get_inputs(with_output=True) if input_tensors is None else input_tensors
            target = "cuda"

            with suppress(Exception):
                target = self.mod.imported_modules[0].type_key

            assert target in ["cuda", "hip"], f"Unknown target: {target}"

            device = tvm.cuda(0) if target == "cuda" else tvm.rocm(0)
            time_evaluator = self.mod.time_evaluator(self.mod.entry_name, device, number=rep, repeat=n_repeat)
            tvm_inputs = [adapt_torch2tvm(inp) for inp in ins]
            # Transform Latency to ms
            return time_evaluator(*tvm_inputs).mean * 1e3
        else:
            raise ValueError(f"Unknown profiler: {profiler}")

    @property
    def func(self):
        assert self.adapter is not None, "adapter should be provided"
        return self.adapter

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.func(*args, **kwds)
