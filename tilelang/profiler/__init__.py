from typing import List, Optional, Callable, Any
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
from tilelang.profiler.bench import do_bench as npu_do_bench
from tilelang.profiler.bench import do_bench_npu as npu_do_bench_msprof
import os

@dataclass
class Profiler:
    """A profiler class for benchmarking and validating kernel implementations.
    
    Attributes:
        params: List of kernel parameters defining the input/output specifications
        result_idx: Indices indicating which parameters are output tensors
        supply_type: Type of tensor supply to use (e.g., random, zeros, etc.)
        adapter: Optional kernel adapter for interfacing with different backends
        direct_func: Optional direct callable function (bypasses adapter)
    """

    params: List[KernelParam]
    result_idx: List[int]
    supply_type: TensorSupplyType
    adapter: Optional[BaseKernelAdapter] = None
    direct_func: Optional[Callable] = None

    def __post_init__(self):
        """Initialize tensor supply after dataclass initialization"""
        self.result_idx = self._legalize_result_idx(self.result_idx)
        self.supply = get_tensor_supply(self.supply_type)

    def _legalize_result_idx(self, result_idx: Optional[List[int]] = None) -> List[int]:
        params = self.params
        # result_idx is a list of indices of the output tensors
        if result_idx is None:
            result_idx = []
        elif isinstance(result_idx, int):
            if result_idx > len(params) or result_idx < -len(params):
                raise ValueError(
                    f"result_idx should be an integer between {-len(params)} and {len(params) - 1}")
            if result_idx < 0:
                result_idx = len(params) + result_idx
            result_idx = [result_idx]
        elif not isinstance(result_idx, list):
            raise ValueError("result_idx should be a list of integers")
        return result_idx

    def with_default_adapter(self, adapter: BaseKernelAdapter) -> "Profiler":
        self.adapter = adapter
        return self
    
    def with_direct_func(self, func: Callable) -> "Profiler":
        self.direct_func = func
        return self

    def _get_inputs(self, with_output=False):
        ins = []
        for i in range(len(self.params)):
            if with_output or i not in self.result_idx:
                ins.append(self.supply(self.params[i]))
        return ins

    def _get_params(self, with_output=False):
        
        params = []
        for i in range(len(self.params)):
            if with_output or i not in self.result_idx:
                params.append(self.params[i])
                
        return params

    def assert_allclose(
        self,
        reference_program: Callable,
        input_tensors: Optional[List[torch.Tensor]] = None,
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
        if hasattr(torch, 'npu') and torch.npu.is_available():
            torch.npu.synchronize()
        elif torch.cuda.is_available():
            torch.cuda.synchronize()
        lib_outs = self.func(*ins)
        if hasattr(torch, 'npu') and torch.npu.is_available():
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

        assert len(lib_tensors) == len(
            ref_tensors), "len(lib_tensors) not equals to len(ref_tensors) !"

        for lhs, rhs in zip(lib_tensors, ref_tensors):

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
        input_tensors: Optional[List[torch.Tensor]] = None,
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
        if hasattr(torch, 'npu') and torch.npu.is_available():
            torch.npu.synchronize()
        elif torch.cuda.is_available():
            torch.cuda.synchronize()
        lib_outs = self.func(*ins)
        if hasattr(torch, 'npu') and torch.npu.is_available():
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
        """Checks for kernel consistency across multiple runs.
        
        Args:
            repeat: Number of times to repeat the consistency check
        """
        # Used to check no race condition inside the kernel
        ins = self._get_inputs()
        ref_outs = self.func(*ins)

        for _ in range(repeat):
            lib_outs = self.func(*ins)
            for lhs, rhs in zip(lib_outs, ref_outs):
                assert torch.allclose(lhs, rhs), [
                    "result is not consistent",
                    lhs,
                    rhs,
                ]

    def run_once(self, func: Optional[Callable] = None):
        ins = self._get_inputs()
        if not func:
            func = self.__call__
        return func(*ins)

    def determine_profiler(self, func: Optional[Callable] = None):
        """Determines which profiler backend to use based on function type.
        
        Args:
            func: Function to be profiled
            profiler: Explicitly specified profiler type or "auto" for automatic detection
        
        Returns:
            str: The determined profiler type ("torch", "tvm", or "npu")
        """
        if func is not None:
            if hasattr(func, '__class__') and func.__class__.__name__ == 'JitKernel_NPU':
                return "npu"
            elif isinstance(func, tvm.runtime.Module):
                return "tvm"
            else:
                return "torch"
        else:
            if self.direct_func is not None:
                if hasattr(self.direct_func, '__class__') and self.direct_func.__class__.__name__ == 'JitKernel_NPU':
                    return "npu"
                else:
                    return "torch"
            elif self.adapter is not None:
                return "torch"
            else:
                raise ValueError("No function or adapter provided")

    def do_bench(
        self,
        func: Optional[Callable] = None,
        warmup: int = 25,
        rep: int = 100,
        n_warmup: int = 1,
        n_repeat: int = 1,
        input_tensors: List[torch.Tensor] = None,
    ) -> float:
        """Benchmarks the execution time of a given function.
        
        Args:
            func: Function to benchmark (uses direct_func or adapter if None)
            warmup: Warmup time in milliseconds
            rep: Number of repetitions for timing
            n_warmup: Number of warmup iterations
            n_repeat: Number of timing iterations
            input_tensors: Optional pre-generated input tensors
            
        Returns:
            float: Average execution time in milliseconds
        """
        profiler = self.determine_profiler(func)
        
        if profiler == "npu":
           
            if func is None:
                if self.direct_func is not None:
                    func = self.direct_func
                else:
                    raise ValueError("No function provided for benchmarking")
            
            ins = self._get_inputs() if input_tensors is None else input_tensors
     
            bench_func = partial(func, *ins)
            
            use_profiling = os.getenv("TILELANG_BENCH_METHOD", "default").lower() == "npu"
            if use_profiling:
                return  npu_do_bench_msprof(bench_func, warmup=warmup, rep=rep)
            
            return npu_do_bench(
                bench_func,
                warmup=warmup,
                rep=rep,
                _n_warmup=n_warmup,
                _n_repeat=n_repeat,
            )
        
        elif profiler == "torch":
            if func is None:
                if self.direct_func is not None:
                    func = self.direct_func
                elif self.adapter is not None:
                    func = self.adapter
                else:
                    raise ValueError("No function provided for benchmarking")
            
            ins = self._get_inputs() if input_tensors is None else input_tensors
            bench_func = partial(func, *ins)
            return npu_do_bench(
                bench_func,
                warmup=warmup,
                rep=rep,
                _n_warmup=n_warmup,
                _n_repeat=n_repeat,
            )
        
        elif profiler == "tvm":
            assert func is not None, "func should not be None"
            assert isinstance(
                func, tvm.runtime.Module), f"func should be a TVM module, but got {type(func)}"

            ins = (self._get_inputs(with_output=True) if input_tensors is None else input_tensors)
            target = "cuda"

            with suppress(Exception):
                target = self.mod.imported_modules[0].type_key

            assert target in ["cuda", "hip"], f"Unknown target: {target}"

            device = tvm.cuda(0) if target == "cuda" else tvm.rocm(0)
            time_evaluator = self.mod.time_evaluator(
                self.mod.entry_name, device, number=rep, repeat=n_repeat)
            tvm_inputs = [adapt_torch2tvm(inp) for inp in ins]
            # Transform Latency to ms
            return time_evaluator(*tvm_inputs).mean * 1e3
        else:
            raise ValueError(f"Unknown profiler: {profiler}")

    @property
    def func(self):
        if self.direct_func is not None:
            return self.direct_func
        elif self.adapter is not None:
            return self.adapter
        else:
            raise ValueError("No available execution function")

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.func(*args, **kwds)
