# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""
This module provides an auto-tuning infrastructure for TileLang (tl) programs.
It includes functionality to JIT-compile TileLang programs into a runnable
kernel adapter using TVM.
"""

from typing import (
    Any,
    List,
    Union,
    Callable,
    Tuple,
    overload,
    Literal,
    Dict,
    Optional,
)
from tilelang import tvm as tvm
from tvm.tir import PrimFunc
from tvm.target import Target

from tilelang.jit.kernel import JITKernel
from tilelang.cache import cached, cached_npu
from os import path, makedirs
from logging import getLogger
import functools
import inspect
from tilelang.jit.param import Kernel, _P, _RProg

logger = getLogger(__name__)


def compile(
    func: PrimFunc = None,
    out_idx: Union[List[int], int, None] = None,
    execution_backend: Literal["dlpack", "ctypes", "cython"] = "cython",
    target: Union[str, Target] = "auto",
    target_host: Union[str, Target] = None,
    verbose: bool = False,
    pass_configs: Optional[Dict[str, Any]] = None,
) -> JITKernel:
    """
    Compile the given TileLang PrimFunc with TVM and build a JITKernel.

    When ``target == 'npuir'`` the result is cached on disk (and in memory)
    using the same ``KernelCache`` infrastructure as the autotune path.  A
    subsequent call with identical arguments will skip compilation entirely and
    return the restored ``JitKernel_NPU`` directly.

    Parameters
    ----------
    func : tvm.tir.PrimFunc, optional
        The TileLang TIR function to compile and wrap.
    out_idx : Union[List[int], int], optional
        Index(es) of the output tensors to return (default: None).
    execution_backend : Literal["dlpack", "ctypes", "cython"], optional
        Execution backend to use for kernel execution (default: "cython").
    target : Union[str, Target], optional
        Compilation target (default: "auto").
    target_host : Union[str, Target], optional
        Target host for cross-compilation (default: None).
    verbose : bool, optional
        Whether to enable verbose output (default: False).
    pass_configs : dict, optional
        Additional keyword arguments to pass to the Compiler PassContext.
    """
    if target == "npuir":
        return cached_npu(
            func=func,
            out_idx=out_idx,
            execution_backend="None",
            target=target,
            target_host=target_host,
            verbose=True,
            pass_configs=pass_configs,
        )

    # --- GPU / other targets ------------------------------------------
    return cached(
        func=func,
        out_idx=out_idx,
        execution_backend=execution_backend,
        target=target,
        target_host=target_host,
        verbose=verbose,
        pass_configs=pass_configs,
    )


class _JitImplementation:
    out_idx: Any
    target: Union[str, Target]
    target_host: Union[str, Target]
    execution_backend: Literal["dlpack", "ctypes", "cython"]
    verbose: bool
    pass_configs: Optional[Dict[str, Any]]
    debug_root_path: Optional[str]
    func: Optional[Callable] = None
    signature: Optional[Any] = None
    wrapper: Optional[Callable] = None

    def __init__(
        self,
        out_idx: Any = None,
        target: Union[str, Target] = "auto",
        target_host: Union[str, Target] = None,
        execution_backend: Literal["dlpack", "ctypes", "cython"] = "cython",
        verbose: bool = False,
        pass_configs: Optional[Dict[str, Any]] = None,
        debug_root_path: Optional[str] = None,
    ):
        self.out_idx = out_idx
        self.execution_backend = execution_backend
        self.target = target
        self.target_host = target_host
        self.verbose = verbose
        self.pass_configs = pass_configs
        self.func = None
        self.signature = None

        self.debug_root_path = debug_root_path
        if self.debug_root_path is not None and not path.isabs(self.debug_root_path):
            try:
                base_path = path.dirname(path.dirname(path.dirname(__file__)))
                self.debug_root_path = path.join(base_path, self.debug_root_path)
            except NameError:
                self.debug_root_path = path.abspath(self.debug_root_path)

        self._kernel_cache: Dict[tuple, Kernel] = {}

    @overload
    def __call__(
        self, func: Callable[_P, _RProg]
    ) -> Callable[_P, Tuple[_RProg, Kernel]]: ...

    @overload
    def __call__(self, func: Callable[_P, _RProg]) -> Callable[_P, Kernel]: ...

    def __call__(self, func: Callable[_P, _RProg]) -> Callable[_P, Any]:
        self.func = func
        self.signature = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> Any:
            tune_params = kwargs.pop("__tune_params", {})

            key_args_tuple = args
            key_kwargs_tuple = tuple(sorted(kwargs.items()))
            key = (key_args_tuple, key_kwargs_tuple)

            if key not in self._kernel_cache:
                program_result_source = func
                if isinstance(program_result_source, PrimFunc):
                    program_result = program_result_source
                elif callable(program_result_source):
                    program_result = program_result_source(
                        *args, **kwargs, **tune_params
                    )
                else:
                    raise ValueError(
                        f"Invalid function type: {type(program_result_source)}"
                    )

                kernel_result = compile(
                    program_result,
                    out_idx=self.out_idx,
                    execution_backend=self.execution_backend,
                    target=self.target,
                    target_host=self.target_host,
                    verbose=self.verbose,
                    pass_configs=self.pass_configs,
                )

                if self.debug_root_path:
                    func_name = getattr(func, "__name__", "jit_kernel")
                    kernel_file = f"tilelang_jit_kernel_{func_name}.c"
                    program_file = f"tilelang_jit_program_{func_name}.py"
                    makedirs(self.debug_root_path, exist_ok=True)
                    with open(path.join(self.debug_root_path, kernel_file), "w") as f:
                        print(kernel_result.get_kernel_source(), file=f)
                    with open(path.join(self.debug_root_path, program_file), "w") as f:
                        print(program_result.script(), file=f)

                self._kernel_cache[key] = kernel_result

            return self._kernel_cache[key]

        wrapper.__jit_impl__ = self
        self.wrapper = wrapper
        return wrapper


def jit(
    func: Union[Callable[_P, _RProg], PrimFunc, None] = None,
    *,
    out_idx: Any = None,
    target: Union[str, Target] = "auto",
    target_host: Union[str, Target] = None,
    execution_backend: Literal["dlpack", "ctypes", "cython"] = "cython",
    verbose: bool = False,
    pass_configs: Optional[Dict[str, Any]] = None,
    debug_root_path: Optional[str] = None,
):
    """
    Just-In-Time (JIT) compiler decorator for TileLang functions.
    """
    if callable(func):
        default_decorator = _JitImplementation(
            out_idx=out_idx,
            target=target,
            target_host=target_host,
            execution_backend=execution_backend,
            verbose=verbose,
            pass_configs=pass_configs,
            debug_root_path=debug_root_path,
        )
        return default_decorator(func)
    elif isinstance(func, PrimFunc):
        raise ValueError("Use tilelang.jit to decorate prim_func is not supported yet.")
    else:
        configured_decorator = _JitImplementation(
            out_idx=out_idx,
            target=target,
            target_host=target_host,
            execution_backend=execution_backend,
            verbose=verbose,
            pass_configs=pass_configs,
            debug_root_path=debug_root_path,
        )
        return configured_decorator
