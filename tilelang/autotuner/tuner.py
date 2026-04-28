"""The auto-tune module for tilelang programs.

This module provides functionality for auto-tuning tilelang programs, including JIT compilation
and performance optimization through configuration search.
"""

from __future__ import annotations
from dataclasses import dataclass

import tilelang
from tilelang import tvm as tvm
from tilelang.jit.jit_npu import JitKernel_NPU
from tilelang.version import __version__  # Import early to avoid circular import
from tvm.tir import PrimFunc, Var
from tvm.target import Target
import inspect
from functools import partial
from typing import Callable, Generic, Literal, Any, TypeVar, TYPE_CHECKING

# Python 3.9 compatibility for ParamSpec
try:
    from typing import ParamSpec
except ImportError:  # Python < 3.10
    from typing_extensions import ParamSpec

if TYPE_CHECKING:
    from tilelang.jit import _JitImplementation as JITImpl
from tqdm.auto import tqdm
import logging
import concurrent.futures
import torch
import os
import sys
import signal
import json
import hashlib
import threading
import traceback
from pathlib import Path

from tilelang import env
from tilelang.autotuner.param import (
    CompileArgs,
    ProfileArgs,
    AutotuneResult,
)

from tilelang.utils.target import determine_target
import time


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Operation timed out")


def run_with_timeout(func, timeout, *args, **kwargs):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    try:
        result = func(*args, **kwargs)
    except Exception as e:
        raise e
    finally:
        signal.alarm(0)
    return result


# Configure logging for the autotuner module
# TODO: Consider creating a common logger in utils
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.propagate = False

# Lazy handler initialization flag
_logger_handlers_initialized = False


class _LazyStreamHandler(logging.StreamHandler):
    """Resolves sys.stdout at emit time so pytest/IPython captures work."""

    def emit(self, record):
        self.stream = sys.stdout
        super().emit(record)


def _init_logger_handlers():
    global _logger_handlers_initialized
    if _logger_handlers_initialized:
        return
    formatter = logging.Formatter("%(asctime)s %(levelname)s:%(message)s")
    file_handler = logging.FileHandler("autotuner.log", mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    console_handler = _LazyStreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    _logger_handlers_initialized = True


def get_available_cpu_count() -> int:
    """Gets the number of CPU cores available to the current process."""
    try:
        cpu_count = len(os.sched_getaffinity(0))
    except AttributeError:
        cpu_count = os.cpu_count()

    return cpu_count or 1


def _normalize_param(value: Any) -> Any:
    """Recursively normalize a parameter value for stable JSON serialisation."""
    if isinstance(value, Var):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_normalize_param(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _normalize_param(v) for k, v in value.items()}
    return value


class AutoTuner:
    """Auto-tuner for tilelang programs.

    This class handles the auto-tuning process by testing different configurations
    and finding the optimal parameters for program execution.

    Args:
        fn: The function to be auto-tuned.
        configs: List of configurations to try during auto-tuning.
    """

    compile_args = CompileArgs()
    profile_args = ProfileArgs()

    _kernel_parameters: tuple[str, ...] | None = None
    _function_parameters: dict[str, Any] | None = None
    _lock = threading.Lock()  # For thread safety
    _memory_cache = {}  # In-memory cache dictionary
    cache_dir: Path = Path(env.TILELANG_CACHE_DIR)

    def __init__(self, fn: Callable, configs):
        self.fn = fn
        self.configs = configs
        self.ref_latency_cache = None
        self.jit_input_tensors = None
        self.ref_input_tensors = None
        self.jit_compile = None

    @classmethod
    def from_kernel(cls, kernel: Callable, configs):
        """Create an AutoTuner instance from a kernel function.

        Args:
            kernel: The kernel function to auto-tune.
            configs: List of configurations to try.

        Returns:
            AutoTuner: A new AutoTuner instance.
        """
        return cls(kernel, configs)

    def set_compile_args(
        self,
        out_idx: list[int] | int | None = None,
        target: Literal["auto", "cuda", "hip"] = "auto",
        execution_backend: Literal["dlpack", "ctypes", "cython"] = "cython",
        target_host: str | Target = None,
        verbose: bool = False,
        pass_configs: dict[str, Any] | None = None,
    ):
        """Set compilation arguments for the auto-tuner.

        Args:
            out_idx: List of output tensor indices.
            target: Target platform.
            execution_backend: Execution backend to use for kernel execution.
            target_host: Target host for cross-compilation.
            verbose: Whether to enable verbose output.
            pass_configs: Additional keyword arguments to pass to the Compiler PassContext.

        Returns:
            AutoTuner: Self for method chaining.
        """
        self.compile_args = CompileArgs(
            out_idx=out_idx,
            target=Target(determine_target(target)),
            execution_backend=execution_backend,
            target_host=target_host,
            verbose=verbose,
            pass_configs=pass_configs,
        )

        return self

    def set_profile_args(
        self,
        warmup: int = 25,
        rep: int = 100,
        timeout: int = 30,
        supply_type: tilelang.TensorSupplyType = tilelang.TensorSupplyType.Auto,
        ref_prog: Callable | None = None,
        supply_prog: Callable | None = None,
        rtol: float = 1e-2,
        atol: float = 1e-2,
        max_mismatched_ratio: float = 0.01,
        skip_check: bool = False,
        manual_check_prog: Callable | None = None,
        cache_input_tensors: bool = False,
    ):
        """Set profiling arguments for the auto-tuner.

        Args:
            supply_type: Type of tensor supply mechanism. Ignored if `supply_prog` is provided.
            ref_prog: Reference program for validation.
            supply_prog: Supply program for input tensors.
            rtol: Relative tolerance for validation.
            atol: Absolute tolerance for validation.
            max_mismatched_ratio: Maximum allowed mismatch ratio.
            skip_check: Whether to skip validation.
            manual_check_prog: Manual check program for validation.
            cache_input_tensors: Whether to cache input tensors.
            warmup: Number of warmup iterations.
            rep: Number of repetitions for timing.
            timeout: Maximum time per configuration.

        Returns:
            AutoTuner: Self for method chaining.
        """
        self.profile_args = ProfileArgs(
            supply_type=supply_type,
            ref_prog=ref_prog,
            supply_prog=supply_prog,
            rtol=rtol,
            atol=atol,
            max_mismatched_ratio=max_mismatched_ratio,
            skip_check=skip_check,
            manual_check_prog=manual_check_prog,
            cache_input_tensors=cache_input_tensors,
            warmup=warmup,
            rep=rep,
            timeout=timeout,
        )

        # If a custom `supply_prog` is provided, the profiler's `supply_type` setting
        # becomes ineffective. The custom supply program will be used instead.
        if supply_prog is not None and supply_type != tilelang.TensorSupplyType.Auto:
            logger.warning(
                "Ignoring `supply_type` passed to `set_profile_args` because "
                "`supply_prog` is not None."
            )

        return self

    def set_kernel_parameters(
        self, k_parameters: tuple[str, ...], f_parameters: dict[str, Any]
    ):
        # for cache key generation
        self._kernel_parameters = k_parameters
        self._function_parameters = f_parameters

    def generate_cache_key(self, parameters: dict[str, Any]) -> AutotuneResult | None:
        """Generate a cache key for the auto-tuning process."""

        # extract parameters from the function signature
        op_parameters = []
        for _, default_value in parameters.items():
            if default_value.default is not inspect.Parameter.empty:
                op_parameters.append(default_value.default)

        if self._kernel_parameters is not None:
            op_parameters += _normalize_param(self._kernel_parameters)

        key_data = {
            "version": __version__,
            "op_parameters": tuple(op_parameters),
            "func_source": inspect.getsource(self.fn),
            "configs": self.configs,
            "compile_args": hash(self.compile_args),
            "profile_args": hash(self.profile_args),
        }
        # Sort keys to ensure consistency
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()

    def _check_cache(self, key: str) -> AutotuneResult | None:
        """Check memory cache then disk cache. Returns a result or None."""
        if not env.is_cache_enabled():
            return None
        with self._lock:
            if key in self._memory_cache:
                logger.warning(
                    "Found kernel in memory cache. For better performance,"
                    " consider using `@tilelang.autotune` instead of direct"
                    " AutoTuner.from_kernel."
                )
                return self._memory_cache[key]
            result = self._load_result_from_disk(key)
            if result is not None:
                self._memory_cache[key] = result
                logger.warning("Found kernel in disk cache.")
                return result
        return None

    def _store_cache(self, key: str, result: AutotuneResult) -> None:
        """Persist result to memory and (if the backend supports it) disk."""
        if self.compile_args.execution_backend in ("dlpack", "torch"):
            logger.warning("DLPack backend does not support cache saving to disk.")
        else:
            with self._lock:
                if env.is_cache_enabled():
                    self._save_result_to_disk(key, result)
        self._memory_cache[key] = result

    def _device_wrapper(
        self, func: Callable, device_type: str, device: int
    ) -> Callable:
        """Return a wrapper that pins the given device before calling *func*."""
        if device_type == "cuda":

            def inner(**config_arg):
                torch.cuda.set_device(device)
                return func(**config_arg)
        else:  # npu

            def inner(**config_arg):
                torch.npu.set_device(device)
                return func(**config_arg)

        return inner

    def _compile_kernel(self, **config_arg) -> tilelang.JitKernel_NPU:
        return self.compile_args.compile_program(self.fn(**config_arg))

    def _build_config_args(self, parameters: dict[str, Any]) -> list[dict[str, Any]]:
        """Convert self.configs into a list of per-run keyword dicts."""
        config_args = []
        for config in self.configs:
            new_kwargs = {name: config[name] for name in parameters if name in config}
            unused = set(config.keys()) - set(new_kwargs.keys())
            if unused:
                raise ValueError(f"Unused keys in config: {unused}")
            config_args.append(new_kwargs)
        if not config_args:
            raise ValueError(
                "No configurations to tune. Please check your `@autotune` decorator."
            )
        return config_args

    def _should_skip_tuning(
        self,
        top_config: dict[str, Any],
        parameters: dict[str, Any],
    ) -> bool:
        """Return True when all tunable params are already provided at call-site."""
        if self._kernel_parameters is None:
            return False
        key_args_tuple, key_kwargs_tuple = self._kernel_parameters
        tunable_arguments = list(top_config.keys())

        def check_tunable_argument_value(key, parameters, key_args_tuple) -> bool:
            params_list = list(parameters.keys())
            assert key in params_list, (
                f"Tunable argument {key} not found in function parameters"
            )
            return params_list.index(key) < len(key_args_tuple)

        # Check if all tunable arguments have been tuned by comparing config keys with key_kwargs_tuple
        return any(key in top_config for key, _ in key_kwargs_tuple) or any(
            check_tunable_argument_value(key, self._function_parameters, key_args_tuple)
            for key in tunable_arguments
        )

    def _resolve_compile_func(self) -> Callable:
        """Wrap jit_compile with a device-pin if a GPU/NPU is active."""
        func = self.jit_compile
        if torch.cuda.is_available():
            return self._device_wrapper(func, "cuda", torch.cuda.current_device())
        if hasattr(torch, "npu") and torch.npu.is_available():
            return self._device_wrapper(func, "npu", torch.npu.current_device())
        return func

    def _determine_num_workers(self) -> int:
        available = get_available_cpu_count()
        cpu_counts = int(env.TILELANG_AUTO_TUNING_CPU_COUNTS)
        max_cpu_count = int(env.TILELANG_AUTO_TUNING_MAX_CPU_COUNT)
        utilization = float(env.TILELANG_AUTO_TUNING_CPU_UTILITIES)

        if cpu_counts > 0:
            num_workers = min(cpu_counts, available)
            logger.info(
                f"Auto-tuning with {cpu_counts} CPU counts,"
                f" {available} CPUs available, {num_workers} will be used."
            )
        else:
            num_workers = max(1, int(available * utilization))
            logger.info(
                f"Auto-tuning with {utilization:.0%} CPU utilisation,"
                f" {available} CPUs available, {num_workers} will be used."
            )

        if 0 < max_cpu_count < num_workers:
            logger.warning(
                f"Capping workers from {num_workers} to max_cpu_count={max_cpu_count}."
            )
            num_workers = max_cpu_count
        return num_workers

    def _compile_all(
        self,
        config_args: list[dict[str, Any]],
        pool: concurrent.futures.ThreadPoolExecutor,
    ) -> list[tuple[tilelang.JitKernel_NPU, dict[str, Any]]]:
        """Submit all compile jobs and collect successful (kernel, config) pairs."""
        compile_func = self._resolve_compile_func()
        future_to_index = {
            pool.submit(compile_func, **cfg): i for i, cfg in enumerate(config_args)
        }
        results = []
        for future in tqdm(
            concurrent.futures.as_completed(future_to_index),
            total=len(future_to_index),
            desc="Compiling configurations",
        ):
            idx = future_to_index[future]
            config = config_args[idx]
            try:
                results.append((future.result(), config))
            except Exception as exc:
                logger.debug(
                    f"Compilation failed for config {config} at index {idx}: {exc}"
                )
        return results

    def _save_result_to_disk(self, key, result: AutotuneResult):
        from tilelang.cache.kernel_cache import KernelCache  # lazy — breaks cycle

        # KernelCache.save(self.cache_dir / key, result, self.compile_args.verbose)
        KernelCache().save_autotune_result(key, result, self.compile_args.verbose)

    def _load_result_from_disk(self, key) -> AutotuneResult:
        from tilelang.cache.kernel_cache import KernelCache  # lazy — breaks cycle

        # return KernelCache.load(self.cache_dir / key, self.compile_args)
        return KernelCache().load_autotune_result(
            key,
            out_idx=self.compile_args.out_idx,
            verbose=self.compile_args.verbose,
        )

    def _get_input_tensors(
        self,
        profiler,
        supply_prog: Callable | None,
        with_output: bool = False,
        config: dict = None,
    ):
        """Return input tensors from supply_prog or the profiler default."""
        params = profiler._get_params(with_output=with_output)
        if supply_prog is not None:
            fn = supply_prog
            params = profiler._get_params(with_output=False)
            if "config" in inspect.signature(fn).parameters:
                input_tensors = fn(params, config=config)
            else:
                input_tensors = fn(params)
            return input_tensors
        return profiler._get_inputs(with_output=with_output)

    def _maybe_refresh_input_tensors(
        self, profiler, supply_prog: Callable | None, config: dict
    ) -> None:
        """Populate or validate self.jit_input_tensors for caching mode."""
        if self.jit_input_tensors is None:
            self.jit_input_tensors = self._get_input_tensors(
                profiler, supply_prog, config=config
            )
            return

        params = profiler._get_params(with_output=False)
        if len(params) != len(self.jit_input_tensors):
            raise ValueError("len(params) != len(self.jit_input_tensors)")
        for p, c in zip(params, self.jit_input_tensors, strict=True):
            if not isinstance(c, torch.Tensor):
                continue
            shape_ok = all(
                a == b or isinstance(a, Var) or isinstance(b, Var)
                for a, b in zip(p.shape, c.shape, strict=True)
            )
            if p.dtype != c.dtype or not shape_ok:
                logger.warning(
                    "Incompatible cached input tensors detected — regenerating. "
                    "Set `cache_input_tensors=False` to avoid this warning."
                )
                self.jit_input_tensors = self._get_input_tensors(
                    profiler, supply_prog, config=config
                )
                break

    def _check_correctness(self, profiler, ref_prog: Callable, input_tensors) -> None:
        """Run the correctness check against ref_prog."""
        pa = self.profile_args
        if pa.manual_check_prog is not None:
            profiler.manual_assert_close(
                ref_prog,
                input_tensors=input_tensors,
                manual_check_prog=pa.manual_check_prog,
            )
        else:
            profiler.assert_allclose(
                ref_prog,
                input_tensors=input_tensors,
                rtol=pa.rtol,
                atol=pa.atol,
                max_mismatched_ratio=pa.max_mismatched_ratio,
            )

    def _measure_latency(
        self,
        jit_kernel: tilelang.JitKernel_NPU,
        warmup: int,
        rep: int,
        config: dict,
    ) -> tuple[float, float | None]:
        """Profile *jit_kernel* and optionally the reference program.

        Returns:
            (latency, ref_latency) — ref_latency is None when no ref_prog is set.
        """
        pa = self.profile_args
        profiler = jit_kernel.get_profiler(tensor_supply_type=pa.supply_type)

        if pa.cache_input_tensors:
            self._maybe_refresh_input_tensors(profiler, pa.supply_prog, config)
        else:
            self.jit_input_tensors = self._get_input_tensors(
                profiler, pa.supply_prog, config=config
            )

        if not pa.skip_check and pa.ref_prog is not None:
            self._check_correctness(profiler, pa.ref_prog, self.jit_input_tensors)

        latency = profiler.do_bench(
            warmup=warmup, rep=rep, input_tensors=self.jit_input_tensors
        )

        if self.ref_latency_cache is None and pa.ref_prog is not None:
            self.ref_input_tensors = self._get_input_tensors(
                profiler, pa.supply_prog, config=config
            )
            self.ref_latency_cache = profiler.do_bench(
                pa.ref_prog,
                n_warmup=warmup,
                n_repeat=rep,
                input_tensors=self.ref_input_tensors,
            )

        return latency, self.ref_latency_cache

    # Retained as the public entry-point for the signal-based timeout path.
    def target_fn(
        self,
        jit_kernel: tilelang.JitKernel_NPU,
        warmup: int,
        rep: int,
        config: dict = None,
    ):
        return self._measure_latency(jit_kernel, warmup, rep, config)

    def run(self, warmup: int = 5, rep: int = 30, timeout: int = 30):
        """Run the auto-tuning process.

        Args:
            warmup: Number of warmup iterations.
            rep: Number of repetitions for timing.
            timeout: Maximum time per configuration.

        Returns:
            AutotuneResult: Results of the auto-tuning process.
        """
        _init_logger_handlers()
        start_time = time.time()

        sig = inspect.signature(self.fn)
        parameters = sig.parameters

        if isinstance(self.configs, Callable):
            self.configs = self.configs(*self._kernel_parameters)

        key = self.generate_cache_key(parameters)

        cached = self._check_cache(key)

        if cached is not None:
            return cached

        best_latency: float = 1e8
        best_config: dict[str, Any] | None = None
        best_kernel: tilelang.JitKernel_NPU | None = None

        if self.jit_compile is None:
            self.jit_compile = self._compile_kernel

        config_args = self._build_config_args(parameters)

        if len(config_args) == 0:
            raise ValueError(
                "No configurations to tune, please check your `@autotune` decorator"
            )

        # check if the tunable arguments has been set.
        # get the back config argument
        top_config, *rest = config_args

        if self._kernel_parameters is not None:
            key_args_tuple, key_kwargs_tuple = self._kernel_parameters
            tunable_arguments = [key for key, _ in top_config.items()]

            if self._should_skip_tuning(top_config, parameters):
                logger.warning(
                    f"Tunable parameters {tunable_arguments} already provided during auto-tuning. Skipping compilation and using direct JIT"
                )
                # compile the kernel with the provided parameters
                jit_kernel = self.jit_compile()
                autotuner_result = AutotuneResult(
                    libcode=jit_kernel.get_kernel_source(),
                    func=jit_kernel.prim_func,
                    kernel=jit_kernel,
                )
                self._memory_cache[key] = autotuner_result
                return autotuner_result

        num_workers = self._determine_num_workers()
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=num_workers)
        futures = []
        future_to_index = {}

        for i, config_arg in enumerate(config_args):
            compile_func = self._resolve_compile_func()

            future = pool.submit(
                compile_func,
                **config_arg,
            )
            futures.append(future)
            future_to_index[future] = i

        results_with_configs = []
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Compiling configurations",
        ):
            idx = future_to_index[future]
            config = config_args[idx]
            try:
                result = future.result()
                results_with_configs.append((result, config))
            except Exception as e:
                logger.debug(
                    f"Compilation failed for config {config} at index {idx} with error: {e}"
                )
                continue

        ref_latency = None
        progress_bar = tqdm(
            range(len(results_with_configs)), desc="Bench configurations"
        )
        use_profiling = os.getenv("TILELANG_BENCH_METHOD", "default").lower() == "npu"
        if use_profiling:
            funcs = []
            configs = []
            jit_kernels = []

            for i in progress_bar:
                jit_kernel, config = results_with_configs[i]
                profile_args = self.profile_args
                supply_type = profile_args.supply_type
                profiler = jit_kernel.get_profiler(tensor_supply_type=supply_type)

                if profile_args.supply_prog is not None:
                    fn = profile_args.supply_prog
                    params = profiler._get_params(with_output=False)
                    if "config" in inspect.signature(fn).parameters:
                        input_tensors = fn(params, config=config)
                    else:
                        input_tensors = fn(params)
                else:
                    input_tensors = profiler._get_inputs(with_output=False)

                ins = self._get_inputs() if input_tensors is None else input_tensors

                if not profile_args.skip_check and profile_args.ref_prog is not None:
                    try:
                        self._check_correctness(profiler, profile_args.ref_prog, ins)
                    except Exception:
                        logger.warning(
                            f"Correctness check failed for config {config}, skipping. "
                            "See autotuner.log for details."
                        )
                        logger.debug(traceback.format_exc())
                        continue  # drop this config; don't add it to funcs/configs/kernels

                if self.ref_latency_cache is None and profile_args.ref_prog is not None:
                    ref_input_tensors = self._get_input_tensors(
                        profiler, profile_args.supply_prog, config=config
                    )
                    self.ref_latency_cache = profiler.do_bench(
                        profile_args.ref_prog,
                        n_warmup=warmup,
                        n_repeat=rep,
                        input_tensors=ref_input_tensors,
                    )

                bench_func = partial(jit_kernel, *ins)
                funcs.append(bench_func)
                configs.append(config)
                jit_kernels.append(jit_kernel)

            try:
                from ..profiler.bench import do_bench_npu

                latencies = do_bench_npu(
                    funcs,
                )
            except Exception:
                logger.warning(
                    "An error occurred while benchmarking configs, checkout autotuner.log for more details"
                )
                logger.debug(f"Error: {traceback.format_exc()}")
                latencies = [float("inf")] * len(funcs)

            def ensure_list(x):
                if isinstance(x, (list, tuple)):
                    return x
                return [x]

            for latency, config, kernel in zip(
                ensure_list(latencies),
                ensure_list(configs),
                ensure_list(jit_kernels),
                strict=True,
            ):
                tqdm.write(f"Tuned Latency {latency} with config {config}")
                if latency < best_latency:
                    best_latency = latency
                    best_config = config
                    best_kernel = kernel
        else:
            for i in progress_bar:
                jit_kernel, config = results_with_configs[i]
                try:
                    # Cannot ThreadPoolExecutor to enforce timeout on target_fn execution
                    # Because tma init may behave strangely with one thread
                    # latency, ref_latency = target_fn(jit_kernel)
                    latency, ref_latency = run_with_timeout(
                        self.target_fn, timeout, jit_kernel, warmup, rep, config
                    )
                except TimeoutException:
                    logger.warning(
                        f"A timeout occurred while testing config {config}, checkout autotuner.log for more details"
                    )
                    continue
                except Exception:
                    logger.warning(
                        f"An error occurred while testing config {config}, checkout autotuner.log for more details"
                    )
                    logger.debug(f"Error: {traceback.format_exc()}")
                    continue
                tqdm.write(f"Tuned Latency {latency} with config {config} at index {i}")
                if latency < best_latency:
                    best_latency = latency
                    best_config = config
                    best_kernel = jit_kernel

            progress_bar.set_postfix({"best_latency": best_latency})

        pool.shutdown()

        if best_kernel is None:
            raise RuntimeError(
                "Auto-tuning failed: no configuration successfully compiled "
                "and passed benchmarking/validation."
            )

        ref_latency = self.ref_latency_cache
        best_kernel = best_kernel.update_tuner_result(
            latency=best_latency,
            config=best_config,
            ref_latency=ref_latency,
        )
        result = AutotuneResult(
            latency=best_latency,
            config=best_config,
            ref_latency=ref_latency,
            libcode=best_kernel.get_kernel_source(),
            func=best_kernel.prim_func,
            kernel=best_kernel,
        )

        self._store_cache(key, result)
        logger.info(f"Auto-tuning finished in {time.time() - start_time:.2f}s")
        return result

    def __call__(self) -> Any:
        """Make the AutoTuner callable, running the auto-tuning process.

        Returns:
            AutotuneResult: Results of the auto-tuning process.
        """
        return self.run()


_P = ParamSpec("_P")
_T = TypeVar("_T")


@dataclass
class AutoTuneImpl(Generic[_P, _T]):
    jit_impl: JITImpl

    warmup: int
    rep: int
    timeout: int
    configs: dict | Callable
    supply_type: tilelang.TensorSupplyType
    ref_prog: Callable | None
    supply_prog: Callable | None
    rtol: float
    atol: float
    max_mismatched_ratio: float
    skip_check: bool
    manual_check_prog: Callable | None
    cache_input_tensors: bool

    def __post_init__(self):
        self._tuner_cache = {}

    def get_tunner(self):
        # Use the real function from jit_impl, not a placeholder
        assert self.jit_impl.func is not None
        autotuner = (
            AutoTuner(self.jit_impl.func, configs=self.configs)
            .set_profile_args(
                supply_type=self.supply_type,
                ref_prog=self.ref_prog,
                supply_prog=self.supply_prog,
                rtol=self.rtol,
                atol=self.atol,
                max_mismatched_ratio=self.max_mismatched_ratio,
                skip_check=self.skip_check,
                manual_check_prog=self.manual_check_prog,
                cache_input_tensors=self.cache_input_tensors,
            )
            .set_compile_args(
                out_idx=self.jit_impl.out_idx,
                execution_backend=self.jit_impl.execution_backend,
                target=self.jit_impl.target,
                target_host=self.jit_impl.target_host,
                verbose=self.jit_impl.verbose,
                pass_configs=self.jit_impl.pass_configs,
            )
        )
        autotuner.run = partial(autotuner.run, self.warmup, self.rep, self.timeout)
        return autotuner

    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> JitKernel_NPU:
        key_args_tuple = args
        key_kwargs_tuple = tuple(sorted(kwargs.items()))
        key = (key_args_tuple, key_kwargs_tuple)
        if key not in self._tuner_cache:

            def jit_compile(**config_arg):
                # Call the wrapper function (which accepts __tune_params)
                # The wrapper function is stored in jit_impl.wrapper
                return self.jit_impl.wrapper(*args, **kwargs, __tune_params=config_arg)

            autotuner = self.get_tunner()
            autotuner.jit_compile = jit_compile
            autotuner.set_kernel_parameters(key, self.jit_impl.signature.parameters)
            artifact = autotuner.run()
            self._tuner_cache[key] = artifact.kernel
        return self._tuner_cache[key]


def autotune(  # This is the new public interface
    func: Callable[_P, _T] | PrimFunc | None = None,
    *,  # Indicates subsequent arguments are keyword-only
    configs: dict | Callable,
    # profile arguments
    warmup: int = 25,
    rep: int = 100,
    timeout: int = 100,
    # compile arguments
    supply_type: tilelang.TensorSupplyType = tilelang.TensorSupplyType.Auto,
    ref_prog: Callable | None = None,
    supply_prog: Callable | None = None,
    rtol: float = 1e-2,
    atol: float = 1e-2,
    max_mismatched_ratio: float = 0.01,
    skip_check: bool = False,
    manual_check_prog: Callable | None = None,
    cache_input_tensors: bool = False,
):
    """
    Just-In-Time (JIT) compiler decorator for TileLang functions.

    This decorator can be used without arguments (e.g., `@tilelang.jit`):
       Applies JIT compilation with default settings.

    Tips:
        - If you want to skip the auto-tuning process, you can set override the tunable parameters in the function signature.
            ```python
                if enable_autotune:
                    kernel = flashattn(batch, heads, seq_len, dim, is_causal)
                else:
                    kernel = flashattn(
                        batch, heads, seq_len, dim, is_causal, groups=groups, block_M=128, block_N=128, num_stages=2, threads=256)
            ```

    Parameters
    ----------
    func_or_out_idx : Any, optional
        If using `@tilelang.jit(...)` to configure, this is the `out_idx` parameter.
        If using `@tilelang.jit` directly on a function, this argument is implicitly
        the function to be decorated (and `out_idx` will be `None`).
    configs : Dict or Callable
        Configuration space to explore during auto-tuning.
    warmup : int, optional
        Number of warmup iterations before timing.
    rep : int, optional
        Number of repetitions for timing measurements.
    timeout : int, optional
    target : Union[str, Target], optional
        Compilation target for TVM (e.g., "cuda", "llvm"). Defaults to "auto".
    target_host : Union[str, Target], optional
        Target host for cross-compilation. Defaults to None.
    execution_backend : Literal["dlpack", "ctypes", "cython"], optional
        Backend for kernel execution and argument passing. Defaults to "cython".
    verbose : bool, optional
        Enables verbose logging during compilation. Defaults to False.
    pass_configs : Optional[Dict[str, Any]], optional
        Configurations for TVM's pass context. Defaults to None.
    debug_root_path : Optional[str], optional
        Directory to save compiled kernel source for debugging. Defaults to None.

    Returns
    -------
    Callable
        Either a JIT-compiled wrapper around the input function, or a configured decorator
        instance that can then be applied to a function.
    """
    if callable(func):
        # Case 1: Used as @autotune (func_or_out_idx is the function, others are defaults)
        # This is a placeholder for a real auto tuner implementation
        raise ValueError(
            "Use tilelang.autotune to decorate func without arguments is not supported yet."
        )
    elif isinstance(func, PrimFunc):
        raise ValueError(
            "Use tilelang.autotune to decorate prim_func is not supported yet."
        )
    else:

        def decorator(impl):
            # impl could be either:
            # 1. A wrapper function from @jit with __jit_impl__ attribute
            # 2. A _JitImplementation instance directly
            if callable(impl) and hasattr(impl, "__jit_impl__"):
                # Case 1: wrapper function from @jit decorator
                jit_impl = impl.__jit_impl__
            else:
                # Case 2: _JitImplementation instance
                jit_impl = impl

            return AutoTuneImpl(
                jit_impl=jit_impl,
                configs=configs,
                warmup=warmup,
                rep=rep,
                timeout=timeout,
                supply_type=supply_type,
                ref_prog=ref_prog,
                supply_prog=supply_prog,
                rtol=rtol,
                atol=atol,
                max_mismatched_ratio=max_mismatched_ratio,
                skip_check=skip_check,
                manual_check_prog=manual_check_prog,
                cache_input_tensors=cache_input_tensors,
            )

        return decorator
