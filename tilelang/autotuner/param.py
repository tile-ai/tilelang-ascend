"""The auto-tune parameters."""

from __future__ import annotations

from tilelang import tvm as tvm
from tvm.tir import PrimFunc
from tvm.target import Target
from typing import Callable, Literal, Any
from dataclasses import dataclass
from pathlib import Path

from tilelang.jit.jit_npu import JitKernel_NPU
import cloudpickle
import shutil
from tilelang import logger
import json
import hashlib

BEST_CONFIG_PATH = "best_config.json"
FUNCTION_PATH = "function.pkl"
LATENCY_PATH = "latency.json"
KERNEL_PATH = "kernel.mlir"
WRAPPED_KERNEL_PATH = "wrapped_kernel.o"
KERNEL_LIB_PATH = "kernel_lib.so"
SO_LAUNCHER_PATH = "main.so"
PARAMS_PATH = "params.pkl"
METADATA_PATH = "metadata.pkl"


@dataclass(frozen=True)
class CompileArgs:
    """Compile arguments for the auto-tuner. Detailed description can be found in `tilelang.jit.compile`.
    Attributes:
        out_idx: List of output tensor indices.
        execution_backend: Execution backend to use for kernel execution (default: "cython").
        target: Compilation target, either as a string or a TVM Target object (default: "auto").
        target_host: Target host for cross-compilation (default: None).
        verbose: Whether to enable verbose output (default: False).
        pass_configs: Additional keyword arguments to pass to the Compiler PassContext.
        Refer to `tilelang.PassConfigKey` for supported options.
    """

    out_idx: list[int] | int | None = None
    execution_backend: Literal["dlpack", "ctypes", "cython"] = "cython"
    target: Literal["auto", "cuda", "hip"] = "auto"
    target_host: str | Target = None
    verbose: bool = False
    pass_configs: dict[str, Any] | None = None

    def compile_program(self, program: PrimFunc):
        import tilelang

        return tilelang.compile(
            program,
            out_idx=self.out_idx,
            target=self.target,
            target_host=self.target_host,
            verbose=self.verbose,
            pass_configs=self.pass_configs,
        )

    def __hash__(self):
        data = {
            "execution_backend": self.execution_backend,
            "target": str(self.target),
            "target_host": str(self.target_host) if self.target_host else None,
            "verbose": self.verbose,
            "pass_configs": json.dumps(self.pass_configs, sort_keys=True)
            if self.pass_configs
            else None,
        }

        hash_obj = hashlib.sha256(json.dumps(data, sort_keys=True).encode("utf-8"))
        return int.from_bytes(hash_obj.digest(), byteorder="big")


@dataclass(frozen=True)
class ProfileArgs:
    """Profile arguments for the auto-tuner.

    Attributes:
        warmup: Number of warmup iterations.
        rep: Number of repetitions for timing.
        timeout: Maximum time per configuration.
        supply_type: Type of tensor supply mechanism.
        ref_prog: Reference program for correctness validation.
        supply_prog: Supply program for input tensors.
        out_idx: Union[List[int], int] = -1
        supply_type: tilelang.TensorSupplyType = tilelang.TensorSupplyType.Auto
        ref_prog: Callable = None
        supply_prog: Callable = None
        rtol: float = 1e-2
        atol: float = 1e-2
        max_mismatched_ratio: float = 0.01
        skip_check: bool = False
        manual_check_prog: Callable = None
        cache_input_tensors: bool = True
    """

    warmup: int = 25
    rep: int = 100
    timeout: int = 30
    supply_type: Any = None
    ref_prog: Callable | None = None
    supply_prog: Callable | None = None
    rtol: float = 1e-2
    atol: float = 1e-2
    max_mismatched_ratio: float = 0.01
    skip_check: bool = False
    manual_check_prog: Callable | None = None
    cache_input_tensors: bool = True

    def __hash__(self):
        data = {
            "warmup": self.warmup,
            "rep": self.rep,
            "timeout": self.timeout,
            "supply_type": str(self.supply_type),
            "rtol": self.rtol,
            "atol": self.atol,
            "max_mismatched_ratio": self.max_mismatched_ratio,
        }
        hash_obj = hashlib.sha256(json.dumps(data, sort_keys=True).encode("utf-8"))
        return int.from_bytes(hash_obj.digest(), byteorder="big")


@dataclass(frozen=True)
class AutotuneResult:
    """Results from auto-tuning process.

    Attributes:
        latency: Best achieved execution latency.
        config: Configuration that produced the best result.
        ref_latency: Reference implementation latency.
        libcode: Generated library code.
        func: Optimized function.
        kernel: Compiled kernel function.
    """

    latency: float | None = None
    config: dict | None = None
    ref_latency: float | None = None
    libcode: str | None = None
    func: Callable | None = None
    kernel: Callable | None = None


class KernelCache:
    """Handles serialising and deserialising AutotuneResult to/from disk.

    Keeps all file-I/O logic in one place so AutotuneResult stays a plain
    data object.
    """

    @staticmethod
    def save(path: Path, result: AutotuneResult, verbose: bool = False) -> None:
        """Persist *result* under *path*.  Creates the directory if needed."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        KernelCache._save_json(path / BEST_CONFIG_PATH, result.config, verbose)
        KernelCache._save_pickle(path / FUNCTION_PATH, result.func, verbose)
        KernelCache._save_json(
            path / LATENCY_PATH,
            {"latency": result.latency, "ref_latency": result.ref_latency},
            verbose,
        )
        KernelCache._save_kernel(path, result.kernel, verbose)

    @staticmethod
    def _save_json(dest: Path, obj: Any, verbose: bool) -> None:
        if verbose:
            logger.debug(f"Saving to {dest}")
        with open(dest, "w") as f:
            json.dump(obj, f)

    @staticmethod
    def _save_pickle(dest: Path, obj: Any, verbose: bool) -> None:
        if verbose:
            logger.debug(f"Saving to {dest}")
        with open(dest, "wb") as f:
            cloudpickle.dump(obj, f)

    @staticmethod
    def _save_kernel(cache_path: Path, kernel: JitKernel_NPU, verbose: bool) -> None:
        """Write all kernel artefacts to *cache_path*."""

        def _try(label: str, fn):
            try:
                fn()
            except Exception as exc:
                logger.error(f"Error saving {label}: {exc}")

        if kernel.mlir_content is not None:
            dest = cache_path / KERNEL_PATH
            if verbose:
                logger.debug(f"Saving kernel MLIR to {dest}")
            _try("kernel MLIR", lambda: dest.write_text(kernel.mlir_content))

        dest = cache_path / WRAPPED_KERNEL_PATH
        if verbose:
            logger.debug(f"Saving wrapped kernel to {dest}")
        _try("wrapped kernel", lambda: dest.write_bytes(kernel.get_kernel_source()))

        dest_launcher = cache_path / SO_LAUNCHER_PATH
        if verbose:
            logger.debug(f"Saving launcher library to {dest_launcher}")
        _try(
            "launcher .so", lambda: shutil.copy(kernel.so_launcher_path, dest_launcher)
        )

        dest_params = cache_path / PARAMS_PATH
        if verbose:
            logger.debug(f"Saving kernel params to {dest_params}")
        _try(
            "kernel params",
            lambda: dest_params.write_bytes(cloudpickle.dumps(kernel.params)),
        )

        metadata = {
            "symbolic": kernel.symbolic,
            "params": kernel.params,
            "out_idx": kernel.out_idx,
            "signature": kernel.signature,
            "primfunc": kernel.prim_func,
            "mlir_content": kernel.mlir_content,
            "shared": kernel.utils_shared,
            "kernel_name": kernel.kernel_name,
            "gridfunc": kernel.gridfunc,
            "mix_mode": kernel.mix_mode,
            "name": kernel.utils_name,
            "tensor_kinds": kernel.tensor_kinds,
            "kernel_src": kernel.utils_kernel_src,
        }
        dest_meta = cache_path / METADATA_PATH
        if verbose:
            logger.debug(f"Saving metadata to {dest_meta}")
        _try("metadata", lambda: dest_meta.write_bytes(cloudpickle.dumps(metadata)))

    @staticmethod
    def load(path: Path, compile_args: CompileArgs) -> AutotuneResult | None:
        from tilelang.cache import _kernel_cache_instance

        key = Path(path).name
        return _kernel_cache_instance.load_autotune_result(
            key,
            out_idx=compile_args.out_idx,
            verbose=compile_args.verbose,
        )
