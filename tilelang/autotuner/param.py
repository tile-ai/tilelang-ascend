"""The auto-tune parameters."""

from __future__ import annotations

import contextlib
from copy import deepcopy
import tilelang
from tilelang import tvm as tvm
from tvm.tirx import PrimFunc
from tvm.target import Target
from typing import TYPE_CHECKING, Literal, Any
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
import errno

from tilelang.jit import JITKernel
from tilelang.jit.adapter.base import CachedTextSource
import cloudpickle
import os
import shutil
from tilelang.engine.param import KernelParam, dump_kernel_params, load_kernel_params
from tilelang import logger
import json
import hashlib
import uuid
from tvm.runtime import Executable

if TYPE_CHECKING:
    from tilelang.backend.module import BackendContext

BEST_CONFIG_PATH = "best_config.json"
FUNCTION_PATH = "function.pkl"
OUT_IDX_PATH = "out_idx.json"
LATENCY_PATH = "latency.json"

# Align file names with cache/kernel_cache.py
DEVICE_KERNEL_PATH = "device_kernel.cu"
HOST_KERNEL_PATH = "host_kernel.cu"
EXECUTABLE_PATH = "executable.so"
KERNEL_LIB_PATH = "kernel_lib.so"
KERNEL_CUBIN_PATH = "kernel.cubin"
KERNEL_PY_PATH = "kernel.py"
PARAMS_PATH = "params.json"
TargetLike = str | dict[str, object] | Target


@dataclass(frozen=True)
class _FrozenCompileRequest:
    context: Any
    compiler_config: Any = None
    compiler_environment: tuple[tuple[str, str], ...] = ()
    layout_cost_model: str = "register-count"


@dataclass(frozen=True)
class CompileArgs:
    """Compile arguments for the auto-tuner. Detailed description can be found in `tilelang.jit.compile`.
    Attributes:
        out_idx: List of output tensor indices.
        execution_backend: Execution backend to use for kernel execution (default: "auto").
        target: Compilation target, either as a string, config dict, or a TVM Target object (default: "auto").
        target_host: Target host for cross-compilation (default: None).
        verbose: Whether to enable verbose output (default: False).
        pass_configs: Additional keyword arguments to pass to the Compiler PassContext.
        compile_flags: Additional compiler options, after derived and pass-config flags.
        workspace_idx: A2A3 compatibility automatic workspace parameter indices.
        Refer to `tilelang.PassConfigKey` for supported options.
    """

    out_idx: list[int] | int | None = None
    execution_backend: Literal["auto", "tvm_ffi", "cython", "nvrtc", "torch", "pto"] = "auto"
    target: TargetLike = "auto"
    target_host: TargetLike | None = None
    verbose: bool = False
    pass_configs: dict[str, Any] | None = None
    compile_flags: list[str] | str | None = None
    workspace_idx: list[int] | int | None = None
    _frozen: _FrozenCompileRequest | None = field(default=None, kw_only=True, repr=False, compare=False)

    def freeze(self, context=None):
        """Resolve an actual tuning request; never called by import or pure hash."""
        from tilelang.backend.module import create_backend_context

        if self._frozen is not None:
            return self
        if context is None:
            from tilelang.ascend.target import is_a2a3_target

            if isinstance(self.target, Target) and not is_a2a3_target(self.target):
                return self
            if isinstance(self.target, dict) and self.target.get("kind") != "ascend":
                return self
            if isinstance(self.target, str):
                target_text = self.target.strip()
                if target_text.startswith("{"):
                    try:
                        if json.loads(target_text).get("kind") != "ascend":
                            return self
                    except ValueError:
                        return self
                elif not target_text or target_text.split()[0] not in {"auto", "asc", "ascend", "ascendc", "pto"}:
                    return self
        context = context or create_backend_context(self.target, self.target_host, self.execution_backend)
        if context.module.name not in {"ascend_a2a3", "pto_a2a3"}:
            return self
        from tilelang.ascend.a2a3.compiler import resolve_compile_config
        from tilelang.ascend.a2a3.pass_config import normalize_a2a3_pass_configs

        effective = normalize_a2a3_pass_configs(context.target, self.pass_configs)
        keys = ("TL_CCE_AUTO_SYNC", "TL_CCE_OPT_LEVEL", "TL_PTO_DEBUG")
        environment = tuple((key, os.environ[key]) for key in keys if key in os.environ)
        base = resolve_compile_config(context.target, effective, self.compile_flags)
        layout_cost_model = tilelang.env.get_default_layout_cost_model() or "register-count"
        return replace(
            self,
            target=context.target,
            target_host=context.target_host,
            execution_backend=context.execution_backend.name,
            out_idx=deepcopy(self.out_idx),
            workspace_idx=deepcopy(self.workspace_idx),
            compile_flags=deepcopy(self.compile_flags),
            pass_configs=deepcopy(self.pass_configs),
            _frozen=_FrozenCompileRequest(context, base, environment, layout_cost_model),
        )

    def _prepare_a2a3_program(self, program):
        from tilelang.jit import _prepare_compile_arguments
        from tilelang.ascend.a2a3.compiler import with_compile_options
        from tilelang.ascend.a2a3.pass_config import normalize_a2a3_pass_configs

        state = self._frozen
        if state is None or state.compiler_config is None:
            raise ValueError("A2A3 candidate compilation requires a frozen tuning request")
        out_idx, pass_configs, flags = _prepare_compile_arguments(program, self.out_idx, self.pass_configs, self.compile_flags)
        # Merge source attributes before freezing the absent environment default:
        # neither a PTO default nor the global environment overrides an explicit
        # function-level pass config. The explicit built-in default also blocks
        # later normalization from introducing a newly changed environment.
        pass_configs = dict(pass_configs or {})
        pass_configs.setdefault("tl.layout_cost_model", state.layout_cost_model)
        effective = normalize_a2a3_pass_configs(state.context.target, pass_configs)
        config = with_compile_options(state.compiler_config, effective, flags, environ=dict(state.compiler_environment))
        return out_idx, effective, flags, config

    def compile_program(self, program: PrimFunc):
        """Compile one candidate program using this compile configuration."""
        if self._frozen is not None and self._frozen.compiler_config is not None:
            from tilelang.ascend.a2a3.kernel_cache import A2A3CythonKernelCache

            out_idx, effective, flags, config = self._prepare_a2a3_program(program)
            return A2A3CythonKernelCache().cached(
                program,
                out_idx,
                workspace_idx=self.workspace_idx,
                backend_context=self._frozen.context,
                compile_config=config,
                verbose=self.verbose,
                pass_configs=effective,
                compile_flags=flags,
            )
        return tilelang.compile(
            program,
            out_idx=self.out_idx,
            execution_backend=self.execution_backend,
            target=self.target,
            target_host=self.target_host,
            verbose=self.verbose,
            pass_configs=self.pass_configs,
            **({"compile_flags": self.compile_flags} if self.compile_flags is not None else {}),
            **({"workspace_idx": self.workspace_idx} if self.workspace_idx is not None else {}),
        )

    def __hash__(self):
        """Return a stable hash for cache key construction."""
        from tilelang.transform.pass_config import normalize_pass_configs

        # Resolve env-var-derived pass-config defaults so a changed environment
        # does not silently reuse tuning results produced under another one.
        a2a3_state = self._frozen is not None and self._frozen.compiler_config is not None
        pass_configs = self.pass_configs if a2a3_state else normalize_pass_configs(self.pass_configs)
        data = {
            "out_idx": self.out_idx,
            "execution_backend": self.execution_backend,
            "target": str(self.target),
            "target_host": str(self.target_host) if self.target_host else None,
            "verbose": self.verbose,
            "pass_configs": json.dumps(pass_configs, sort_keys=True) if pass_configs else None,
        }
        if self.compile_flags is not None:
            data["compile_flags"] = self.compile_flags
        if self.workspace_idx is not None:
            data["workspace_idx"] = self.workspace_idx
        if a2a3_state:
            data["target"] = str(self._frozen.context.target)
            data["target_host"] = str(self._frozen.context.target_host) if self._frozen.context.target_host else None
            data["execution_backend"] = self._frozen.context.execution_backend.name
            data["compiler_config"] = self._frozen.compiler_config.to_metadata()
            # Explicit O2 and absent O2 can yield different VS candidates, even
            # when their base compiler flags are the same. Keep that distinction.
            data["compiler_environment"] = self._frozen.compiler_environment
            data["layout_cost_model"] = self._frozen.layout_cost_model
        hash_obj = hashlib.sha256(json.dumps(data, sort_keys=True).encode("utf-8"))
        return int.from_bytes(hash_obj.digest(), byteorder="big")


@dataclass(frozen=True)
class ProfileArgs:
    """Profile arguments for the auto-tuner.

    Attributes:
        warmup: Number of warmup iterations.
        rep: Number of repetitions for timing.
        timeout: Maximum time per configuration.
        backend: Profiler backend - "event", "cupti", or "cudagraph".
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
    backend: Literal["event", "cupti", "cudagraph"] = "event"
    supply_type: tilelang.TensorSupplyType = tilelang.TensorSupplyType.Auto
    ref_prog: Callable = None
    supply_prog: Callable = None
    rtol: float = 1e-2
    atol: float = 1e-2
    max_mismatched_ratio: float = 0.01
    skip_check: bool = False
    manual_check_prog: Callable = None
    cache_input_tensors: bool = True

    def __hash__(self):
        """Return a stable hash for profiling configuration reuse."""
        data = {
            "warmup": self.warmup,
            "rep": self.rep,
            "timeout": self.timeout,
            "backend": self.backend,
            "supply_type": str(self.supply_type),
            "rtol": self.rtol,
            "atol": self.atol,
            "max_mismatched_ratio": self.max_mismatched_ratio,
            "skip_check": self.skip_check,
            "cache_input_tensors": self.cache_input_tensors,
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

    @staticmethod
    def _load_binary(path: str):
        """Load binary content from a cache artifact."""
        with open(path, "rb") as file:
            binary = file.read()
        return binary

    @staticmethod
    def _safe_write_file(path: str, mode: str, operation: Callable[[Any], None]):
        """Atomically write one cache file through a temporary sibling file."""
        directory, filename = os.path.split(path)
        temp_path = os.path.join(directory, f".{filename}.{os.getpid()}_{uuid.uuid4().hex}.tmp")
        try:
            with open(temp_path, mode) as temp_file:
                operation(temp_file)
                # Without this barrier a crash can persist the rename below
                # before the file data, publishing a truncated file.
                temp_file.flush()
                os.fsync(temp_file.fileno())
            os.replace(temp_path, path)
        finally:
            with contextlib.suppress(OSError):
                os.remove(temp_path)

    @staticmethod
    def _safe_write_executable(executable: Executable, path: str):
        """Atomically export one runtime executable to disk."""
        directory, filename = os.path.split(path)
        stem, suffix = os.path.splitext(filename)
        temp_path = os.path.join(directory, f".{stem}.{os.getpid()}_{uuid.uuid4().hex}.tmp{suffix}")
        try:
            executable.export_library(temp_path)
            # Flush exported data before the rename publishes the file.
            with open(temp_path, "rb+") as temp_file:
                os.fsync(temp_file.fileno())
            os.replace(temp_path, path)
        finally:
            with contextlib.suppress(OSError):
                os.remove(temp_path)

    def _save_kernel_to_disk(self, cache_path: Path, kernel: JITKernel, verbose: bool = False):
        """
        Persists a compiled kernel to disk cache.

        Args:
            cache_path (Path): The root path for the cache files.
            kernel (JITKernel): The compiled kernel to be saved.
            verbose (bool): Enable verbose log messages.

        Note:
            Saves the following files:
            - kernel.cu: The compiled kernel source code
            - wrapped_kernel.cu: The wrapped kernel source code
            - kernel_lib.so: The compiled kernel library
            - params.json: The serialized kernel parameters
        """
        os.makedirs(cache_path, exist_ok=True)  # Ensure directory exists.

        # Save device kernel source code
        device_kernel_path = os.path.join(cache_path, DEVICE_KERNEL_PATH)
        if verbose:
            logger.debug(f"Saving kernel source code to file: {device_kernel_path}")
        device_kernel_source = kernel.kernel_source
        if kernel.execution_backend == "cutedsl":
            device_kernel_source = kernel.adapter.get_kernel_source(kernel_only=True)
        if device_kernel_source is not None:
            self._safe_write_file(device_kernel_path, "w", lambda f: f.write(device_kernel_source))

        # Save host kernel source code (wrapped)
        host_kernel_path = os.path.join(cache_path, HOST_KERNEL_PATH)
        if verbose:
            logger.debug(f"Saving wrapped kernel source code to file: {host_kernel_path}")
        # Match kernel_cache behavior: use host source for tvm_ffi, otherwise wrapped kernel
        if kernel.execution_backend == "tvm_ffi" or kernel.execution_backend == "cutedsl":
            self._safe_write_file(host_kernel_path, "w", lambda f: f.write(kernel.adapter.get_host_source()))
        else:
            self._safe_write_file(host_kernel_path, "w", lambda f: f.write(kernel.adapter.get_kernel_source()))

        # Save kernel library (backend-specific)
        kernel_lib_file = self._get_kernel_lib_file(kernel.execution_backend)

        kernel_lib_path = os.path.join(cache_path, kernel_lib_file)

        if kernel.execution_backend == "nvrtc":
            # Save cubin and python helper file
            src_lib_path = kernel.adapter.libpath
            kernel_py_path = os.path.join(cache_path, KERNEL_PY_PATH)
            py_src_path = src_lib_path.replace(".cubin", ".py")
            if verbose:
                logger.debug(f"Saving kernel nvrtc python code to file: {kernel_py_path}")
            self._safe_write_file(kernel_py_path, "wb", lambda f: f.write(self._load_binary(py_src_path)))
            if verbose:
                logger.debug(f"Saving kernel library to file: {kernel_lib_path}")
            self._safe_write_file(kernel_lib_path, "wb", lambda f: f.write(self._load_binary(src_lib_path)))
        elif kernel.execution_backend == "tvm_ffi":
            if hasattr(kernel.adapter, "libpath") and kernel.adapter.libpath:
                src_lib_path = kernel.adapter.libpath
                if verbose:
                    logger.debug(f"Copying kernel library to file: {kernel_lib_path}")
                self._safe_write_file(kernel_lib_path, "wb", lambda f: f.write(self._load_binary(src_lib_path)))
            else:
                executable = kernel.adapter.get_exportable_executable()
                if verbose:
                    logger.debug(f"Saving kernel executable to file: {kernel_lib_path}")
                self._safe_write_executable(executable, kernel_lib_path)
        elif kernel.execution_backend == "cutedsl":
            # Save the Python source file (CuTeDSL "library" is a .py, not a .so)
            src_lib_path = kernel.adapter.libpath
            if verbose:
                logger.debug(f"Saving CuTeDSL kernel Python source to file: {kernel_lib_path}")
            self._safe_write_file(kernel_lib_path, "wb", lambda f: f.write(self._load_binary(src_lib_path)))

            # Save launcher .so if present (compiled C++ launcher for TMA etc.)
            lib_gen = kernel.adapter.lib_generator
            launcher_src = getattr(lib_gen, "launcher_libpath", None)
            if launcher_src and os.path.exists(launcher_src):
                launcher_name = getattr(lib_gen, "launcher_libname", os.path.basename(launcher_src))
                dst_launcher = os.path.join(cache_path, launcher_name)
                if verbose:
                    logger.debug(f"Saving CuTeDSL launcher library to file: {dst_launcher}")
                self._safe_write_file(dst_launcher, "wb", lambda f: f.write(self._load_binary(launcher_src)))

            # Save cubin if already generated (generated during autotuning benchmark)
            src_dir = os.path.dirname(src_lib_path)
            src_cubin = os.path.join(src_dir, "kernel.cubin")
            if os.path.exists(src_cubin):
                dst_cubin = os.path.join(cache_path, KERNEL_CUBIN_PATH)
                if verbose:
                    logger.debug(f"Saving CuTeDSL cubin to file: {dst_cubin}")
                self._safe_write_file(dst_cubin, "wb", lambda f: f.write(self._load_binary(src_cubin)))
        else:
            src_lib_path = kernel.adapter.libpath
            if verbose:
                logger.debug(f"Saving kernel library to file: {kernel_lib_path}")
            self._safe_write_file(kernel_lib_path, "wb", lambda f: f.write(self._load_binary(src_lib_path)))

        # Save kernel parameters
        params_path = os.path.join(cache_path, PARAMS_PATH)
        if verbose:
            logger.debug(f"Saving kernel parameters to disk: {params_path}")
        self._safe_write_file(params_path, "w", lambda f: f.write(dump_kernel_params(kernel.params)))
        if getattr(getattr(kernel, "backend", None), "name", None) in {"ascend_a2a3", "pto_a2a3"}:
            from tilelang.ascend.a2a3.kernel_cache import A2A3CythonKernelCache, runtime_metadata

            metadata_path = os.path.join(cache_path, A2A3CythonKernelCache.runtime_metadata_path)
            self._safe_write_file(metadata_path, "w", lambda file: json.dump(runtime_metadata(kernel), file, sort_keys=True))

    def _load_kernel_from_disk(
        self,
        cache_path: Path,
        backend_context: BackendContext,
        out_idx: list[int] | int | None = None,
        pass_configs: dict = None,
        compile_flags: list[str] | str | None = None,
        func: Callable = None,
        verbose: bool = False,
        compile_args: CompileArgs | None = None,
    ) -> JITKernel:
        """
        Loads a previously compiled kernel from disk cache.

        Args:
            key (str): The hash key identifying the kernel.
            backend_context: Resolved backend state for this compilation.
            out_idx (List[int], optional): Indices specifying which outputs to return.
            pass_configs (dict, optional): Configuration for compiler passes.
            func (Callable, optional): The original function.
            verbose (bool): Enable verbose log messages.

        Returns:
            JITKernel: The loaded kernel if found, None otherwise.
        """

        if not os.path.exists(cache_path):
            return None

        # Resolve backend to pick correct file names
        execution_backend = backend_context.execution_backend.name
        kernel_lib_file = self._get_kernel_lib_file(execution_backend)

        device_kernel_path = os.path.join(cache_path, DEVICE_KERNEL_PATH)
        host_kernel_path = os.path.join(cache_path, HOST_KERNEL_PATH)
        kernel_lib_path = os.path.join(cache_path, kernel_lib_file)
        params_path = os.path.join(cache_path, PARAMS_PATH)

        required_files = [*self._get_required_kernel_files(Path(cache_path), execution_backend), Path(params_path)]
        if backend_context.module.name in {"ascend_a2a3", "pto_a2a3"}:
            from tilelang.ascend.a2a3.kernel_cache import A2A3CythonKernelCache

            required_files.append(Path(cache_path) / A2A3CythonKernelCache.runtime_metadata_path)
        if not all(file.exists() for file in required_files):
            return None

        device_kernel_source: str | None = None
        host_kernel_source: str | None = None
        kernel_params: list[KernelParam] | None = None

        # Load optional device kernel source
        try:
            if verbose:
                logger.debug(f"Loading kernel source code from file: {device_kernel_path}")
            with open(device_kernel_path) as f:
                device_kernel_source = f.read()
        except Exception as e:
            logger.error(f"Error loading kernel source code from disk: {e}")

        # Load optional host kernel source
        try:
            if verbose:
                logger.debug(f"Loading wrapped kernel source code from file: {host_kernel_path}")
            with open(host_kernel_path) as f:
                host_kernel_source = f.read()
        except Exception as e:
            logger.error(f"Error loading host kernel source code from disk: {e}")

        # Load kernel parameters
        try:
            if verbose:
                logger.debug(f"Loading kernel parameters from file: {params_path}")
            with open(params_path) as f:
                kernel_params = load_kernel_params(f.read())
        except Exception as e:
            logger.error(f"Error loading kernel parameters from disk: {e}")

        if host_kernel_source and device_kernel_source and kernel_params:
            if backend_context.module.name in {"ascend_a2a3", "pto_a2a3"}:
                from tilelang.ascend.a2a3.kernel_cache import A2A3CythonKernelCache, _is_missing_own_library, stamp_runtime_request

                if compile_args is None:
                    raise ValueError("A2A3 tuning restore requires the frozen compile request")
                if compile_args._frozen is None:
                    compile_args = compile_args.freeze(backend_context)
                requested_out, effective, flags, config = compile_args._prepare_a2a3_program(func)
                derived = stamp_runtime_request(func, compile_args.workspace_idx, config)
                try:
                    return A2A3CythonKernelCache()._build_kernel(
                        derived,
                        CachedTextSource(text=host_kernel_source),
                        CachedTextSource(text=device_kernel_source),
                        kernel_lib_path,
                        kernel_params,
                        backend_context,
                        requested_out,
                        effective,
                        flags,
                    )
                except OSError as error:
                    if not _is_missing_own_library(error, kernel_lib_path):
                        raise
                    logger.warning("A2A3 tuning kernel library disappeared during cache restore: %s", kernel_lib_path)
                    return None
            return JITKernel.from_database(
                func=func,
                host_kernel_source=CachedTextSource(text=host_kernel_source),
                device_kernel_source=CachedTextSource(text=device_kernel_source),
                kernel_lib_path=kernel_lib_path,
                params=kernel_params,
                target=backend_context.target,
                target_host=backend_context.target_host,
                out_idx=out_idx,
                execution_backend=execution_backend,
                pass_configs=pass_configs,
                compile_flags=compile_flags,
                backend_context=backend_context,
            )
        else:
            return None

    def save_to_disk(self, path: Path, verbose: bool = False):
        """Persist autotune result to disk using atomic directory rename.

        All files are written into a temporary staging directory under the
        shared namespace staging root. Once complete, the staging directory is
        atomically renamed to *path* so that concurrent readers never see a
        half-written result.
        """
        # Already saved (e.g. another process won the race with a complete entry).
        a2a3 = getattr(getattr(self.kernel, "backend", None), "name", None) in {"ascend_a2a3", "pto_a2a3"}
        variant_options = {"a2a3": True} if a2a3 else {}
        if self._is_complete_result_dir(path, self.kernel.execution_backend, **variant_options):
            return

        # Keep autotuner staging under the shared namespace staging root so stale cleanup
        # never needs to scan the full cache directory.
        staging_path = path.parent.parent / ".staging" / f"{Path(path).name}_{os.getpid()}_{uuid.uuid4().hex[:8]}"
        os.makedirs(staging_path)
        # Ensure the parent of the final path exists (e.g. ~/.tilelang/cache/<namespace>/autotuner/)
        os.makedirs(Path(path).parent, exist_ok=True)

        try:
            # save best config
            if verbose:
                logger.debug(f"Saving best config to file: {staging_path / BEST_CONFIG_PATH}")
            self._safe_write_file(str(staging_path / BEST_CONFIG_PATH), "w", lambda f: json.dump(self.config, f))

            # save function
            if verbose:
                logger.debug(f"Saving function to file: {staging_path / FUNCTION_PATH}")
            self._safe_write_file(str(staging_path / FUNCTION_PATH), "wb", lambda f: cloudpickle.dump(self.func, f))

            # save out idx
            if verbose:
                logger.debug(f"Saving out idx to file: {staging_path / OUT_IDX_PATH}")
            self._safe_write_file(
                str(staging_path / OUT_IDX_PATH),
                "w",
                lambda f: json.dump(
                    {
                        "out_idx": list(self.func.attrs["tilelang_out_idx"])
                        if (self.func.attrs and "tilelang_out_idx" in self.func.attrs)
                        else None,
                    },
                    f,
                ),
            )

            # save latency
            if verbose:
                logger.debug(f"Saving latency to file: {staging_path / LATENCY_PATH}")
            self._safe_write_file(
                str(staging_path / LATENCY_PATH),
                "w",
                lambda f: json.dump(
                    {
                        "latency": self.latency,
                        "ref_latency": self.ref_latency,
                    },
                    f,
                ),
            )

            # save kernel
            self._save_kernel_to_disk(staging_path, self.kernel, verbose)

            missing_files = self._get_missing_complete_result_files(staging_path, self.kernel.execution_backend, **variant_options)
            if missing_files:
                missing_names = ", ".join(path.name for path in missing_files)
                raise RuntimeError(f"Incomplete autotune staging directory is missing required file(s): {missing_names}")

            # Repair stale/incomplete entries before making the new directory visible.
            self._remove_incomplete_result_dir(path, self.kernel.execution_backend, **variant_options)

            # Durability barrier: keep the staged data ahead of the rename so a
            # crash cannot publish a truncated cache entry.
            for entry in os.scandir(staging_path):
                if entry.is_file(follow_symlinks=False):
                    with open(entry.path, "rb+") as staged_file:
                        os.fsync(staged_file.fileno())

            # Atomic rename — directory becomes visible in one step.
            try:
                os.rename(str(staging_path), str(path))
            except OSError as exc:
                if not self._is_rename_collision(exc):
                    raise
                # Another process won the race with a complete cache entry.
                shutil.rmtree(str(staging_path), ignore_errors=True)
        except Exception:
            shutil.rmtree(str(staging_path), ignore_errors=True)
            logger.exception("Error during atomic autotune result save")

    @classmethod
    def load_from_disk(cls, path: Path, compile_args: CompileArgs) -> AutotuneResult:
        """Load a complete autotune result and its compiled kernel from disk."""
        if not os.path.exists(path):
            return None

        verbose = compile_args.verbose
        from tilelang.backend.module import create_backend_context

        compile_args = compile_args.freeze()
        state = compile_args._frozen
        backend_context = (
            state.context
            if state is not None
            else create_backend_context(compile_args.target, compile_args.target_host, compile_args.execution_backend)
        )
        # load best config
        if verbose:
            logger.debug(f"Loading best config from file: {path / BEST_CONFIG_PATH}")
        with open(path / BEST_CONFIG_PATH) as f:
            config = json.load(f)
        if backend_context.module.name in {"ascend_a2a3", "pto_a2a3"}:
            if compile_args._frozen is None:
                compile_args = compile_args.freeze(backend_context)
            # The public result stores the original config dictionary. The
            # internal __pass_configs__ key exists only in compile-unit kwargs.
            per_config = (config or {}).get("pass_configs")
            if per_config:
                merged = dict(compile_args.pass_configs or {})
                merged.update(per_config)
                compile_args = replace(compile_args, pass_configs=merged)

        # load function
        if verbose:
            logger.debug(f"Loading function from file: {path / FUNCTION_PATH}")
        with open(path / FUNCTION_PATH, "rb") as f:
            func = cloudpickle.load(f)

        # load out idx (optional — older caches may not have this file)
        out_idx_override = None
        out_idx_file = path / OUT_IDX_PATH
        if out_idx_file.exists():
            if verbose:
                logger.debug(f"Loading out idx from file: {out_idx_file}")
            with open(out_idx_file) as f:
                out_idx_override = json.load(f)["out_idx"]

        # load latency
        if verbose:
            logger.debug(f"Loading latency from file: {path / LATENCY_PATH}")
        with open(path / LATENCY_PATH) as f:
            latency = json.load(f)
            latency, ref_latency = latency["latency"], latency["ref_latency"]

        kernel = cls._load_kernel_from_disk(
            cls,
            path,
            backend_context,
            out_idx_override if out_idx_override is not None else compile_args.out_idx,
            compile_args.pass_configs,
            compile_args.compile_flags,
            func,
            **({"compile_args": compile_args} if backend_context.module.name in {"ascend_a2a3", "pto_a2a3"} else {}),
        )
        if kernel is None:
            return None
        kernel.update_tuner_result(
            config=config,
            latency=latency,
            ref_latency=ref_latency,
        )
        result = cls(
            config=config,
            func=func,
            kernel=kernel,
            libcode=kernel.get_kernel_source(),
            latency=latency,
            ref_latency=ref_latency,
        )
        return result

    @staticmethod
    def _get_kernel_lib_file(execution_backend: str) -> str:
        """Return the cache filename for one backend's executable artifact."""
        if execution_backend == "nvrtc":
            return KERNEL_CUBIN_PATH
        if execution_backend == "tvm_ffi":
            return EXECUTABLE_PATH
        if execution_backend == "cutedsl":
            return KERNEL_PY_PATH
        return KERNEL_LIB_PATH

    @classmethod
    def _get_required_kernel_files(cls, path: Path, execution_backend: str) -> list[Path]:
        """Return backend-specific files required to reload a kernel."""
        files = [path / cls._get_kernel_lib_file(execution_backend)]
        if execution_backend == "nvrtc":
            files.append(path / KERNEL_PY_PATH)
        return files

    @classmethod
    def _get_complete_result_files(cls, path: Path, execution_backend: str, *, a2a3=False) -> list[Path]:
        """Return the full file set that marks an autotune result complete."""
        files = list(
            dict.fromkeys(
                [
                    path / BEST_CONFIG_PATH,
                    path / FUNCTION_PATH,
                    path / LATENCY_PATH,
                    path / DEVICE_KERNEL_PATH,
                    path / HOST_KERNEL_PATH,
                    *cls._get_required_kernel_files(path, execution_backend),
                    path / PARAMS_PATH,
                ]
            )
        )
        if a2a3:
            from tilelang.ascend.a2a3.kernel_cache import A2A3CythonKernelCache

            files.append(path / A2A3CythonKernelCache.runtime_metadata_path)
        return files

    @classmethod
    def _get_missing_complete_result_files(cls, path: Path, execution_backend: str, *, a2a3=False) -> list[Path]:
        """Return complete-result files missing from a candidate cache path."""
        return [file for file in cls._get_complete_result_files(path, execution_backend, a2a3=a2a3) if not file.exists()]

    @classmethod
    def _is_complete_result_dir(cls, path: Path, execution_backend: str, *, a2a3=False) -> bool:
        """Return whether a cache directory contains all required files."""
        return path.is_dir() and not cls._get_missing_complete_result_files(path, execution_backend, a2a3=a2a3)

    @classmethod
    def _remove_incomplete_result_dir(cls, path: Path, execution_backend: str, *, a2a3=False) -> bool:
        """Remove a stale incomplete result directory when present."""
        if not path.is_dir() or cls._is_complete_result_dir(path, execution_backend, a2a3=a2a3):
            return False
        shutil.rmtree(path)
        return True

    @staticmethod
    def _is_rename_collision(exc: OSError) -> bool:
        """Return whether a rename failure came from a benign race."""
        return exc.errno in {errno.EEXIST, errno.ENOTEMPTY}
