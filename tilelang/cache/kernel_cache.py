# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""The cache utils with class and database persistence - KernelCache Class"""

import os
import json
import shutil
from pathlib import Path
from typing import Callable, List, Literal, Union, Optional
from tvm.target import Target
from tvm.tir import PrimFunc
from tilelang.jit import JITKernel
from tilelang.jit.jit_npu import JitKernel_NPU
from tilelang.engine.param import KernelParam
import threading
import cloudpickle
import logging
from tilelang.utils.npu_utils import compute_sha256_hash

from tilelang import env
from tilelang.version import __version__
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from tilelang.autotuner.param import AutotuneResult

KERNEL_PATH = "kernel.cu"
WRAPPED_KERNEL_PATH = "wrapped_kernel.cu"
KERNEL_LIB_PATH = "kernel_lib.so"
PARAMS_PATH = "params.pkl"

AUTOTUNE_SUBDIR = "autotune"
AUTOTUNE_BEST_CONFIG_PATH = "best_config.json"
AUTOTUNE_FUNCTION_PATH = "function.pkl"
AUTOTUNE_LATENCY_PATH = "latency.json"
# NPU kernel artefacts (written at the entry root, same level as GPU files)
AUTOTUNE_KERNEL_MLIR_PATH = "kernel.mlir"
AUTOTUNE_WRAPPED_KERNEL_PATH = "wrapped_kernel.o"
AUTOTUNE_SO_LAUNCHER_PATH = "main.so"
AUTOTUNE_METADATA_PATH = "metadata.pkl"
COMPILE_SUBDIR = "compile"


class KernelCache:
    """
    Caches compiled kernels using a class and database persistence to avoid redundant compilation.
    Cache files:
        kernel.cu: The compiled kernel source code
        wrapped_kernel.cu: The compiled wrapped kernel source code
        kernel_lib.so: The compiled kernel library
        params.pkl: The compiled kernel parameters
        Directory layout::

        <cache_dir>/
            <kernel_hash>/
                # GPU artefacts
                kernel.cu
                wrapped_kernel.cu
                kernel_lib.so
                params.pkl
                # NPU artefacts (compile path)
                kernel.mlir
                wrapped_kernel.o
                npu_utils.so
                main.so
                npu_params.pkl
                metadata.pkl            <- presence signals a valid NPU compile entry
                # autotune extras
                autotune/
                    best_config.json
                    function.pkl
                    latency.json
    """

    _instance = None  # For implementing singleton pattern
    _lock = threading.Lock()  # For thread safety
    _memory_cache = {}  # In-memory cache dictionary

    cache_dir: Path = Path(env.TILELANG_CACHE_DIR)

    def __new__(cls, cache_dir=env.TILELANG_CACHE_DIR):
        """
        Implements singleton pattern for KernelCache class.

        Args:
            cache_dir (str): Directory path for storing kernel cache. Defaults to TILELANG_CACHE_DIR.

        Returns:
            KernelCache: The singleton instance of KernelCache.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:  # Double-checked locking
                    instance = super().__new__(cls)
                    instance.cache_dir = Path(cache_dir)
                    os.makedirs(instance.cache_dir, exist_ok=True)
                    instance._memory_cache = {}  # Initialize memory cache
                    cls._instance = instance
        return cls._instance

    def _generate_key(
        self,
        func: Callable,
        out_idx: List[int],
        workspace_idx: List[int],
        execution_backend: Literal["dlpack", "ctypes", "cython"] = "cython",
        args=None,
        target: Union[str, Target] = "auto",
        target_host: Union[str, Target] = None,
        platform: Literal["A2", "A3", "A5"] = "A3",
        pass_configs: dict = None,
    ) -> str:
        """
        Generates a unique hash key for caching compiled kernels.

        Args:
            func (Callable): The function to be compiled.
            out_idx (List[int]): Indices specifying which outputs to return.
            workspace_idx (List[int]): Indices specifying auto-allocated workspace tensors.
            execution_backend (Literal): Backend type for execution. Defaults to "cython".
            args: Arguments passed to the function.
            target (Union[str, Target]): Compilation target platform. Defaults to "auto".
            target_host (Union[str, Target], optional): Host target platform.
            platform (Literal): Specifies the target hardware platform generation. Defaults to "A3".

        Returns:
            str: SHA256 hash key for the kernel configuration.
        """
        func_binary = cloudpickle.dumps(func.script())
        key_data = {
            "version": __version__,
            "func": compute_sha256_hash(func_binary),
            "out_idx": (
                tuple(out_idx) if isinstance(out_idx, (list, tuple)) else [out_idx]
            ),
            "workspace_idx": (
                tuple(workspace_idx)
                if isinstance(workspace_idx, (list, tuple))
                else [workspace_idx]
            ),
            "args_repr": tuple(
                repr(arg) for arg in args
            ),  # Use repr to serialize arguments, may need more robust serialization
            "target": str(target),
            "target_host": str(target_host) if target_host else None,
            "platform": str(platform),
            "execution_backend": execution_backend,
            "pass_configs": pass_configs,
        }
        key_string = json.dumps(
            key_data, sort_keys=True
        )  # Sort keys to ensure consistency
        return compute_sha256_hash(key_string)

    def _generate_compile_key(
        self,
        func,
        out_idx,
        target,
        target_host,
        execution_backend,
        verbose,
        pass_configs,
    ) -> str:
        """Stable hash key for a ``compile()`` call targeting NPU IR.

        The key captures every input that could produce a different binary:
        the serialised PrimFunc source, output-index spec, target strings,
        execution backend, and pass configurations.  ``verbose`` is included
        so debug builds are never confused with release builds.
        """
        func_binary = cloudpickle.dumps(func.script())
        key_data = {
            "version": __version__,
            "func": compute_sha256_hash(func_binary),
            "out_idx": (
                list(out_idx)
                if isinstance(out_idx, (list, tuple))
                else ([out_idx] if out_idx is not None else [])
            ),
            "target": str(target),
            "target_host": str(target_host) if target_host else None,
            "execution_backend": execution_backend,
            "verbose": verbose,
            "pass_configs": (
                json.dumps(pass_configs, sort_keys=True) if pass_configs else None
            ),
        }
        return compute_sha256_hash(json.dumps(key_data, sort_keys=True))

    def cached(
        self,
        func: PrimFunc = None,
        out_idx: List[int] = None,
        workspace_idx: List[int] = None,
        *args,
        target: Union[str, Target] = "auto",
        target_host: Union[str, Target] = None,
        platform: Literal["A2", "A3", "A5"] = "A3",
        execution_backend: Literal["dlpack", "ctypes", "cython"] = "cython",
        verbose: bool = False,
        pass_configs: dict = None,
    ) -> JITKernel:
        """
        Caches and reuses compiled kernels to avoid redundant compilation.

        Args:
            func: Function to be compiled or a prepared PrimFunc
            out_idx: Indices specifying which outputs to return
            workspace_idx: Indices specifying auto-allocated workspace tensors
            target: Compilation target platform
            target_host: Host target platform
            platform: Specifies the target hardware platform generation. Defaults to "A3".
            *args: Arguments passed to func

        Returns:
            JITKernel: The compiled kernel, either freshly compiled or from cache
        """
        if not env.is_cache_enabled():
            return JITKernel(
                func,
                out_idx=out_idx,
                workspace_idx=workspace_idx,
                execution_backend=execution_backend,
                target=target,
                target_host=target_host,
                platform=platform,
                verbose=verbose,
                pass_configs=pass_configs,
            )

        key = self._generate_key(
            func=func,
            out_idx=out_idx,
            workspace_idx=workspace_idx,
            execution_backend=execution_backend,
            args=args,
            target=target,
            target_host=target_host,
            platform=platform,
            pass_configs=pass_configs,
        )
        with self._lock:
            # First check in-memory cache
            if key in self._memory_cache:
                logger.warning(
                    "Found kernel in memory cache. For better performance,"
                    " consider using `@tilelang.jit` instead of direct kernel caching."
                )
                return self._memory_cache[key]

            # Then check disk cache
            kernel = self._load_kernel_from_disk(
                key,
                target,
                target_host,
                platform,
                out_idx,
                workspace_idx,
                execution_backend,
                pass_configs,
                func,
            )
            if kernel is not None:
                # Populate memory cache with disk result
                self._memory_cache[key] = kernel
                return kernel

        # Compile kernel if cache miss; leave critical section
        kernel = JITKernel(
            func,
            out_idx=out_idx,
            workspace_idx=workspace_idx,
            execution_backend=execution_backend,
            target=target,
            target_host=target_host,
            platform=platform,
            verbose=verbose,
            pass_configs=pass_configs,
        )
        if execution_backend == "dlpack":
            logger.warning("DLPack backend does not support cache saving to disk.")
        else:
            with self._lock:
                if env.is_cache_enabled():
                    self._save_kernel_to_disk(key, kernel, func)

        # Store in memory cache after compilation
        self._memory_cache[key] = kernel
        return kernel

    def cached_npu(
        self,
        func,
        out_idx=None,
        execution_backend="cython",
        target="npuir",
        target_host=None,
        verbose=False,
        pass_configs=None,
    ):
        """Compile *func* for the ``npuir`` target with memory+disk caching.

        This is the NPU-side counterpart of :func:`tilelang.cache.cached` and owns
        the full lookup → compile → store lifecycle:

        1. In-memory cache  (``KernelCache._memory_cache``)  — cheapest, per-process.
        2. Disk cache       (``KernelCache.load_compile_result``) — survives restarts.
        3. Compile from scratch via ``compiler_npu().compile(func, out_idx)``.

        Parameters
        ----------
        func:
            The ``tvm.tir.PrimFunc`` to compile.
        out_idx:
            Index(es) of the output buffer(s) to expose to the caller.
        execution_backend:
            Forwarded to ``_generate_compile_key`` for cache-key stability only;
            the NPU compiler does not use it directly.
        target:
            Must be ``"npuir"``; kept as a parameter so callers can forward their
            ``target`` variable without an extra ``if`` branch.
        target_host:
            Forwarded to ``_generate_compile_key`` only.
        verbose:
            Enable debug-level cache-hit/miss logging.
        pass_configs:
            Forwarded to ``_generate_compile_key`` only.

        Returns
        -------
        JitKernel_NPU
            A compiled, ready-to-run NPU kernel wrapper.
        """
        from tilelang.jit.jit_npu import compiler_npu
        from tilelang import env

        _key = self._generate_compile_key(
            func=func,
            out_idx=out_idx,
            target=target,
            target_host=target_host,
            execution_backend=execution_backend,
            verbose=verbose,
            pass_configs=pass_configs,
        )
        if env.is_cache_enabled():
            mem_hit = self._memory_cache.get(_key)
            if mem_hit is not None:
                if verbose:
                    logger.debug(f"cached_npu(): memory cache hit for key {_key[:8]}…")
                return mem_hit

            disk_hit = self.load_compile_result(
                _key, func=func, out_idx=out_idx, verbose=verbose
            )
            if disk_hit is not None:
                logger.debug(f"cached_npu(): disk cache hit for key {_key[:8]}…")
                self._memory_cache[_key] = disk_hit
                return disk_hit

        kernel = compiler_npu().compile(func, out_idx)

        if env.is_cache_enabled():
            self.save_compile_result(_key, kernel, verbose=verbose)
            self._memory_cache[_key] = kernel

        return kernel

    def set_cache_dir(self, cache_dir: str):
        """
        Sets the cache directory for the kernel cache.
        """
        self.cache_dir = Path(cache_dir)

    def get_cache_dir(self) -> Path:
        """
        Gets the cache directory for the kernel cache.
        """
        return self.cache_dir

    def clear_cache(self):
        """
        Clears the entire kernel cache, including both in-memory and disk cache.
        """
        with self._lock:
            self._memory_cache.clear()  # Clear in-memory cache
            self._clear_disk_cache()  # Clear disk cache

    def _get_cache_path(self, key: str) -> str:
        """
        Gets the filesystem path for a cached kernel.

        Args:
            key (str): The hash key identifying the kernel.

        Returns:
            str: Absolute path to the cache directory for this kernel.
        """
        return self.cache_dir / key

    def _save_kernel_to_disk(self, key: str, kernel: JITKernel, func: Callable = None):
        """
        Persists a compiled kernel to disk cache.

        Args:
            key (str): The hash key identifying the kernel.
            kernel (JITKernel): The compiled kernel to be saved.
            func (Callable, optional): The original function.

        Note:
            Saves the following files:
            - kernel.cu: The compiled kernel source code
            - wrapped_kernel.cu: The wrapped kernel source code
            - kernel_lib.so: The compiled kernel library
            - params.pkl: The serialized kernel parameters
        """
        cache_path = self._get_cache_path(key)
        os.makedirs(cache_path, exist_ok=True)  # Ensure directory exists

        # Save kernel source code
        try:
            kernel_path = os.path.join(cache_path, KERNEL_PATH)
            with open(kernel_path, "w") as f:
                f.write(kernel.artifact.kernel_source)
        except Exception as e:
            logger.error(f"Error saving kernel source code to disk: {e}")

        # Save wrapped kernel source code
        try:
            wrapped_kernel_path = os.path.join(cache_path, WRAPPED_KERNEL_PATH)
            with open(wrapped_kernel_path, "w") as f:
                f.write(kernel.adapter.get_kernel_source())
        except Exception as e:
            logger.error(f"Error saving wrapped kernel source code to disk: {e}")

        # Save kernel library
        try:
            kernel_lib_path = os.path.join(cache_path, KERNEL_LIB_PATH)
            src_lib_path = kernel.adapter.libpath
            shutil.copy(src_lib_path, kernel_lib_path)
        except Exception as e:
            logger.error(f"Error saving kernel library to disk: {e}")

        # Save kernel parameters
        try:
            params_path = os.path.join(cache_path, PARAMS_PATH)
            with open(params_path, "wb") as f:
                cloudpickle.dump(kernel.params, f)
        except Exception as e:
            logger.error(f"Error saving kernel parameters to disk: {e}")

    def _load_kernel_from_disk(
        self,
        key: str,
        target: Union[str, Target] = "auto",
        target_host: Union[str, Target] = None,
        platform: Literal["A2", "A3", "A5"] = "A3",
        out_idx: List[int] = None,
        workspace_idx: List[int] = None,
        execution_backend: Literal["dlpack", "ctypes", "cython"] = "cython",
        pass_configs: dict = None,
        func: Callable = None,
    ) -> JITKernel:
        """
        Loads a previously compiled kernel from disk cache.

        Args:
            key (str): The hash key identifying the kernel.
            target (Union[str, Target]): Compilation target platform. Defaults to "auto".
            target_host (Union[str, Target], optional): Host target platform.
            platform (Literal): Specifies the target hardware platform generation. Defaults to "A3".
            out_idx (List[int], optional): Indices specifying which outputs to return.
            workspace_idx (List[int], optional): Indices specifying auto-allocated workspace tensors.
            execution_backend (Literal): Backend type for execution. Defaults to "cython".
            pass_configs (dict, optional): Configuration for compiler passes.
            func (Callable, optional): The original function.

        Returns:
            JITKernel: The loaded kernel if found, None otherwise.
        """
        cache_path = self._get_cache_path(key)
        if not os.path.exists(cache_path):
            return None

        kernel_global_source: Optional[str] = None
        kernel_params: Optional[List[KernelParam]] = None

        try:
            wrapped_kernel_path = os.path.join(cache_path, WRAPPED_KERNEL_PATH)
            with open(wrapped_kernel_path, "r") as f:
                kernel_global_source = f.read()
        except Exception as e:
            logger.error(f"Error loading wrapped kernel source code from disk: {e}")

        kernel_lib_path = os.path.join(cache_path, KERNEL_LIB_PATH)

        # Load kernel parameters
        try:
            params_path = os.path.join(cache_path, PARAMS_PATH)
            with open(params_path, "rb") as f:
                kernel_params = cloudpickle.load(f)
        except Exception as e:
            logger.error(f"Error loading kernel parameters from disk: {e}")

        if kernel_global_source and kernel_params:
            return JITKernel.from_database(
                func=func,
                kernel_global_source=kernel_global_source,
                kernel_lib_path=kernel_lib_path,
                params=kernel_params,
                target=target,
                target_host=target_host,
                platform=platform,
                out_idx=out_idx,
                workspace_idx=workspace_idx,
                execution_backend=execution_backend,
                pass_configs=pass_configs,
            )
        else:
            return None

    def _clear_disk_cache(self):
        """
        Removes all cached kernels from disk.

        Note:
            This operation will delete the entire cache directory and recreate it empty.
            Use with caution as this operation cannot be undone.
        """
        try:
            if os.path.exists(self.cache_dir):
                shutil.rmtree(self.cache_dir)  # Delete entire cache directory
            os.makedirs(self.cache_dir, exist_ok=True)  # Re-create cache directory
        except Exception as e:
            logger.error(f"Error clearing disk cache: {e}")

    def save_compile_result(
        self,
        key: str,
        kernel: JitKernel_NPU,
        verbose: bool = False,
    ) -> None:
        """Persist a ``JitKernel_NPU`` produced by ``compile()`` to disk.

        A sentinel sub-directory ``compile/`` is created so that
        ``load_compile_result`` can distinguish a compile-cache entry from an
        autotune-cache entry that happens to share the same root directory.

        Layout::

            <cache_dir>/<key>/
                compile/          ← sentinel sub-directory
                kernel.mlir
                wrapped_kernel.o
                npu_utils.so
                main.so
                npu_params.pkl
                metadata.pkl
        """
        cache_path = self._get_cache_path(key)
        cache_path.mkdir(parents=True, exist_ok=True)

        self._save_npu_kernel_to_disk(cache_path, kernel, verbose)

        if verbose:
            logger.debug(f"Compile result saved to {cache_path}")

    def load_compile_result(
        self,
        key: str,
        func,
        out_idx,
        verbose: bool = False,
    ) -> Optional[JitKernel_NPU]:
        """Load a ``JitKernel_NPU`` saved by ``save_compile_result``.

        Returns ``None`` on cache miss (sentinel directory absent or any
        required artefact missing/corrupt).
        """
        cache_path = self._get_cache_path(key)

        if not (cache_path / AUTOTUNE_METADATA_PATH).exists():
            return None

        kernel = self._load_npu_kernel_from_disk(cache_path, func=func, out_idx=out_idx)
        if verbose and kernel is not None:
            logger.debug(f"Compile result loaded from {cache_path}")
        return kernel

    def save_autotune_result(
        self,
        key: str,
        result: "AutotuneResult",
        verbose: bool = False,
    ) -> None:
        """Persist an autotune result under ``<cache_dir>/<key>/``.

        NPU kernel artefacts are written at the entry root (alongside the GPU
        files).  The three tuning metadata files go into the ``autotune/``
        subfolder so they are easy to spot and never collide with kernel files.
        """
        cache_path = self._get_cache_path(key)
        autotune_path = cache_path / AUTOTUNE_SUBDIR
        cache_path.mkdir(parents=True, exist_ok=True)
        autotune_path.mkdir(exist_ok=True)

        # --- NPU kernel artefacts (entry root) ---
        self._save_npu_kernel_to_disk(cache_path, result.kernel, verbose)

        # --- Tuning metadata (autotune/ subfolder) ---
        self._try_save(
            "best config",
            lambda: _write_json(
                autotune_path / AUTOTUNE_BEST_CONFIG_PATH, result.config
            ),
        )
        self._try_save(
            "function",
            lambda: (autotune_path / AUTOTUNE_FUNCTION_PATH).write_bytes(
                cloudpickle.dumps(result.func)
            ),
        )
        self._try_save(
            "latency",
            lambda: _write_json(
                autotune_path / AUTOTUNE_LATENCY_PATH,
                {"latency": result.latency, "ref_latency": result.ref_latency},
            ),
        )

        if verbose:
            logger.debug(f"Autotune result saved to {autotune_path}")

    def load_autotune_result(
        self,
        key: str,
        out_idx: Optional[List[int]],
        verbose: bool = False,
    ) -> Optional["AutotuneResult"]:
        """Load a previously saved autotune result.  Returns ``None`` on miss."""

        cache_path = self._get_cache_path(key)
        autotune_path = cache_path / AUTOTUNE_SUBDIR
        if not autotune_path.exists():
            return None

        try:
            config = _read_json(autotune_path / AUTOTUNE_BEST_CONFIG_PATH)
            func = cloudpickle.loads(
                (autotune_path / AUTOTUNE_FUNCTION_PATH).read_bytes()
            )
            latency_data = _read_json(autotune_path / AUTOTUNE_LATENCY_PATH)
        except Exception as exc:
            logger.error(
                f"Failed to load autotune metadata from {autotune_path}: {exc}"
            )
            return None

        latency = latency_data["latency"]
        ref_latency = latency_data["ref_latency"]

        kernel = self._load_npu_kernel_from_disk(cache_path, func=func, out_idx=out_idx)
        if kernel is None:
            return None

        kernel.update_tuner_result(
            config=config, latency=latency, ref_latency=ref_latency
        )
        from tilelang.autotuner.param import AutotuneResult

        return AutotuneResult(
            config=config,
            func=func,
            kernel=kernel,
            libcode=kernel.get_kernel_source(),
            latency=latency,
            ref_latency=ref_latency,
        )

    def _save_npu_kernel_to_disk(
        self, cache_path: Path, kernel: JitKernel_NPU, verbose: bool = False
    ) -> None:
        if kernel.mlir_content is not None:
            self._try_save(
                "kernel MLIR",
                lambda: (cache_path / AUTOTUNE_KERNEL_MLIR_PATH).write_text(
                    kernel.mlir_content
                ),
            )
        self._try_save(
            "wrapped kernel",
            lambda: (cache_path / AUTOTUNE_WRAPPED_KERNEL_PATH).write_bytes(
                kernel.get_kernel_source()
            ),
        )

        self._try_save(
            "main.so",
            lambda: shutil.copy(
                kernel.so_launcher_path, cache_path / AUTOTUNE_SO_LAUNCHER_PATH
            ),
        )

        metadata = {
            "symbolic": kernel.symbolic,
            "params": kernel.params,
            "param_info": kernel.param_info,
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
        self._try_save(
            "metadata",
            lambda: (cache_path / AUTOTUNE_METADATA_PATH).write_bytes(
                cloudpickle.dumps(metadata)
            ),
        )

    def _load_npu_kernel_from_disk(
        self,
        cache_path: Path,
        func: Callable,
        out_idx: Optional[List[int]],
    ) -> Optional[JitKernel_NPU]:
        if not cache_path.exists():
            return None

        kernel_source: Optional[str] = None
        kernel_global_source: Optional[bytes] = None
        metadata: Optional[dict] = None

        try:
            kernel_source = (cache_path / AUTOTUNE_KERNEL_MLIR_PATH).read_text()
        except Exception as exc:
            logger.error(f"Error loading kernel MLIR: {exc}")

        try:
            kernel_global_source = (
                cache_path / AUTOTUNE_WRAPPED_KERNEL_PATH
            ).read_bytes()
        except Exception as exc:
            logger.error(f"Error loading wrapped kernel: {exc}")

        try:
            metadata = cloudpickle.loads(
                (cache_path / AUTOTUNE_METADATA_PATH).read_bytes()
            )
        except Exception as exc:
            logger.error(f"Error loading metadata: {exc}")

        if not (kernel_global_source and metadata):
            logger.warning(f"Incomplete NPU kernel artefacts at {cache_path}.")
            return None

        return JitKernel_NPU.from_database(
            mod=func,
            kernel_source=kernel_source,
            kernel_launcher_path=str(cache_path / AUTOTUNE_SO_LAUNCHER_PATH),
            kernel_utils_path=None,
            metadata=metadata,
            out_idx=out_idx,
        )

    def _try_save(self, label: str, fn: Callable) -> None:
        try:
            fn()
        except Exception as exc:
            logger.error(f"Error saving {label}: {exc}")


def _write_json(path: Path, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f)


def _read_json(path: Path):
    with open(path) as f:
        return json.load(f)
