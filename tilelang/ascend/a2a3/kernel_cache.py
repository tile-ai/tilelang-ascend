"""TA-compatible runtime metadata on the shared atomic Cython kernel cache."""

from __future__ import annotations

import errno
import json
import os
from pathlib import Path

from tilelang import env, tvm
from tilelang.backend.module import create_backend_context
from tilelang.engine.param import dump_kernel_params
from tilelang.jit.adapter.cython.kernel_cache import CythonKernelCache
from tilelang.jit.kernel import JITKernel

from .compiler import CompileConfig, resolve_compile_config
from .execution_backend import COMPILER_CONFIG_ATTR, WORKSPACE_IDX_ATTR
from .pass_config import normalize_a2a3_pass_configs
from .runtime import dump_runtime_signature, load_runtime_signature, normalize_memory_indices


def _is_missing_own_library(error: OSError, library_path: str | Path) -> bool:
    """Recognize removal of this requested SO without hiding other load errors."""
    requested = os.fspath(library_path)
    if isinstance(error, PermissionError):
        return False
    if error.errno == errno.ENOENT:
        if error.filename is None or os.fspath(error.filename) != requested:
            return False
    elif error.errno is None:
        # ctypes reports the failed object, which can instead be a dependency.
        if str(error) != f"{requested}: cannot open shared object file: No such file or directory":
            return False
    else:
        return False
    try:
        os.stat(requested)
    except OSError as stat_error:
        return stat_error.errno == errno.ENOENT
    return False


def _workspace_request(indices):
    if indices is None:
        return []
    if isinstance(indices, int):
        return [indices]
    if isinstance(indices, list) and all(isinstance(index, int) for index in indices):
        return indices.copy()
    raise ValueError("workspace_idx should be a list of integers")


def runtime_metadata(kernel):
    """Describe the final ABI alongside the frozen compiler identity."""
    return {
        "schema_version": 1,
        "workspace_idx": kernel.workspace_idx,
        "auto_gm_idx": kernel.auto_gm_idx,
        "compiler_config": kernel.compile_config.to_metadata(),
        "runtime_signature": dump_runtime_signature(kernel.adapter.prim_func),
    }


def stamp_runtime_request(func, workspace_idx, config):
    """Derive the cache identity without mutating a frontend function."""
    return func.with_attr(WORKSPACE_IDX_ATTR, _workspace_request(workspace_idx)).with_attr(
        COMPILER_CONFIG_ATTR, json.dumps(config.to_metadata(), sort_keys=True, separators=(",", ":"))
    )


class A2A3CythonKernelCache(CythonKernelCache):
    """Keep A2/A3 compiler identity and final GM argument metadata together."""

    _instance = None
    device_kernel_path = "device_kernel.asc"
    host_kernel_path = "host_kernel.asc"
    runtime_metadata_path = "a2a3_runtime.json"

    def cached(self, func=None, out_idx=None, *args, workspace_idx=None, compile_config=None, **kwargs):
        context = kwargs.get("backend_context")
        if context is None:
            context = create_backend_context(
                kwargs.get("target") or env.get_default_target(),
                kwargs.get("target_host"),
                kwargs.get("execution_backend") or env.get_default_execution_backend(),
            )
        if context.module.name not in {"ascend_a2a3", "pto_a2a3"} or context.execution_backend.name != "cython":
            raise ValueError("A2A3CythonKernelCache requires the A2A3 Cython backend")
        effective = normalize_a2a3_pass_configs(context.target, kwargs.get("pass_configs"))
        # Resolve once, before the inherited memory/disk lookup. A derived
        # PrimFunc makes this request immutable and avoids singleton cache state.
        config = compile_config or resolve_compile_config(context.target, effective, kwargs.get("compile_flags"))
        if str(context.target.attrs.get("model", "")) != config.platform:
            raise ValueError("A2A3 compiler snapshot requires the same frozen target.model platform")
        if config.dialect != ("pto" if "pto" in context.target.keys else "ascendc"):
            raise ValueError("A2A3 compiler snapshot requires the same target dialect keys")
        derived = stamp_runtime_request(func, workspace_idx, config)
        kwargs.update(backend_context=context, pass_configs=effective)
        return super().cached(derived, out_idx, *args, **kwargs)

    def _get_required_files(self, cache_path):
        return [*super()._get_required_files(cache_path), os.path.join(cache_path, self.runtime_metadata_path)]

    def _save_kernel_source_code_to_disk(self, kernel, cache_path, verbose=False):
        super()._save_kernel_source_code_to_disk(kernel, cache_path, verbose)
        metadata = runtime_metadata(kernel)
        self._safe_write_file(
            os.path.join(cache_path, self.runtime_metadata_path),
            "w",
            lambda file: json.dump(metadata, file, sort_keys=True),
        )

    def _build_kernel(
        self,
        func,
        host_kernel_source,
        device_kernel_source,
        kernel_lib_path,
        kernel_params,
        backend_context,
        out_idx,
        pass_configs,
        compile_flags,
    ):
        path = Path(kernel_lib_path).parent / self.runtime_metadata_path
        with path.open() as file:
            metadata = json.load(file)
        if (
            set(metadata) != {"schema_version", "workspace_idx", "auto_gm_idx", "compiler_config", "runtime_signature"}
            or metadata["schema_version"] != 1
        ):
            raise ValueError("Invalid A2A3 runtime cache metadata")
        saved = CompileConfig.from_metadata(metadata["compiler_config"])
        requested = CompileConfig.from_metadata(json.loads(str(func.attrs[COMPILER_CONFIG_ATTR])))
        if saved.cache_identity != requested.cache_identity:
            raise ValueError("A2A3 cached compiler identity conflicts with the requested snapshot")
        signature = load_runtime_signature(metadata["runtime_signature"])
        if kernel_params is None or len(kernel_params) != len(signature.params):
            raise ValueError("A2A3 cached runtime parameters do not match the final device signature")
        workspace = normalize_memory_indices(metadata["workspace_idx"], len(kernel_params), "workspace_idx")
        requested_workspace = normalize_memory_indices(
            [int(index) for index in func.attrs[WORKSPACE_IDX_ATTR]], len(kernel_params), "workspace_idx"
        )
        if workspace != requested_workspace:
            raise ValueError("A2A3 cached workspace indices conflict with the requested allocation")
        auto_gm = normalize_memory_indices(metadata["auto_gm_idx"], len(kernel_params), "auto_gm_idx")
        if auto_gm != [int(index) for index in signature.attrs.get("auto_gm_indices", [])]:
            raise ValueError("A2A3 cached auto-GM indices conflict with the final device signature")
        from tilelang.engine.lower import extrac_params

        # Validate the complete shape graph in one comparison, including shared
        # symbol relationships across different parameters.
        tvm.ir.assert_structural_equal(
            tvm.ir.load_json(dump_kernel_params(kernel_params)),
            tvm.ir.load_json(dump_kernel_params(extrac_params(signature))),
            map_free_vars=True,
        )
        return JITKernel.from_database(
            func=func,
            host_kernel_source=host_kernel_source,
            device_kernel_source=device_kernel_source,
            kernel_lib_path=kernel_lib_path,
            params=kernel_params,
            target=backend_context.target,
            target_host=backend_context.target_host,
            out_idx=out_idx,
            workspace_idx=workspace,
            auto_gm_idx=auto_gm,
            compile_config=saved,
            runtime_func=signature,
            execution_backend=backend_context.execution_backend.name,
            pass_configs=pass_configs,
            compile_flags=compile_flags,
            backend_context=backend_context,
        )


def clear_cache() -> None:
    """Clear the existing A2/A3 Cython cache's memory and disk entries."""
    A2A3CythonKernelCache().clear_cache()
