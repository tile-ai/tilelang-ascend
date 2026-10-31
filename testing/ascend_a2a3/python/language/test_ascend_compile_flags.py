"""Tests for kernel-scoped Bisheng compile flags (issue #1386).

#1346 disabled Bisheng auto-sync by writing TL_CCE_AUTO_SYNC / TL_CCE_OPT_LEVEL
into ``os.environ`` (process-wide, never restored), so compiling one kernel
changed how later kernels were compiled. Flags are now derived per kernel and
threaded through the JIT pipeline, with ``compile_flags`` appended last for
caller overrides. The flag-resolution tests are hermetic (no NPU / bisheng);
the final NPU-gated regression covers the intentionally unsynchronized runtime
case.
"""

import types
from unittest import mock

import pytest
import torch

import tilelang
import tilelang.ascend.a2a3.language as T
from tilelang import tvm
from tilelang.ascend.a2a3 import compiler
from tilelang.ascend.a2a3.pass_config import normalize_a2a3_pass_configs
from tilelang.transform.pass_config import (
    PassConfigKey,
    normalize_pass_configs,
)
from tilelang.ascend.a2a3.compiler import LibraryGenerator, resolve_compile_flags, normalize_compiler_options
from tilelang.ascend.a2a3.kernel_cache import A2A3CythonKernelCache, stamp_runtime_request

VS_ON = {PassConfigKey.TL_ASCEND_AUTO_SYNC_VS: True}
VS_OFF = {PassConfigKey.TL_ASCEND_AUTO_SYNC_VS: False}


@pytest.fixture()
def clean_env():
    """Run with the TL_CCE_* / TL_PTO_DEBUG env vars absent."""
    with mock.patch.dict("os.environ", {}, clear=False) as env:
        for k in ("TL_CCE_AUTO_SYNC", "TL_CCE_OPT_LEVEL", "TL_PTO_DEBUG"):
            env.pop(k, None)
        yield


def _pc(target, vs):
    """Effective VS input for the compiler-only tests, without target probing.

    The public PTO default is tested by the native normalization/compile cases;
    these pure flag tests supply that known effective default explicitly.
    """
    if vs is None and target == "pto":
        vs = True
    return normalize_pass_configs(None if vs is None else (VS_ON if vs else VS_OFF))


def _target(dialect="ascendc", platform="A3"):
    return tvm.target.Target({"kind": "ascend", "arch": "dav-2201", "model": platform, "keys": [dialect, "ascend"]})


# ---------------------------------------------------------------------------
# resolve_compile_flags: derived defaults, env fallbacks, caller overrides
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "target, vs, env, user, expected",
    [
        # framework-derived defaults
        ("ascendc", False, {}, None, ["-O2"]),
        ("ascendc", True, {}, None, ["-O3", "--cce-auto-sync=off"]),  # VS pairing (#1346)
        ("pto", None, {}, None, ["-O3"]),  # pto defaults VS on; never gets the auto-sync flag
        # env vars are lenient legacy fallbacks
        ("ascendc", False, {"TL_CCE_AUTO_SYNC": "off"}, None, ["-O2", "--cce-auto-sync=off"]),
        ("ascendc", False, {"TL_CCE_AUTO_SYNC": "garbage"}, None, ["-O2"]),  # malformed -> on, never raises
        ("pto", False, {"TL_PTO_DEBUG": "1"}, None, ["-O2", "-D_DEBUG", "--cce-enable-print"]),
        ("ascendc", False, {"TL_PTO_DEBUG": "1"}, None, ["-O2"]),  # AscendC never gets the PTO debug flags
        # caller flags are appended last, so they win (bisheng is last-wins)
        ("ascendc", True, {}, ["-O0"], ["-O3", "--cce-auto-sync=off", "-O0"]),
        ("ascendc", False, {}, "-O3", ["-O2", "-O3"]),  # a bare str is accepted
    ],
)
def test_resolve_compile_flags(clean_env, target, vs, env, user, expected):
    with mock.patch.dict("os.environ", env):
        assert resolve_compile_flags(target, _pc(target, vs), user) == expected


def test_env_opt_level_invalid_raises(clean_env):
    with mock.patch.dict("os.environ", {"TL_CCE_OPT_LEVEL": "9"}), pytest.raises(ValueError):
        resolve_compile_flags("ascendc")


def test_no_environ_mutation(clean_env):
    import os

    before = dict(os.environ)
    # pto's VS default previously wrote TL_CCE_* into the environment.
    resolve_compile_flags("pto", _pc("pto", None))
    normalize_compiler_options(_pc("ascendc", True))
    assert dict(os.environ) == before
    assert not any(k in os.environ for k in ("TL_CCE_AUTO_SYNC", "TL_CCE_OPT_LEVEL", "TL_PTO_DEBUG"))


# ---------------------------------------------------------------------------
# LibraryGenerator: flags reach the bisheng command
# ---------------------------------------------------------------------------


def _bisheng_command(target, compile_flags, env=None):
    """Freeze a real native target request and capture its compiler argv."""
    captured = {}

    def _run(cmd, *a, **k):
        from pathlib import Path

        captured["cmd"] = list(cmd)
        Path(cmd[-1]).write_bytes(b"captured compiler output")
        return types.SimpleNamespace(returncode=0)

    with (
        mock.patch("tilelang.ascend.a2a3.compiler._get_ascend_home_path", return_value="/a"),
        mock.patch("tilelang.ascend.a2a3.compiler._get_tl_root", return_value="/t"),
        mock.patch("tilelang.ascend.a2a3.compiler.shutil.which", return_value="/a/bin/bisheng"),
        mock.patch("tilelang.ascend.a2a3.compiler.subprocess.run", _run),
        mock.patch.dict("os.environ", env or {}, clear=True),
    ):
        config = compiler.resolve_compile_config(_target(target), compile_flags=compile_flags, run_mode="npu")
        gen = LibraryGenerator(config)
        gen.update_lib_code("// x")
        try:
            gen.compile_lib()
        finally:
            from pathlib import Path

            for path in (gen.srcpath, gen.libpath):
                if path:
                    Path(path).unlink(missing_ok=True)
    return captured["cmd"]


@pytest.mark.a2a3_native
def test_libgen_appends_flags():
    cmd = _bisheng_command("ascendc", ["-O3", "--cce-auto-sync=off"])
    assert "-O3" in cmd and "--cce-auto-sync=off" in cmd
    assert cmd.index("-O3") < cmd.index("-o")  # appended before the output arg
    # a whitespace-joined flag string is split, matching upstream tilelang
    assert "--cce-enable-print" in _bisheng_command("pto", ["-D_DEBUG --cce-enable-print"])
    # a flag already present in the base command is not duplicated
    assert _bisheng_command("ascendc", ["-fPIC"]).count("-fPIC") == 1


@pytest.mark.a2a3_native
def test_libgen_direct_caller_falls_back_to_derived():
    # compile_flags=None (direct/legacy caller) -> framework defaults, env still honored.
    assert "-O2" in _bisheng_command("ascendc", None)
    assert "-O3" in _bisheng_command("ascendc", None, {"TL_CCE_OPT_LEVEL": "3"})


# ---------------------------------------------------------------------------
# compile(): flags resolved once, then threaded down to the kernel
# ---------------------------------------------------------------------------


def _copy_kernel():
    # Keep the original kernel body; defer native elaboration so pure flag
    # regressions remain importable with USE_ASCEND=OFF.
    @T.prim_func
    def main(A: T.Tensor((16, 16), "float16"), B: T.Tensor((16, 16), "float16")):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            T.copy(A, B)

    return main


@pytest.mark.a2a3_native
def test_compile_resolves_and_threads_flags(clean_env, isolated_a2a3_cache):
    """Public compile freezes derived flags once before adapter construction."""
    from tilelang.jit.kernel import JITKernel
    from tilelang.env import CacheState

    seen = {}

    def _spy(self, func, out_idx):
        seen["flags"] = self.compile_flags
        seen["config"] = self.compile_config
        return types.SimpleNamespace(func=lambda *a, **k: None)

    with (
        mock.patch.object(CacheState, "_enabled", False),
        mock.patch.object(JITKernel, "_compile_and_create_adapter", _spy),
        mock.patch.object(JITKernel, "_set_adapter"),
    ):
        tilelang.compile(_copy_kernel(), target=_target(), out_idx=[1], compile_flags=["-O3"])
    assert seen["flags"] == ["-O3"]  # current public API retains the caller request
    assert seen["config"].flags == ("-O2", "-O3")  # actual command has last-wins order


# ---------------------------------------------------------------------------
# Cache key + autotuner disk-restore
# ---------------------------------------------------------------------------


@pytest.mark.a2a3_native
def test_cache_key_scopes_flags(clean_env, isolated_a2a3_cache):
    func = _copy_kernel()
    target = _target()
    cache = A2A3CythonKernelCache()

    def key(vs):
        configs = normalize_a2a3_pass_configs(target, VS_ON if vs else VS_OFF)
        config = compiler.resolve_compile_config(target, configs)
        derived = stamp_runtime_request(func, [], config)
        return cache._generate_key(
            func=derived,
            out_idx=[1],
            execution_backend="cython",
            args=(),
            target=target,
            target_host=None,
            pass_configs=configs,
            compile_flags=None,
        )

    assert key(True) != key(False)
    assert key(True) == key(True)


@pytest.mark.a2a3_native
def test_autotuner_restore_threads_full_signature(tmp_path, clean_env):
    """Modern native restore delegates the complete frozen ABI to the cache."""
    import json
    from tilelang.autotuner import param as ap
    from tilelang.engine.lower import extrac_params
    from tilelang.engine.param import dump_kernel_params
    from tilelang.ascend.a2a3.execution_backend import COMPILER_CONFIG_ATTR, WORKSPACE_IDX_ATTR

    program = _copy_kernel()
    request = ap.CompileArgs(target=_target(), out_idx=[1], workspace_idx=[], compile_flags=["-O3"]).freeze()
    context = request._frozen.context
    (tmp_path / ap.HOST_KERNEL_PATH).write_text("// wrapped")
    (tmp_path / ap.DEVICE_KERNEL_PATH).write_text("// device")
    (tmp_path / ap.KERNEL_LIB_PATH).write_bytes(b"")
    (tmp_path / ap.PARAMS_PATH).write_text(dump_kernel_params(extrac_params(program)))
    (tmp_path / A2A3CythonKernelCache.runtime_metadata_path).write_text("{}")
    captured = {}

    def _spy(self, func, host_source, device_source, library, params, backend_context, out_idx, pass_configs, compile_flags):
        captured.update(
            func=func,
            host=host_source,
            device=device_source,
            library=library,
            params=params,
            context=backend_context,
            out_idx=out_idx,
            pass_configs=pass_configs,
            compile_flags=compile_flags,
        )
        return "K"

    with (
        mock.patch.object(A2A3CythonKernelCache, "_build_kernel", _spy),
        mock.patch.dict("os.environ", {"TL_PLATFORM": "A2", "TL_CCE_OPT_LEVEL": "0"}),
    ):
        result = ap.AutotuneResult._load_kernel_from_disk(
            ap.AutotuneResult, tmp_path, backend_context=context, func=program, compile_args=request
        )
    assert result == "K"
    assert captured["context"] is context
    assert str(context.target.attrs["model"]) == "A3"
    assert captured["out_idx"] == [1]
    assert captured["compile_flags"] == ["-O3"]
    assert len(captured["params"]) == 2
    assert list(captured["func"].attrs[WORKSPACE_IDX_ATTR]) == []
    saved = compiler.CompileConfig.from_metadata(json.loads(str(captured["func"].attrs[COMPILER_CONFIG_ATTR])))
    assert saved.platform == "A3" and saved.flags == ("-O2", "-O3")
    # Sidecar schema/autoGM validation and actual fresh-process restoration are
    # covered by the real A2A3 cache tests, rather than the retired pickle files.


@pytest.mark.a2a3_native
def test_compile_args_hash_scopes_platform_and_flags(clean_env):
    from tilelang.autotuner.param import CompileArgs

    assert hash(CompileArgs(target=_target(platform="A2")).freeze()) != hash(CompileArgs(target=_target(platform="A3")).freeze())
    assert hash(CompileArgs(target=_target(), compile_flags=["-O3"]).freeze()) != hash(CompileArgs(target=_target()).freeze())


# ---------------------------------------------------------------------------
# Runtime regression: negative synchronization case belongs in testing
# ---------------------------------------------------------------------------


def _build_sync_dependency_kernel(auto_sync):
    M = N = 256
    block_M = block_N = 64
    vec_num = 2
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)
    pass_configs = {
        PassConfigKey.TL_ASCEND_AUTO_SYNC: auto_sync,
        PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    }

    @tilelang.jit(
        out_idx=[1],
        target="ascendc",
        pass_configs=pass_configs,
        compile_flags=["--cce-auto-sync=off", "-O3"],
    )
    def build():
        @T.prim_func
        def main(A: T.Tensor((M, N), "float"), B: T.Tensor((M, N), "float")):
            with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
                bx = cid // n_num
                by = cid % n_num
                row_offset = bx * block_M + vid * block_M // vec_num

                a_ub = T.alloc_shared((block_M // vec_num, block_N), "float")
                b_ub = T.alloc_shared((block_M // vec_num, block_N), "float")
                zero_ub = T.alloc_shared((block_M // vec_num, block_N), "float")

                T.copy(A[row_offset, by * block_N], a_ub)
                T.tile.fill(zero_ub, 0.0)
                T.tile.sub(a_ub, zero_ub, a_ub)
                T.tile.exp(a_ub, a_ub)
                T.tile.add(a_ub, a_ub, 1.0)
                T.tile.reciprocal(b_ub, a_ub)
                T.copy(b_ub, B[row_offset, by * block_N])

        return main

    return build()


@pytest.mark.a2a3_device
def test_auto_sync_pass_prevents_dependency_hazard(clean_env):
    torch.manual_seed(0)
    a = torch.randn(256, 256).npu()
    ref = torch.sigmoid(a)

    kernel_with_sync = _build_sync_dependency_kernel(auto_sync=True)
    kernel_without_sync = _build_sync_dependency_kernel(auto_sync=False)
    out_with_sync = kernel_with_sync(a)
    out_without_sync = kernel_without_sync(a)

    torch.testing.assert_close(out_with_sync, ref, rtol=1e-2, atol=1e-2)
    assert not torch.allclose(out_without_sync, ref, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
