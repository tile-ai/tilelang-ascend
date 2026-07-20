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
import tilelang.language as T
from tilelang.transform.pass_config import (
    PassConfigKey,
    normalize_compiler_options,
    process_default_pass_config,
)
from tilelang.jit.adapter.libgen import LibraryGenerator, resolve_compile_flags
from tilelang.cache.kernel_cache import KernelCache

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
    """pass_configs for `target`; vs=None means the target's own default."""
    return process_default_pass_config(target, None if vs is None else (VS_ON if vs else VS_OFF))


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
    """Build the bisheng command without running it."""
    captured = {}

    class _Ret:
        returncode = 0

    def _run(cmd, *a, **k):
        captured["cmd"] = list(cmd)
        return _Ret()

    gen = LibraryGenerator(target, "A3", compile_flags)
    gen.update_lib_code("// x")
    with (
        mock.patch("tilelang.jit.adapter.libgen._get_ascend_home_path", return_value="/a"),
        mock.patch("tilelang.jit.adapter.libgen._get_tl_root", return_value="/t"),
        mock.patch("tilelang.jit.adapter.libgen.subprocess.run", _run),
        mock.patch.dict("os.environ", env or {}, clear=True),
    ):
        gen.compile_lib()
    return captured["cmd"]


def test_libgen_appends_flags():
    cmd = _bisheng_command("ascendc", ["-O3", "--cce-auto-sync=off"])
    assert "-O3" in cmd and "--cce-auto-sync=off" in cmd
    assert cmd.index("-O3") < cmd.index("-o")  # appended before the output arg
    # a whitespace-joined flag string is split, matching upstream tilelang
    assert "--cce-enable-print" in _bisheng_command("pto", ["-D_DEBUG --cce-enable-print"])
    # a flag already present in the base command is not duplicated
    assert _bisheng_command("ascendc", ["-fPIC"]).count("-fPIC") == 1


def test_libgen_direct_caller_falls_back_to_derived():
    # compile_flags=None (direct/legacy caller) -> framework defaults, env still honored.
    assert "-O2" in _bisheng_command("ascendc", None)
    assert "-O3" in _bisheng_command("ascendc", None, {"TL_CCE_OPT_LEVEL": "3"})


# ---------------------------------------------------------------------------
# compile(): flags resolved once, then threaded down to the kernel
# ---------------------------------------------------------------------------


@T.prim_func
def _copy_kernel(A: T.Tensor((16, 16), "float16"), B: T.Tensor((16, 16), "float16")):
    with T.Kernel(1, is_npu=True) as (cid, vid):
        T.copy(A, B)


def test_compile_resolves_and_threads_flags(clean_env):
    """compile() resolves the flags once and hands them to JITKernel -- the hop
    that makes them kernel-scoped. Adapter creation (and lowering) is stubbed."""
    from tilelang.jit.kernel import JITKernel

    seen = {}

    def _spy(self, func, out_idx, workspace_idx):
        seen["flags"] = self.compile_flags
        return types.SimpleNamespace(func=lambda *a, **k: None)

    with (
        mock.patch("tilelang.cache.kernel_cache.is_cache_enabled", return_value=False),
        mock.patch.object(JITKernel, "_compile_and_create_adapter", _spy),
    ):
        tilelang.compile(_copy_kernel, target="ascendc", out_idx=[1], compile_flags=["-O3"])

    # derived "-O2" first, caller "-O3" appended last (last-wins).
    assert seen["flags"] == ["-O2", "-O3"]


# ---------------------------------------------------------------------------
# Cache key + autotuner disk-restore
# ---------------------------------------------------------------------------


def test_cache_key_scopes_flags(clean_env):
    def key(flags):
        return KernelCache()._generate_key(
            func=_copy_kernel,
            out_idx=[1],
            workspace_idx=[],
            auto_gm_idx=[],
            execution_backend="cython",
            args=(),
            target="ascendc",
            target_host=None,
            platform="A3",
            pass_configs={},
            compile_flags=flags,
        )

    # VS-on and VS-off kernels resolve to different flags and must not alias.
    on = resolve_compile_flags("ascendc", _pc("ascendc", True))
    off = resolve_compile_flags("ascendc", _pc("ascendc", False))
    assert on != off and key(on) != key(off)
    assert key(on) == key(on)  # deterministic


def test_autotuner_restore_threads_full_signature(tmp_path):
    """Regression: the disk-restore call to JITKernel.from_database dropped the
    now-required platform / workspace_idx / auto_gm_idx args; it must supply every
    required arg and thread compile_flags + the persisted auto_gm_idx + the
    resolved platform."""
    import inspect
    import json as _json

    import cloudpickle

    from tilelang.jit.kernel import JITKernel
    from tilelang.autotuner import param as ap

    required = {
        n
        for n, p in inspect.signature(JITKernel.from_database).parameters.items()
        if p.default is inspect.Parameter.empty and n not in ("cls", "self")
    }
    (tmp_path / ap.WRAPPED_KERNEL_PATH).write_text("// x")
    (tmp_path / ap.KERNEL_LIB_PATH).write_bytes(b"")
    with open(tmp_path / ap.PARAMS_PATH, "wb") as f:
        cloudpickle.dump(["p"], f)
    with open(tmp_path / ap.AUTO_GM_IDX_PATH, "w") as f:
        _json.dump([2], f)

    captured = {}

    def _spy(**kw):
        captured.update(kw)
        return "K"

    with mock.patch.object(JITKernel, "from_database", _spy), mock.patch.dict("os.environ", {"TL_PLATFORM": "A5"}):
        result = ap.AutotuneResult._load_kernel_from_disk(
            ap.AutotuneResult,
            tmp_path,
            target="ascendc",
            compile_flags=["-O3"],
            platform="auto",
        )

    assert result == "K"
    assert not (required - set(captured))  # all required args supplied (the bug)
    assert captured["auto_gm_idx"] == [2]  # persisted auto_gm_idx restored
    assert captured["compile_flags"] == ["-O3"]
    assert captured["platform"] == "A5"  # "auto" resolved via TL_PLATFORM (sim path)


def test_compile_args_hash_scopes_platform_and_flags():
    from tilelang.autotuner.param import CompileArgs

    # Distinct platforms / compile flags -> distinct auto-tuner cache keys.
    assert hash(CompileArgs(platform="A2")) != hash(CompileArgs(platform="A3"))
    assert hash(CompileArgs(compile_flags=["-O3"])) != hash(CompileArgs())


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


@pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="requires an Ascend NPU",
)
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
