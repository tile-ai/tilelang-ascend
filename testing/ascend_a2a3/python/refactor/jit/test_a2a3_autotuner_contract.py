"""A2A3 autotuner candidate: frozen flags and final ABI persistence.

TA source 0deb8b93da11b7f0becbcded732d9633b7ca4e34 tracks flags/autoGM but
not declared workspace in the tuner. Workspace forwarding here is an explicit
migration increment. No tuning search, benchmarking, or candidate selection is
replaced by these contracts.
"""

from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path
import types

import pytest
import tilelang
from tilelang import tvm, env
from tilelang.ascend.a2a3 import compiler
from tilelang.ascend.a2a3.kernel_cache import A2A3CythonKernelCache
from tilelang.autotuner import AutoTuner
from tilelang.autotuner.param import CompileArgs, AutotuneResult
from tilelang.backend import create_backend_context
from tilelang.env import CacheState

from tilelang.jit.adapter.cython import adapter as adapter_module

from test_a2a3_public_execution import TARGET, _factories, _small_func


def _factory(size=16):
    return _small_func()


@pytest.fixture
def snapshot_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(compiler, "_get_ascend_home_path", lambda **kwargs: str(tmp_path / "sdk"))
    monkeypatch.setattr(compiler, "_get_tl_root", lambda **kwargs: str(Path(__file__).parents[5]))
    monkeypatch.setattr(compiler.shutil, "which", lambda *args, **kwargs: "/unused/bisheng")
    monkeypatch.setattr(compiler, "_get_simulator_lib_path", lambda *args: str(tmp_path / "sim"))
    monkeypatch.setenv("TL_RUN_MODE", "npu")
    for key in ("TL_CCE_OPT_LEVEL", "TL_CCE_AUTO_SYNC", "TL_PTO_DEBUG", "TILELANG_LAYOUT_COST_MODEL"):
        monkeypatch.delenv(key, raising=False)


def _args(**kwargs):
    return CompileArgs(target=kwargs.pop("target", TARGET), execution_backend="cython", **kwargs)


def test_default_creation_and_offline_key_have_no_target_or_compiler_probe(monkeypatch):
    import tilelang.backend.module as backend_module

    def forbidden(*args, **kwargs):
        pytest.fail("offline/default CompileArgs must not resolve a device/compiler")

    monkeypatch.setattr(backend_module, "create_backend_context", forbidden)
    monkeypatch.setattr(compiler, "resolve_compile_config", forbidden)
    args = CompileArgs()
    tuner = AutoTuner(_factory, [{"size": 16}])
    assert args._frozen is None and tuner.compile_args._frozen is None
    assert isinstance(hash(args), int)
    assert tuner.generate_cache_key(inspect.signature(_factory).parameters, {})


def test_non_a2a3_frozen_request_retains_the_exact_original_hash(snapshot_paths):
    args = CompileArgs(target={"kind": "llvm"}, execution_backend="tvm_ffi")
    data = {
        "out_idx": None,
        "execution_backend": "tvm_ffi",
        "target": str(args.target),
        "target_host": None,
        "verbose": False,
        "pass_configs": None,
    }
    expected = int.from_bytes(hashlib.sha256(json.dumps(data, sort_keys=True).encode()).digest(), "big")
    assert args.__hash__() == expected
    frozen = args.freeze()
    assert frozen.__hash__() == expected
    assert args._frozen is None and frozen is args


def test_actual_run_freezes_before_first_lookup_without_mutating_shared_default(monkeypatch, snapshot_paths):
    import tilelang.backend.module as backend_module

    context = create_backend_context(TARGET, execution_backend="cython")
    resolutions = []
    monkeypatch.setattr(backend_module, "create_backend_context", lambda *args, **kwargs: resolutions.append(args) or context)
    tuner = AutoTuner(_factory, [{"size": 16}])
    default = tuner.compile_args
    sentinel = types.SimpleNamespace(func=None)
    monkeypatch.setattr(tuner, "_memory_cache", {"frozen-key": sentinel})
    monkeypatch.setattr(
        tuner,
        "generate_cache_key",
        lambda *args: "frozen-key" if tuner.compile_args._frozen is not None else pytest.fail("lookup before freeze"),
    )
    monkeypatch.setattr(CacheState, "_enabled", True)
    monkeypatch.setattr(vars(type(env))["TILELANG_DISABLE_CACHE"], "_forced_value", "0")
    monkeypatch.setattr(vars(type(env))["TILELANG_AUTO_TUNING_DISABLE_CACHE"], "_forced_value", "0")
    assert tuner.run() is sentinel
    assert len(resolutions) == 1
    assert default._frozen is None
    assert tuner.compile_args._frozen.compiler_config.platform == "A3"
    assert tuner.compile_args.target.same_as(context.target)
    assert tuner.compile_args.execution_backend == "cython"


def test_candidate_flags_derive_from_the_frozen_environment(snapshot_paths, monkeypatch):
    base = _args(compile_flags=["-DEXPLICIT=1"]).freeze()
    before = hash(base)
    monkeypatch.setenv("TL_PLATFORM", "A2")
    monkeypatch.setenv("TL_RUN_MODE", "sim")
    monkeypatch.setenv("TL_CCE_OPT_LEVEL", "0")
    monkeypatch.setenv("TL_CCE_AUTO_SYNC", "on")
    candidate = replace(base, pass_configs={"tl.ascend_auto_sync_vs": True, "tl.device_compile_flags": ["-DCANDIDATE=1"]})
    _, _, _, config = candidate._prepare_a2a3_program(_small_func())
    assert config.flags == ("-O3", "--cce-auto-sync=off", "-DCANDIDATE=1", "-DEXPLICIT=1")
    assert config.platform == "A3" and config.run_mode == "npu"
    assert hash(base) == before
    assert config.working_directory == base._frozen.compiler_config.working_directory
    # A later tuning request captures the new environment, not an old memo.
    fresh = _args(compile_flags=["-DEXPLICIT=1"]).freeze()
    assert fresh._frozen.compiler_config.flags == ("-O0", "-DEXPLICIT=1")
    assert hash(fresh) != hash(base)


def test_explicit_o2_environment_is_distinct_from_absent_default_o2(snapshot_paths, monkeypatch):
    absent = _args().freeze()
    monkeypatch.setenv("TL_CCE_OPT_LEVEL", "2")
    explicit = _args().freeze()
    assert absent._frozen.compiler_config.cache_identity == explicit._frozen.compiler_config.cache_identity
    assert hash(absent) != hash(explicit)
    a = replace(absent, pass_configs={"tl.ascend_auto_sync_vs": True})._prepare_a2a3_program(_small_func())[3]
    b = replace(explicit, pass_configs={"tl.ascend_auto_sync_vs": True})._prepare_a2a3_program(_small_func())[3]
    assert a.flags[0] == "-O3" and b.flags[0] == "-O2"


def test_frontend_attrs_and_candidate_overrides_are_preserved(snapshot_paths, monkeypatch):
    program = _small_func().with_attr("tilelang_out_idx", [2]).with_attr("tilelang_compile_flags", ["-DFUNCTION=1"])
    program = program.with_attr("tilelang_pass_configs", {"tl.ascend_auto_sync_vs": True})
    before = tvm.ir.save_json(program)
    args = _args(workspace_idx=[-2], compile_flags=["-DEXPLICIT=1"], pass_configs={"tl.ascend_auto_sync_vs": False}).freeze()
    capture = {}

    def cached(self, func, out_idx, **kwargs):
        capture.update(func=func, out_idx=out_idx, **kwargs)
        return "captured"

    monkeypatch.setattr(A2A3CythonKernelCache, "cached", cached)
    assert args.compile_program(program) == "captured"
    assert capture["out_idx"] == [2] and capture["workspace_idx"] == [-2]
    assert capture["compile_config"].flags == ("-O2", "-DFUNCTION=1", "-DEXPLICIT=1")
    assert capture["pass_configs"]["tl.ascend_auto_sync_vs"] is False
    assert tvm.ir.save_json(program) == before
    tuner = AutoTuner(_factory, [])
    tuner.compile_args = args
    merged = tuner._merge_pass_configs_into_compile_args({"tl.ascend_auto_sync_vs": True})
    assert merged._frozen is args._frozen
    assert merged.workspace_idx == [-2] and merged.compile_flags == ["-DEXPLICIT=1"]
    assert merged._prepare_a2a3_program(_small_func())[3].flags[:2] == ("-O3", "--cce-auto-sync=off")


def test_original_decorator_flags_and_workspace_reach_tuner_compile_args(snapshot_paths):
    from tilelang.autotuner import autotune

    @autotune(configs=[{"size": 16}])
    @tilelang.jit(target=TARGET, execution_backend="cython", out_idx=[2], workspace_idx=[-2], compile_flags=["-DORIGINAL=1"])
    def factory(size: int):
        return _small_func()

    tuner = factory.get_tunner()
    assert tuner.compile_args.workspace_idx == [-2]
    assert tuner.compile_args.compile_flags == ["-DORIGINAL=1"]
    assert tuner.compile_args._frozen is None
    args = tuner.compile_args.freeze()
    merged = replace(args, pass_configs={"tl.ascend_auto_sync_vs": True})
    assert merged._prepare_a2a3_program(_small_func())[3].flags == ("-O3", "--cce-auto-sync=off", "-DORIGINAL=1")


@pytest.mark.parametrize("dialect", ["ascendc", "pto"])
@pytest.mark.parametrize("platform", ["A2", "A3"])
def test_frozen_candidates_keep_dialect_platform_and_source_overrides(snapshot_paths, monkeypatch, dialect, platform):
    target = {"kind": "ascend", "arch": "dav-2201", "model": platform, "keys": [dialect]}
    args = CompileArgs(target=target, execution_backend="cython").freeze()
    program = _small_func().with_attr("tilelang_pass_configs", {"tl.ascend_auto_sync_vs": False, "tl.layout_cost_model": "io-aware"})
    monkeypatch.setenv("TL_PLATFORM", "A2" if platform == "A3" else "A3")
    monkeypatch.setenv("TL_RUN_MODE", "sim")
    monkeypatch.setenv("TL_CCE_OPT_LEVEL", "invalid-after-freeze")
    monkeypatch.setenv("TILELANG_LAYOUT_COST_MODEL", "invalid-after-freeze")
    _, pass_configs, _, config = args._prepare_a2a3_program(program)
    assert config.platform == platform and config.dialect == dialect and config.run_mode == "npu"
    assert config.flags[0] == "-O2"
    assert pass_configs["tl.ascend_auto_sync_vs"] is False
    assert pass_configs["tl.layout_cost_model"] == "io-aware"
    assert args.freeze() is args
    candidate = replace(args, pass_configs={"tl.ascend_auto_sync_vs": True, "tl.device_compile_flags": ["-mllvm", "candidate"]})
    _, _, _, config = candidate._prepare_a2a3_program(program)
    assert config.flags[0] == "-O3"
    assert config.flags[config.flags.index("-mllvm") + 1] == "candidate"


def test_unset_layout_environment_does_not_enter_later_candidate_normalization(snapshot_paths, monkeypatch):
    args = _args().freeze()
    original_key = hash(args)
    monkeypatch.setenv("TILELANG_LAYOUT_COST_MODEL", "io-aware")
    _, pass_configs, _, _ = args._prepare_a2a3_program(_small_func())
    assert pass_configs["tl.layout_cost_model"] == "register-count"
    assert hash(args) == original_key
    fresh = _args().freeze()
    assert hash(fresh) != original_key
    assert fresh._prepare_a2a3_program(_small_func())[1]["tl.layout_cost_model"] == "io-aware"


def test_internal_cache_rejects_frozen_dialect_or_platform_conflict(snapshot_paths):
    state = _args().freeze()._frozen
    for target in [
        {"kind": "ascend", "arch": "dav-2201", "model": "A2", "keys": ["ascendc"]},
        {"kind": "ascend", "arch": "dav-2201", "model": "A3", "keys": ["pto"]},
    ]:
        context = create_backend_context(target, execution_backend="cython")
        with pytest.raises(ValueError, match="same.*(platform|dialect)"):
            A2A3CythonKernelCache().cached(_small_func(), backend_context=context, compile_config=state.compiler_config)


def test_shared_compile_forwarding_keeps_existing_defaults_and_new_requested_flags(monkeypatch):
    captured = []
    monkeypatch.setattr(tilelang, "compile", lambda program, **kwargs: captured.append(kwargs))
    program = _small_func()
    args = CompileArgs(target="llvm", execution_backend="tvm_ffi")
    args.compile_program(program)
    assert "workspace_idx" not in captured[-1] and "compile_flags" not in captured[-1]
    replace(args, compile_flags=["-DREQUESTED=1"]).compile_program(program)
    assert captured[-1]["compile_flags"] == ["-DREQUESTED=1"]
    assert captured[-1]["target"] == "llvm" and captured[-1]["execution_backend"] == "tvm_ffi"


@pytest.mark.parametrize("dialect", ["ascendc", "pto"])
def test_real_decorator_callbacks_freeze_outer_cache_and_keep_debug_artifacts(snapshot_paths, monkeypatch, tmp_path, dialect):
    from tilelang.autotuner import autotune
    from tilelang.autotuner.param import AutotuneResult
    import tilelang.ascend.a2a3.language as T

    target = {"kind": "ascend", "arch": "dav-2201", "model": "A3", "keys": [dialect]}
    records = []
    runs = []

    @autotune(configs=[{"size": 8}, {"size": 16, "pass_configs": {"tl.ascend_auto_sync_vs": True}}])
    @tilelang.jit(
        target=target,
        execution_backend="cython",
        out_idx=[2],
        workspace_idx=[-2],
        compile_flags=["-DEXPLICIT=1"],
        pass_configs={"tl.ascend_auto_sync_vs": False},
        debug_root_path=str(tmp_path / "debug"),
    )
    def factory(size: int):
        current = tvm.target.Target.current(allow_none=True)
        assert current.attrs["model"] == "A3" and dialect in current.keys

        @T.prim_func
        def program(A: T.Tensor((16, 16), "float16"), B: T.Tensor((16, 16), "float16"), C: T.Tensor((16, 16), "float16")):
            with T.Kernel(1, is_npu=True):
                T.evaluate(0)

        return program.with_attr("tune_size", size)

    def cached(self, program, out_idx, **kwargs):
        record = dict(program=program, out_idx=out_idx, **kwargs)
        records.append(record)
        return types.SimpleNamespace(get_kernel_source=lambda: f"// source {int(program.attrs['tune_size'])}")

    def run(tuner, *args, **kwargs):
        runs.append(tuner.compile_args)
        assert tuner._kernel_parameters == ((), ())
        # A candidate must use the request frozen before this environment change.
        with monkeypatch.context() as changed:
            changed.setenv("TL_PLATFORM", "A2")
            changed.setenv("TL_RUN_MODE", "sim")
            changed.setenv("TL_CCE_OPT_LEVEL", "invalid-after-lookup")
            changed.setenv("TILELANG_LAYOUT_COST_MODEL", "io-aware")
            first = tuner.jit_compile(size=8)
            best = tuner.jit_compile(size=16, __pass_configs__={"tl.ascend_auto_sync_vs": True})
        assert first is not best
        return AutotuneResult(kernel=best, config={"size": 16, "pass_configs": {"tl.ascend_auto_sync_vs": True}})

    monkeypatch.setattr(A2A3CythonKernelCache, "cached", cached)
    monkeypatch.setattr(AutoTuner, "run", run)
    original_configs = factory.jit_impl.pass_configs.copy()
    first = factory()
    assert factory() is first and len(runs) == 1
    assert [record["compile_config"].flags[0] for record in records] == ["-O2", "-O3"]
    for record in records:
        assert record["workspace_idx"] == [-2] and record["out_idx"] == [2]
        assert record["compile_config"].dialect == dialect and record["compile_config"].platform == "A3"
        assert record["compile_config"].run_mode == "npu" and record["compile_config"].flags[-1] == "-DEXPLICIT=1"
        assert record["pass_configs"]["tl.layout_cost_model"] == "register-count"
    assert factory.jit_impl.pass_configs == original_configs
    debug = tmp_path / "debug"
    assert (debug / "tilelang_jit_kernel_factory.c").read_text() == "// source 16\n"
    assert (debug / "tilelang_jit_program_factory.py").read_text() == records[-1]["program"].script() + "\n"
    monkeypatch.setenv("TL_CCE_OPT_LEVEL", "0")
    second = factory()
    assert second is not first and len(runs) == 2
    assert [record["compile_config"].flags[0] for record in records[-2:]] == ["-O0", "-O0"]


def test_shared_outer_cache_keeps_the_original_argument_key(monkeypatch):
    from tilelang.autotuner import autotune
    from tilelang.autotuner.param import AutotuneResult

    @autotune(configs=[{"size": 16}])
    @tilelang.jit(target="llvm", execution_backend="tvm_ffi")
    def factory(size: int = 16):
        return _small_func()

    sentinel = object()
    calls = []
    monkeypatch.setattr(
        AutoTuner,
        "run",
        lambda tuner, *args, **kwargs: calls.append(tuner.compile_args) or AutotuneResult(kernel=sentinel, config={"size": 16}),
    )
    assert factory() is sentinel
    monkeypatch.setenv("TL_CCE_OPT_LEVEL", "1")
    assert factory() is sentinel and len(calls) == 1
    assert calls[0]._frozen is None
    assert next(iter(factory._tuner_cache)) == ((), ())


def test_npu_compile_threads_bind_the_device_captured_by_the_caller(snapshot_paths, monkeypatch):
    import threading
    import torch

    caller = threading.get_ident()
    events = []
    local = threading.local()

    def current_device():
        assert threading.get_ident() == caller
        events.append(("capture", caller, 7))
        return 7

    def set_device(device):
        local.device = device
        events.append(("bind", threading.get_ident(), device))

    monkeypatch.setattr(
        torch, "npu", types.SimpleNamespace(is_available=lambda: True, current_device=current_device, set_device=set_device), raising=False
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    tuner = AutoTuner(_factory, [])
    tuner.compile_args = _args().freeze()
    monkeypatch.setattr(tuner, "_resolve_num_compile_workers", lambda: 2)

    def compile_candidate(**config):
        assert threading.get_ident() != caller and local.device == 7
        return config["size"]

    pool, futures, _, _ = tuner._prepare_compile_execution([{"size": 8}, {"size": 16}], False, 1, compile_candidate, lambda **kwargs: None)
    try:
        assert [future.result()[0][2] for future in futures] == [8, 16]
    finally:
        pool.shutdown(wait=True)
    assert events[0] == ("capture", caller, 7)
    assert all(event[1] != caller and event[2] == 7 for event in events[1:])


@pytest.fixture
def actual_toolchain(monkeypatch, tmp_path):
    if tvm.ffi.get_global_func("tl.transform.A2A3AscendVectorMaskLegalize", allow_missing=True) is None:
        pytest.skip("requires fresh USE_ASCEND native components")
    monkeypatch.setenv("TL_RUN_MODE", "npu")
    monkeypatch.delenv("TL_CCE_OPT_LEVEL", raising=False)
    monkeypatch.delenv("TL_CCE_AUTO_SYNC", raising=False)
    try:
        compiler.resolve_compile_config(tvm.target.Target(TARGET))
    except (FileNotFoundError, ValueError) as error:
        pytest.skip(f"requires actual AscendC toolchain/resources: {error}")
    # Preserve the raw descriptor override, including None/live environment
    # behavior, instead of restoring a previously resolved directory string.
    monkeypatch.setattr(vars(type(env))["TILELANG_CACHE_DIR"], "_forced_value", str(tmp_path / "kernel_cache"))
    monkeypatch.setattr(vars(type(env))["TILELANG_DISABLE_CACHE"], "_forced_value", "0")
    monkeypatch.setattr(vars(type(env))["TILELANG_AUTO_TUNING_DISABLE_CACHE"], "_forced_value", "0")
    monkeypatch.setattr(CacheState, "_enabled", True)
    monkeypatch.setattr(adapter_module.CythonKernelAdapter, "_process_buffer_device", lambda self: {})
    import torch

    monkeypatch.setattr(adapter_module, "_device_providers", lambda: (lambda: torch.device("cpu"), lambda device: 0))
    monkeypatch.setattr(A2A3CythonKernelCache(), "_memory_cache", {})
    monkeypatch.setattr(AutoTuner, "_memory_cache", {})


@pytest.mark.parametrize("dialect", ["ascendc", "pto"])
def test_actual_result_restore_preserves_candidate_flags_workspace_and_environment(actual_toolchain, monkeypatch, tmp_path, dialect):
    target = {"kind": "ascend", "arch": "dav-2201", "model": "A3", "keys": [dialect]}
    configs = {
        "tl.ascend_auto_sync": True,
        "tl.ascend_auto_cross_core_sync": True,
        "tl.ascend_auto_cv_combine": True,
        "tl.ascend_memory_planning": True,
        "tl.ascend_auto_sync_vs": False,
    }
    args = _args(target=target, out_idx=[2], workspace_idx=[-1], compile_flags=["-DEXPLICIT=1"], pass_configs=configs).freeze()
    per_config = {"tl.device_compile_flags": ["-DCANDIDATE=1"]}
    candidate = replace(args, pass_configs={**args.pass_configs, **per_config})
    monkeypatch.setenv("TL_CCE_OPT_LEVEL", "0")
    program = _factories().matmul_add(256, 256, 128, 128, 128, 64)
    kernel = candidate.compile_program(program)
    assert kernel.compile_config.flags == ("-O2", "-DCANDIDATE=1", "-DEXPLICIT=1")
    result = AutotuneResult(latency=1.0, config={"size": 128, "pass_configs": per_config}, func=kernel.prim_func, kernel=kernel)
    path = tmp_path / "results" / "workspace"
    result.save_to_disk(path)
    assert (path / "a2a3_runtime.json").exists()
    monkeypatch.setattr(compiler.LibraryGenerator, "compile_lib", lambda *args, **kwargs: pytest.fail("result restore compiled"))
    restored = AutotuneResult.load_from_disk(path, args)
    assert restored.kernel.workspace_idx == [4] and restored.kernel.out_idx == [2]
    assert restored.kernel.compile_config.cache_identity == kernel.compile_config.cache_identity
    assert restored.kernel.artifact is None
    # A fresh request from the changed environment cannot accept this snapshot.
    with pytest.raises(ValueError, match="compiler identity"):
        AutotuneResult.load_from_disk(
            path, _args(target=target, out_idx=[2], workspace_idx=[-1], compile_flags=["-DEXPLICIT=1"], pass_configs=configs).freeze()
        )
    with pytest.raises(ValueError, match="workspace"):
        AutotuneResult.load_from_disk(path, replace(args, workspace_idx=[1]))
    (path / "a2a3_runtime.json").unlink()
    assert not AutotuneResult._is_complete_result_dir(path, "cython", a2a3=True)
    assert AutotuneResult.load_from_disk(path, args) is None


@pytest.mark.parametrize("dialect", ["ascendc", "pto"])
def test_actual_auto_gm_result_survives_new_process_restore(actual_toolchain, monkeypatch, tmp_path, dialect):
    import tilelang.ascend.a2a3.language as T
    import subprocess
    import sys

    @T.prim_func
    def automatic(A: T.Tensor((16, 16), "float16"), B: T.Tensor((16, 16), "float16"), C: T.Tensor((16, 16), "float16")):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a = T.alloc_L1((16, 16), "float16")
            b = T.alloc_L1((16, 16), "float16")
            c0 = T.alloc_L0C((16, 16), "float")
            ub = T.alloc_ub((16, 16), "float16")
            with T.Scope("C"):
                T.copy(A, a)
                T.copy(B, b)
                T.gemm_v0(a, b, c0, init=True)
                T.copy(c0, ub)
            with T.Scope("V"):
                T.copy(ub, C)

    target = {"kind": "ascend", "arch": "dav-2201", "model": "A3", "keys": [dialect]}
    args = _args(target=target, out_idx=[2]).freeze()
    kernel = args.compile_program(automatic)
    assert len(automatic.params) == 3 and len(kernel.params) == 4
    assert kernel.auto_gm_idx == [3]
    path = tmp_path / "results" / "auto_gm"
    AutotuneResult(latency=1.0, config={}, func=kernel.prim_func, kernel=kernel).save_to_disk(path)
    script = tmp_path / "restore_gm.py"
    script.write_text("""import importlib.util,json,os,sys
if os.environ.get("A2A3_TEST_CYTHON_MODULE"):
 s=importlib.util.spec_from_file_location("tilelang_cython_wrapper",os.environ["A2A3_TEST_CYTHON_MODULE"]);m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m)
from pathlib import Path
import tilelang,tilelang_cython_wrapper
from tilelang.autotuner.param import CompileArgs,AutotuneResult
from tilelang.ascend.a2a3 import compiler
from tilelang.jit.adapter.cython import adapter
adapter.CythonKernelAdapter._process_buffer_device=lambda self:{}
compiler.LibraryGenerator.compile_lib=lambda *a,**kw:(_ for _ in ()).throw(AssertionError("restore compiled"))
args=CompileArgs(target=json.loads(sys.argv[2]),execution_backend="cython",out_idx=[2]).freeze()
os.environ.update(TL_PLATFORM="A2",TL_CCE_OPT_LEVEL="invalid-after-freeze",TL_RUN_MODE="sim",TILELANG_LAYOUT_COST_MODEL="io-aware")
k=AutotuneResult.load_from_disk(Path(sys.argv[1]),args).kernel
assert k.artifact is None
print("RESTORE_GM="+json.dumps({"count":len(k.params),"gm":k.auto_gm_idx,"shape":[int(d)for d in k.params[3].shape],"flags":list(k.compile_config.flags),"module":tilelang.__file__,"cython":tilelang_cython_wrapper.__file__,"native":sorted({line.split()[-1]for line in Path("/proc/self/maps").read_text().splitlines()if "libtilelang.so"in line})}))
""")
    child = subprocess.run([sys.executable, str(script), str(path), json.dumps(target)], capture_output=True, text=True)
    (tmp_path / "child_restore.stdout").write_text(child.stdout)
    (tmp_path / "child_restore.stderr").write_text(child.stderr)
    assert child.returncode == 0, child.stdout + child.stderr
    metadata = json.loads(next(line.removeprefix("RESTORE_GM=") for line in child.stdout.splitlines() if line.startswith("RESTORE_GM=")))
    module = metadata.pop("module")
    native = metadata.pop("native")
    cython = metadata.pop("cython")
    assert Path(module).resolve() == Path(tilelang.__file__).resolve()
    import tilelang_cython_wrapper

    parent_native = sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines() if "libtilelang.so" in line})
    assert native and native == parent_native
    assert Path(cython).resolve() == Path(tilelang_cython_wrapper.__file__).resolve()
    assert metadata == {
        "count": 4,
        "gm": [3],
        "shape": [512] if dialect == "pto" else [1, 16, 16],
        "flags": ["-O3"] if dialect == "pto" else ["-O2"],
    }
    print("fresh-process modules:", module, cython, native)
    sidecar = path / "a2a3_runtime.json"
    invalid = json.loads(sidecar.read_text())
    invalid["auto_gm_idx"] = []
    sidecar.write_text(json.dumps(invalid))
    with pytest.raises(ValueError, match="auto-GM"):
        AutotuneResult.load_from_disk(path, args)
    print("fresh-process tuning result ABI:", metadata)


def test_eager_autotune_uses_the_frozen_candidate_and_frontend_outputs(snapshot_paths, monkeypatch):
    import torch
    import tilelang.ascend.a2a3.language as T
    from tilelang.autotuner import autotune

    @autotune(configs=[{"block": 8}])
    @tilelang.jit(target=TARGET, execution_backend="cython", compile_flags=["-DEAGER=1"])
    def eager(A: T.Tensor((16,), "float32"), block: int):
        C = T.empty((16,), "float32")
        with T.Kernel(1, is_npu=True):  # noqa: SIM117
            with T.Scope("V"):
                for i in T.serial(16):
                    C[i] = A[i] + T.float32(block)
        return C

    records = []
    kernel = object()
    monkeypatch.setattr(
        A2A3CythonKernelCache, "cached", lambda self, program, out_idx, **kwargs: records.append((program, out_idx, kwargs)) or kernel
    )

    def run(tuner, *args, **kwargs):
        with monkeypatch.context() as changed:
            changed.setenv("TL_CCE_OPT_LEVEL", "invalid-after-freeze")
            compiled = tuner.jit_compile(block=8)
        return AutotuneResult(kernel=compiled, config={"block": 8})

    monkeypatch.setattr(AutoTuner, "run", run)
    assert eager(torch.zeros(16), __return_kernel=True) is kernel
    [(program, out_idx, kwargs)] = records
    assert out_idx == [-1] and len(program.params) == 2
    assert kwargs["compile_config"].flags == ("-O2", "-DEAGER=1")


def test_public_jit_get_tir_compile_and_debug_output_keep_their_default_route(monkeypatch, tmp_path):
    import importlib
    import tilelang.language as T

    jit_module = importlib.import_module("tilelang.jit")

    @tilelang.jit(target="llvm", execution_backend="tvm_ffi", compile_flags=["-DPUBLIC=1"], debug_root_path=str(tmp_path))
    def public_factory(size: int = 16):
        @T.prim_func
        def program(A: T.Tensor((size,), "float32")):
            T.evaluate(0)

        return program

    source = "// ordinary public JIT source"
    kernel = types.SimpleNamespace(get_kernel_source=lambda: source)
    captured = []
    monkeypatch.setattr(jit_module, "compile", lambda program, **kwargs: captured.append((program, kwargs)) or kernel)
    program = public_factory.get_tir(16)
    assert public_factory.compile(16) is kernel
    [(compiled, kwargs)] = captured
    tvm.ir.assert_structural_equal(program, compiled)
    assert kwargs["target"] == "llvm" and kwargs["compile_flags"] == ["-DPUBLIC=1"]
    assert (tmp_path / "tilelang_jit_kernel_public_factory.c").read_text() == source + "\n"
    assert (tmp_path / "tilelang_jit_program_public_factory.py").read_text() == compiled.script() + "\n"


@pytest.mark.parametrize("dialect", ["ascendc", "pto"])
def test_actual_default_tuner_candidate_memory_and_result_cache(actual_toolchain, monkeypatch, tmp_path, dialect):
    target = {"kind": "ascend", "arch": "dav-2201", "model": "A3", "keys": [dialect]}
    config = {
        "M": 256,
        "N": 256,
        "K": 128,
        "block_M": 128,
        "block_N": 128,
        "block_K": 64,
        "pass_configs": {"tl.ascend_auto_sync_vs": True, "tl.device_compile_flags": ["-DCANDIDATE=1"]},
    }
    tuner = (
        AutoTuner(_factories().matmul_add, [config])
        .set_compile_args(
            target=target,
            execution_backend="cython",
            out_idx=[2],
            workspace_idx=-1,
            compile_flags=["-DREQUEST=1"],
            pass_configs={
                "tl.ascend_auto_sync": True,
                "tl.ascend_auto_cv_combine": True,
                "tl.ascend_memory_planning": True,
                "tl.ascend_auto_cross_core_sync": True,
                "tl.ascend_auto_sync_vs": False,
            },
        )
        .set_profile_args(skip_check=True, warmup=1, rep=1, timeout=0)
    )
    benchmarked = []
    monkeypatch.setattr(tuner, "_resolve_num_compile_workers", lambda: 1)

    def benchmark(*, jit_kernel, **kwargs):
        # Hardware-independent execution of the real tuner orchestration and
        # Bisheng candidates; this stub is explicitly not a numerical benchmark.
        benchmarked.append(jit_kernel)
        assert jit_kernel.workspace_idx == [4]
        assert jit_kernel.compile_config.flags == (
            "-O3",
            *(["--cce-auto-sync=off"] if dialect == "ascendc" else []),
            "-DCANDIDATE=1",
            "-DREQUEST=1",
        )
        monkeypatch.setenv("TL_CCE_OPT_LEVEL", "invalid-after-candidate")
        return 1.0, None

    monkeypatch.setattr(tuner, "_benchmark_target", benchmark)
    cold = tuner.run()
    assert len(benchmarked) == 1
    assert tuner.run() is cold
    assert list(tuner.cache_dir.rglob("a2a3_runtime.json"))
    tuner._memory_cache.clear()
    monkeypatch.setattr(compiler.LibraryGenerator, "compile_lib", lambda *args, **kwargs: pytest.fail("autotune result cache recompiled"))
    restored = tuner.run()
    assert restored.kernel.artifact is None and len(benchmarked) == 1
    assert restored.kernel.workspace_idx == [4]
    assert restored.kernel.compile_config.cache_identity == cold.kernel.compile_config.cache_identity


@pytest.mark.parametrize("timeout", [0, 1])
def test_npu_benchmark_scope_reaches_worker_and_timeout_call_threads(snapshot_paths, monkeypatch, timeout):
    from contextlib import contextmanager
    import queue
    import threading
    import torch
    from tilelang.autotuner.tuner import _BenchmarkWorkerState

    caller = threading.get_ident()
    events = []
    local = threading.local()

    def current_device():
        assert threading.get_ident() == caller
        events.append(("capture", caller, 7))
        return 7

    @contextmanager
    def device(index):
        local.device = index
        events.append(("enter", threading.get_ident(), index))
        try:
            yield
        finally:
            events.append(("exit", threading.get_ident(), index))
            del local.device

    monkeypatch.setattr(
        torch, "npu", types.SimpleNamespace(is_available=lambda: True, current_device=current_device, device=device), raising=False
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: pytest.fail("A2A3 benchmarking consulted the CUDA device"))
    tuner = AutoTuner(_factory, [])
    tuner.compile_args = _args().freeze()
    active, devices = tuner._resolve_benchmark_devices(benchmark_multi_gpu=False, benchmark_devices=None, target_kind="ascend")
    assert not active and devices == [7]
    work, results = queue.Queue(), queue.Queue()
    start = threading.Event()
    work.put((object(), {}, 0))
    work.put(None)

    def benchmark(**kwargs):
        assert kwargs["benchmark_device"] is None and local.device == 7
        events.append(("benchmark", threading.get_ident(), local.device))
        return 1.0, None

    worker = threading.Thread(
        target=tuner._benchmark_worker_loop, args=(7, work, results, start, "ascend", benchmark, timeout, _BenchmarkWorkerState())
    )
    worker.start()
    start.set()
    result = results.get(timeout=5)
    worker.join(timeout=5)
    assert not worker.is_alive() and result[3:5] == (1.0, None) and result[5] is None and result[6] == ""
    assert events[0] == ("capture", caller, 7)
    assert [event[0] for event in events[1:]] == ["enter", "benchmark", "exit"]
    invocation_thread = events[2][1]
    assert invocation_thread != caller
    assert (invocation_thread == worker.ident) is (timeout == 0)


@pytest.mark.parametrize("dialect", ["ascendc", "pto"])
def test_actual_dynamic_workspace_result_restores_the_shape_graph_and_host_abi(actual_toolchain, monkeypatch, tmp_path, dialect):
    import subprocess
    import sys

    target = {"kind": "ascend", "arch": "dav-2201", "model": "A3", "keys": [dialect]}
    if dialect == "pto":
        from test_a2a3_pto_public_execution import _dynamic_workspace_kernel

        program = _dynamic_workspace_kernel()
        options = dict(out_idx=5, workspace_idx=6)
        names = ["M", "N"]
    else:
        program = _factories().indexer_dynamic(1, 32, 64, 1024, 512, 4, 128, 256, 64)
        options = dict(out_idx=-1, workspace_idx=-3)
        names = ["B", "S1", "S2"]
    args = _args(target=target, pass_configs={"tl.ascend_memory_planning": True}, **options).freeze()
    kernel = args.compile_program(program)
    saved = tmp_path / "result"
    AutotuneResult(latency=1.0, config={}, func=kernel.prim_func, kernel=kernel).save_to_disk(saved)
    sources = [kernel.get_host_source(), kernel.get_kernel_source()]
    declaration = next(line for source in sources if source for line in source.splitlines() if "void call(" in line)
    positions = [declaration.index(f"int64_t {name}") for name in names]
    assert positions == sorted(positions)
    expected = {
        "symbols": names,
        "slots": [list(slot) for slot in kernel.adapter.dynamic_symbolic_map.values()],
        "workspace": kernel.workspace_idx,
        "gm": kernel.auto_gm_idx,
        "library_sha256": hashlib.sha256(Path(kernel.adapter.libpath).read_bytes()).hexdigest(),
    }
    script = tmp_path / "restore_dynamic.py"
    script.write_text("""import hashlib,json,os,sys
from pathlib import Path
import tilelang
from tilelang.autotuner.param import CompileArgs,AutotuneResult
from tilelang.ascend.a2a3 import compiler
from tilelang.jit.adapter.cython import adapter
adapter.CythonKernelAdapter._process_buffer_device=lambda self:{}
compiler.LibraryGenerator.compile_lib=lambda *a,**kw:(_ for _ in ()).throw(AssertionError("result restore compiled"))
args=CompileArgs(target=json.loads(sys.argv[2]),execution_backend="cython",pass_configs={"tl.ascend_memory_planning":True},**json.loads(sys.argv[3])).freeze()
os.environ.update(TL_CCE_OPT_LEVEL="invalid-after-freeze",TL_RUN_MODE="sim",TILELANG_LAYOUT_COST_MODEL="io-aware")
k=AutotuneResult.load_from_disk(Path(sys.argv[1]),args).kernel
assert k.artifact is None
print("DYNAMIC_RESULT="+json.dumps({"symbols":[str(v)for v in k.adapter.dynamic_symbolic_map],"slots":list(k.adapter.dynamic_symbolic_map.values()),"workspace":k.workspace_idx,"gm":k.auto_gm_idx,"library_sha256":hashlib.sha256(Path(k.adapter.libpath).read_bytes()).hexdigest(),"module":tilelang.__file__}))
""")
    child = subprocess.run(
        [sys.executable, str(script), str(saved), json.dumps(target), json.dumps(options)], text=True, capture_output=True
    )
    (tmp_path / "dynamic_child.stdout").write_text(child.stdout)
    (tmp_path / "dynamic_child.stderr").write_text(child.stderr)
    assert child.returncode == 0, child.stdout + child.stderr
    row = json.loads(next(line.removeprefix("DYNAMIC_RESULT=") for line in child.stdout.splitlines() if line.startswith("DYNAMIC_RESULT=")))
    assert Path(row.pop("module")).resolve() == Path(tilelang.__file__).resolve()
    assert row == expected
    print("dynamic host ABI:", declaration)
    print("new-process autotune result:", row)
