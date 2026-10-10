"""JIT specialization identity must include the autotuner's candidate parameters."""

import importlib
from types import SimpleNamespace

import pytest

jit_module = importlib.import_module("tilelang.jit")


@pytest.fixture
def compiled_programs(monkeypatch):
    programs = []

    def compile_program(program, **kwargs):
        programs.append(program)
        return SimpleNamespace(program=program)

    monkeypatch.setattr(jit_module, "compile", compile_program)
    return programs


def test_jit_cache_distinguishes_tuning_candidates(compiled_programs):
    generated = []

    @jit_module.jit(target="ascendc", platform="A3")
    def factory(n, block_m=64, block_n=32):
        program = (n, block_m, block_n)
        generated.append(program)
        return program

    a = factory(1024, __tune_params={"block_m": 64, "block_n": 32})
    b = factory(1024, __tune_params={"block_m": 128, "block_n": 32})
    a_again = factory(1024, __tune_params={"block_m": 64, "block_n": 32})
    a_reordered = factory(1024, __tune_params={"block_n": 32, "block_m": 64})

    assert a.program == (1024, 64, 32)
    assert b.program == (1024, 128, 32)
    assert a is not b
    assert a_again is a_reordered is a
    assert generated == compiled_programs == [(1024, 64, 32), (1024, 128, 32)]


def test_empty_tuning_parameters_preserve_cache_reuse(compiled_programs):
    @jit_module.jit(target="ascendc", platform="A3")
    def factory(n):
        return n

    kernel = factory(1024)
    assert factory(1024, __tune_params={}) is kernel
    assert compiled_programs == [1024]


def test_tuning_parameters_do_not_override_duplicate_kwargs(compiled_programs):
    @jit_module.jit(target="ascendc", platform="A3")
    def factory(n, block_m):
        return n, block_m

    with pytest.raises(TypeError, match="multiple values.*block_m"):
        factory(1024, block_m=64, __tune_params={"block_m": 128})

    assert compiled_programs == []


@pytest.mark.parametrize(
    ("first", "second"),
    [([64, 32], [128, 32]), ({"block_m": 64}, {"block_m": 128})],
)
def test_unhashable_tuning_parameters_reach_factory(compiled_programs, first, second):
    @jit_module.jit(target="ascendc", platform="A3")
    def factory(n, tile_shape):
        return n, tile_shape

    a = factory(1024, __tune_params={"tile_shape": first})
    b = factory(1024, __tune_params={"tile_shape": second})

    assert a.program == (1024, first)
    assert b.program == (1024, second)
    assert compiled_programs == [(1024, first), (1024, second)]


def test_unhashable_parameters_preserve_factory_errors(compiled_programs):
    @jit_module.jit(target="ascendc", platform="A3")
    def factory(n, tile_shape):
        raise TypeError("factory failure")

    with pytest.raises(TypeError, match="factory failure"):
        factory(1024, __tune_params={"tile_shape": [64, 32]})

    assert compiled_programs == []


def test_public_autotune_keeps_single_worker_candidates_separate(monkeypatch):
    tuner_module = importlib.import_module("tilelang.autotuner.tuner")
    generated = []
    measured = []

    class Kernel:
        def __init__(self, program):
            self.program = self.prim_func = program

        def get_profiler(self, tensor_supply_type):
            def do_bench(**kwargs):
                measured.append(self.program)
                return 256.0 - self.program[1]

            return SimpleNamespace(_get_inputs=lambda **kwargs: [], do_bench=do_bench)

        def get_kernel_source(self):
            return repr(self.program)

        def update_tuner_result(self, latency, config, ref_latency):
            self.config = config
            return self

    monkeypatch.setattr(jit_module, "compile", lambda program, **kwargs: Kernel(program))
    monkeypatch.setattr(tuner_module, "_init_logger_handlers", lambda: None)
    monkeypatch.setattr(tuner_module.env, "is_cache_enabled", lambda: False)
    monkeypatch.setattr(tuner_module.AutoTuner, "_memory_cache", {})
    monkeypatch.setattr(tuner_module.env, "TILELANG_AUTO_TUNING_CPU_COUNTS", "1")
    monkeypatch.setattr(tuner_module.env, "TILELANG_AUTO_TUNING_MAX_CPU_COUNT", "1")
    monkeypatch.setattr(tuner_module.torch.cuda, "is_available", lambda: False)
    if hasattr(tuner_module.torch, "npu"):
        monkeypatch.setattr(tuner_module.torch.npu, "is_available", lambda: False)

    @tuner_module.autotune(
        configs=[{"block_m": 64}, {"block_m": 128}],
        warmup=0,
        rep=1,
        skip_check=True,
    )
    @jit_module.jit(target="ascendc", platform="A3", execution_backend="cython")
    def factory(n, block_m=64):
        program = (n, block_m)
        generated.append(program)
        return program

    best = factory(1024)

    assert generated == [(1024, 64), (1024, 128)]
    assert sorted(measured) == generated
    assert best.program == (1024, 128)
    assert best.config == {"block_m": 128}
