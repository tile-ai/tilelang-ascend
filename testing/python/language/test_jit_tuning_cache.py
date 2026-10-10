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
