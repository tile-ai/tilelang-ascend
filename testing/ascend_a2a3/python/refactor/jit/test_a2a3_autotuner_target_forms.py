"""Freeze every supported A2A3 target form without reinterpreting manual A5 inputs."""

import json

import pytest
from tilelang import tvm
from tilelang.autotuner.param import CompileArgs
from tilelang.backend import create_backend_context
from tilelang.ascend import target as ascend_target

from test_a2a3_autotuner_contract import snapshot_paths  # noqa: F401


@pytest.mark.parametrize(
    "alias,backend", [("ascend", "ascend_a2a3"), ("asc", "ascend_a2a3"), ("ascendc", "ascend_a2a3"), ("pto", "pto_a2a3")]
)
def test_registered_aliases_freeze_the_resolved_request(monkeypatch, alias, backend):
    monkeypatch.setenv("TL_PLATFORM", "A3")
    monkeypatch.setattr(ascend_target, "_get_soc_name", lambda: pytest.fail("known environment must not probe hardware"))
    context = create_backend_context(alias)
    args = CompileArgs(target=alias)
    frozen = args.freeze()
    assert context.module.name == backend
    assert frozen._frozen is not None and frozen._frozen.context.module.name == backend
    assert frozen.execution_backend == "cython" and frozen.target.attrs["arch"] == "dav-2201"


_CASES = [
    ({"kind": "ascend"}, "ascend"),
    ({"kind": "ascend", "model": "A3", "keys": ["pto", "ascend"]}, "pto"),
    ({"kind": "ascend", "arch": "dav-2201", "model": "A3", "keys": ["ascendc"]}, "ascend_a2a3"),
    ({"kind": "ascend", "arch": "dav-2201", "model": "A3", "keys": ["pto", "ascend"]}, "pto_a2a3"),
    ({"kind": "ascend", "arch": "dav-3510", "model": "A3", "keys": ["ascendc"]}, "ascend"),
    ({"kind": "llvm"}, "cpu"),
]


@pytest.mark.usefixtures("snapshot_paths")
@pytest.mark.parametrize("form", ["dict", "target", "json"])
@pytest.mark.parametrize("config,backend", _CASES)
def test_manual_target_forms_match_the_existing_context_contract(monkeypatch, form, config, backend):
    monkeypatch.setenv("TL_PLATFORM", "A3")
    monkeypatch.setattr(ascend_target, "_get_soc_name", lambda: pytest.fail("manual targets must not probe hardware"))
    request = config.copy() if form == "dict" else (tvm.target.Target(config) if form == "target" else json.dumps(config))
    context = create_backend_context(request)
    args = CompileArgs(target=request)
    before = args.__hash__()
    frozen = args.freeze()
    assert context.module.name == backend
    if backend in {"ascend_a2a3", "pto_a2a3"}:
        assert frozen._frozen is not None and frozen._frozen.context.module.name == backend
        assert frozen._frozen.compiler_config.dialect == ("pto" if backend == "pto_a2a3" else "ascendc")
        assert frozen._frozen.compiler_config.platform == "A3"
    else:
        assert frozen is args and frozen._frozen is None
        assert frozen.__hash__() == before
    if config.get("arch") == "dav-3510":
        assert context.target.attrs["arch"] == "dav-3510"
    if config["kind"] == "ascend" and "arch" not in config:
        # Bare alias strings are detected; manually constructed dictionaries,
        # Targets and JSON configs keep the existing no-arch official backend.
        assert "arch" not in context.target.attrs


@pytest.mark.parametrize("form", ["str", "dict", "target", "json"])
def test_non_ascend_fast_path_never_resolves_a_context(monkeypatch, form):
    import tilelang.backend.module as backend_module

    monkeypatch.setattr(backend_module, "create_backend_context", lambda *args, **kwargs: pytest.fail("non-Ascend freeze probed a context"))
    config = {"kind": "llvm"}
    request = {"str": "llvm", "dict": config, "target": tvm.target.Target(config), "json": json.dumps(config)}[form]
    args = CompileArgs(target=request)
    assert args.freeze() is args
