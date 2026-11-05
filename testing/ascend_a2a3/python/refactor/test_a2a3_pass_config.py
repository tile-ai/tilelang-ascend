"""A2/A3 source defaults and local standard-key compatibility."""

from enum import Enum

import pytest

from tilelang import tvm
from tilelang.ascend.a2a3.pass_config import normalize_a2a3_pass_configs
from tilelang.backend.pass_pipeline.pipeline_utils import allow_vectorize
from tilelang.transform.pass_config import PassConfigKey, normalize_pass_configs

_NATIVE_OPTIONS = {
    "TL_ASCEND_AUTO_SYNC": "tl.ascend_auto_sync",
    "TL_ASCEND_AUTO_SYNC_VS": "tl.ascend_auto_sync_vs",
    "TL_ASCEND_MEMORY_PLANNING": "tl.ascend_memory_planning",
    "TL_ASCEND_AUTO_CV_COMBINE": "tl.ascend_auto_cv_combine",
    "TL_ASCEND_AUTO_CV_SYNC": "tl.ascend_auto_cross_core_sync",
    "TL_ASCEND_VECTOR_MASK_REUSE": "tl.ascend_vector_mask_reuse",
    "TL_ASCEND_TAIL_MASK": "tl.ascend_tail_mask",
    "TL_ASCEND_PTO_USE_PIPE_IN_CV_COPY": "tl.ascend_pto_use_pipe_in_cv_copy",
    "TL_ASCEND_EXCEPTION_DUMP": "tl.ascend_exception_dump",
}


def _target(dialect="ascendc", model="A3"):
    return tvm.target.Target({"kind": "ascend", "arch": "dav-2201", "keys": [dialect], "model": model})


@pytest.mark.parametrize("model", ["A2", "A3"])
def test_ascendc_keeps_empty_python_defaults_for_both_hardware_models(model):
    assert normalize_a2a3_pass_configs(_target(model=model), {}) == normalize_pass_configs({})


def test_pto_default_does_not_replace_an_explicit_false():
    target = _target("pto")
    assert normalize_a2a3_pass_configs(target)["tl.ascend_auto_sync_vs"] is True
    assert normalize_a2a3_pass_configs(target, {PassConfigKey.TL_ASCEND_AUTO_SYNC_VS: False})["tl.ascend_auto_sync_vs"] is False


def test_nine_native_names_remain_strings_and_round_trip_without_mutating_input():
    original = {getattr(PassConfigKey, name): index % 2 == 0 for index, name in enumerate(_NATIVE_OPTIONS)}
    snapshot = dict(original)
    result = normalize_a2a3_pass_configs(_target(), original)
    assert original == snapshot
    for name, value in _NATIVE_OPTIONS.items():
        assert getattr(PassConfigKey, name).value == value
        assert result[value] == snapshot[getattr(PassConfigKey, name)]
        assert type(next(k for k in result if k == value)) is str


def test_legacy_enum_keys_and_tirx_spellings_share_the_current_gate():
    class Legacy(str, Enum):
        DISABLE_VECTORIZE = "tir.disable_vectorize"
        MERGE = "tir.merge_static_smem"

    normalized = normalize_a2a3_pass_configs(_target(), {Legacy.DISABLE_VECTORIZE: True, Legacy.MERGE: False})
    assert normalized["tirx.disable_vectorize"] is True
    assert normalized["tirx.merge_static_smem"] is False
    assert "tir.disable_vectorize" not in normalized
    assert "tir.merge_static_smem" not in normalized
    with tvm.transform.PassContext(config=normalized):
        assert allow_vectorize() is False


@pytest.mark.parametrize(
    "old,new",
    [
        ("tir.disable_cse_tir", "tirx.disable_cse_tir"),
        ("tir.Simplify", "tirx.Simplify"),
        ("tir.disable_storage_rewrite", "tirx.disable_storage_rewrite"),
        ("tir.disable_vectorize", "tirx.disable_vectorize"),
        ("tir.use_async_copy", "tirx.use_async_copy"),
        ("tir.enable_debug", "tirx.enable_debug"),
        ("tir.merge_static_smem", "tirx.merge_static_smem"),
        ("tir.add_lower_pass", "tirx.add_lower_pass"),
        ("tir.noalias", "tirx.noalias"),
    ],
)
def test_source_standard_options_translate_locally_without_modifying_global_normalizer(old, new):
    assert normalize_a2a3_pass_configs(_target(), {old: False})[new] is False
    assert old in normalize_pass_configs({old: False})


def test_target_ownership_is_resolved_before_backend_defaults():
    with pytest.raises(ValueError, match="resolved ascend Target"):
        normalize_a2a3_pass_configs("ascendc")
    with pytest.raises(ValueError, match="arch=dav-2201"):
        normalize_a2a3_pass_configs(tvm.target.Target({"kind": "ascend", "arch": "dav-3510"}))


def test_all_nine_native_bool_options_are_accepted_without_re_registration():
    missing = set(_NATIVE_OPTIONS.values()) - set(tvm.transform.PassContext.list_configs())
    if missing:
        pytest.skip("A2/A3 native config owners are not compiled")
    for flag in [False, True]:
        options = normalize_a2a3_pass_configs(_target(), {key: flag for key in _NATIVE_OPTIONS.values()})
        with tvm.transform.PassContext(config=options) as context:
            assert all(bool(context.config[key]) == flag for key in _NATIVE_OPTIONS.values())
