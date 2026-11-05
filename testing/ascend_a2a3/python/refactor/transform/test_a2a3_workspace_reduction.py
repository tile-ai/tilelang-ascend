"""Cross-core workspace rewriting and PTO pipe lifecycle source contracts."""

import pytest

from tilelang import tvm
import tilelang.ascend.a2a3.language as T


tir = tvm.tirx
_CONFIG = "tl.ascend_pto_use_pipe_in_cv_copy"


def _kernel(direction="v2c", threads=2, dtype="float32", reads=2, overwrite=False):
    src_scope, dst_scope = ("shared.ub", "shared.l1") if direction == "v2c" else ("wmma.accumulator", "shared.ub")
    spelling = {"float16": "half", "float32": "float", "int32": "int", "bfloat16": "bfloat16_t"}[dtype]
    template = (
        f"tl::ascend::copy_ub_to_l1<{spelling}, 16, 8>"
        if direction == "v2c"
        else f"tl::ascend::copy_l0c_to_ub<{spelling}, {spelling}, layout::RowMajor, 8, 16, false>"
    )

    @T.prim_func
    def main():
        with T.Kernel(4, threads=threads, is_npu=True):
            src = T.alloc_fragment((8, 16), dtype, scope=src_scope)
            dst = T.alloc_fragment((8, 16), dtype, scope=dst_scope)
            T.evaluate(tir.call_extern("handle", template, src.access_ptr("r"), dst.access_ptr("w"), 16, 8, 8, 16))
            for _i in T.serial(reads):
                T.evaluate(tir.call_extern("handle", "consume", dst.access_ptr("r")))
            if overwrite:
                T.evaluate(tir.call_extern("handle", "overwrite", dst.access_ptr("w")))

    return main


def _target(backend="ascendc", model="A3"):
    return tvm.target.Target({"kind": "ascend", "arch": "dav-2201", "keys": [backend, "ascend"], "model": model})


def _run(func, backend="ascendc", use_pipe=True, platform="A3", model="A3"):
    func = func.with_attr("target", _target(backend, model)).with_attr("npu_platform", platform)
    factory = tvm.ffi.get_global_func("tl.transform.A2A3AscendWorkspaceReduction")
    with tvm.transform.PassContext(config={_CONFIG: use_pipe}):
        return factory()(tvm.IRModule({"main": func}))["main"]


def _calls(func):
    calls = []
    tir.stmt_functor.post_order_visit(func.body, lambda node: calls.append(node) if isinstance(node, tir.Call) else None)
    return calls


def _extern_names(func):
    return [call.args[0].value for call in _calls(func) if call.op.name == "tirx.call_extern"]


@pytest.mark.parametrize("direction", ["v2c", "c2v"])
@pytest.mark.parametrize("threads", [None, 1, 2])
@pytest.mark.parametrize("dtype", ["float16", "float32", "int32", "bfloat16"])
def test_real_kernel_modes_preserve_workspace_shape_and_offsets(direction, threads, dtype):
    func = _kernel(direction, threads, dtype)
    result = _run(func)
    assert len(result.params) == 1
    workspace = result.buffer_map[result.params[-1]]
    # Both original tags recorded vector count. V2C reconstructs the whole
    # C-side tile from each vector's M; C2V already has the C-side full shape.
    rows = 8 * (2 if threads is None else threads) if direction == "v2c" else 8
    assert [int(x) for x in workspace.shape] == [4, rows, 16]
    assert workspace.dtype == dtype
    assert [int(x) for x in result.attrs["auto_gm_indices"]] == [0]
    names = _extern_names(result)
    assert any("copy_ub_to_gm" in name if direction == "v2c" else "copy_l0c_to_gm" in name for name in names)
    assert any("copy_gm_to_l1" in name if direction == "v2c" else "copy_gm_to_ub" in name for name in names)
    if threads is not None:
        assert result.attrs["npu_cv_ratio"] == func.attrs["npu_cv_ratio"]


@pytest.mark.parametrize("direction", ["v2c", "c2v"])
@pytest.mark.parametrize("threads", [None, 1, 2])
@pytest.mark.parametrize("use_pipe", [False, True])
def test_pto_flag_controls_complete_pipe_or_workspace_path(direction, threads, use_pipe):
    result = _run(_kernel(direction, threads), "pto", use_pipe)
    names = _extern_names(result)
    if use_pipe:
        assert len(result.attrs["pipe_infos"]) == 1
        info = list(result.attrs["pipe_infos"].values())[0]
        assert int(info["dir_type"]) == (2 if direction == "v2c" else 1)
        assert int(info["slot_num"]) == 1
        assert any("copy_ub_to_pipe" in name if direction == "v2c" else "copy_l0c_to_pipe" in name for name in names)
        assert any("copy_pipe_to_l1" in name if direction == "v2c" else "copy_pipe_to_ub" in name for name in names)
        free = [call for call in _calls(result) if call.op.name == "tl.ascend_free_pipe"]
        assert len(free) == 1
        assert free[0].args[0].value == ("free_pipe_C" if direction == "v2c" else "free_pipe_V")
    else:
        assert "pipe_infos" not in result.attrs
        assert all("pipe" not in name for name in names)


@pytest.mark.parametrize("direction", ["v2c", "c2v"])
def test_read_only_consumption_inserts_one_copy_back(direction):
    result = _run(_kernel(direction, reads=3, overwrite=True))
    assert sum("copy_gm_to_" in name for name in _extern_names(result)) == 1


def test_model_does_not_override_the_pto_key():
    result = _run(_kernel(), backend="ascendc", model="pto")
    assert "pipe_infos" not in result.attrs
    result = _run(_kernel(), backend="pto", model="A2")
    assert "pipe_infos" in result.attrs


def test_pto_pipe_option_retains_its_original_true_default():
    func = _kernel().with_attr("target", _target("pto"))
    factory = tvm.ffi.get_global_func("tl.transform.A2A3AscendWorkspaceReduction")
    result = factory()(tvm.IRModule({"main": func}))["main"]
    assert "pipe_infos" in result.attrs


@pytest.mark.parametrize("platform", ["A2", "A3", "A5"])
@pytest.mark.parametrize("direction", ["v2c", "c2v"])
def test_source_platform_branches_retain_gm_workspace_ownership(platform, direction):
    result = _run(_kernel(direction), "pto", platform=platform)
    assert len(result.params) == (0 if platform == "A5" else 1)
    assert "pipe_infos" in result.attrs


def _skip_kernel(threads=2, missing_shapes=False):
    func = _kernel("v2c", threads)
    buffers = {}

    def visit(node):
        if isinstance(node, tir.SBlock):
            buffers.update({buffer.name: buffer for buffer in node.alloc_buffers})

    tir.stmt_functor.post_order_visit(func.body, visit)
    func = func.with_attr("buffers_skip_vid_reduction", ["src"])
    if not missing_shapes:
        func = func.with_attr("buffer_shapess", {buffers["src"].data: [8, 16], buffers["dst"].data: [32, 16]})
    return func


@pytest.mark.parametrize("threads", [None, 1, 2])
def test_skip_ub_uses_the_full_recorded_destination_shape(threads):
    result = _run(_skip_kernel(threads))
    workspace = result.buffer_map[result.params[-1]]
    assert [int(x) for x in workspace.shape] == [4, 32, 16]
    assert any("copy_gm_to_l1<float, 32, 16>" in name for name in _extern_names(result))


def test_skip_ub_without_shape_metadata_retains_source_diagnostic():
    with pytest.raises(tvm.error.InternalError, match="not found in buffer_shapes"):
        _run(_skip_kernel(missing_shapes=True))
