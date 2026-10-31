"""Hardware-free contract check for the first migrated A2/A3 native pass."""

from tilelang import tvm
import pytest


@pytest.mark.parametrize("with_symbolic_launch", [False, True])
def test_a2a3_host_legalize_preserves_body_and_tiling_attrs(with_symbolic_launch):
    make_pass = tvm.ffi.get_global_func("tl.transform.A2A3HostLegalize")
    original = tvm.tirx.PrimFunc([], tvm.tirx.Evaluate(1)).with_attr("global_symbol", "main")
    if with_symbolic_launch:
        tirx = tvm.tirx
        n, bound, index, cid = [tirx.Var(name, "int32") for name in ("n", "bound", "i", "cid")]
        # A TIR LetStmt's scoped body becomes a Bind followed by its uses in
        # TIRX. Exercise that sequence, thread_extent, and symbolic loop bounds.
        thread = tirx.IterVar(tvm.ir.Range(0, 4), cid, tirx.IterVar.ThreadIndex, "blockIdx.x")
        loop = tirx.For(index, cid, bound, tirx.ForKind.SERIAL, tirx.Evaluate(index + bound))
        body = tirx.SeqStmt([tirx.Bind(bound, n + 1), tirx.AttrStmt(thread, "thread_extent", 4, loop)])
        original = tirx.PrimFunc([n], body).with_attr("global_symbol", "main")
    original = original.with_attr("test_metadata", "keep")

    result = make_pass()(tvm.IRModule.from_expr(original))["main"]

    tvm.ir.assert_structural_equal(result.body, original.body)
    assert "tiling_map" in result.attrs
    assert "var_sequence" in result.attrs
    assert len(result.attrs["tiling_map"]) == 0
    assert len(result.attrs["var_sequence"]) == 0
    assert result.attrs["test_metadata"] == "keep"
    tvm.ir.assert_structural_equal(result.params, original.params)
    # The pass must not modify the input PrimFunc in place.
    assert "tiling_map" not in original.attrs
