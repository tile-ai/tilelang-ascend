"""TA launch, resource scopes and allocation contracts on the eager frontend."""

import pytest

import tilelang
import tilelang.ascend.a2a3.language as T
from tilelang import tvm
from tilelang.language.kernel import KernelLaunchFrame


def _nodes(func, node_type):
    found = []
    tvm.tirx.stmt_functor.post_order_visit(func.body, lambda n: found.append(n) if isinstance(n, node_type) else None)
    return found


def _materialize(func):
    mod = tvm.IRModule.from_expr(func)
    mod = tvm.tirx.transform.BindTarget(tvm.target.Target({"kind": "ascend", "arch": "dav-2201"}))(mod)
    mod = tilelang.transform.MaterializeKernelLaunch(
        lower_grid_binding=True,
        lower_thread_binding=False,
        default_threads=None,
        unsupported_annotations=["cluster_dims"],
        launch_dim_tags=["cthread"],
    )(mod)
    return mod[func.attrs["global_symbol"]]


@pytest.mark.parametrize("threads", [None, 1, 2, [1], [2], (1,), (2,)])
def test_npu_launch_preserves_ta_binding_forms(threads):
    vector_count = 2 if threads is None else threads if isinstance(threads, int) else threads[0]

    @T.prim_func
    def main(A: T.Tensor((16,), "int32")):
        if threads is None:
            with T.Kernel(4, is_npu=True) as (cid, vid):
                A[cid * 2 + vid] = T.get_thread_binding()
        else:
            with T.Kernel(4, threads=threads, is_npu=True) as cid:
                vid = T.get_thread_binding()
                A[cid * 2 + vid] = T.get_thread_extent()

    if threads is None:
        assert "npu_cv_ratio" not in main.attrs
    else:
        assert main.attrs["npu_cv_ratio"].value == f"cv_1_{vector_count}"
    launch_block = next(block for block in _nodes(main, tvm.tirx.SBlock) if block.name_hint == "tilelang_root")
    marker = "tilelang.is_npu_kernel_frame" if threads is None else "tilelang.is_npu_kernel_frame_dev_mode"
    assert launch_block.annotations[marker]
    lowered = _materialize(main)
    extents = {attr.node.thread_tag: int(attr.value) for attr in _nodes(lowered, tvm.tirx.AttrStmt) if attr.attr_key == "thread_extent"}
    assert extents == {"blockIdx.x": 4, "cthread": vector_count}
    assert not any(
        isinstance(bind.value, tvm.tirx.Call) and bind.value.op.name == "tl.launch_thread_idx" for bind in _nodes(lowered, tvm.tirx.Bind)
    )
    assert KernelLaunchFrame.Current() is None


def test_symbolic_grid_and_legacy_parenthesized_cid():
    n = T.symbolic("n")

    @T.prim_func
    def main(A: T.Tensor((n,), "int32")):
        with T.Kernel(T.ceildiv(n, 2), threads=1, is_npu=True) as (cid):
            A[cid] = T.get_block_binding()

    lowered = _materialize(main)
    core = next(
        attr for attr in _nodes(lowered, tvm.tirx.AttrStmt) if attr.attr_key == "thread_extent" and attr.node.thread_tag == "blockIdx.x"
    )
    assert tvm.ir.structural_equal(core.value, tvm.tirx.ceildiv(n, 2))


@pytest.mark.parametrize(
    ("threads", "error", "message"),
    [
        (0, ValueError, "1 or 2"),
        (3, ValueError, "1 or 2"),
        ([], ValueError, "one thread dimension"),
        ([1, 1], ValueError, "one thread dimension"),
        ("2", TypeError, "int, list or tuple"),
    ],
)
def test_invalid_npu_thread_specification(threads, error, message):
    with pytest.raises(error, match=message):

        @T.prim_func
        def main(A: T.Tensor((1,), "int32")):
            with T.Kernel(1, threads=threads, is_npu=True):
                A[0] = 1

    assert KernelLaunchFrame.Current() is None


def test_invalid_npu_grid():
    with pytest.raises(ValueError, match="one block dimension"):

        @T.prim_func
        def main(A: T.Tensor((1,), "int32")):
            with T.Kernel(1, 1, is_npu=True):
                A[0] = 1


def test_resource_scopes_and_nested_macros():
    @T.macro
    def inner(A, index):
        with T.Scope("V"):
            value = T.alloc_var("int32", init=7)
            A[index] = value

    @T.macro
    def outer(A, index):
        with T.Scope("C"):
            inner(A, index)

    @T.prim_func
    def main(A: T.Tensor((4,), "int32")):
        with T.Kernel(2, is_npu=True) as (cid, vid):
            outer(A, cid * 2 + vid)

    lowered = _materialize(main)
    scopes = [attr for attr in _nodes(lowered, tvm.tirx.AttrStmt) if attr.attr_key == "resource_scope"]
    assert [int(scope.value) for scope in scopes] == [1, 0]
    assert all(int(scope.node) == 0 for scope in scopes)
    assert any(attr.same_as(scopes[0]) for attr in _nodes(tvm.tirx.PrimFunc([], scopes[1].body), tvm.tirx.AttrStmt))
    block = next(block for block in _nodes(main, tvm.tirx.SBlock) if "tl.local_var_init" in block.annotations)
    init_map = block.annotations["tl.local_var_init"]
    assert [int(value) for value in init_map.values()] == [7]


def test_ta_allocation_scopes():
    @T.prim_func
    def main(A: T.Tensor((4,), "int32")):
        with T.Kernel(1, threads=1, is_npu=True):
            _shared = T.alloc_shared((4,), "float32")
            _boolean = T.alloc_shared((4,), "bool")
            _fragment = T.alloc_fragment((4,), "float32")
            _l1 = T.alloc_L1((4,), "float32")
            _l0a = T.alloc_L0A((4,), "float32")
            _l0b = T.alloc_L0B((4,), "float32")
            _l0c = T.alloc_L0C((4,), "float32")
            _ub = T.alloc_ub((4,), "float32")
            A[0] = 1

    scopes = [buffer.scope() for block in _nodes(main, tvm.tirx.SBlock) for buffer in block.alloc_buffers]
    assert scopes == [
        "shared",
        "shared.ub",
        "local.fragment",
        "shared.l1",
        "wmma.matrix_a",
        "wmma.matrix_b",
        "wmma.accumulator",
        "shared.ub",
    ]


@pytest.mark.parametrize("form", ["init", "init_scope", "kw_init", "scope", "empty"])
def test_alloc_var_original_argument_forms(form):
    @T.prim_func
    def main(A: T.Tensor((1,), "int32")):
        with T.Kernel(1, threads=1, is_npu=True):
            if form == "init":
                value = T.alloc_var("int32", 3)
            elif form == "init_scope":
                value = T.alloc_var("int32", 3, "local.var")
            elif form == "kw_init":
                value = T.alloc_var("int32", init=3)
            elif form == "scope":
                value = T.alloc_var("int32", "local.var")
            else:
                value = T.alloc_var("int32")
            A[0] = value

    annotations = [
        block.annotations["tl.local_var_init"] for block in _nodes(main, tvm.tirx.SBlock) if "tl.local_var_init" in block.annotations
    ]
    assert bool(annotations) == (form not in ("scope", "empty"))
    if annotations:
        assert [int(value) for value in annotations[0].values()] == [3]


def test_alloc_var_preserves_ta_positional_scope_and_keyword_init_rejection():
    # The TA docstring lists this form, but its argument parser rejects it.
    # Keep the migration aligned with the actual source behavior.
    with pytest.raises(TypeError, match="Initializer specified multiple times"):

        @T.prim_func
        def main(A: T.Tensor((1,), "int32")):
            with T.Kernel(1, threads=1, is_npu=True):
                value = T.alloc_var("int32", "local.var", init=3)
                A[0] = value


def test_alloc_var_dynamic_initializer_is_a_store():
    @T.prim_func
    def main(A: T.Tensor((2,), "int32")):
        with T.Kernel(1, threads=1, is_npu=True):
            value = T.alloc_var("int32", A[0])
            A[1] = value

    stores = _nodes(main, tvm.tirx.BufferStore)
    assert any(store.buffer.scope() == "local.var" and isinstance(store.value, tvm.tirx.BufferLoad) for store in stores)


def test_gpu_warpgroup_helper_is_preserved():
    @T.prim_func
    def main(A: T.Tensor((128,), "int32")):
        with T.Kernel(1, threads=128), T.ws(0):
            A[T.get_thread_binding()] = 1

    assert _nodes(main, tvm.tirx.IfThenElse)


def test_dialects_coexist_without_changing_launches():
    import tilelang.ascend.language as A5
    import tilelang.cuda.language as CUDA

    @CUDA.prim_func
    def cuda(A: CUDA.Tensor((4,), "int32")):
        with CUDA.Kernel(4, threads=32) as bx:
            A[bx] = 1

    @A5.prim_func
    def a5(A: A5.Tensor((4,), "int32")):
        with A5.MixedKernel(2, sids=2) as (bx, sid):
            A[bx * 2 + sid] = 1

    @T.prim_func
    def a2a3(A: T.Tensor((4,), "int32")):
        with T.Kernel(2, is_npu=True) as (cid, vid):
            A[cid * 2 + vid] = 1

    assert "npu_cv_ratio" not in cuda.attrs
    assert "npu_cv_ratio" not in a5.attrs
    assert not any("tilelang.is_npu_kernel_frame" in block.annotations for block in _nodes(a5, tvm.tirx.SBlock))
    assert any("tilelang.is_npu_kernel_frame" in block.annotations for block in _nodes(a2a3, tvm.tirx.SBlock))
    assert CUDA.Kernel is not T.Kernel
    assert A5.Kernel is not T.Kernel
