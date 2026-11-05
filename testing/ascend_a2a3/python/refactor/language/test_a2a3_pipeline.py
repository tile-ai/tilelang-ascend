"""Complete AscendC compiler pipeline with original example pass settings."""

import pytest

from tilelang import tvm
import tilelang.ascend.a2a3.language as T
from tilelang.ascend.a2a3 import pipeline
from tilelang.ascend.a2a3.pass_config import normalize_a2a3_pass_configs
from tilelang.backend import create_backend_context
from tilelang.engine.lower import lower_with_context
from tvm.ir import CallingConv

from _a2a3_source_kernels import indexer, indexer_dynamic, matmul, matmul_add

_NATIVE = tvm.ffi.get_global_func("tl.transform.A2A3AscendVectorMaskLegalize", allow_missing=True) is not None
pytestmark = pytest.mark.skipif(not _NATIVE, reason="complete A2/A3 native compiler is not built")

# Exact @jit pass_configs from TA 0deb8b93 examples. The factory-only fixture
# removes decorators, so restore these instead of silently using defaults.
_EXAMPLE_CONFIGS = {
    "matmul": {},
    "matmul_add": {
        "tl.ascend_auto_sync": True,
        "tl.ascend_auto_cross_core_sync": True,
        "tl.ascend_auto_cv_combine": True,
        "tl.ascend_memory_planning": True,
    },
    "indexer": {"tl.ascend_memory_planning": True},
    "indexer_dynamic": {"tl.ascend_memory_planning": True},
}


def _context():
    return create_backend_context({"kind": "ascend", "arch": "dav-2201", "model": "A3", "keys": ["ascendc"]}, "c", "cython")


@tvm.instrument.pass_instrument
class _Trace:
    def __init__(self):
        self.names = []
        self.mask_kernel = None

    def run_before_pass(self, _mod, info):
        self.names.append(info.name)

    def run_after_pass(self, mod, info):
        if info.name == "tl.A2A3AscendVectorMaskLegalize":
            self.mask_kernel = pipeline.get_device_kernel(mod)


@pytest.mark.parametrize(
    "factory,args",
    [
        (matmul, (256, 256, 128, 128, 128, 64)),
        (matmul_add, (256, 256, 128, 128, 128, 64)),
        (indexer, (2, 1, 32, 512, 4096, 64, 1024, 512, 4, 128, 256, 64)),
        (indexer_dynamic, (1, 32, 64, 1024, 512, 4, 128, 256, 64)),
    ],
)
def test_original_factories_lower_through_complete_pipeline_and_codegen(factory, args):
    context = _context()
    trace = _Trace()
    configs = normalize_a2a3_pass_configs(context.target, _EXAMPLE_CONFIGS[factory.__name__])
    with tvm.transform.PassContext(opt_level=3, config=configs, instruments=[trace]):
        artifact = lower_with_context(factory(*args), context)
    func = pipeline.get_device_kernel(artifact.device_mod)
    assert len(artifact.params) == len(func.params)
    assert len(artifact.host_mod.functions) == 0
    assert int(func.attrs["calling_conv"]) == int(CallingConv.DEVICE_KERNEL_LAUNCH)
    assert func.attrs["npu_platform"] == "A3"
    assert 'extern "C" void call(' in artifact.kernel_source
    assert "main_kernel" in artifact.kernel_source
    if factory is matmul_add:
        assert "AscendC::SetFlag<" in artifact.kernel_source
        assert "AscendC::CrossCore" in artifact.kernel_source
    mask = trace.names.index("tl.A2A3AscendVectorMaskLegalize")
    assert trace.names[mask + 1 :] == ["tirx.Filter", "tirx.Filter"]
    assert func.same_as(trace.mask_kernel)
    assert func.body.same_as(trace.mask_kernel.body)
    assert "tl.A2A3AnnotateDeviceRegions" not in trace.names  # proposal-retained, inactive in TA.
    assert "tl.LowerIntrin" not in trace.names
    assert "address_map" in func.attrs and "size_map" in func.attrs


def _simple_without_kernel():
    @T.prim_func
    def simple(A: T.Tensor((16,), "float32"), B: T.Tensor((16,), "float32")):
        with T.Scope("V"):
            for i in T.serial(16):
                B[i] = A[i] + 1

    return simple


def test_source_compatible_export_without_kernel_marker_still_compiles():
    artifact = lower_with_context(_simple_without_kernel(), _context())
    assert "B.SetValue" in artifact.kernel_source
    assert len(artifact.params) == 2
    assert pipeline.get_device_kernel(artifact.device_mod).attrs["global_symbol"] == "simple"


def test_source_kernel_metadata_rejects_explicit_host_convention_and_multiple_exports():
    context = _context()
    original = _simple_without_kernel()
    with pytest.raises(ValueError, match="explicit host calling convention"):
        context.lower(tvm.IRModule.from_expr(original.with_attr("calling_conv", CallingConv.C_PACKED_FUNC)))
    with pytest.raises(ValueError, match="exactly one exported source"):
        context.lower(tvm.IRModule({"simple": original, "other": original.with_attr("global_symbol", "other")}))
    with pytest.raises(ValueError, match="exactly one exported source"):
        context.lower(tvm.IRModule({}))


def test_missing_native_component_fails_before_partial_lowering(monkeypatch):
    original_get = tvm.ffi.get_global_func
    missing = "tl.transform.A2A3AscendVectorMaskLegalize"

    def get(name, allow_missing=False):
        return None if name == missing else original_get(name, allow_missing=allow_missing)

    context = _context()
    monkeypatch.setattr(tvm.ffi, "get_global_func", get)
    trace = _Trace()
    with tvm.transform.PassContext(instruments=[trace]), pytest.raises(RuntimeError, match="complete A2/A3 native compiler.*USE_ASCEND=ON"):
        context.lower(tvm.IRModule.from_expr(_simple_without_kernel()))
    assert not trace.names


def test_pto_pipeline_applies_the_same_single_export_contract():
    target = tvm.target.Target({"kind": "ascend", "arch": "dav-2201", "model": "A3", "keys": ["pto", "ascend"]})
    with pytest.raises(ValueError, match="exactly one exported source"):
        pipeline.PTO_PIPELINE.lower(tvm.IRModule({}), target)


def test_real_workspace_pass_appends_gm_argument_and_public_metadata():
    @T.prim_func
    def automatic(A: T.Tensor((16, 16), "float16"), B: T.Tensor((16, 16), "float16"), C: T.Tensor((16, 16), "float16")):
        with T.Kernel(1, is_npu=True) as (_cid, _vid):
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

    artifact = lower_with_context(automatic, _context())
    func = pipeline.get_device_kernel(artifact.device_mod)
    assert len(automatic.params) == 3
    assert len(func.params) == len(artifact.params) == 4
    assert [int(index) for index in func.attrs["auto_gm_indices"]] == [3]
    assert [p.shape for p in artifact.params[:3]] == [[16, 16]] * 3
    assert artifact.params[3].shape == [1, 16, 16]
    assert str(artifact.params[3].dtype) == "float16"
    assert "workspace_1_handle" in artifact.kernel_source
