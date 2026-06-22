import os
import pathlib
import pytest
import shutil
import subprocess
import tempfile
import tilelang
import tilelang.language as T
import torch

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


def lower_pto_source(func):
    with tilelang.tvm.transform.PassContext(opt_level=3, config=PASS_CONFIGS):
        artifact = tilelang.lower(func, target="pto")
    return artifact.kernel_source


def pto_runtime_jit_available():
    bisheng = shutil.which("bisheng")
    if bisheng is None:
        return False

    tl_root = pathlib.Path(__file__).resolve().parents[4]
    pto_include = tl_root / "3rdparty" / "pto-isa" / "include"
    if not (pto_include / "pto" / "pto-inst.hpp").is_file():
        return False

    ascend_home = os.environ.get("ASCEND_HOME_PATH") or os.environ.get(
        "ASCEND_HOME", "/usr/local/Ascend/ascend-toolkit/latest"
    )
    ascend_include = pathlib.Path(ascend_home) / "include"
    if not ascend_include.is_dir():
        return False

    with tempfile.TemporaryDirectory() as temp_dir:
        source_path = pathlib.Path(temp_dir) / "probe.cpp"
        object_path = pathlib.Path(temp_dir) / "probe.o"
        source_path.write_text(
            '#include <pto/pto-inst.hpp>\nextern "C" __global__ __aicore__ void probe() {}\n',
            encoding="utf-8",
        )
        result = subprocess.run(
            [
                bisheng,
                "--cce-aicore-arch=dav-c220",
                "-DMEMORY_BASE",
                "-O2",
                "-std=gnu++17",
                "-xcce",
                f"-I{pto_include}",
                f"-I{ascend_include}",
                "-c",
                str(source_path),
                "-o",
                str(object_path),
            ],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
        )
        return result.returncode == 0


requires_pto_runtime_jit = pytest.mark.skipif(
    not pto_runtime_jit_available(),
    reason="PTO runtime JIT compiler cannot compile pto-isa probe in this environment",
)


@pytest.fixture(scope="session", autouse=True)
def clear_cache():
    tilelang.cache.clear_cache()
    yield


@pytest.fixture
def setup_random_seed():
    torch.manual_seed(0)
    yield


def make_compact_to_aligned_kernel(groups=8, compact_cols=4, aligned_cols=8, dtype="float"):
    @T.prim_func
    def main(A: T.Tensor((groups, compact_cols), dtype), C: T.Tensor((groups, aligned_cols), dtype)):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            src = T.alloc_ub((groups, compact_cols), dtype)
            dst = T.alloc_ub((groups, aligned_cols), dtype)
            with T.Scope("V"):
                T.copy(A, src)
                T.tile.fill(dst, 0.0)
                for i, j in T.Parallel(groups, compact_cols):
                    dst[i, j] = src[i, j]
                T.copy(dst, C)

    return main


def compile_compact_to_aligned(target="pto"):
    return tilelang.compile(
        make_compact_to_aligned_kernel(),
        out_idx=[-1],
        pass_configs=PASS_CONFIGS,
        target=target,
    )


def test_parallel_compact_to_aligned_ub_copy_emits_strided_pto_copy():
    source = lower_pto_source(make_compact_to_aligned_kernel())

    assert "copy_ub_to_ub_strided" in source
    assert ", 8, 4," in source


@requires_pto_runtime_jit
def test_parallel_compact_to_aligned_ub_copy_preserves_padding(setup_random_seed):
    kernel = compile_compact_to_aligned(target="pto")

    a = torch.arange(32, dtype=torch.float32, device="npu").reshape(8, 4)
    expected = torch.zeros((8, 8), dtype=torch.float32, device="npu")
    expected[:, :4] = a

    torch.npu.synchronize()
    out = kernel(a)

    torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)


def make_parallel_1d_multi_store_kernel(length=128, dtype="float"):
    @T.prim_func
    def main(A: T.Tensor((length,), dtype), C: T.Tensor((length,), dtype)):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            src = T.alloc_ub((length,), dtype)
            dst = T.alloc_ub((length,), dtype)
            with T.Scope("V"):
                T.copy(A, src)
                for i in T.Parallel(length):
                    dst[i] = src[i]
                    dst[i] = dst[i] + 1.0
                T.copy(dst, C)

    return main


def test_parallel_1d_multi_store_codegen_does_not_reference_v_thread():
    source = lower_pto_source(make_parallel_1d_multi_store_kernel())

    assert "v_thread" not in source


@requires_pto_runtime_jit
def test_parallel_1d_multi_store_runs_correctly(setup_random_seed):
    kernel = tilelang.compile(
        make_parallel_1d_multi_store_kernel(),
        out_idx=[-1],
        pass_configs=PASS_CONFIGS,
        target="pto",
    )
    a = torch.randn(128, device="npu")

    torch.npu.synchronize()
    out = kernel(a)

    torch.testing.assert_close(out, a + 1.0, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
