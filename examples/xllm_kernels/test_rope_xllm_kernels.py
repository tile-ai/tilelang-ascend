import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


XLLM_ROPE_CASES = [
    pytest.param(16, 4, 128, 128, 0, 20260213, id="tokens16_heads4_dim128_rope128_start0"),
    pytest.param(2051, 2, 128, 128, 0, 20260214, id="tokens2051_heads2_dim128_rope128_start0"),
    pytest.param(1, 1, 128, 128, 0, 101, id="tokens1_heads1_dim128_rope128_start0"),
    pytest.param(7, 3, 128, 128, 0, 102, id="tokens7_heads3_dim128_rope128_start0"),
    pytest.param(64, 4, 128, 128, 0, 107, id="tokens64_heads4_dim128_rope128_start0"),
    pytest.param(8, 5, 128, 128, 0, 103, id="tokens8_heads5_dim128_rope128_start0"),
    pytest.param(9, 5, 128, 128, 0, 104, id="tokens9_heads5_dim128_rope128_start0"),
    pytest.param(4, 64, 128, 128, 0, 108, id="tokens4_heads64_dim128_rope128_start0"),
    pytest.param(127, 8, 128, 128, 0, 105, id="tokens127_heads8_dim128_rope128_start0"),
    pytest.param(33, 16, 128, 128, 0, 106, id="tokens33_heads16_dim128_rope128_start0"),
    pytest.param(1, 1, 576, 64, 512, 20260226, id="tokens1_heads1_dim576_rope64_start512"),
    pytest.param(8, 1, 576, 64, 512, 20260227, id="tokens8_heads1_dim576_rope64_start512"),
    pytest.param(47, 1, 576, 64, 512, 20260301, id="tokens47_heads1_dim576_rope64_start512"),
    pytest.param(48, 1, 576, 64, 512, 20260302, id="tokens48_heads1_dim576_rope64_start512"),
    pytest.param(49, 1, 576, 64, 512, 20260303, id="tokens49_heads1_dim576_rope64_start512"),
    pytest.param(95, 1, 576, 64, 512, 20260304, id="tokens95_heads1_dim576_rope64_start512"),
    pytest.param(96, 1, 576, 64, 512, 20260305, id="tokens96_heads1_dim576_rope64_start512"),
    pytest.param(97, 1, 576, 64, 512, 20260306, id="tokens97_heads1_dim576_rope64_start512"),
    pytest.param(128, 1, 576, 64, 512, 20260228, id="tokens128_heads1_dim576_rope64_start512"),
    pytest.param(512, 1, 576, 64, 512, 20260307, id="tokens512_heads1_dim576_rope64_start512"),
    pytest.param(1024, 1, 576, 64, 512, 20260308, id="tokens1024_heads1_dim576_rope64_start512"),
    pytest.param(2048, 1, 576, 64, 512, 20260225, id="tokens2048_heads1_dim576_rope64_start512"),
]


def _load_rope_example() -> ModuleType:
    source = Path(__file__).with_name("rope.py")
    spec = importlib.util.spec_from_file_location("_xllm_rope_example_for_test", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load example module: {source}")

    module = importlib.util.module_from_spec(spec)
    original_argv = sys.argv
    original_sys_path = list(sys.path)
    try:
        sys.argv = [str(source)]
        sys.path.insert(0, str(source.parent))
        sys.path.insert(0, str(source.parents[2]))
        spec.loader.exec_module(module)
    finally:
        sys.argv = original_argv
        sys.path[:] = original_sys_path
    return module


@pytest.mark.parametrize(
    ("num_tokens", "num_heads", "head_dim", "rope_dim", "start_dim", "seed"),
    XLLM_ROPE_CASES,
)
def test_rope_accuracy(
    num_tokens: int,
    num_heads: int,
    head_dim: int,
    rope_dim: int,
    start_dim: int,
    seed: int,
) -> None:
    example = _load_rope_example()

    example._run_ref_check(
        num_tokens=num_tokens,
        num_heads=num_heads,
        head_dim=head_dim,
        rope_dim=rope_dim,
        start_dim=start_dim,
        seed=seed,
        vec_core_num=example.detect_vec_core_num(),
        ub_buffer_bytes=example.FIXED_UB_BUFFER_BYTES,
    )
