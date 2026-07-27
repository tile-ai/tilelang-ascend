import importlib.util
import sys
from pathlib import Path
from types import ModuleType

# Values mirror the argparse defaults in flash_attn_bhsd.py's __main__ block.
B, S, H, D = 1, 128, 1, 512


def _load_flash_attn_example() -> ModuleType:
    source = Path(__file__).with_name("flash_attn_bhsd.py")
    spec = importlib.util.spec_from_file_location("_flash_attn_bhsd_example_for_test", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load example module: {source}")

    module = importlib.util.module_from_spec(spec)
    original_argv = sys.argv
    try:
        # Hide Pytest arguments while loading the example without changing it.
        sys.argv = [str(source)]
        spec.loader.exec_module(module)
    finally:
        sys.argv = original_argv

    return module


def _ref_flash_attn(q, k, v):
    # Mirrors ref_flash_attn defined inside the example's __main__ block, which
    # is not importable from the module scope.
    import torch

    q = q.float()
    k = k.float()
    v = v.float()

    acc = torch.einsum("bhsd,bhkd->bhsk", q, k) * (1.0 / q.shape[-1]) ** 0.5
    acc = acc.softmax(dim=-1)
    o = torch.einsum("bhsk,bhkd->bhsd", acc, v)
    return o.to(torch.float16)


def test_flash_attn_bhsd_accuracy() -> None:
    import torch

    example = _load_flash_attn_example()

    torch.manual_seed(0)

    func = example.flash_attention_fwd(
        batch=B,
        seq_len=S,
        heads=H,
        dim=D,
    )

    q = torch.randn((B, H, S, D), dtype=torch.float16, device="npu")
    k = torch.randn((B, H, S, D), dtype=torch.float16, device="npu")
    v = torch.randn((B, H, S, D), dtype=torch.float16, device="npu")

    output = func(q, k, v)
    ref_output = _ref_flash_attn(q, k, v)
    torch.npu.synchronize()

    torch.testing.assert_close(ref_output, output, rtol=1e-2, atol=1e-2)
