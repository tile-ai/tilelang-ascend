import runpy
import sys
from pathlib import Path

import pytest
import torch


def _run_example(*argv):
    source = Path(__file__).with_name("flash_attn_bhsd_auto_pipeline_h16_d128.py")

    # The example compiles the kernel and compares against the reference under
    # a __main__ guard, so it has to be executed under that name; importing it
    # would only define the module and check nothing. Its own CLI is where the
    # shape comes from, which is why the arguments go through sys.argv.
    #
    # It also sets the default device at module level. Restoring it keeps that
    # choice from reaching whatever else shares the process.
    previous = torch.get_default_device()
    original_argv = sys.argv
    try:
        sys.argv = [str(source), *argv]
        runpy.run_path(str(source), run_name="__main__")
    finally:
        sys.argv = original_argv
        torch.set_default_device(previous)


@pytest.mark.parametrize(
    ("q_heads", "kv_heads"),
    [(16, 16), (16, 4)],
    ids=["mha_16", "gqa_16q_4kv"],
)
def test_flash_attn_bhsd_auto_pipeline_h16_d128_accuracy(q_heads: int, kv_heads: int) -> None:
    # The head dimension is asserted to be 128 by the kernel and the sequence
    # length has to stay a multiple of its block, so those keep the example's
    # own values; the batch shrinks so several of these fit on one device.
    # The second case exercises the grouped path, where the kernel repeats the
    # key and value heads rather than reading one per query head.
    _run_example(
        "--B",
        "1",
        "--S",
        "1024",
        "--q-heads",
        str(q_heads),
        "--kv-heads",
        str(kv_heads),
    )
