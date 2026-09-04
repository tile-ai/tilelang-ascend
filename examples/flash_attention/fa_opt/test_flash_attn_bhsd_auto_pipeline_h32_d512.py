import runpy
import sys
from pathlib import Path

import torch


def _run_example(*argv):
    source = Path(__file__).with_name("flash_attn_bhsd_auto_pipeline_h32_d512.py")

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


def test_flash_attn_bhsd_auto_pipeline_h32_d512_accuracy() -> None:
    # Head count and head dimension are what this variant is tuned for and stay
    # at the example's values; the batch and sequence length come down so that
    # a 512-wide head does not take the device on its own.
    _run_example("--B", "1", "--S", "1024")
