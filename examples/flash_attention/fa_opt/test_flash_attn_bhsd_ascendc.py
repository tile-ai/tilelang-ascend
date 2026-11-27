import runpy
import sys
from pathlib import Path

import torch


def _run_example(*argv):
    source = Path(__file__).with_name("flash_attn_bhsd_ascendc.py")

    # The example builds its tensors and compares against the reference under a
    # __main__ guard, so it has to be executed under that name; importing it
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


def test_flash_attn_bhsd_ascendc_accuracy() -> None:
    # A batch of one over a quarter of the example's sequence length: the same
    # kernel and the same reference, sized so that several of these can be in
    # flight on one device without exhausting it.
    _run_example("--B", "1", "--S", "1024")
