import runpy
import sys
from pathlib import Path

import torch


def test_xattention_accuracy() -> None:
    source = Path(__file__).with_name("xattention.py")

    # Shapes, head counts and the decode step are module-level constants here,
    # and the comparison against the reference happens under a __main__ guard,
    # so the script is run under that name rather than imported. One of those
    # constants reads the device's cube core count, which is why this cannot be
    # collected anywhere but on the accelerator.
    previous = torch.get_default_device()
    original_argv = sys.argv
    try:
        sys.argv = [str(source)]
        runpy.run_path(str(source), run_name="__main__")
    finally:
        sys.argv = original_argv
        torch.set_default_device(previous)
