import runpy
import sys
from pathlib import Path

import torch


def test_xattention_paged_accuracy() -> None:
    source = Path(__file__).with_name("xattention_paged.py")

    # Same arrangement as the unpaged variant next to it: shapes and the decode
    # step are module-level constants, the comparison against the reference is
    # under a __main__ guard, and one constant reads the device's cube core
    # count. What differs is that the shared keys and values are addressed
    # through a block table rather than laid out contiguously.
    previous = torch.get_default_device()
    original_argv = sys.argv
    try:
        sys.argv = [str(source)]
        runpy.run_path(str(source), run_name="__main__")
    finally:
        sys.argv = original_argv
        torch.set_default_device(previous)
