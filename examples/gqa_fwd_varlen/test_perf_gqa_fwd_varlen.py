import runpy
import sys
from pathlib import Path

import torch


def _run_example(*argv):
    source = Path(__file__).with_name("perf_gqa_fwd_varlen.py")

    # The script is a benchmark, but every shape it benchmarks is compared
    # against the reference first and it exits non-zero if any of them differ,
    # so running it is a correctness check as well. That happens under a
    # __main__ guard, hence run_path rather than an import, and the preset is
    # chosen through the script's own CLI.
    previous = torch.get_default_device()
    original_argv = sys.argv
    try:
        sys.argv = [str(source), *argv]
        runpy.run_path(str(source), run_name="__main__")
    finally:
        sys.argv = original_argv
        torch.set_default_device(previous)


def test_perf_gqa_fwd_varlen_accuracy() -> None:
    # The smallest of the script's presets. The larger ones sweep sequence
    # length to time the kernel, which is not what is being checked here, and
    # they compare the same kernel against the same reference while doing it.
    _run_example("--preset", "small")
