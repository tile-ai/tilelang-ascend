"""Synchronization-free performance gate for compiler-managed Vector masks.

Run this file under ``msprof op`` in a clean checkout.  The timed kernels use
only UB-local Vector operations: TileLang auto-sync and BiSheng CCE auto-sync
are both disabled, and there are no source-level barriers or events.  Run the
same file from the old and new checkout roots so that the DSL workload and
profiler settings stay identical while the imported compiler changes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import tilelang
import tilelang.language as T


PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}
ITERATIONS = 4096
NUM_AIVS = 48
COUNTER_CONSUMERS = 8
MODE_SWITCH_CONSUMERS = 4


@T.prim_func
def counter_chain(
    a: T.Tensor((NUM_AIVS, 128), "float16"),
    b: T.Tensor((NUM_AIVS, 128), "float16"),
    c: T.Tensor((NUM_AIVS, 128), "float16"),
):
    """Eight consecutive equal-count consumers with no DMA or synchronization."""
    with T.Kernel(NUM_AIVS, threads=1, is_npu=True) as cid:  # noqa: F841
        a_ub = T.alloc_ub((128,), "float16")
        b_ub = T.alloc_ub((128,), "float16")
        c_ub = T.alloc_ub((128,), "float16")
        T.tile.fill(a_ub, 1.0)
        T.tile.fill(b_ub, 0.0)
        T.tile.fill(c_ub, 0.0)
        for _ in T.serial(ITERATIONS):
            T.tile.add(c_ub, a_ub, b_ub)
            T.tile.add(a_ub, c_ub, b_ub)
            T.tile.add(c_ub, a_ub, b_ub)
            T.tile.add(a_ub, c_ub, b_ub)
            T.tile.add(c_ub, a_ub, b_ub)
            T.tile.add(a_ub, c_ub, b_ub)
            T.tile.add(c_ub, a_ub, b_ub)
            T.tile.add(a_ub, c_ub, b_ub)


@T.prim_func
def mode_switch(
    a: T.Tensor((NUM_AIVS, 128), "float16"),
    b: T.Tensor((NUM_AIVS, 128), "float16"),
    c: T.Tensor((NUM_AIVS, 128), "float16"),
):
    """Alternate raw NORMAL reductions and raw COUNTER arithmetic without sync."""
    with T.Kernel(NUM_AIVS, threads=1, is_npu=True) as cid:  # noqa: F841
        a_ub = T.alloc_ub((128,), "float16")
        b_ub = T.alloc_ub((128,), "float16")
        c_ub = T.alloc_ub((128,), "float16")
        reduced_ub = T.alloc_ub((8,), "float16")
        T.tile.fill(a_ub, 1.0)
        T.tile.fill(b_ub, 0.0)
        T.tile.fill(c_ub, 0.0)
        T.tile.fill(reduced_ub, 0.0)
        for _ in T.serial(ITERATIONS):
            T.tile.block_reduce_max(reduced_ub, a_ub, 1, 128, 1, 1, 8)
            T.tile.add(c_ub, a_ub, b_ub)
            T.tile.block_reduce_max(reduced_ub, c_ub, 1, 128, 1, 1, 8)
            T.tile.add(a_ub, c_ub, b_ub)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["counter_chain", "mode_switch"], required=True)
    parser.add_argument("--launches", type=int, default=100)
    args = parser.parse_args()

    # Every benchmark process must lower its current checkout afresh.
    tilelang.disable_cache()

    program = counter_chain if args.case == "counter_chain" else mode_switch
    consumers = COUNTER_CONSUMERS if args.case == "counter_chain" else MODE_SWITCH_CONSUMERS
    kernel = tilelang.compile(
        program,
        out_idx=[2],
        pass_configs=PASS_CONFIGS,
        compile_flags=["--cce-auto-sync=off"],
        target="ascendc",
        platform="A3",
    )
    source = kernel.get_kernel_source()

    a = torch.empty((NUM_AIVS, 128), dtype=torch.float16, device="npu")
    b = torch.empty((NUM_AIVS, 128), dtype=torch.float16, device="npu")
    torch.npu.synchronize()
    for _ in range(args.launches):
        kernel(a, b)
    torch.npu.synchronize()

    result = {
        "case": args.case,
        "tilelang_root": str(Path(tilelang.__file__).resolve().parent.parent),
        "launches": args.launches,
        "iterations": ITERATIONS,
        "aiv_count": NUM_AIVS,
        "consumers_per_iteration": consumers,
        "count_form_add_count_in_source": source.count("AscendC::Add("),
        "raw_add_count_in_source": source.count("AscendC::Add<"),
        "set_mode_count_in_source": source.count("AscendC::SetMaskCount();") + source.count("AscendC::SetMaskNorm();"),
        "set_payload_count_in_source": source.count("AscendC::SetVectorMask"),
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
