"""Layered correctness tests for the improved Lightning Indexer."""

import argparse
import sys

import tilelang

try:
    from .example_lightning_indexer_dynamic_shape_improved import (
        auto_s2_splits,
        test_indexer as run_indexer_case,
        validate_indexer_config,
    )
except ImportError:
    from example_lightning_indexer_dynamic_shape_improved import (
        auto_s2_splits,
        test_indexer as run_indexer_case,
        validate_indexer_config,
    )


L0_CASES = [
    ("smoke", 1, 64, 1024, 256, 64, 64),
]

L1_CASES = [
    ("standard_d64", 1, 256, 2048, 1024, 256, 64),
    ("standard_d128", 2, 512, 4096, 1024, 512, 128),
]

BOUNDARY_CASES = [
    ("single_trunk_top_k_equals_s2", 1, 64, 512, 512, 512, 64),
    ("maximum_top_k", 1, 512, 4096, 2048, 128, 64),
]

VALID_CONFIG = {
    "n2": 1,
    "groups": 32,
    "dimension": 64,
    "top_k": 256,
    "vector_basen": 256,
    "vector_baseg": 16,
    "block_m": 64,
    "block_n": 256,
    "block_k": 64,
    "max_s2": 2048,
    "s2_splits": 1,
    "max_cores": 20,
    "batch": 1,
    "s1": 64,
    "s2": 2048,
}


def _run_precision_cases(level, cases):
    for name, batch, s1, s2, top_k, block_n, dimension in cases:
        print(f"[{level.upper()}] {name}")
        run_indexer_case(
            s1=s1,
            s2=s2,
            top_k=top_k,
            block_n=block_n,
            vector_basen=block_n,
            batch=batch,
            dimension=dimension,
        )


def validate_test_case_configurations():
    """Validate every precision case without compiling or running the kernel."""
    for cases in (L0_CASES, L1_CASES, BOUNDARY_CASES):
        for name, batch, s1, s2, top_k, block_n, dimension in cases:
            s2_splits = auto_s2_splits(batch, s1, s2, block_n)
            validate_indexer_config(
                n2=1,
                groups=32,
                dimension=dimension,
                top_k=top_k,
                vector_basen=block_n,
                vector_baseg=16,
                block_m=64,
                block_n=block_n,
                block_k=dimension,
                max_s2=s2,
                s2_splits=s2_splits,
                batch=batch,
                s1=s1,
                s2=s2,
            )
            print(f"[CONFIG_PASS] {name} (S2_SPLITS={s2_splits})")


def test_l0():
    """Run the blocking smoke test."""
    _run_precision_cases("l0", L0_CASES)


def test_l1():
    """Run representative supported shapes and configurations."""
    _run_precision_cases("l1", L1_CASES)


def test_l2():
    """Verify that invalid host-side configurations are rejected."""
    invalid_cases = [
        ("zero_batch", {"batch": 0}),
        ("unsupported_dimension", {"dimension": 96, "block_k": 96}),
        ("block_k_mismatch", {"block_k": 128}),
        ("unsupported_block_m", {"block_m": 32}),
        ("unsupported_block_n", {"block_n": 192, "vector_basen": 192}),
        ("vector_basen_mismatch", {"vector_basen": 128}),
        ("groups_not_divisible", {"groups": 24}),
        ("unaligned_s1", {"s1": 65}),
        ("unaligned_s2", {"s2": 2000}),
        ("top_k_exceeds_s2", {"top_k": 2049}),
        ("top_k_exceeds_ub_limit", {"block_n": 512, "vector_basen": 512, "top_k": 1537}),
        ("invalid_s2_split", {"s2_splits": 3}),
        ("invalid_max_cores", {"max_cores": 0}),
        ("index_exceeds_fp32_exact_range", {"max_s2": 2**25, "s2": 2**25}),
        ("unsupported_input_dtype", {"input_dtype": "float32"}),
    ]
    for name, overrides in invalid_cases:
        arguments = VALID_CONFIG | overrides
        try:
            validate_indexer_config(**arguments)
        except ValueError:
            print(f"[BOUNDARY_PASS] l2 {name}")
        else:
            raise AssertionError(f"Invalid configuration was not rejected: {name}")


def test_boundary():
    """Run supported boundary shapes as blocking correctness tests."""
    _run_precision_cases("boundary", BOUNDARY_CASES)


def run_layered_tests(level):
    """Run one test level or the complete layered suite."""
    tilelang.disable_cache()
    validate_test_case_configurations()
    if level in ("l0", "all"):
        test_l0()
    if level in ("l1", "all"):
        test_l1()
    if level in ("l2", "all"):
        test_l2()
    if level in ("boundary", "all"):
        test_boundary()
    print("Test Passed!")


def main():
    parser = argparse.ArgumentParser(description="Lightning Indexer layered correctness tests")
    parser.add_argument(
        "--level",
        default="all",
        choices=["l0", "l1", "l2", "boundary", "all"],
        help="Test level to run (default: l0)",
    )
    arguments = parser.parse_args()
    try:
        run_layered_tests(arguments.level)
    except Exception as error:
        print(f"Test Failed: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
