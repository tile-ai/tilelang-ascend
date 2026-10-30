#!/usr/bin/env python3
"""Check complete auto ownership, real versions, source producers and final A3 IR."""

from _native_checks import parser, run

NODES = (
    "testing/python/ascend_migration/test_s8_integration_ir.py",
    "testing/python/ascend_migration/test_s8_sync_contract.py",
    "testing/python/ascend_migration/test_s8_cache.py",
)


def main():
    args = parser(__doc__, step="S8").parse_args()
    return run(args, "s8_check_integration_ir", NODES, step="S8", require_solver=True)


if __name__ == "__main__":
    raise SystemExit(main())
