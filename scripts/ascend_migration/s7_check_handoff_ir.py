#!/usr/bin/env python3
"""Check joint C/V READY/RELEASE, event drain and synchronization ownership."""

from _native_checks import parser, run

NODES = (
    "testing/python/ascend_migration/test_s7_handoff_ir.py",
    "testing/python/ascend_migration/test_s7_pipeline_ownership.py",
)


def main():
    args = parser(__doc__, step="S7").parse_args()
    return run(args, "s7_check_handoff_ir", NODES, step="S7", require_solver=True)


if __name__ == "__main__":
    raise SystemExit(main())
