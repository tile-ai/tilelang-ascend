#!/usr/bin/env python3
"""Check canonical owner, view, raw-pointer and allocation identities without an NPU."""

from _native_checks import parser, run

NODES = (
    "testing/python/ascend_migration/test_s6_owner_ir.py",
    "testing/python/ascend_migration/test_s6_cache.py",
)


def main():
    args = parser(__doc__, step="S6").parse_args()
    return run(args, "s6_check_owner_ir", NODES, step="S6", require_solver=True)


if __name__ == "__main__":
    raise SystemExit(main())
