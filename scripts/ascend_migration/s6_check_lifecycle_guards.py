#!/usr/bin/env python3
"""Check source readiness, storage activations, reuse and exit paths without an NPU."""

from _native_checks import parser, run

NODES = (
    "testing/python/ascend_migration/test_s6_lifecycle.py",
    "testing/python/ascend_migration/test_s6_sync_coverage.py",
)


def main():
    args = parser(__doc__, step="S6").parse_args()
    return run(args, "s6_check_lifecycle_guards", NODES, step="S6", require_solver=True)


if __name__ == "__main__":
    raise SystemExit(main())
