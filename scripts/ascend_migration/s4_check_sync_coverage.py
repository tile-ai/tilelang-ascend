#!/usr/bin/env python3
"""Check final S4 completion coverage, event pairing and physical-storage reuse."""

from _native_checks import parser, run


NODES = ["testing/python/ascend_migration/test_s4_sync_coverage.py"]


def main() -> int:
    args = parser(__doc__, step="S4").parse_args()
    return run(args, "s4_check_sync_coverage", NODES, step="S4", require_solver=True)


if __name__ == "__main__":
    raise SystemExit(main())
