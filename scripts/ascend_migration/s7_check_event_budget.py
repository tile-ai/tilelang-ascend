#!/usr/bin/env python3
"""Check A3 event domains, live budgets and actual physical Vector hazards."""

from _native_checks import parser, run

NODES = (
    "testing/python/ascend_migration/test_s7_event_budget.py",
    "testing/python/ascend_migration/test_s7_local_sync.py",
    "testing/python/ascend_migration/test_s7_cache.py",
)


def main():
    args = parser(__doc__, step="S7").parse_args()
    return run(args, "s7_check_event_budget", NODES, step="S7", require_solver=True)


if __name__ == "__main__":
    raise SystemExit(main())
