#!/usr/bin/env python3
"""Check S4 strategy/cache isolation, injector ABI and supported-input boundaries."""

from _native_checks import parser, run


NODES = [
    "testing/python/ascend_migration/test_s4_pipeline_ir.py",
    "testing/python/ascend_migration/test_s4_cache.py",
]


def main() -> int:
    args = parser(__doc__, step="S4").parse_args()
    return run(args, "s4_check_pipeline_ir", NODES, step="S4", require_solver=True)


if __name__ == "__main__":
    raise SystemExit(main())
