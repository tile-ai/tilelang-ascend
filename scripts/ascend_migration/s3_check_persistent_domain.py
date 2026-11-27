#!/usr/bin/env python3
"""Check Persistent exact-once coverage by interpreting actual generated loop IR."""

from _native_checks import parser, run


NODES = [
    "testing/python/ascend_migration/test_s3_persistent_domain.py",
    "testing/python/ascend_migration/test_s3_loop_ir.py::test_persistent_invalid_domain_is_rejected",
]


def main() -> int:
    args = parser(__doc__).parse_args()
    return run(args, "s3_check_persistent_domain", NODES)


if __name__ == "__main__":
    raise SystemExit(main())
