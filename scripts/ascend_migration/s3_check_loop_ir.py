#!/usr/bin/env python3
"""Check S3 loop interfaces, annotation placement and validated Task/Stage markers."""

from _native_checks import parser, run


NODES = [
    "testing/python/ascend_migration/test_s3_loop_ir.py",
    "testing/python/ascend_migration/test_s3_schedule_hints.py",
]


def main() -> int:
    args = parser(__doc__).parse_args()
    return run(args, "s3_check_loop_ir", NODES)


if __name__ == "__main__":
    raise SystemExit(main())
