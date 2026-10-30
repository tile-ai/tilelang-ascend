#!/usr/bin/env python3
"""Check physical versions, guarded domains and final sync using native IR fixtures.

Synthetic protocol-cost fixtures do not constitute measured hardware timing.
"""

from _native_checks import parser, run

NODES = [
    "testing/python/ascend_migration/test_s5_versions_ir.py",
    "testing/python/ascend_migration/test_s5_sync_coverage.py",
    "testing/python/ascend_migration/test_s5_mask_contract.py",
    "testing/python/ascend_migration/test_s5_cost_model.py",
    "testing/python/ascend_migration/test_s5_calibration.py",
    "testing/python/ascend_migration/test_s5_native_snapshot.py",
    "testing/python/ascend_migration/test_s5_cache.py",
    "testing/python/ascend_migration/test_s5_npu_evidence.py",
    "testing/python/ascend_migration/test_s5_calibration_export.py",
    "testing/python/ascend_migration/test_s5_runner_contracts.py",
]


def main() -> int:
    args = parser(__doc__, step="S5").parse_args()
    return run(args, "s5_check_versions_ir", NODES, step="S5", require_solver=True)


if __name__ == "__main__":
    raise SystemExit(main())
