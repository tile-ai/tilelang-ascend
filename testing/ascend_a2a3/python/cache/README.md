# Cache publication and recovery regressions

These assets originate from TA `0deb8b93da11b7f0becbcded732d9633b7ca4e34`
(`testing/python/cache`: README, atomic_copy and cache_recovery). The nine
publication and 32 recovery source scenarios remain individually traceable.
They exercise the current shared/A2A3 cache interfaces; the retired file helper,
standalone `AutoTunerCache` and synthetic `sys.modules` compiler framework are
not restored. The adaptation changes the APIs and several outcomes, so it is
not a claim of original whole-file AST or behavioral equality.

Run with a built/importable modern TileLang package and pytest:

```bash
python -m pytest testing/ascend_a2a3/python/cache -q
```

Unlike the original stdlib-only file-loader tests, these require the actual TVM
and TileLang Python/runtime dependencies. They do not allocate or initialize an
NPU or execute a device kernel. Pure filesystem/shared-cache tests also run with
an actual USE_ASCEND=OFF library. Four strict A2A3 loader scenarios require the
registered Ascend Target kind and compiler configuration dependencies, and skip
individually without that kind. The real-library publication tests require Linux
and `cc`; they compile tiny **host** libraries and set core size to zero in the
loader subprocess. All directories are temporary and cache descriptors, original
singleton/memory dictionaries and raw cache state are restored after each case.
The actual tuner uses a scoped test-owned stream logger and initialization flag,
leaving any caller cwd `autotuner.log`, logger/handler identities and ordering
untouched. Fault logs remain real and capturable; only the owned handler closes.

## Publication safety

The current `KernelCache._safe_write_file` and
`AutotuneResult._safe_write_file` independently create a temporary sibling,
write/flush/fsync it, and call `os.replace`. Existing descriptors/mappings retain
the old inode, and new readers see the complete replacement. The test operation
reads/writes the source bytes in the same way as the current library publishers;
the actual implementation owns publication and cleanup. Both implementations
run all nine source scenarios (18 collected tests), including four publishing
processes with 40 writes each, and a real host SO returning 17 before replacement
and 29 in a new reader. No writer truncates an already mapped destination inode.

The original source permission input 0o750 remains tested. Current writers create
files using the caller's normal umask instead of inheriting source permissions
as `shutil.copy` did. The expectation compares a normal control file's mode;
no test-side chmod is added to disguise this difference. Dlopen of a shared
library does not require inheriting its source executable bits.

Modern kernel/result publication additionally stages an entire cache entry under
`<namespace>/.staging`, checks required files, and renames the complete directory.
Kernel manifests record library content hashes and source sizes; A2A3 runtime
metadata is a required sidecar, retaining the frozen compiler and final ABI.
Single-file replacement itself does not coordinate concurrent deletion or promise
cross-entry transactions, and the tests do not claim crash-proof persistence.

## Recovery boundaries

Shared `KernelCache` treats incomplete entries and integrity failures as misses.
A failure during the actual loader/build step is logged, the entry removed, and
returned as a miss. This intentionally differs from old tests which expected
Permission/ELF/foreign dependency errors to propagate from that particular class.

The strict A2A3 `AutotuneResult` restore path propagates those errors. It recovers
only when its own complete library path is reported missing, and a second stat
still returns ENOENT. A ctypes errno-less error must match the exact full path
and missing-object message; permission errors, malformed/truncated ELF, missing
dependencies, another directory's basename, recreated files and metadata/ABI
inconsistencies are not converted to that narrow miss. AscendC and PTO each run
the strict loader boundary. Placeholder loader files only test control flow;
they do not count as valid ABI, ELF, public compilation or NPU numerical evidence.
Real public/Bisheng recovery remains covered by the existing
`testing/ascend_a2a3/python/refactor/jit/test_a2a3_tuner_library_recovery.py` integration gate, whose
benchmark is explicitly a stub and must not be called a numerical/performance run.

The modern metadata reader propagates a missing best_config/function/latency
file. Modern staging-directory creation occurs before the save try and propagates
its errors; failures within the atomic save try (including serialization) are
logged, staging removed, and no final entry published. The tests preserve the
original fault inputs/stages but assert these existing main contracts instead
of adding fallback catches. `AutoTuner.run` replaces the retired cache class:
the full-control-flow test uses real scheduling/save/memory/disk flow, with only
compile, benchmark and loader boundaries stubbed. Disk hits must not compile or
benchmark; its latency/config values are state-machine fixtures.

## Source scenario mapping

The names below preserve source IDs, including historical names saying
"propagates" or "not_a_miss" where the modern expectation differs. Comments and
the table make that difference explicit; a retained name does not promise the
old implementation's contract. The nine publication scenarios are exercised on
both actual writers, with the permissions difference described above.

| Source recovery scenario | Current mapping / explicit difference |
| --- | --- |
| `TestIsFileMissing.test_missing_entry_is_missing` | backend-private _is_missing_own_library predicate; original error/path/stat conditions retained |
| `TestIsFileMissing.test_existing_entry_is_not_missing` | backend-private _is_missing_own_library predicate; original error/path/stat conditions retained |
| `TestIsFileMissing.test_permission_error_is_not_missing` | backend-private _is_missing_own_library predicate; original error/path/stat conditions retained |
| `TestIsFileMissing.test_not_a_directory_is_not_missing` | backend-private _is_missing_own_library predicate; original error/path/stat conditions retained |
| `TestErrorReportsMissingPath.test_real_missing_target_is_reported` | backend-private _is_missing_own_library predicate; original error/path/stat conditions retained |
| `TestErrorReportsMissingPath.test_real_truncated_library_is_not_reported` | backend-private _is_missing_own_library predicate; original error/path/stat conditions retained |
| `TestErrorReportsMissingPath.test_dependency_name_containing_target_is_not_reported` | backend-private _is_missing_own_library predicate; original error/path/stat conditions retained |
| `TestErrorReportsMissingPath.test_same_basename_in_other_directory_is_not_reported` | backend-private _is_missing_own_library predicate; original error/path/stat conditions retained |
| `TestKernelCacheRecovery.test_recovers_when_library_deleted_before_load` | actual shared KernelCache staging/manifest/memory/disk transitions; strict error propagation separately covered at A2 tuning loader |
| `TestKernelCacheRecovery.test_oserror_without_errno_is_recovered` | actual shared KernelCache staging/manifest/memory/disk transitions; strict error propagation separately covered at A2 tuning loader |
| `TestKernelCacheRecovery.test_permission_error_is_not_a_miss` | shared loader logs/removes/misses; strict A2 tuning loader preserves original propagation safety at its own API |
| `TestKernelCacheRecovery.test_unloadable_library_is_not_a_miss` | shared loader logs/removes/misses; strict A2 tuning loader preserves original propagation safety at its own API |
| `TestKernelCacheRecovery.test_missing_entry_directory_returns_none` | actual shared KernelCache staging/manifest/memory/disk transitions; strict error propagation separately covered at A2 tuning loader |
| `TestKernelCacheRecovery.test_cold_and_hot_cache_do_not_reload` | actual shared KernelCache staging/manifest/memory/disk transitions; strict error propagation separately covered at A2 tuning loader |
| `TestKernelCacheRecovery.test_permission_error_with_deletion_is_not_a_miss` | shared loader logs/removes/misses; strict A2 tuning loader preserves original propagation safety at its own API |
| `TestKernelCacheRecovery.test_dependency_error_with_deletion_is_not_a_miss` | shared loader logs/removes/misses; strict A2 tuning loader preserves original propagation safety at its own API |
| `TestKernelCacheRecovery.test_miss_compiles_and_returns_result` | actual shared KernelCache staging/manifest/memory/disk transitions; strict error propagation separately covered at A2 tuning loader |
| `TestKernelCacheRecovery.test_disk_hot_cache_does_not_compile` | actual shared KernelCache staging/manifest/memory/disk transitions; strict error propagation separately covered at A2 tuning loader |
| `TestKernelCacheRecovery.test_truncated_library_with_deletion_is_not_a_miss` | shared loader logs/removes/misses; strict A2 tuning loader preserves original propagation safety at its own API |
| `TestKernelCacheRecovery.test_recreated_library_after_deletion_reraises` | shared loader logs/removes/misses; strict A2 tuning loader preserves original propagation safety at its own API |
| `TestKernelCacheRecovery.test_makedirs_failure_still_returns_result` | exact current mkdir stage must be isolated; no blanket old deletion suppression |
| `TestKernelCacheRecovery.test_makedirs_permission_error_propagates` | actual shared KernelCache staging/manifest/memory/disk transitions; strict error propagation separately covered at A2 tuning loader |
| `TestTunerCacheRecovery.test_recovers_when_library_deleted_before_load` | current AutoTuner.run/result load-save; retired AutoTunerCache call-count details are not API |
| `TestTunerCacheRecovery.test_unloadable_library_is_not_a_miss` | current AutoTuner.run/result load-save; retired AutoTunerCache call-count details are not API |
| `TestTunerCacheRecovery.test_full_control_flow` | current AutoTuner.run/result load-save; retired AutoTunerCache call-count details are not API |
| `TestAutotuneResultRecovery.test_recovers_when_library_deleted_before_load` | current AutotuneResult native loader and staged save/load metadata boundaries |
| `TestAutotuneResultRecovery.test_unloadable_library_is_not_a_miss` | current AutotuneResult native loader and staged save/load metadata boundaries |
| `TestAutotuneResultRecovery.test_load_from_disk_metadata_deleted_returns_none` | current metadata read propagates FileNotFoundError; no product fallback |
| `TestAutotuneResultRecovery.test_save_to_disk_survives_cache_deletion` | current AutotuneResult native loader and staged save/load metadata boundaries |
| `TestAutotuneResultRecovery.test_save_to_disk_serialization_error_propagates` | modern staged-save catches/logs, removes staging, no publication; matches existing main tests |
| `TestAutotuneResultRecovery.test_save_to_disk_makedirs_deletion_skips_save` | staging mkdir before try propagates FileNotFoundError; preserve real current stage |
| `TestAutotuneResultRecovery.test_save_to_disk_makedirs_permission_error_propagates` | current AutotuneResult native loader and staged save/load metadata boundaries |
