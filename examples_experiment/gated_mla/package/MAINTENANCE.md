# Source records after reviewed changes

`source_manifest.json` constrains both production code and tests. In particular,
`tests/test_runtime_npu.py` is hashed under `new_package_files`; a reviewed core
test patch requires its hash to be refreshed. `FILELIST.sha256` additionally covers
README, packaging metadata, source_manifest and all exported source files.

After reviewing a tests/tools/documentation-only patch, from this checkout:

```bash
python tools/refresh_records.py
python tools/audit_package.py
python -m pytest -m 'not npu' -q
python -m pytest --collect-only -q
```

`refresh_records.py` refuses any changed or unregistered production `gmla/*.py`
and any changed retained reference/fixture. It records old/new hashes for reviewed
nonproduction updates, updates those manifest entries, then rebuilds FILELIST.
It is not a way to bypass production or oracle review. Apply repository Ruff
before refreshing, using the actual repository-root pyproject configuration.

Audit checks only package-relative files. It never opens a historical development
origin ID and does not require an original D-drive/remote workspace. Successful
integrity checks do not prove the historical origins, hardware correctness or
benchmark qualification.

The source-checkout audit/test files are intentionally outside the runtime wheel.
If packaging configuration changes, build the wheel without downloading or
replacing the native stack and verify its `gmla/*.py` bytes against this checkout.
Run installed-import checks outside the checkout to avoid pytest's local
`pythonpath` shadowing. The repository now registers nine package anchors plus
the preserved L0 suite; changes to those mappings still require selection review.
See [VALIDATION](VALIDATION.md) for the separate full-boundary cold-test verdict.
