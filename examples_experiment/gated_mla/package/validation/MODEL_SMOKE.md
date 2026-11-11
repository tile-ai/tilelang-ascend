# Reproduce the installed-package one-layer smoke

This tool is the exact executed runner (SHA256
`1b649c1cdbf15dea0c7d22a5f5e3f0a3986298d21accea3034bf20c773b88ac9`).
It normally imports `gmla.create_cache`, `gmla.build_packed_adapter` and
`gmla.integration.kimi.FrozenKimiLayerAdapter`; it never injects a prototype
provider. The supplied manifest must match all25 installed production files.

Supply the [pinned Moonshot source file](https://huggingface.co/moonshotai/Kimi-K3/resolve/c5d1dd4c428bd1ce8b88c5044f3b6ccde9e3b721/modeling_kimi_linear.py) yourself, under its applicable terms.
It is not bundled or downloaded by this tool. Required SHA256 is
`9e3564c70ac21854ce5a090cc946c5dc76b70d1050ef50840449181a20fff44a`,
revision `c5d1dd4c428bd1ce8b88c5044f3b6ccde9e3b721`. The loader checks the complete
file hash before loading the four frozen definitions. No checkpoint is used.

Activate the matching post4/CANN9.2/TileLang environment, install the package
normally, and run fresh processes serially on the owned NPU. From repository root:

```bash
PKG="$PWD/examples_experiment/gated_mla/package"
python -m pip install --no-deps --no-build-isolation "$PKG"
# Set these to your external source and a new evidence directory:
GMLA_OFFICIAL_SOURCE=/path/to/modeling_kimi_linear.py
GMLA_MODEL_OUT=/new/model-evidence
mkdir -p "$GMLA_MODEL_OUT"
python "$PKG/benchmark/verify_installed.py" --source-root "$PKG" \
  --output "$GMLA_MODEL_OUT/installed_identity.json"
cd "$GMLA_MODEL_OUT"  # keep the checkout off the normal import path
python "$PKG/validation/model_smoke.py" --official-source "$GMLA_OFFICIAL_SOURCE" \
  --package-manifest "$PKG/source_manifest.json" --dtype float16 --target pto \
  --output "$GMLA_MODEL_OUT/pto_float16"
```

Repeat separately for `--target ascendc`; for BF16 use `--dtype bfloat16
--collect-diagnostics`, with fresh output paths. The recorded jobs used a1200s
outer process limit, not a latency target. Do not run these alongside another
NPU job. The script preloads the normal package before NPU operations.

Configuration is fixed B2/H96/hidden7168/qrank1536/cap65; phases are17→3→1.
It checks two independent FP64 core forms, before/after W_O, actual official
incremental-cache forward, exact external Q/C/rawKV/gate correspondence,
readonly weights/hidden inputs and cache prefix/append/future contents. Prefill
must show actual Cube Query and Value R3. Selected CPP/hashes are exported.

Read `package_model_result.json` without conflating these fields:

- `candidate_FP64_all_three_phases_passed`: both candidate FP64 gates in all phases.
- `actual_official_external_inputs_equal_all_phases`: external wiring identity.
- `steps[].comparisons`: every original candidate-vs-FP64 and candidate-vs-HF gate.
- `steps[].official_vs_same_fp64_diagnostic_only`: HF's own oracle qualification.
- `failed_original_gates`, `status`, exit code and failure artifacts: preserved
  comparison failures; BF16 `diagnostic_only_failed`/exit1 is not relabelled PASS.

`--collect-diagnostics` continues only cross-HF failures while candidate FP64
and input-identity checks pass. Candidate/hardware failures stop. Budgets remain
rtol0.01/atol0.002/normalized max0.005. Full tensor failures may be large and are
not included in the compact summary. The emitted event/wall cost fields include
cold initialization/JIT/host gaps; they are not steady-state performance gates.

Recorded results and limitations are in [VALIDATION](../VALIDATION.md) and
[model_summary.json](model_summary.json).
