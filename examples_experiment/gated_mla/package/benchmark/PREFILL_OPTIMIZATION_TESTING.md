# Reproduce the reviewed prefill optimization

The candidate changes only the prefill implementation. The parent control is
PR #1970 commit `de16598e67bce8168775fce23f9b207dd6259401`.
Build a normal isolated installation from that parent; keep the current candidate
source outside that installation. Do not silently substitute an optimized
installation for the original performance control.

The two tools preserve the exact code used for numerical/timing experiments.
Their historical machine-specific defaults are overridden by all paths below.
Activate the built TileLang/CANN environment and compatible Torch-NPU stack first.

```bash
PARENT_PKG=<parent-checkout>/examples_experiment/gated_mla/package
PKG=<candidate-checkout>/examples_experiment/gated_mla/package
BASELINE_SITE=<normal-parent-installation-site>
python -m pip install --no-deps --no-build-isolation --target "$BASELINE_SITE" "$PARENT_PKG"
PYTHONPATH="$BASELINE_SITE:$PYTHONPATH" python "$PKG/benchmark/verify_installed.py"   --source-root "$PARENT_PKG" --output <parent-inventory.json>

PYTHONPATH="$BASELINE_SITE:$PYTHONPATH" python "$PKG/benchmark/prefill_optimization_regression.py"   --variant-file "$PKG/gmla/kernels/prefill.py"   --installed-root "$BASELINE_SITE" --inventory <parent-inventory.json>   --protocol-root "$PKG/benchmark/protocol" --output <new-regression-directory>

PYTHONPATH="$BASELINE_SITE:$PYTHONPATH" python "$PKG/benchmark/prefill_optimization_benchmark.py"   --variant-file "$PKG/gmla/kernels/prefill.py" --scope core   --installed-root "$BASELINE_SITE" --inventory <parent-inventory.json>   --alignment-tools "$PKG/benchmark" --protocol-root "$PKG/benchmark/protocol"   --dtype float16 --target pto --mode benchmark --order-seed 1700   --output <new-trial-directory>
```

Use three new processes/output directories with order seeds1700/1701/1702.
Repeat for both dtypes and backends, and `--scope full-chain`.
Each trial qualifies seeds0/1/17; only seed0 is timed. Core BF16 is a shared-LP
component test, while full-chain retains the real production query HiLo.
The full-chain third comparator is the historical FP32-projection/FIA/gate
composition; it is not the low-precision vLLM source-prefill path.

For hardware counters, first validate the same variant and arguments, then run
`msprof op --kernel-name=main_kernel --launch-count=1 --warm-up=5
--replay-mode=kernel --aic-metrics=PipeUtilization,Memory,MemoryL0,ArithmeticUtilization,ResourceConflictRatio`
with the benchmark application in `--scope core --mode validate` and a new entry
directory. The core tool deliberately launches the variant before the control.
Check the application JSON precision/source status and profiler records; exit0
alone is insufficient. Hardware Task and Event/wall/Graph remain separate.

The regression includes13 shape/tile templates, two seeds, two dtypes and two
backends (104 cases), numerical inactive zeros, unchanged-input bytes and
same-pointer C/E changes in ordinary calls and Graph replay. E-only perturbations
are nonuniform: a uniform E shift alone cancels in softmax and proves no reload.
Public-interface tests additionally update Q and CPU-owned metadata on replay.
