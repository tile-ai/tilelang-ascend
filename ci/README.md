# Operator test manifest

`operator_test_manifest.yaml` maps an operator under `examples/` to the Pytest
test that covers it:

```yaml
  examples/xattention/xattention.py: examples/xattention/test_xattention.py
```

`examples/bench_test.sh` decides whether an operator passed by grepping its
stdout for `Kernel Output Match!` or `TEST PASSED!`. A wording change in the
operator fails the run, and an operator that prints the phrase while computing
the wrong answer passes it. A registered operator is skipped there and judged by
its test's assertions instead.

## Adding a test

An entry is already reserved for every operator the legacy runner collects, so
writing the test file is the whole change — the manifest needs no edit.

1. Find your operator in this file. The path after the colon is the file to
   create, exactly as written.
2. Write the test. `examples/batch_gemm/test_batch_gemm.py` is the smallest
   sample; `examples/flash_attention/test_flash_attn_bhsd.py` shows what to do
   when the example keeps its reference implementation inside `__main__`.
3. Run it, then tighten the tolerance and run it again. It should fail — for
   float16 the difference bottoms out around `2^-11`. Passing at a tolerance
   that tight usually means the test compares a value with itself.

Leave the operator itself unchanged.

### Do not import the example at module level

Most examples touch the device while being imported — `.npu()`, `parse_args()`,
`torch.npu.synchronize()` at module level. Pytest executes that during
collection, once per xdist worker, before any test runs. Load the example inside
the test instead:

```python
def _load_example() -> ModuleType:
    source = Path(__file__).with_name("xattention.py")
    spec = importlib.util.spec_from_file_location("_xattention_for_test", source)
    module = importlib.util.module_from_spec(spec)
    original_argv = sys.argv
    try:
        sys.argv = [str(source)]      # keep Pytest's arguments away from argparse
        spec.loader.exec_module(module)
    finally:
        sys.argv = original_argv
    return module
```

Both samples do this.

## States

An entry only takes effect once its test file exists. Until then the operator
keeps running in the legacy runner, so reserving a name changes nothing on its
own.

| | |
|---|---|
| **live** | the test exists; the operator is skipped in the legacy runner and Pytest runs the test |
| **reserved** | the test does not exist yet; the operator runs where it always did |

`python scripts/ci/resolve_operator_tests.py validate` prints both counts.

## Checks

Two run before the legacy runner collects anything, and both fail the script:

- `validate` — the file parses, every source exists, no duplicate source or test.
- `check-orphans` — every `test_*.py` under `examples/` matches an entry. The
  legacy runner skips all of them by name, so one that matches nothing is run by
  nothing at all while CI stays green. Tests a sibling shell script invokes are
  exempt.

A test whose name does not match its reserved entry fails `check-orphans`, and
the message names the entry it was probably meant to match.

## Not listed

- `examples/gemm_aot/example_gemm.py` — the derived name `test_example_gemm.py`
  already belongs to a standalone script that `run_example_gemm_aot.sh` runs
  directly. Handing that script to Pytest executes its module level in every
  worker and collects no test.
- `dispatch_combine/`, `shmem/`, `generative_recommendation/{golden,testcase}.py`,
  `fa_opt` entries other than `flash_*`, and `bench_sfa` — the legacy runner does
  not collect these, so registering them could never take effect.
