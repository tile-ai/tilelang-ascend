# Experimental Gated MLA decode

This example implements a dense, causal Gated MLA attention core with five manually
scheduled Vector kernels on Ascend. It includes query absorption, latent and auxiliary
attention, latent readout, value restoration, and a channelwise sigmoid gate. The output
is before the model's output projection. Input projection, normalization and cache writes
belong to the caller.

## Interface and supported domain

`example_gated_mla_decode.gated_mla_core` accepts these contiguous FP16 tensors on one NPU:

| Input | Shape |
|---|---|
| `q_content` | `[1, 1, 1, 128]` |
| `q_aux` | `[1, 1, 1, 64]` |
| `latent_kv` | `[1, 64, 512]` |
| `key_aux` | `[1, 64, 64]` |
| `w_k` | `[1, 128, 512]` |
| `w_v` | `[1, 128, 512]` |
| `gate_logits` | `[1, 1, 1, 128]` |

The result has shape `[1, 1, 1, 128]` and dtype FP16. Intermediate arithmetic is FP32.
All input values must be finite; the metadata validator does not read device values.
Inputs and caches are read-only. Other shapes and dtypes are rejected explicitly.

`q_lens`, `kv_lens` and `query_start` are host tuples/lists with one integer each.
An active query sees `min(kv_lens[0], query_start[0] + 1)` keys. The metadata must satisfy
`0 <= query_start[0]` and `query_start[0] + q_lens[0] <= kv_lens[0] <= 64`.
`q_lens=(0,)` produces zeros. The default positive scale is `1 / sqrt(128 + 64)`;
it is based on the original query dimensions, not the 512-dimensional latent width.

For the visible prefix, the computation is

```text
q_latent = q_content @ w_k
scores = (latent_kv @ q_latent + key_aux @ q_aux) * scale
probabilities = softmax(scores)
latent = probabilities @ latent_kv
output = (w_v @ latent) * sigmoid(gate_logits)
```

`gated_mla_golden.py` supplies two independent FP64 mathematical references: expanded
keys/values with per-query prefix slicing, and absorbed queries with a causal mask.
They are validation code, never runtime fallbacks. No model download or external
workspace files are required.

## Programming model and backend

This is an Expert example with explicit UB addresses, Vector scopes and barriers;
the four automatic synchronization/memory/CV passes remain disabled. The five stages
make intermediate ownership and precision inspectable. AscendC is selected explicitly.
PTO is not validated or claimed. Cube execution is not used by these kernels.

The latent readout clears the contiguous invalid rows with `T.tile.fill`. Each FP32 row
is 256 bytes, so every supported prefix gives a 32-byte-aligned offset and a whole-vector
fill length. A full prefix skips the empty fill. This avoids an elementwise scalar mask
without changing the existing multiplication, reduction or synchronization.

## Run and test

Use a built TileLang-Ascend environment with CANN and torch_npu enabled. From the repository
root:

```bash
python examples_experiment/gated_mla/example_gated_mla_decode.py
python -m pytest examples_experiment/gated_mla/test_gated_mla_decode.py -q
python -m pytest examples_experiment/gated_mla/test_gated_mla_decode.py -q -k test_cpu
python examples_experiment/gated_mla/tools/benchmark/measure_decode.py --output /tmp/gated_mla_benchmark
```

The complete test entry requires an NPU and fails if dependencies or a device are absent.
The selected `test_cpu` cases require only PyTorch. There is no opt-in environment variable
that silently skips the device tests. The CI manifest registers the example and its test.

FP16 acceptance uses `rtol=1e-2`, `atol=2e-3` and maximum absolute error normalized by the
reference maximum amplitude at most `5e-3`. These are acceptance limits for the tested
synthetic domain, not an error guarantee for arbitrary finite input magnitudes.

## Performance and limitations

See [PERFORMANCE.md](PERFORMANCE.md) for the measured protocol, candidate and reference
precision, before/after results and remaining work. This is a fixed-shape experimental
example. It does not provide long-cache, prefill, multiple heads/batches, quantization,
backward or a complete model integration.

The attention boundary follows the Gated MLA structure in the
[fixed Kimi K3 model reference](https://huggingface.co/moonshotai/Kimi-K3/blob/c5d1dd4c428bd1ce8b88c5044f3b6ccde9e3b721/modeling_kimi_linear.py).
The model source is linked for provenance and is not redistributed here. Related latent
attention examples are available under `examples_experiment/deepseek_mla/`.


## CI test lanes

```bash
python -m pytest examples_experiment/gated_mla/test_gated_mla_decode.py -q -m 'not (low_priority or ci_skip)'
python -m pytest examples_experiment/gated_mla/test_gated_mla_decode.py -q -m 'not ci_skip'
```

The PR lane contains25 cases and the complete lane29. Extended cases use the repository's
`low_priority` marker. Key prefix-mask regressions and device negative-infinity checks
remain in the PR lane. Tests use the uniquely named `gated_mla_golden` helper module to
avoid colliding with other examples' top-level `utils` modules during combined collection.
The benchmark rejects nonempty output directories. Missing optional metadata commands
such as `git` or `npu-smi` are recorded as unavailable; actual NPU and precision failures
still fail the run.
