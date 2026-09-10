# JIT compilation on Ascend

TileLang offers two public entry points:

- `tilelang.compile(program, ...)` compiles an existing `T.prim_func`.
- `@tilelang.jit(...)` compiles the program returned by a Python kernel factory.

Both entry points accept `pass_configs` for TileLang compiler passes and
`compile_flags` for extra Bisheng command-line options.

## Kernel-scoped Bisheng flags

`compile_flags` accepts `list[str]`, `str`, or `None`:

```python
@tilelang.jit(
    out_idx=[1],
    target="ascendc",
    compile_flags=["-O3"],
)
def build_kernel(M, N):
    ...
```

Without the decorator:

```python
kernel = tilelang.compile(program, target="ascendc", compile_flags=["-O3"])
```

Flags are resolved per kernel and included in its cache key. TileLang derives
Bisheng defaults from `pass_configs`, applies supported legacy environment
overrides, then appends explicit `compile_flags`. For optimization level and
auto-sync, explicit flags override the defaults because Bisheng uses the last value.
`pass_configs` still controls which TileLang passes run.

The legacy variables are `TL_CCE_AUTO_SYNC`, `TL_CCE_OPT_LEVEL`, and
`TL_PTO_DEBUG`; the last affects only `target="pto"`. They are read-only:
compiling a kernel does not change the process environment or later kernels'
defaults. Prefer `compile_flags` for per-kernel options.

## Synchronization and debugging

For `target="ascendc"`, setting `TL_ASCEND_AUTO_SYNC_VS=True` in `pass_configs`
enables the TileLang VS synchronization pass and defaults Bisheng to
`-O3 --cce-auto-sync=off`. Passing
`compile_flags=["-O2", "--cce-auto-sync=on"]` changes Bisheng to O2 with auto-sync
enabled; the TileLang VS pass still runs.

Likewise, `TL_ASCEND_AUTO_SYNC=True` still enables TileLang synchronization
insertion when `compile_flags=["--cce-auto-sync=off"]` disables Bisheng's later
auto-sync stage. Disabling Bisheng auto-sync requires all dependencies to be
covered by the TileLang passes or manual synchronization.

Enable Ascend device printing for one kernel with:

```python
@tilelang.jit(
    target="ascendc",
    compile_flags=["-D_DEBUG", "--cce-enable-print"],
)
def debug_kernel(...):
    ...
```

Each entry is whitespace-split; exact duplicates of base-command flags are omitted.
TileLang does not validate target support: use one supported option per list entry. See
[`examples/compile_flags/compile_flags_example.py`](../../examples/compile_flags/compile_flags_example.py)
for an executable kernel-scoping example.

## Ahead-of-time compilation

For a deployable shared library, use the framework's `LibraryGenerator`
workflow instead of maintaining a copied Bisheng command. The canonical
end-to-end example is
[`examples/gemm_aot`](https://github.com/tile-ai/tilelang-ascend/tree/ascendc_pto/examples/gemm_aot).
