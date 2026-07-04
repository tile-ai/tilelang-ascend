# TVM submodule patches

Minimal local patches applied on top of the **pinned** `3rdparty/tvm` submodule
commit. `install_ascend.sh` applies every `tvm_*.patch` in this directory right
after `git submodule update --init --recursive`, so every build (including CI)
picks them up. Application is:

- **idempotent** — an already-applied patch is detected (`git apply --reverse
  --check`) and skipped, so re-running install / incremental builds is safe;
- **fatal on failure** — if a patch cannot apply (e.g. the pinned tvm was bumped
  and the context no longer matches) the install aborts instead of silently
  building an unpatched TVM.

Files must be named `tvm_*.patch` to be picked up.

## Patches

- **`tvm_slice_step_fix.patch`** — `Buffer.__getitem__` treats an explicit unit
  step (`a:b:1`) the same as no step, so a runtime-dynamic slice such as
  `buf[0, 0:idx]` takes the `BufferRegion` path instead of the `Ramp` path (whose
  `int(lanes)` requires a compile-time-constant length and crashes on a dynamic
  `idx`). The TVMScript evaluator auto-fills `step=1` for sliced call arguments,
  which is what triggers the crash without this fix. Fixes dynamic `T.tile.fill`
  (issue #1207). Upstream TVM fixed the same thing in a large refactor that cannot
  be cherry-picked into the pinned commit.

## Regenerating a patch after a submodule bump

    cd 3rdparty/tvm
    # apply the intended edit to the source file, then:
    git diff -- python/tvm/tir/buffer.py > ../patches/tvm_slice_step_fix.patch
    git checkout -- python/tvm/tir/buffer.py
