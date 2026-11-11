"""In-place NPU compressed-cache initialization/append; candidate only."""

from functools import lru_cache

import tilelang
import tilelang.language as T

EXPERT_PASSES = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
}


def _plan(B, rows):
    if any(type(v) is not int or v <= 0 for v in (B, rows)):
        raise ValueError("B and physical row capacity must be positive integers")
    if B * rows * 576 > 2**31 - 1:
        raise ValueError("cache/source offsets exceed the declared int32 indexing domain")
    tasks = B * rows
    blocks = min(24, (tasks + 1) // 2)
    workers = blocks * 2
    return tasks, blocks, workers, (tasks + workers - 1) // workers


def _options(dtype, target):
    if dtype not in ("float16", "bfloat16"):
        raise TypeError("cache dtype must be float16 or bfloat16")
    if target not in ("pto", "ascendc"):
        raise ValueError("target must be pto or ascendc")


@lru_cache(maxsize=128)
def initialize_cache_kernel(B, capacity, dtype="float16", *, target="pto"):
    _options(dtype, target)
    tasks, blocks, workers, rounds = _plan(B, capacity)

    @T.prim_func
    def main(CacheC: T.Tensor((B, capacity, 512), dtype), CacheE: T.Tensor((B, capacity, 64), dtype)):
        with T.Kernel(blocks, is_npu=True) as (cid, vid):
            c = T.alloc_ub((512,), dtype)
            e = T.alloc_ub((64,), dtype)
            T.annotate_address({c: 0, e: 1024})
            with T.Scope("V"):
                T.tile.fill(c, 0.0)
                T.tile.fill(e, 0.0)
                T.barrier_all()
                for turn in T.serial(rounds):
                    task = cid * 2 + vid + turn * workers
                    if task < tasks:
                        b = task // capacity
                        t = task % capacity
                        T.copy(c, CacheC[b, t, 0])
                        T.copy(e, CacheE[b, t, 0])
                        T.barrier_all()

    return tilelang.compile(main, out_idx=[], target=target, pass_configs=EXPERT_PASSES)


@lru_cache(maxsize=128)
def append_cache_kernel(B, Qcap, capacity, dtype="float16", *, target="pto"):
    _options(dtype, target)
    tasks, blocks, workers, rounds = _plan(B, Qcap)
    _plan(B, capacity)

    @T.prim_func
    def main(
        Cnorm: T.Tensor((B, Qcap, 512), dtype),
        RawKV: T.Tensor((B, Qcap, 576), dtype),
        Metadata: T.Tensor((B, 8), "int32"),
        CacheC: T.Tensor((B, capacity, 512), dtype),
        CacheE: T.Tensor((B, capacity, 64), dtype),
    ):
        with T.Kernel(blocks, is_npu=True) as (cid, vid):
            meta = T.alloc_ub((8,), "int32")
            c = T.alloc_ub((512,), dtype)
            e = T.alloc_ub((64,), dtype)
            T.annotate_address({meta: 0, c: 32, e: 1056})
            with T.Scope("V"):
                for turn in T.serial(rounds):
                    task = cid * 2 + vid + turn * workers
                    if task < tasks:
                        b = task // Qcap
                        t = task % Qcap
                        T.copy(Metadata[b, 0], meta)
                        T.barrier_all()
                        q_len = meta[0]
                        kv_len = meta[1]
                        start = meta[2]
                        destination = start + t
                        if t < q_len and start >= 0 and destination < capacity and destination < kv_len:
                            T.copy(Cnorm[b, t, 0], c)
                            T.copy(RawKV[b, t, 512], e)
                            T.barrier_all()
                            T.copy(c, CacheC[b, destination, 0])
                            T.copy(e, CacheE[b, destination, 0])
                            T.barrier_all()

    return tilelang.compile(main, out_idx=[], target=target, pass_configs=EXPERT_PASSES)
