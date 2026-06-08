"""
Test T.sync_all("AICOnly") — cross-block visibility.
"""
import tilelang, tilelang.language as T, torch

tilelang.cache.clear_cache()
NB = 4


@tilelang.jit(out_idx=[0], target="pto")
def syncall_aiconly():
    @T.prim_func
    def main(Out: T.Tensor((NB, NB), "int32")):
        with T.Kernel(NB, is_npu=True) as (cid, vid):
            bid = cid
            val = T.alloc_ub((1,), "int32")
            buf = T.alloc_ub((NB,), "int32")
            with T.Scope("V"):
                T.tile.fill(val, bid + 1)
                T.barrier_all()
                T.copy(val, Out[0, bid])
                T.barrier_all()
                T.sync_all("AICOnly")
                T.copy(Out[0, :NB], buf)
                T.barrier_all()
                T.copy(buf, Out[bid, :NB])
    return main


func = syncall_aiconly()
torch.npu.synchronize()
out = func()
torch.npu.synchronize()

for b in range(NB):
    got = out[b].cpu().tolist()
    exp = list(range(1, NB + 1))
    assert got == exp, f"block {b}: got {got}, expected {exp}"
print("AICOnly: PASS — all blocks see whole shared row")