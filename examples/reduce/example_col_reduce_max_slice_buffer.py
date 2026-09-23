import tilelang
from tilelang import language as T
import torch


def _check_precision(actual, golden, dtype):
    table = {
        "float16": (2**-14, 2**-9, 0.1),
        "bfloat16": (2**-10, 2**-6, 1.0),
        "float32": (2**-16, 2**-10, 0.01),
        "hifloat32": (2**-16, 2**-10, 0.01),
        "float8_e4m3": (2**-4, 2**-2, 1.0),
        "float8_e5m2": (2**-3, 2**-1, 0.1),
    }
    actual, golden = actual.detach().cpu().float(), golden.detach().cpu().float()
    if actual.shape != golden.shape:
        return False, 0.0, float("inf")
    dtype_name = str(dtype).replace("torch.", "")
    dtype_name = "float8_e4m3" if "float8_e4m3" in dtype_name else "float8_e5m2" if "float8_e5m2" in dtype_name else dtype_name
    atol, rtol, limit = table.get(dtype_name, table["float16"])
    special = ~torch.isfinite(golden)
    if special.any() and (
        not torch.equal(torch.isnan(actual[special]), torch.isnan(golden[special]))
        or not torch.equal(torch.isinf(actual[special]), torch.isinf(golden[special]))
    ):
        return False, 0.0, float("inf")
    finite = torch.isfinite(golden)
    if not finite.any():
        return True, 1.0, 0.0
    error = (actual[finite] - golden[finite]).abs()
    error = torch.where(torch.isfinite(error), error, torch.full_like(error, float("inf")))
    ratio, maximum = (error <= atol + rtol * golden[finite].abs()).float().mean().item(), error.max().item()
    return ratio >= 0.99 and maximum <= limit, ratio, maximum


torch.set_default_device("npu")
torch.manual_seed(42)

tilelang.disable_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[1], target="pto", pass_configs=pass_configs)
def reduce_max_slice_buffer():
    dtype = "float"

    @T.prim_func
    def main(
        Input: T.Tensor([5, 8], dtype),
        Output: T.Tensor([1, 8], dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            in_shared = T.alloc_ub((5, 8), dtype)
            out_shared = T.alloc_ub((1, 8), dtype=dtype)

            if vid == 0:
                T.copy(Input, in_shared)
                T.reduce_max(in_shared, out_shared, dim=0, real_shape=[3, 8])
                T.copy(out_shared, Output)

    return main


func = reduce_max_slice_buffer()
print("init successful!")

dtype = torch.float
input = torch.randn((5, 8), dtype=dtype)
output = torch.empty((1, 8), dtype=dtype)
torch.npu.synchronize()

output = func(input)
torch.npu.synchronize()

ref_output = torch.max(input[:3, :], dim=0, keepdim=True).values
torch.npu.synchronize()

# print(f"ref_output: {ref_output}")
# print(f"input: {input}")
# print(f"output: {output[:, :4]}")

passed, ratio, max_abs = _check_precision(output, ref_output, output.dtype)
assert passed, f"dtype={output.dtype}, matched_ratio={ratio:.4f}, max_abs_error={max_abs:.6e}"
print("Kernel Output Match!")

# kernel = tilelang.engine.lower(func,target="pto")
# print(kernel.kernel_source)
