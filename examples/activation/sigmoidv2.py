import tilelang
from tilelang import language as T
import torch


def _check_precision(actual, golden, dtype):
    name = str(dtype).replace("torch.", "")
    if name.startswith("float8_e4m3"):
        name = "float8_e4m3"
    elif name.startswith("float8_e5m2"):
        name = "float8_e5m2"
    table = {
        "float16": (2**-14, 2**-9, 1e-1, 0.99),
        "bfloat16": (2**-10, 2**-6, 1.0, 0.99),
        "float32": (2**-16, 2**-10, 1e-2, 0.99),
        "hifloat32": (2**-16, 2**-10, 1e-2, 0.99),
        "float8_e4m3": (2**-4, 2**-2, 1.0, 0.99),
        "float8_e5m2": (2**-3, 2**-1, 1e-1, 0.99),
    }
    actual_cpu, golden_cpu = actual.detach().cpu(), golden.detach().cpu()
    if actual_cpu.shape != golden_cpu.shape:
        return False, 0.0, float("inf")
    if name in {"int8", "int16", "int32", "int64", "uint8"}:
        mismatches, total = (actual_cpu != golden_cpu).sum().item(), max(actual_cpu.numel(), 1)
        return mismatches == 0, 1.0 - mismatches / total, 0.0 if mismatches == 0 else float("inf")
    atol, rtol, max_limit, required_ratio = table.get(name, table["float16"])
    actual_fp32, golden_fp32 = actual_cpu.float(), golden_cpu.float()
    special = ~torch.isfinite(golden_fp32)
    if special.any() and (
        not torch.equal(torch.isnan(actual_fp32[special]), torch.isnan(golden_fp32[special]))
        or not torch.equal(torch.isinf(actual_fp32[special]), torch.isinf(golden_fp32[special]))
        or not torch.equal(actual_fp32[special][torch.isinf(golden_fp32[special])], golden_fp32[special][torch.isinf(golden_fp32[special])])
    ):
        return False, 0.0, float("inf")
    finite = torch.isfinite(golden_fp32)
    if finite.sum().item() == 0:
        return True, 1.0, 0.0
    abs_error = (actual_fp32[finite] - golden_fp32[finite]).abs()
    abs_error = torch.where(torch.isfinite(abs_error), abs_error, torch.full_like(abs_error, float("inf")))
    ratio = (abs_error <= atol + rtol * golden_fp32[finite].abs()).float().mean().item()
    max_abs = abs_error.max().item()
    return ratio >= required_ratio and max_abs <= max_limit, ratio, max_abs


torch.set_default_device("npu")
torch.manual_seed(42)

tilelang.disable_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def sigmoidv2():
    dtype = "float"

    @T.prim_func
    def main(
        input: T.Tensor([4, 8], dtype),
        output: T.Tensor([4, 8], dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            input_shared = T.alloc_ub((4, 8), dtype)
            output_shared = T.alloc_ub((4, 8), dtype)

            T.copy(input, input_shared)
            T.tile.sigmoid(output_shared, input_shared)
            T.copy(output_shared, output)

    return main


dtype = torch.float
input = torch.randn([4, 8], dtype=dtype)
func = sigmoidv2()
print("init successful!")
output = func(input)

torch.npu.synchronize()

ref_output = torch.sigmoid(input)

passed, ratio, max_abs = _check_precision(output, ref_output, output.dtype)
assert passed, f"dtype={output.dtype}, matched_ratio={ratio:.4f}, max_abs_error={max_abs:.6e}"
print("Kernel Output Match!")
