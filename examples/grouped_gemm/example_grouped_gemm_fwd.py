"""High-performance GroupedMatmul with an explicit-MMA pipeline.

All non-empty workloads share the explicit-MMA kernel. This frozen variant
predates the Catlass float32 L1-to-L0B fix, so non-transposed float32 weights
are transposed and made contiguous on the host before kernel dispatch.
Bias remains in the Vector epilogue; C2/BiasTable support is not required.
"""

import math
from typing import Dict, List, Optional

import tilelang
import torch
from tilelang import language as T

tilelang.cache.clear_cache()

CAST_MODE = "CAST_RINT"
VEC_NUM = 2
MTE2_TO_V_EVENT = 1
V_TO_MTE3_EVENT = 2

MANUAL_SYNC_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
}

C2V_EVENT_BASE = 0
V2C_EVENT_BASE = 2

L1_BUFFER_NUM = 2
L0_BUFFER_NUM = 2
L1_C0_ELEMS = {"float16": 16, "bfloat16": 16, "float32": 8}
L1_EVENT_BASE = 0
L0_EVENT_BASE = 2
L0C_EVENT = 4

TILING_PROFILES: Dict[str, Dict[str, int]] = {
    "m16n128k32": {"block_M": 16, "block_N": 128, "block_K": 32, "kL0Size": 32},
    "m64n128": {"block_M": 64, "block_N": 128, "block_K": 64, "kL0Size": 32},
    "m32n128": {"block_M": 32, "block_N": 128, "block_K": 64, "kL0Size": 32},
}

LARGE_TILING_CONFIG = {
    "float16": {"block_M": 128, "block_N": 256, "block_K": 128, "kL0Size": 64},
    "bfloat16": {"block_M": 128, "block_N": 256, "block_K": 128, "kL0Size": 64},
    "float32": {"block_M": 128, "block_N": 256, "block_K": 64, "kL0Size": 32},
}

PHYSICAL_CORE_NUM = 20


def torch_dtype_to_tl(dtype: torch.dtype) -> str:
    """Map the supported PyTorch dtypes to TileLang dtype names."""
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.bfloat16:
        return "bfloat16"
    if dtype == torch.float32:
        return "float"
    raise ValueError(f"Unsupported dtype: {dtype}")


@tilelang.jit(out_idx=[4], workspace_idx=[5], pass_configs=MANUAL_SYNC_PASS_CONFIGS)
def grouped_matmul_mma_kernel(
    batch_sizes_list,
    M,
    K,
    N,
    block_M,
    block_N,
    block_K,
    kL0Size,
    transpose_weight=False,
    dtype="float16",
    out_dtype="float16",
    has_bias=True,
    bias_dtype="float16",
    need_cast=True,
):
    """固定物理核数、显式 MMA 流水和双槽 workspace 的 kernel。"""
    accum_dtype = "float32"
    batch_count = len(batch_sizes_list)
    total_m_blocks = sum((size + block_M - 1) // block_M for size in batch_sizes_list)
    n_num = (N + block_N - 1) // block_N
    total_tasks = total_m_blocks * n_num
    core_num = min(PHYSICAL_CORE_NUM, total_tasks)
    tasks_per_core = (total_tasks + core_num - 1) // core_num
    block_M_2 = block_M // VEC_NUM
    l1_c0_elems = L1_C0_ELEMS[dtype]
    use_tail_l1 = block_K < K and K % block_K != 0

    if transpose_weight:
        w_shape = [batch_count, N, K]
        b_l1_shape = (L1_BUFFER_NUM, block_N, block_K)
    else:
        w_shape = [batch_count, K, N]
        b_l1_shape = (L1_BUFFER_NUM, block_K, block_N)
    tail_a_shape = (block_M, block_K) if use_tail_l1 else (1, 1)
    tail_b_shape = b_l1_shape[1:] if use_tail_l1 else (1, 1)

    @T.prim_func
    def kernel(
        X: T.Tensor([M, K], dtype),  # type: ignore
        W: T.Tensor(w_shape, dtype),  # type: ignore
        block_metadata: T.Tensor([total_m_blocks, 3], "int32"),  # type: ignore
        bias: T.Tensor([batch_count, N], bias_dtype),  # type: ignore
        Y: T.Tensor([M, N], out_dtype),  # type: ignore
        workspace: T.Tensor([core_num, 2, block_M, block_N], accum_dtype),  # type: ignore
    ):
        with T.Kernel(core_num, is_npu=True) as (cid, vid):
            A_L1 = T.alloc_L1((L1_BUFFER_NUM, block_M, block_K), dtype)
            B_L1 = T.alloc_L1(b_l1_shape, dtype)
            # A dedicated tile-base buffer makes GM→L1 K-tail clearing explicit:
            # dynamic ping-pong offsets are not recognized as tile bases by the
            # current codegen's need_clear=(dst_offset==0) rule.
            A_tail_L1 = T.alloc_L1(tail_a_shape, dtype)
            B_tail_L1 = T.alloc_L1(tail_b_shape, dtype)
            A_L0 = T.alloc_L0A((L0_BUFFER_NUM, block_M, kL0Size), dtype)
            B_L0 = T.alloc_L0B((L0_BUFFER_NUM, kL0Size, block_N), dtype)
            C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)

            with T.Scope("C"):
                for event_offset in T.unroll(L1_BUFFER_NUM):
                    T.set_flag("MTE1", "MTE2", L1_EVENT_BASE + event_offset)
                for event_offset in T.unroll(L0_BUFFER_NUM):
                    T.set_flag("M", "MTE1", L0_EVENT_BASE + event_offset)
                T.set_flag("FIX", "M", L0C_EVENT)

                for task_iter in T.serial(tasks_per_core):
                    task_id = task_iter * core_num + cid
                    if task_id < total_tasks:
                        slot = task_iter % 2

                        # Before reusing a slot, wait until both Vector subcores
                        # have consumed the previous slot into private UB.
                        if task_iter >= 2:
                            T.wait_cross_flag(V2C_EVENT_BASE + slot)

                        bx = task_id // n_num
                        by = task_id % n_num
                        expert = block_metadata[bx, 0]
                        m_start = block_metadata[bx, 1]
                        valid_m = block_metadata[bx, 2]

                        loop_k = T.ceildiv(K, block_K)
                        loop_kk = T.ceildiv(block_K, kL0Size)

                        # Prologue: load the first K_L1 tile.
                        T.wait_flag("MTE1", "MTE2", L1_EVENT_BASE)
                        T.copy(X[m_start : m_start + valid_m, 0:block_K], A_L1[0, :, :])
                        if transpose_weight:
                            T.copy(W[expert, by * block_N : (by + 1) * block_N, 0:block_K], B_L1[0, :, :])
                        else:
                            T.copy(W[expert, 0:block_K, by * block_N : (by + 1) * block_N], B_L1[0, :, :])
                        T.set_flag("MTE2", "MTE1", L1_EVENT_BASE)

                        # C_L0 is single-buffered because a 128x256 fp32
                        # accumulator already occupies all 128KB of L0C.
                        T.wait_flag("FIX", "M", L0C_EVENT)

                        for k in T.serial(loop_k):
                            l1_side = k % L1_BUFFER_NUM

                            # Prefetch the next GM→L1 tile while the current
                            # tile proceeds through MTE1 and MMA.
                            if k < loop_k - 1:
                                next_l1_side = (k + 1) % L1_BUFFER_NUM
                                T.wait_flag("MTE1", "MTE2", L1_EVENT_BASE + next_l1_side)
                                if use_tail_l1 and k == loop_k - 2:
                                    T.copy(X[m_start : m_start + valid_m, (k + 1) * block_K : (k + 2) * block_K], A_tail_L1)
                                    if transpose_weight:
                                        T.copy(
                                            W[expert, by * block_N : (by + 1) * block_N, (k + 1) * block_K : (k + 2) * block_K], B_tail_L1
                                        )
                                    else:
                                        T.copy(
                                            W[expert, (k + 1) * block_K : (k + 2) * block_K, by * block_N : (by + 1) * block_N], B_tail_L1
                                        )
                                else:
                                    T.copy(X[m_start : m_start + valid_m, (k + 1) * block_K : (k + 2) * block_K], A_L1[next_l1_side, :, :])
                                    if transpose_weight:
                                        T.copy(
                                            W[expert, by * block_N : (by + 1) * block_N, (k + 1) * block_K : (k + 2) * block_K],
                                            B_L1[next_l1_side, :, :],
                                        )
                                    else:
                                        T.copy(
                                            W[expert, (k + 1) * block_K : (k + 2) * block_K, by * block_N : (by + 1) * block_N],
                                            B_L1[next_l1_side, :, :],
                                        )
                                T.set_flag("MTE2", "MTE1", L1_EVENT_BASE + next_l1_side)

                            for kk in T.serial(loop_kk):
                                l0_side = kk % L0_BUFFER_NUM
                                if kk == 0:
                                    T.wait_flag("MTE2", "MTE1", L1_EVENT_BASE + l1_side)
                                T.wait_flag("M", "MTE1", L0_EVENT_BASE + l0_side)
                                # TileLang's current L1 index lowering selects
                                # different logical-to-fractal mappings after
                                # specializing away the Vector bias path. Keep
                                # the logical form for no-bias kernels; for
                                # bias kernels spell the equivalent physical
                                # offsets used by gemm_v0 in common.h.
                                if use_tail_l1 and k == loop_k - 1:
                                    if has_bias:
                                        T.copy(A_tail_L1[kk * block_M * kL0Size // block_K, 0], A_L0[l0_side, :, :])
                                        if transpose_weight:
                                            T.copy(B_tail_L1[kk * block_N * kL0Size // block_K, 0], B_L0[l0_side, :, :], transpose=True)
                                        else:
                                            T.copy(B_tail_L1[0, kk * l1_c0_elems * kL0Size], B_L0[l0_side, :, :])
                                    else:
                                        T.copy(A_tail_L1[0, kk * kL0Size], A_L0[l0_side, :, :])
                                        if transpose_weight:
                                            T.copy(B_tail_L1[0, kk * kL0Size], B_L0[l0_side, :, :], transpose=True)
                                        else:
                                            T.copy(B_tail_L1[kk * kL0Size, 0], B_L0[l0_side, :, :])
                                elif has_bias:
                                    T.copy(A_L1[l1_side, kk * block_M * kL0Size // block_K, 0], A_L0[l0_side, :, :])
                                    if transpose_weight:
                                        T.copy(B_L1[l1_side, kk * block_N * kL0Size // block_K, 0], B_L0[l0_side, :, :], transpose=True)
                                    else:
                                        T.copy(B_L1[l1_side, 0, kk * l1_c0_elems * kL0Size], B_L0[l0_side, :, :])
                                else:
                                    T.copy(A_L1[l1_side, 0, kk * kL0Size], A_L0[l0_side, :, :])
                                    if transpose_weight:
                                        T.copy(B_L1[l1_side, 0, kk * kL0Size], B_L0[l0_side, :, :], transpose=True)
                                    else:
                                        T.copy(B_L1[l1_side, kk * kL0Size, 0], B_L0[l0_side, :, :])
                                if kk == loop_kk - 1:
                                    T.set_flag("MTE1", "MTE2", L1_EVENT_BASE + l1_side)
                                T.set_flag("MTE1", "M", L0_EVENT_BASE + l0_side)
                                T.wait_flag("MTE1", "M", L0_EVENT_BASE + l0_side)
                                T.mma(A_L0[l0_side, :, :], B_L0[l0_side, :, :], C_L0, init=T.And(k == 0, kk == 0))
                                T.set_flag("M", "MTE1", L0_EVENT_BASE + l0_side)

                        T.set_flag("M", "FIX", L0C_EVENT)
                        T.wait_flag("M", "FIX", L0C_EVENT)
                        T.copy(C_L0, workspace[cid, slot, 0, 0])
                        T.set_cross_flag("FIX", C2V_EVENT_BASE + slot)
                        T.set_flag("FIX", "M", L0C_EVENT)

                for event_offset in T.unroll(L1_BUFFER_NUM):
                    T.wait_flag("MTE1", "MTE2", L1_EVENT_BASE + event_offset)
                for event_offset in T.unroll(L0_BUFFER_NUM):
                    T.wait_flag("M", "MTE1", L0_EVENT_BASE + event_offset)
                T.wait_flag("FIX", "M", L0C_EVENT)

            with T.Scope("V"):
                c_ub = T.alloc_ub((block_M_2, block_N), accum_dtype)
                bias_ub = T.alloc_ub((block_N,), accum_dtype)
                c_out = T.alloc_ub((block_M_2, block_N), out_dtype)
                bias_in_ub = T.alloc_ub((block_N,), bias_dtype)

                for task_iter in T.serial(tasks_per_core):
                    task_id = task_iter * core_num + cid
                    if task_id < total_tasks:
                        slot = task_iter % 2
                        bx = task_id // n_num
                        by = task_id % n_num
                        expert = block_metadata[bx, 0]
                        m_start = block_metadata[bx, 1]
                        valid_m = block_metadata[bx, 2]

                        T.wait_cross_flag(C2V_EVENT_BASE + slot)

                        v_start = m_start + vid * block_M_2
                        v_len = T.if_then_else(
                            valid_m > vid * block_M_2,
                            T.if_then_else(valid_m - vid * block_M_2 > block_M_2, block_M_2, valid_m - vid * block_M_2),
                            0,
                        )

                        T.copy(workspace[cid, slot, vid * block_M_2, 0], c_ub)
                        # The MTE2-pipe cross flag is ordered after the
                        # workspace read. Both Vector subcores issue it, so the
                        # Cube wait only completes once both UB copies are safe.
                        T.set_cross_flag("MTE2", V2C_EVENT_BASE + slot)

                        if has_bias:
                            T.copy(bias[expert, by * block_N : (by + 1) * block_N], bias_in_ub)

                        T.set_flag("MTE2", "V", MTE2_TO_V_EVENT)
                        T.wait_flag("MTE2", "V", MTE2_TO_V_EVENT)

                        if has_bias:
                            if bias_dtype != accum_dtype:
                                T.tile.cast(bias_ub, bias_in_ub, mode="CAST_NONE", count=block_N)
                            else:
                                T.copy(bias_in_ub, bias_ub)

                            T.pipe_barrier("V")
                            for i, j in T.Parallel(block_M_2, block_N):
                                c_ub[i, j] = c_ub[i, j] + bias_ub[j]

                            if need_cast:
                                T.pipe_barrier("V")

                        if need_cast:
                            T.tile.cast(c_out, c_ub, mode=CAST_MODE, count=block_M_2 * block_N)
                            T.set_flag("V", "MTE3", V_TO_MTE3_EVENT)
                            T.wait_flag("V", "MTE3", V_TO_MTE3_EVENT)
                            T.copy(c_out, Y[v_start : v_start + v_len, by * block_N : (by + 1) * block_N])
                        else:
                            T.set_flag("V", "MTE3", V_TO_MTE3_EVENT)
                            T.wait_flag("V", "MTE3", V_TO_MTE3_EVENT)
                            T.copy(c_ub, Y[v_start : v_start + v_len, by * block_N : (by + 1) * block_N])

    return kernel


def _select_tiling(tl_dtype: str, K: int, N: int, group_sizes: List[int]) -> Dict[str, int]:
    """基于 shape 与 group 分布选择经过实测的 tiling profile。"""

    def task_count(block_M: int, block_N: int) -> int:
        m_tiles = sum((group_m + block_M - 1) // block_M for group_m in group_sizes if group_m > 0)
        return m_tiles * ((N + block_N - 1) // block_N)

    nonempty_groups = [group_m for group_m in group_sizes if group_m > 0]
    if not nonempty_groups:
        return LARGE_TILING_CONFIG.get(tl_dtype, LARGE_TILING_CONFIG["float16"])

    max_group_m = max(nonempty_groups)
    large_tasks = task_count(128, 256)

    # Tiny explicit-MMA profile: pad N to the proven 128-column Cube tile. The
    # current AscendC Mmad/L0C path returns undefined data for a 32x32 tile,
    # even though it compiles successfully.
    if K <= 32 and N <= 32 and max_group_m <= 16 and task_count(16, 128) <= PHYSICAL_CORE_NUM:
        return TILING_PROFILES["m16n128k32"]

    # fp32 small-N kernels are under-parallel with the large tile. Keep each
    # candidate around 2M scalar multiply-adds per logical tile.
    if tl_dtype == "float32" and N <= 256 and large_tasks <= 4:
        if K <= 256 and task_count(64, 128) <= PHYSICAL_CORE_NUM:
            return TILING_PROFILES["m64n128"]
        if K <= 512 and task_count(32, 128) <= PHYSICAL_CORE_NUM:
            return TILING_PROFILES["m32n128"]

    if tl_dtype != "float32":
        # Small K/N: keep the exact-N 64-row profile when it still fits.
        if K <= 128 and N <= 128 and task_count(64, 128) <= PHYSICAL_CORE_NUM:
            return TILING_PROFILES["m64n128"]

        # Small groups: exact 32-row M tiles compensate for the extra N tasks.
        if K <= 256 and N <= 512 and max_group_m <= 32 and task_count(32, 128) <= PHYSICAL_CORE_NUM:
            return TILING_PROFILES["m32n128"]

    # Throughput profile: shapes outside the validated small-shape domains,
    # or small profiles whose direct-launch task count would exceed 20.
    return LARGE_TILING_CONFIG.get(tl_dtype, LARGE_TILING_CONFIG["float16"])


def _select_fixed_core(total_tasks: int) -> bool:
    """任务超过 20 个物理 Cube 核时启用 Fixed Core。"""
    return total_tasks > PHYSICAL_CORE_NUM


_KERNEL_CACHE = {}


def _get_kernel(batch_sizes_tuple, M, K, N, transpose_weight, tl_dtype, out_dtype, has_bias, bias_dtype, need_cast, config):
    config_tuple = (config["block_M"], config["block_N"], config["block_K"], config["kL0Size"])
    key = (batch_sizes_tuple, M, K, N, transpose_weight, tl_dtype, out_dtype, has_bias, bias_dtype, need_cast, config_tuple)
    if key not in _KERNEL_CACHE:
        _KERNEL_CACHE[key] = grouped_matmul_mma_kernel(
            batch_sizes_tuple,
            M,
            K,
            N,
            *config_tuple,
            transpose_weight=transpose_weight,
            dtype=tl_dtype,
            out_dtype=out_dtype,
            has_bias=has_bias,
            bias_dtype=bias_dtype,
            need_cast=need_cast,
        )
    return _KERNEL_CACHE[key]


def grouped_matmul(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    group_list=None,
    split_item: int = 0,
    transpose_weight: bool = False,
) -> List[torch.Tensor]:
    """Run all non-empty GroupedMatmul workloads through explicit MMA."""
    assert x.dim() == 2, "x must be 2D [M, K]"
    assert weight.dim() == 3, "weight must be 3D"

    M, K = x.shape
    expert_count = weight.shape[0]
    N = weight.shape[1] if transpose_weight else weight.shape[2]
    device = x.device
    dtype = x.dtype

    tl_dtype = torch_dtype_to_tl(dtype)
    if tl_dtype == "float":
        tl_dtype = "float32"

    if isinstance(group_list, torch.Tensor):
        ends = group_list.to(torch.int64).tolist()
    else:
        ends = list(group_list)
    assert len(ends) == expert_count, f"group_list length {len(ends)} != E {expert_count}"
    assert ends[-1] == M, f"group_list last value {ends[-1]} must equal M {M}"

    starts = [0] + ends[:-1]
    group_sizes = [ends[expert] - starts[expert] for expert in range(expert_count)]
    config = _select_tiling(tl_dtype, K, N, group_sizes)
    block_M = config["block_M"]
    total_m_blocks = sum(math.ceil(group_m / block_M) for group_m in group_sizes if group_m > 0)
    n_num = math.ceil(N / config["block_N"])
    total_tasks = total_m_blocks * n_num

    # A zero-task kernel cannot launch. The empty host result preserves the
    # public split semantics without introducing a second compute kernel.
    if total_tasks == 0:
        output = torch.empty((M, N), dtype=dtype, device=device)
        if split_item in (0, 1):
            return [output[starts[expert] : ends[expert]] for expert in range(expert_count)]
        return [output]

    # Compatibility path before the Catlass float32 zN(L1) -> nZ(L0B) fix.
    kernel_tw = transpose_weight
    if dtype == torch.float32 and not transpose_weight:
        weight = weight.transpose(-2, -1).contiguous()
        kernel_tw = True

    metadata_list = []
    for expert, group_m in enumerate(group_sizes):
        for block_idx in range(math.ceil(group_m / block_M)):
            metadata_list.append([expert, starts[expert] + block_idx * block_M, min(block_M, group_m - block_idx * block_M)])

    block_metadata = torch.tensor(metadata_list, device=device, dtype=torch.int32)
    has_bias = bias is not None
    need_cast = tl_dtype != "float32"
    out_dtype = "float32" if tl_dtype == "float32" else tl_dtype

    if has_bias:
        bias_tl_dtype = torch_dtype_to_tl(bias.dtype)
        if bias_tl_dtype == "float":
            bias_tl_dtype = "float32"
        bias_input = bias.contiguous()
    else:
        bias_tl_dtype = "float32"
        bias_input = torch.empty((expert_count, N), dtype=torch.float32, device=device)

    kernel = _get_kernel(tuple(group_sizes), M, K, N, kernel_tw, tl_dtype, out_dtype, has_bias, bias_tl_dtype, need_cast, config)
    output = kernel(x, weight, block_metadata, bias_input)

    if split_item in (0, 1):
        return [output[starts[expert] : ends[expert]] for expert in range(expert_count)]
    return [output]


def grouped_matmul_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    group_list=None,
    split_item: int = 0,
    transpose_weight: bool = False,
) -> List[torch.Tensor]:
    """PyTorch reference for the grouped, cumulative-end-index interface."""
    M = x.shape[0]
    expert_count = weight.shape[0]
    N = weight.shape[1] if transpose_weight else weight.shape[2]
    ends = group_list.to(torch.int64).tolist() if isinstance(group_list, torch.Tensor) else list(group_list)
    starts = [0] + ends[:-1]
    output = torch.zeros((M, N), dtype=x.dtype, device=x.device)
    x_fp32 = x.float()

    for expert in range(expert_count):
        start, end = starts[expert], ends[expert]
        if start == end:
            continue
        expert_weight = weight[expert].float()
        if transpose_weight:
            expert_weight = expert_weight.transpose(-2, -1)
        result = torch.matmul(x_fp32[start:end], expert_weight)
        if bias is not None:
            result = result + bias[expert].float()
        output[start:end] = result.to(x.dtype)

    if split_item in (0, 1):
        return [output[starts[expert] : ends[expert]] for expert in range(expert_count)]
    return [output]


if __name__ == "__main__":
    # This shape is the fp32/non-transposed compatibility case. It verifies
    # that the example runs on TileLang before the Catlass L1-to-L0B fix by
    # exercising the host-side transpose-and-contiguous route.
    torch.manual_seed(0)
    M, K, N = 256, 256, 256
    group_list = [256]
    x = torch.randn((M, K), dtype=torch.float32, device="npu")
    weight = torch.randn((1, K, N), dtype=torch.float32, device="npu")
    bias = torch.randn((1, N), dtype=torch.float32, device="npu")

    output = grouped_matmul(x, weight, bias, group_list, split_item=0, transpose_weight=False)
    reference = grouped_matmul_reference(x, weight, bias, group_list, split_item=0, transpose_weight=False)
    torch.npu.synchronize()

    torch.testing.assert_close(output[0], reference[0], rtol=1e-3, atol=1e-3)
    print("Kernel Output Match!")
