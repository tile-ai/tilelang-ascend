import os
from importlib.util import find_spec
from typing import Optional

import torch
import tilelang

import torch_npu
from tilelang import language as T

_ = torch_npu
tilelang.cache.clear_cache()


def align(x: int, alignment: int) -> int:
    return (x + alignment - 1) // alignment * alignment


def ceil_div(x: int, divisor: int) -> int:
    return (x + divisor - 1) // divisor


pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(pass_configs=pass_configs)
def get_expand_to_fused_kernel(
    hidden: int,
    num_topk: int,
    num_per_channels: Optional[int],
    use_tma_aligned_col_major_sf: Optional[bool],
    use_packed_ue8m0: Optional[bool],
    x_dtype: str,
    sf_dtype: str,
):
    num_tokens = T.symbolic("num_tokens")
    num_expanded_tokens = T.symbolic("num_expanded_tokens")
    num_cores = 24

    rows_per_vec = 64 if hidden <= 1024 else 16 if hidden <= 2048 else 4 if hidden <= 4096 else 2
    tokens_per_block = rows_per_vec * 2
    num_token_blocks = T.ceildiv(num_tokens, tokens_per_block)
    num_iters = T.ceildiv(num_token_blocks, num_cores)
    stages = 2
    aligned_topk = ((num_topk + 7) // 8) * 8

    hidden_aligned = hidden
    if num_per_channels is not None:
        hidden_sf = ceil_div(hidden, num_per_channels)
        if use_packed_ue8m0:
            hidden_sf = ceil_div(hidden_sf, 4)
        hidden_sf_aligned = hidden_sf
    else:
        hidden_sf, hidden_sf_aligned = 1, 1

    sf_shape = (hidden_sf, num_expanded_tokens) if use_tma_aligned_col_major_sf else (num_expanded_tokens, hidden_sf)

    @T.prim_func
    def expand_to_fused_kernel(
        x: T.Tensor((num_tokens, hidden), x_dtype),
        x_sf: T.Tensor((num_tokens, hidden_sf), sf_dtype),
        expanded_x_sf: T.Tensor(sf_shape, sf_dtype),
        token_topk_to_pos: T.Tensor((num_tokens, num_topk), "int32"),
        pos_to_expert: T.Tensor((num_expanded_tokens,), "int32"),
        expanded_x: T.Tensor((num_expanded_tokens, hidden), x_dtype),
    ):
        with T.Kernel(num_cores, is_npu=True) as (cid, vid), T.Scope("V"):
            pos_ub = T.alloc_ub((stages, rows_per_vec, aligned_topk), "int32")
            x_ub = T.alloc_ub((stages, rows_per_vec, hidden_aligned), x_dtype)
            x_sf_ub = T.alloc_ub((stages, rows_per_vec, hidden_sf_aligned), sf_dtype)
            block_id = T.alloc_var("int", init=0)
            next_block_id = T.alloc_var("int", init=0)
            token_start = T.alloc_var("int", init=0)
            next_token_start = T.alloc_var("int", init=0)
            token_id = T.alloc_var("int", init=0)

            for stage in T.serial(stages):
                T.set_flag("mte3", "v", stage)

            if cid < num_token_blocks:
                token_start = cid * tokens_per_block + vid * rows_per_vec

                T.wait_flag("mte3", "v", 0)
                T.set_flag("v", "mte2", 0)
                T.wait_flag("v", "mte2", 0)
                for row in T.serial(rows_per_vec):
                    token_id = token_start + row

                    if token_id < num_tokens:
                        T.copy(token_topk_to_pos[token_id, :], pos_ub[0, row, :])
                        T.copy(x[token_id, :], x_ub[0, row, 0:hidden])
                        if num_per_channels is not None:
                            T.copy(x_sf[token_id, :], x_sf_ub[0, row, 0:hidden_sf])

                T.set_flag("mte2", "s", 0)
                T.set_flag("mte2", "v", 0)

            for i in T.serial(num_iters):
                cur = i % stages
                nxt = (i + 1) % stages

                block_id = cid + i * num_cores
                token_start = block_id * tokens_per_block + vid * rows_per_vec

                if block_id < num_token_blocks:
                    T.wait_flag("mte2", "s", cur)
                    T.wait_flag("mte2", "v", cur)

                    next_block_id = cid + (i + 1) * num_cores
                    if next_block_id < num_token_blocks:
                        next_token_start = next_block_id * tokens_per_block + vid * rows_per_vec
                        T.wait_flag("mte3", "v", nxt)
                        T.set_flag("v", "mte2", nxt)
                        T.wait_flag("v", "mte2", nxt)
                        for row in T.serial(rows_per_vec):
                            token_id = next_token_start + row
                            if token_id < num_tokens:
                                T.copy(token_topk_to_pos[token_id, :], pos_ub[nxt, row, :])
                                T.copy(x[token_id, :], x_ub[nxt, row, 0:hidden])
                                if num_per_channels is not None:
                                    T.copy(x_sf[token_id, :], x_sf_ub[nxt, row, 0:hidden_sf])

                        T.set_flag("mte2", "s", nxt)
                        T.set_flag("mte2", "v", nxt)

                    T.set_flag("v", "mte3", cur)
                    T.wait_flag("v", "mte3", cur)
                    for row in T.serial(rows_per_vec):
                        token_id = token_start + row

                        if token_id < num_tokens:
                            for k in T.serial(num_topk):
                                pos = pos_ub[cur, row, k]

                                if pos >= 0:
                                    T.copy(x_ub[cur, row, 0:hidden], expanded_x[pos, :])
                                    if num_per_channels is not None:
                                        if use_tma_aligned_col_major_sf:
                                            T.copy(x_sf_ub[cur, row, 0:hidden_sf], expanded_x_sf[:, pos])
                                        else:
                                            T.copy(x_sf_ub[cur, row, 0:hidden_sf], expanded_x_sf[pos, :])

                    T.set_flag("mte3", "v", cur)

            for stage in T.serial(stages):
                T.wait_flag("mte3", "v", stage)

    return expand_to_fused_kernel


HAS_NPU = find_spec("torch_npu") is not None


def get_device() -> str:
    if HAS_NPU and torch.npu.is_available():
        return "npu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def npu_sync_if_needed(device: str) -> None:
    if device == "npu":
        torch.npu.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def dtype_to_tilelang_str(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def get_expand_kernel_for_tensors(
    x: torch.Tensor,
    token_topk_to_pos: torch.Tensor,
    pos_to_expert: torch.Tensor,
):
    assert x.is_contiguous()
    assert token_topk_to_pos.is_contiguous()
    assert pos_to_expert.is_contiguous()
    assert x.dim() == 2
    assert token_topk_to_pos.dim() == 2
    assert pos_to_expert.dim() == 1
    assert x.shape[0] == token_topk_to_pos.shape[0]

    num_tokens, hidden = x.shape
    _num_tokens, num_topk = token_topk_to_pos.shape
    x_dtype_str = dtype_to_tilelang_str(x.dtype)

    kernel = get_expand_to_fused_kernel(
        hidden,
        num_topk,
        None,
        None,
        None,
        x_dtype_str,
        x_dtype_str,
    )

    return kernel


def ascend_expand_to_fused(
    x: torch.Tensor,
    token_topk_to_pos: torch.Tensor,
    pos_to_expert: torch.Tensor,
) -> torch.Tensor:
    num_tokens, hidden = x.shape
    num_expanded_tokens = pos_to_expert.shape[0]

    out = torch.zeros((num_expanded_tokens, hidden), dtype=x.dtype, device=x.device)
    if num_tokens == 0:
        return out

    kernel = get_expand_kernel_for_tensors(x, token_topk_to_pos, pos_to_expert)
    dummy_x_sf = torch.empty((num_tokens, 1), dtype=x.dtype, device=x.device)
    dummy_expanded_x_sf = torch.empty((num_expanded_tokens, 1), dtype=x.dtype, device=x.device)
    kernel(x, dummy_x_sf, dummy_expanded_x_sf, token_topk_to_pos, pos_to_expert, out)
    return out


def ascend_expand_to_fused_profile_only(
    x: torch.Tensor,
    token_topk_to_pos: torch.Tensor,
    pos_to_expert: torch.Tensor,
) -> torch.Tensor:
    num_tokens, hidden = x.shape
    num_expanded_tokens = pos_to_expert.shape[0]

    out = torch.empty((num_expanded_tokens, hidden), dtype=x.dtype, device=x.device)
    if num_tokens == 0:
        return out

    kernel = get_expand_kernel_for_tensors(x, token_topk_to_pos, pos_to_expert)
    dummy_x_sf = torch.empty((num_tokens, 1), dtype=x.dtype, device=x.device)
    dummy_expanded_x_sf = torch.empty((num_expanded_tokens, 1), dtype=x.dtype, device=x.device)
    kernel(x, dummy_x_sf, dummy_expanded_x_sf, token_topk_to_pos, pos_to_expert, out)
    return out


def ascend_expand_to_fused_with_sf(
    x: tuple,
    num_per_channels: int,
    token_topk_to_pos: torch.Tensor,
    pos_to_expert: torch.Tensor,
    use_tma_aligned_col_major_sf: bool = False,
) -> tuple:
    x_data, x_sf = x
    assert x_data.is_contiguous()
    assert x_sf.is_contiguous()
    assert token_topk_to_pos.is_contiguous()
    assert pos_to_expert.is_contiguous()
    assert x_data.dim() == 2 and token_topk_to_pos.dim() == 2 and pos_to_expert.dim() == 1
    assert x_sf.dtype in (torch.float32, torch.int32)
    assert num_per_channels in [32, 128]

    num_tokens, hidden = x_data.shape
    num_topk = token_topk_to_pos.shape[1]
    num_expanded_tokens = pos_to_expert.shape[0]
    assert num_tokens == token_topk_to_pos.shape[0]
    assert num_tokens == x_sf.shape[0]

    num_expanded_sf_tokens = align(num_expanded_tokens, 4) if use_tma_aligned_col_major_sf else num_expanded_tokens
    hidden_sf = ceil_div(hidden, num_per_channels)

    use_packed_ue8m0 = False
    if x_sf.dtype == torch.int32:
        use_packed_ue8m0 = True
        hidden_sf = ceil_div(hidden_sf, 4)
        assert use_tma_aligned_col_major_sf

    assert hidden_sf == x_sf.shape[1]

    x_dtype_str = dtype_to_tilelang_str(x_data.dtype)
    sf_dtype_str = dtype_to_tilelang_str(x_sf.dtype)

    kernel = get_expand_to_fused_kernel(
        hidden,
        num_topk,
        num_per_channels,
        use_tma_aligned_col_major_sf,
        use_packed_ue8m0,
        x_dtype_str,
        sf_dtype_str,
    )

    out = torch.zeros((num_expanded_tokens, hidden), dtype=x_data.dtype, device=x_data.device)
    out_sf = torch.zeros(
        (hidden_sf, num_expanded_sf_tokens) if use_tma_aligned_col_major_sf else (num_expanded_tokens, hidden_sf),
        dtype=x_sf.dtype,
        device=x_sf.device,
    )
    out_sf = out_sf[:, :num_expanded_tokens] if use_tma_aligned_col_major_sf else out_sf

    if num_tokens > 0:
        kernel(x_data, x_sf, out_sf, token_topk_to_pos, pos_to_expert, out)
    out_sf = out_sf.T if use_tma_aligned_col_major_sf else out_sf

    return out, out_sf


def ascendc_expand_to_fused(
    x: torch.Tensor,
    expert_idx: torch.Tensor,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens, _ = x.shape
    num_topk = expert_idx.shape[1]
    active_num = num_tokens * num_topk

    (
        expanded_x,
        expanded_row_idx,
        _expert_tokens_count,
        _expanded_scale,
    ) = torch.ops.npu.npu_moe_init_routing_v2(
        x,
        expert_idx,
        scale=None,
        offset=None,
        active_num=active_num,
        expert_capacity=-1,
        expert_num=num_experts,
        drop_pad_mode=0,
        expert_tokens_num_type=1,
        expert_tokens_num_flag=True,
        quant_mode=-1,
        active_expert_range=[0, num_experts],
        row_idx_type=0,
    )

    return expanded_x, expanded_row_idx


def build_tilelang_mapping_from_ascendc_row_idx(
    x_cpu: torch.Tensor,
    expert_idx_cpu: torch.Tensor,
    ascendc_output_cpu: torch.Tensor,
    expanded_row_idx_cpu: torch.Tensor,
    num_tokens: int,
    num_topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:

    x_cpu = x_cpu.detach().to(device="cpu").contiguous()
    expert_idx_cpu = expert_idx_cpu.detach().to(device="cpu").contiguous()
    ascendc_output_cpu = ascendc_output_cpu.detach().to(device="cpu").contiguous()
    expanded_row_idx_cpu = expanded_row_idx_cpu.detach().to(device="cpu").contiguous()

    total_routes = num_tokens * num_topk
    scatter = expanded_row_idx_cpu.to(dtype=torch.int64).reshape(-1)
    assert scatter.numel() == total_routes

    valid = (scatter >= 0) & (scatter < total_routes)
    if not bool(valid.all()):
        bad = torch.where(~valid)[0][:16]
        raise AssertionError(f"expanded_row_idx contains invalid values at {bad.tolist()}: {scatter[bad].tolist()}")

    if torch.unique(scatter).numel() != total_routes:
        raise AssertionError("expanded_row_idx is not a permutation of [0, N*K).")

    sample_cols = min(16, x_cpu.shape[1])
    route_ids = torch.arange(
        total_routes,
        dtype=torch.int64,
        device="cpu",
    )
    token_ids_token_major = torch.div(
        route_ids,
        num_topk,
        rounding_mode="floor",
    )

    ref_sample = torch.empty(
        (ascendc_output_cpu.shape[0], sample_cols),
        dtype=ascendc_output_cpu.dtype,
        device="cpu",
    )
    ref_sample[scatter] = x_cpu[token_ids_token_major, :sample_cols]

    if not torch.equal(ref_sample, ascendc_output_cpu[:, :sample_cols]):
        raise AssertionError(
            "The installed npu_moe_init_routing_v2 row-index semantics do not match the official token-major scatter formula."
        )

    token_topk_to_pos_cpu = scatter.to(torch.int32).reshape(num_tokens, num_topk).contiguous()

    source_expert_token_major = expert_idx_cpu.reshape(-1).to(torch.int32)
    pos_to_expert_cpu = torch.empty(
        (total_routes,),
        dtype=torch.int32,
        device="cpu",
    )
    pos_to_expert_cpu[scatter] = source_expert_token_major

    if pos_to_expert_cpu.numel() > 1 and not bool(torch.all(pos_to_expert_cpu[1:] >= pos_to_expert_cpu[:-1])):
        raise AssertionError("expanded_row_idx does not place routes in expert-sorted order.")

    return token_topk_to_pos_cpu, pos_to_expert_cpu


def verify_result(
    output: torch.Tensor,
    golden: torch.Tensor,
    rtol: float = 1e-3,
    atol: float = 1e-3,
    error_tol: float = 1e-4,
) -> bool:
    output = output.reshape(-1)
    golden = golden.reshape(-1)
    assert output.dtype == golden.dtype

    if output.dtype in (torch.float16, torch.bfloat16, torch.float32):
        output = output.to(torch.float64)
        golden = golden.to(torch.float64)

    close_mask = torch.isclose(
        output,
        golden,
        rtol=rtol,
        atol=atol,
        equal_nan=True,
    )
    different_element_indexes = torch.where(~close_mask)[0]

    error_ratio = float(different_element_indexes.numel()) / golden.numel()
    return error_ratio <= error_tol


def get_test_configs() -> list[dict[str, int]]:

    return [
        {"num_tokens": 4001, "hidden": 576, "num_topk": 8, "num_experts": 4},
        {"num_tokens": 4001, "hidden": 2048, "num_topk": 8, "num_experts": 4},
        {"num_tokens": 4001, "hidden": 4096, "num_topk": 8, "num_experts": 4},
        {"num_tokens": 4001, "hidden": 7168, "num_topk": 8, "num_experts": 4},
        {"num_tokens": 4001, "hidden": 576, "num_topk": 9, "num_experts": 4},
        {"num_tokens": 4001, "hidden": 2048, "num_topk": 9, "num_experts": 4},
        {"num_tokens": 4001, "hidden": 4096, "num_topk": 9, "num_experts": 4},
        {"num_tokens": 4001, "hidden": 7168, "num_topk": 9, "num_experts": 4},
    ]


def run_one_case(config: dict[str, int]) -> None:
    num_tokens = config["num_tokens"]
    hidden = config["hidden"]
    num_topk = config["num_topk"]
    num_experts = config["num_experts"]

    x_cpu = torch.randn(
        (num_tokens, hidden),
        dtype=torch.bfloat16,
        device="cpu",
    )
    expert_idx_cpu = torch.randint(
        0,
        num_experts,
        (num_tokens, num_topk),
        dtype=torch.int32,
        device="cpu",
    )

    x = x_cpu.npu().contiguous()
    expert_idx = expert_idx_cpu.npu().contiguous()

    ascendc_output, ascendc_expanded_row_idx = ascendc_expand_to_fused(
        x,
        expert_idx,
        num_experts,
    )
    torch.npu.synchronize()

    assert ascendc_expanded_row_idx.dtype == torch.int32
    assert ascendc_expanded_row_idx.numel() == num_tokens * num_topk

    ascendc_output_cpu = ascendc_output.cpu()
    expanded_row_idx_cpu = ascendc_expanded_row_idx.cpu()
    token_topk_to_pos_cpu, pos_to_expert_cpu = build_tilelang_mapping_from_ascendc_row_idx(
        x_cpu=x_cpu,
        expert_idx_cpu=expert_idx_cpu,
        ascendc_output_cpu=ascendc_output_cpu,
        expanded_row_idx_cpu=expanded_row_idx_cpu,
        num_tokens=num_tokens,
        num_topk=num_topk,
    )

    token_topk_to_pos = token_topk_to_pos_cpu.npu().contiguous()
    pos_to_expert = pos_to_expert_cpu.npu().contiguous()

    output = ascend_expand_to_fused(
        x,
        token_topk_to_pos,
        pos_to_expert,
    )
    torch.npu.synchronize()

    output_cpu = output.cpu()

    passed_ascendc = verify_result(
        output_cpu,
        ascendc_output_cpu,
        rtol=0.0,
        atol=0.0,
    )
    assert passed_ascendc

    torch.testing.assert_close(
        output_cpu,
        ascendc_output_cpu,
        rtol=0.0,
        atol=0.0,
        check_dtype=True,
    )
    case = f"expand_to_fused T={num_tokens} H={hidden} K={num_topk} E={num_experts}"
    print(f"pass {case}")


def main() -> None:
    torch.manual_seed(0)
    os.environ.setdefault("TILELANG_PRINT_ON_COMPILATION", "0")

    run_times = int(os.getenv("RUN_TIMES", "1"))
    test_configs = get_test_configs()

    for _ in range(run_times):
        for config in test_configs:
            run_one_case(config)

    print("TEST PASSED!")


if __name__ == "__main__":
    main()
