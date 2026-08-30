import math

import tilelang
import torch
from tilelang import language as T

VEC_NUM = 2
MAX_UB_BYTES = 192 * 1024


def _estimate_ub_bytes(mhc: int, h_blk: int, num_layers: int, num_post: int, tbs: int) -> int:
    sub_h = h_blk // VEC_NUM
    mix_per_vec = num_layers * tbs * mhc * 4 + num_post * tbs * mhc * 4 + num_post * tbs * mhc * mhc * 4
    work_per_vec = (
        mhc * sub_h * 4 + mhc * sub_h * 4 + sub_h * 4 + sub_h * 4 + mhc * sub_h * 2 + num_post * sub_h * 2 + sub_h * 2 + mhc * sub_h * 2
    )
    return (mix_per_vec + work_per_vec) * VEC_NUM


def _compute_safe_params(hidden: int, mhc: int, num_layers: int, num_post: int, max_h_blk: int = 2048) -> tuple[int, int]:
    for h_blk_candidate in [2048, 1024, 512, 256, 128]:
        if h_blk_candidate > max_h_blk:
            continue
        h_blk = math.gcd(h_blk_candidate, hidden)
        if h_blk % VEC_NUM != 0:
            continue
        for tbs_candidate in [128, 64, 32, 16, 8]:
            ub = _estimate_ub_bytes(mhc, h_blk, num_layers, num_post, tbs_candidate)
            if ub <= MAX_UB_BYTES:
                return h_blk, tbs_candidate
    raise ValueError(f"No safe h_blk/tbs for hidden={hidden} mhc={mhc} L={num_layers} L_post={num_post}")


_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def _mhc_multilayer_recompute_kernel(
    mhc_mult: int,
    hidden: int,
    num_layers: int,
    num_post: int,
    n_thr: int = 16,
    h_blk: int = 2048,
) -> tilelang.JITKernel:
    n = T.symbolic("num_tokens")
    dtype = "float32"
    dbtype = "bfloat16"
    h = hidden
    mhc = mhc_mult
    L = num_layers
    L_post = num_post
    tbs = n_thr

    h_blk = math.gcd(h_blk, hidden)
    assert h_blk % VEC_NUM == 0, f"h_blk={h_blk} must be divisible by VEC_NUM={VEC_NUM}"

    sub_h = h_blk // VEC_NUM
    num_blocks_h = h // h_blk

    block_pre_size = L * tbs * mhc
    block_post_size = L_post * tbs * mhc
    block_comb_size = L_post * tbs * mhc * mhc

    @T.prim_func
    def kernel(
        initial_residual: T.Tensor[(n, mhc, h), dbtype],
        pre_mix_tensors: T.Tensor[(L, n, mhc), dtype],
        layer_output_tensors: T.Tensor[(L_post, n, h), dbtype],
        post_mix_tensors: T.Tensor[(L_post, n, mhc), dtype],
        comb_mix_tensors: T.Tensor[(L_post, n, mhc, mhc), dtype],
        layer_input_tensors: T.Tensor[(L, n, h), dbtype],
        residual_tensors: T.Tensor[(L_post, n, mhc, h), dbtype],
    ) -> None:
        pre_mix_flat = T.decl_buffer(shape=(L * n * mhc,), data=pre_mix_tensors.data)
        post_mix_flat = T.decl_buffer(shape=(L_post * n * mhc,), data=post_mix_tensors.data)
        comb_mix_flat = T.decl_buffer(shape=(L_post * n * mhc * mhc,), data=comb_mix_tensors.data)

        with T.Kernel(T.ceildiv(n, tbs), is_npu=True) as (cid, vid):
            all_pre_mix_ub = T.alloc_ub((block_pre_size,), dtype)
            all_post_mix_ub = T.alloc_ub((block_post_size,), dtype)
            all_comb_mix_ub = T.alloc_ub((block_comb_size,), dtype)

            res_ub = T.alloc_ub((mhc, sub_h), dtype)
            new_res_ub = T.alloc_ub((mhc, sub_h), dtype)
            layer_input_ub = T.alloc_ub((sub_h,), dtype)
            layer_output_ub = T.alloc_ub((sub_h,), dtype)

            initial_res_bf16_ub = T.alloc_ub((mhc, sub_h), dbtype)
            all_layer_output_bf16_ub = T.alloc_ub((L_post, sub_h), dbtype)
            layer_input_bf16_ub = T.alloc_ub((sub_h,), dbtype)
            new_res_bf16_ub = T.alloc_ub((mhc, sub_h), dbtype)

            pid_n_first = cid * tbs

            for l_idx in range(L):
                src_offset = l_idx * (n * mhc) + pid_n_first * mhc
                T.copy(
                    pre_mix_flat[src_offset : src_offset + tbs * mhc], all_pre_mix_ub[l_idx * (tbs * mhc) : l_idx * (tbs * mhc) + tbs * mhc]
                )

            for l_idx in range(L_post):
                src_offset_pm = l_idx * (n * mhc) + pid_n_first * mhc
                src_offset_cm = l_idx * (n * mhc * mhc) + pid_n_first * (mhc * mhc)
                T.copy(
                    post_mix_flat[src_offset_pm : src_offset_pm + tbs * mhc],
                    all_post_mix_ub[l_idx * (tbs * mhc) : l_idx * (tbs * mhc) + tbs * mhc],
                )
                T.copy(
                    comb_mix_flat[src_offset_cm : src_offset_cm + tbs * mhc * mhc],
                    all_comb_mix_ub[l_idx * (tbs * mhc * mhc) : l_idx * (tbs * mhc * mhc) + tbs * mhc * mhc],
                )

            T.set_flag("mte2", "v", 4)
            T.wait_flag("mte2", "v", 4)

            for i_token in T.serial(tbs):
                pid_n = cid * tbs + i_token

                if pid_n < n:
                    i_n_offset = i_token * mhc

                    for i0_h in T.serial(num_blocks_h):
                        h_offset = i0_h * h_blk + vid * sub_h

                        T.copy(initial_residual[pid_n, 0:mhc, h_offset : h_offset + sub_h], initial_res_bf16_ub[:, :])
                        for l_idx in range(L_post):
                            T.copy(layer_output_tensors[l_idx, pid_n, h_offset : h_offset + sub_h], all_layer_output_bf16_ub[l_idx, :])
                        T.set_flag("mte2", "v", 0)
                        T.wait_flag("mte2", "v", 0)

                        T.tile.cast(res_ub, initial_res_bf16_ub, "CAST_NONE", mhc * sub_h)

                        for i_layer in range(L_post):
                            pre_offset = i_layer * (tbs * mhc) + i_n_offset
                            comb_offset = i_layer * (tbs * mhc * mhc) + i_token * (mhc * mhc)

                            T.tile.cast(layer_output_ub, all_layer_output_bf16_ub[i_layer, :], "CAST_NONE", sub_h)

                            T.tile.mul(layer_input_ub, res_ub[0, :], all_pre_mix_ub[pre_offset])
                            for i_mhc in range(1, mhc):
                                T.tile.axpy(layer_input_ub, res_ub[i_mhc, :], all_pre_mix_ub[pre_offset + i_mhc])

                            for i_mhc in range(mhc):
                                T.tile.mul(new_res_ub[i_mhc, :], layer_output_ub, all_post_mix_ub[pre_offset + i_mhc])
                                for i_mhci in range(mhc):
                                    T.tile.axpy(
                                        new_res_ub[i_mhc, :], res_ub[i_mhci, :], all_comb_mix_ub[comb_offset + i_mhci * mhc + i_mhc]
                                    )

                            T.tile.cast(layer_input_bf16_ub, layer_input_ub, "CAST_ROUND", sub_h)
                            T.tile.cast(new_res_bf16_ub, new_res_ub, "CAST_ROUND", mhc * sub_h)

                            T.set_flag("v", "mte3", 2)

                            T.copy(new_res_ub[:, :], res_ub[:, :])

                            T.wait_flag("v", "mte3", 2)
                            T.copy(layer_input_bf16_ub[0:sub_h], layer_input_tensors[i_layer, pid_n, h_offset : h_offset + sub_h])
                            T.copy(new_res_bf16_ub[:, :], residual_tensors[i_layer, pid_n, :, h_offset : h_offset + sub_h])
                            T.set_flag("mte3", "v", 2)

                            T.wait_flag("mte3", "v", 2)

                        if L_post < L:
                            bnd_pre_offset = L_post * (tbs * mhc) + i_n_offset

                            T.tile.mul(layer_input_ub, res_ub[0, :], all_pre_mix_ub[bnd_pre_offset])
                            for i_mhc in range(1, mhc):
                                T.tile.axpy(layer_input_ub, res_ub[i_mhc, :], all_pre_mix_ub[bnd_pre_offset + i_mhc])

                            T.tile.cast(layer_input_bf16_ub, layer_input_ub, "CAST_ROUND", sub_h)

                            T.set_flag("v", "mte3", 3)
                            T.wait_flag("v", "mte3", 3)
                            T.copy(layer_input_bf16_ub[0:sub_h], layer_input_tensors[L_post, pid_n, h_offset : h_offset + sub_h])
                            T.set_flag("mte3", "v", 3)
                            T.wait_flag("mte3", "v", 3)

    return kernel


def mhc_multilayer_recompute_reference(initial_residual, pre_mix, layer_output, post_mix, comb_mix, num_layers, num_post):
    device = initial_residual.device
    N, mhc, H = initial_residual.shape

    layer_input_ref = torch.zeros((num_layers, N, H), dtype=torch.bfloat16, device=device)
    residual_ref = torch.zeros((num_post, N, mhc, H), dtype=torch.bfloat16, device=device)

    current_res = initial_residual.to(torch.float32)

    for i_layer in range(num_post):
        p_mix = pre_mix[i_layer].unsqueeze(-1)
        l_in = torch.sum(p_mix * current_res, dim=1)
        layer_input_ref[i_layer] = l_in.to(torch.bfloat16)

        l_out = layer_output[i_layer].to(torch.float32).unsqueeze(1)
        p_out = post_mix[i_layer].unsqueeze(-1)
        gated_out = p_out * l_out

        c_mix = comb_mix[i_layer]
        transformed_res = torch.bmm(c_mix.transpose(1, 2), current_res)

        current_res = gated_out + transformed_res
        residual_ref[i_layer] = current_res.to(torch.bfloat16)

    if num_layers > num_post:
        p_mix = pre_mix[num_post].unsqueeze(-1)
        l_in = torch.sum(p_mix * current_res, dim=1)
        layer_input_ref[num_post] = l_in.to(torch.bfloat16)

    return layer_input_ref, residual_ref


_CORRECTNESS_CASES = [
    (1, 1, 2560),
    (3, 2, 2560),
    (10, 9, 2560),
    (10, 10, 2560),
    (10, 9, 4096),
    (10, 10, 4096),
    (10, 9, 7168),
    (10, 10, 7168),
    (10, 9, 8192),
    (10, 10, 8192),
]


def _run_test_case(num_layers: int, num_post: int, hidden: int, n: int = 8192, mhc_mult: int = 4):
    device = "npu"

    h_blk, tbs = _compute_safe_params(hidden, mhc_mult, num_layers, num_post)
    n_padded = ((n + tbs - 1) // tbs) * tbs

    torch.manual_seed(42)
    initial_residual = torch.randn((n_padded, mhc_mult, hidden), dtype=torch.bfloat16, device=device)
    pre_mix_tensors = torch.randn((num_layers, n_padded, mhc_mult), dtype=torch.float32, device=device)
    layer_output_tensors = torch.randn((num_post, n_padded, hidden), dtype=torch.bfloat16, device=device)
    post_mix_tensors = torch.randn((num_post, n_padded, mhc_mult), dtype=torch.float32, device=device)
    comb_mix_tensors = torch.randn((num_post, n_padded, mhc_mult, mhc_mult), dtype=torch.float32, device=device)

    layer_input_tensors = torch.zeros((num_layers, n_padded, hidden), dtype=torch.bfloat16, device=device)
    residual_tensors = torch.zeros((num_post, n_padded, mhc_mult, hidden), dtype=torch.bfloat16, device=device)

    cpu_initial = initial_residual[:n].cpu()
    cpu_pre_mix = pre_mix_tensors[:, :n, :].cpu()
    cpu_layer_output = layer_output_tensors[:, :n, :].cpu()
    cpu_post_mix = post_mix_tensors[:, :n, :].cpu()
    cpu_comb_mix = comb_mix_tensors[:, :n, :].cpu()

    ref_layer_input, ref_residual = mhc_multilayer_recompute_reference(
        cpu_initial, cpu_pre_mix, cpu_layer_output, cpu_post_mix, cpu_comb_mix, num_layers, num_post
    )

    del cpu_initial, cpu_pre_mix, cpu_layer_output, cpu_post_mix, cpu_comb_mix

    fwd_func = _mhc_multilayer_recompute_kernel(
        mhc_mult=mhc_mult, hidden=hidden, num_layers=num_layers, num_post=num_post, h_blk=h_blk, n_thr=tbs
    )

    fwd_func(
        initial_residual[:n_padded],
        pre_mix_tensors,
        layer_output_tensors,
        post_mix_tensors,
        comb_mix_tensors,
        layer_input_tensors,
        residual_tensors,
    )

    torch.testing.assert_close(layer_input_tensors[:, :n, :].cpu(), ref_layer_input, atol=4e-2, rtol=1e-2)
    torch.testing.assert_close(residual_tensors[:, :n, :, :].cpu(), ref_residual, atol=4e-2, rtol=1e-2)

    print("Kernel Output Match!")

    del initial_residual, pre_mix_tensors, layer_output_tensors, post_mix_tensors, comb_mix_tensors
    del layer_input_tensors, residual_tensors, ref_layer_input, ref_residual, fwd_func
    torch.npu.empty_cache()


def test_all_correctness():
    for num_layers, num_post, hidden in _CORRECTNESS_CASES:
        _run_test_case(num_layers, num_post, hidden)
        torch.npu.empty_cache()


def test_fwd():
    _run_test_case(num_layers=6, num_post=5, hidden=1280)


if __name__ == "__main__":
    test_all_correctness()
