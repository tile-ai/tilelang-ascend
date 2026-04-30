import os

import torch
import tilelang
from tilelang import language as T

dtype = "bfloat16"
accum_dtype = "float32"


@tilelang.jit(target="npuir")
def get_engram_gate_bwd_kernel(
    num_tokens: int,
    hidden_size: int,
    scalar: float,
    num_persistent_blocks: int,
    clamp_value: float = 1e-6,
    hc_mult: int = 4,
):
    """NPU backward kernel, written close to the reference kernel style.

    NPU-specific correctness differences:
      - logical warp/lane are expanded as serial loops
      - x/k/w tile loop is outside lane loops
      - grad_w uses a separate UB tile reduction pass to avoid GM +=
    """
    assert hc_mult == 4

    warp_size = 32
    warps_per_head = 2
    num_warps = hc_mult * warps_per_head
    threads = warp_size * num_warps
    threads_per_head = warp_size * warps_per_head

    assert hidden_size % threads == 0

    elems_per_thread = hidden_size // threads

    go_vec_size = 8
    x_vec_size = 4

    def _choose_v_vec_size(elems_per_thread):
        for vec in [4, 2, 1]:
            if elems_per_thread % vec == 0:
                return vec
        return 1

    def _choose_go_blk_d(hidden_size, go_tile):
        result = go_tile
        for blk in range(go_tile, hidden_size // 2 + 1, go_tile):
            if hidden_size % blk == 0:
                result = blk
        return result

    def _choose_x_blk_d(hidden_size, x_tile, hc_mult, warps_per_head):
        result = x_tile
        x_limit = min(1024, hidden_size // 2)
        for blk in range(x_tile, x_limit + 1, x_tile):
            if hidden_size % blk == 0:
                result = blk
        return result

    v_vec_size = _choose_v_vec_size(elems_per_thread)
    go_blk_d = _choose_go_blk_d(hidden_size, threads_per_head * go_vec_size)
    x_blk_d = _choose_x_blk_d(
        hidden_size, threads_per_head * x_vec_size, hc_mult, warps_per_head
    )

    assert hidden_size % go_blk_d == 0
    assert hidden_size % x_blk_d == 0
    assert go_blk_d % (threads_per_head * go_vec_size) == 0
    assert x_blk_d % (threads_per_head * x_vec_size) == 0
    assert go_blk_d + x_blk_d <= hidden_size

    num_go_tiles = hidden_size // go_blk_d
    num_x_tiles = hidden_size // x_blk_d
    go_sub_blks = go_blk_d // (threads_per_head * go_vec_size)
    x_sub_blks = x_blk_d // (threads_per_head * x_vec_size)

    per_block_static = (num_tokens + num_persistent_blocks - 1) // num_persistent_blocks

    @T.prim_func
    def engram_gate_bwd_kernel(
        grad_out: T.Tensor([num_tokens, hc_mult, hidden_size], dtype),
        hidden_states: T.Tensor([num_tokens, hc_mult, hidden_size], dtype),
        k: T.Tensor([num_tokens, hc_mult, hidden_size], dtype),
        v: T.Tensor([num_tokens, hidden_size], dtype),
        weight_fused: T.Tensor([hc_mult, hidden_size], accum_dtype),
        dot_in: T.Tensor([num_tokens, hc_mult], accum_dtype),
        gate_in: T.Tensor([num_tokens, hc_mult], accum_dtype),
        rstd_x_in: T.Tensor([num_tokens, hc_mult], accum_dtype),
        rstd_k_in: T.Tensor([num_tokens, hc_mult], accum_dtype),
        grad_x: T.Tensor([num_tokens, hc_mult, hidden_size], dtype),
        grad_k: T.Tensor([num_tokens, hc_mult, hidden_size], dtype),
        grad_v: T.Tensor([num_tokens, hidden_size], dtype),
        grad_w_partial: T.Tensor(
            [num_persistent_blocks, hc_mult, hidden_size], accum_dtype
        ),
    ):
        with T.Kernel(num_persistent_blocks, is_npu=True) as (pid_p, _):
            go_local = T.alloc_shared((go_vec_size,), accum_dtype)
            v_local = T.alloc_shared((go_vec_size,), accum_dtype)

            go_v_local = T.alloc_shared((v_vec_size,), accum_dtype)
            go_x_local = T.alloc_shared((x_vec_size,), accum_dtype)

            x_local = T.alloc_shared((x_vec_size,), accum_dtype)
            k_local = T.alloc_shared((x_vec_size,), accum_dtype)
            w_fused_local = T.alloc_shared((x_vec_size,), accum_dtype)

            grad_v_partial = T.alloc_shared((v_vec_size,), accum_dtype)
            grad_w_tile = T.alloc_shared((x_blk_d,), accum_dtype)

            xw_val = T.alloc_shared((x_vec_size,), accum_dtype)
            kdotk_val = T.alloc_shared((x_vec_size,), accum_dtype)
            gx_val = T.alloc_shared((x_vec_size,), accum_dtype)
            gk_val = T.alloc_shared((x_vec_size,), accum_dtype)

            dldg_local = T.alloc_shared((1,), accum_dtype)
            dldg_r = T.alloc_shared((1,), accum_dtype)

            gate_local = T.alloc_shared((hc_mult,), accum_dtype)
            gate_local_hc = T.alloc_shared((1,), accum_dtype)
            rstd_x_local = T.alloc_shared((1,), accum_dtype)
            rstd_k_local = T.alloc_shared((1,), accum_dtype)
            dot_in_local = T.alloc_shared((1,), accum_dtype)
            dot_x_local = T.alloc_shared((1,), accum_dtype)
            dot_k_local = T.alloc_shared((1,), accum_dtype)
            dot_in_abs = T.alloc_shared((1,), accum_dtype)

            sqrt_in = T.alloc_shared((8,), accum_dtype)
            sqrt_out = T.alloc_shared((8,), accum_dtype)

            go_smem = T.alloc_shared((hc_mult, hidden_size), dtype)
            v_smem = T.alloc_shared((hidden_size,), dtype)

            x_smem = T.alloc_shared((2, hc_mult, x_blk_d), dtype)
            k_smem = T.alloc_shared((2, hc_mult, x_blk_d), dtype)
            w_smem = T.alloc_shared((2, hc_mult, x_blk_d), accum_dtype)

            dldg_smem = T.alloc_shared((hc_mult, warps_per_head), accum_dtype)
            dldg_r_cache = T.alloc_shared((per_block_static, hc_mult), accum_dtype)

            per_block = T.ceildiv(num_tokens, num_persistent_blocks)
            t_start = T.min(per_block * pid_p, num_tokens)
            t_end = T.min(per_block * (pid_p + 1), num_tokens)

            for i_off in T.serial(per_block_static):
                i_s = t_start + i_off

                if i_s < t_end:
                    T.copy(v[i_s, :], v_smem)
                    T.copy(grad_out[i_s, :, 0:go_blk_d], go_smem[:, 0:go_blk_d])
                    T.copy(gate_in[i_s, :], gate_local)

                    for head_id in T.serial(hc_mult):
                        for sub_warp_id in T.serial(warps_per_head):
                            T.clear(dldg_local)

                            for i_b_off in T.serial(num_go_tiles - 1):
                                i_b = i_b_off + 1
                                prev = i_b_off

                                T.copy(
                                    grad_out[
                                        i_s, :, i_b * go_blk_d : (i_b + 1) * go_blk_d
                                    ],
                                    go_smem[:, i_b * go_blk_d : (i_b + 1) * go_blk_d],
                                )

                                for i_sub in T.serial(go_sub_blks):
                                    go_base = (
                                        prev * go_blk_d
                                        + i_sub * threads_per_head * go_vec_size
                                        + sub_warp_id * warp_size * go_vec_size
                                    )

                                    for lane_id in T.serial(warp_size):
                                        for i_k in T.Parallel(go_vec_size):
                                            go_local[i_k] = go_smem[
                                                head_id,
                                                go_base + lane_id * go_vec_size + i_k,
                                            ]
                                            v_local[i_k] = v_smem[
                                                go_base + lane_id * go_vec_size + i_k
                                            ]

                                        for i_k in T.serial(go_vec_size):
                                            dldg_local[0] += (
                                                go_local[i_k] * v_local[i_k]
                                            )

                            for i_sub in T.serial(go_sub_blks):
                                go_base = (
                                    (num_go_tiles - 1) * go_blk_d
                                    + i_sub * threads_per_head * go_vec_size
                                    + sub_warp_id * warp_size * go_vec_size
                                )

                                for lane_id in T.serial(warp_size):
                                    for i_k in T.Parallel(go_vec_size):
                                        go_local[i_k] = go_smem[
                                            head_id,
                                            go_base + lane_id * go_vec_size + i_k,
                                        ]
                                        v_local[i_k] = v_smem[
                                            go_base + lane_id * go_vec_size + i_k
                                        ]

                                    for i_k in T.serial(go_vec_size):
                                        dldg_local[0] += go_local[i_k] * v_local[i_k]

                            dldg_smem[head_id, sub_warp_id] = dldg_local[0]

                    # === Pass 1b: grad_v ===
                    for i in T.serial(elems_per_thread // v_vec_size):
                        for tid in T.serial(threads):
                            T.clear(grad_v_partial)

                            for i_h in T.serial(hc_mult):
                                for i_k in T.Parallel(v_vec_size):
                                    go_v_local[i_k] = go_smem[
                                        i_h,
                                        i * threads * v_vec_size
                                        + tid * v_vec_size
                                        + i_k,
                                    ]

                                for i_k in T.Parallel(v_vec_size):
                                    grad_v_partial[i_k] += (
                                        go_v_local[i_k] * gate_local[i_h]
                                    )

                            for i_k in T.Parallel(v_vec_size):
                                grad_v[
                                    i_s,
                                    i * threads * v_vec_size + tid * v_vec_size + i_k,
                                ] = grad_v_partial[i_k]

                    # === Pass 2: grad_x / grad_k ===
                    for head_id in T.serial(hc_mult):
                        gate_local_hc[0] = gate_in[i_s, head_id]
                        rstd_x_local[0] = rstd_x_in[i_s, head_id]
                        rstd_k_local[0] = rstd_k_in[i_s, head_id]
                        dot_in_local[0] = dot_in[i_s, head_id]

                        if dot_in_local[0] < 0.0:
                            dot_in_abs[0] = -dot_in_local[0]
                        else:
                            dot_in_abs[0] = dot_in_local[0]

                        T.clear(sqrt_in)
                        T.clear(sqrt_out)

                        if dot_in_abs[0] < 1.0e-30:
                            sqrt_in[0] = 0.0
                        else:
                            sqrt_in[0] = (
                                scalar
                                * rstd_x_local[0]
                                * rstd_k_local[0]
                                / dot_in_abs[0]
                            )

                        T.vsqrt(sqrt_in, sqrt_out)

                        dldg_r[0] = 0.0

                        if (
                            dot_in_abs[0] * scalar * rstd_x_local[0] * rstd_k_local[0]
                            >= clamp_value
                        ):
                            dldg_r[0] = (
                                (dldg_smem[head_id, 0] + dldg_smem[head_id, 1])
                                * gate_local_hc[0]
                                * (1.0 - gate_local_hc[0])
                                * 0.5
                                * sqrt_out[0]
                            )

                        dldg_r_cache[i_off, head_id] = dldg_r[0]

                        dot_x_local[0] = (
                            dot_in_local[0]
                            * rstd_x_local[0]
                            * rstd_x_local[0]
                            / hidden_size
                        )
                        dot_k_local[0] = (
                            dot_in_local[0]
                            * rstd_k_local[0]
                            * rstd_k_local[0]
                            / hidden_size
                        )

                        T.copy(hidden_states[i_s, :, 0:x_blk_d], x_smem[0, :, :])
                        T.copy(k[i_s, :, 0:x_blk_d], k_smem[0, :, :])
                        T.copy(weight_fused[:, 0:x_blk_d], w_smem[0, :, :])

                        for tile_id in T.serial(num_x_tiles):
                            cur_phase = tile_id % 2

                            if tile_id + 1 < num_x_tiles:
                                next_tile = tile_id + 1
                                next_phase = next_tile % 2

                                T.copy(
                                    hidden_states[
                                        i_s,
                                        :,
                                        next_tile * x_blk_d : (next_tile + 1) * x_blk_d,
                                    ],
                                    x_smem[next_phase, :, :],
                                )
                                T.copy(
                                    k[
                                        i_s,
                                        :,
                                        next_tile * x_blk_d : (next_tile + 1) * x_blk_d,
                                    ],
                                    k_smem[next_phase, :, :],
                                )
                                T.copy(
                                    weight_fused[
                                        :,
                                        next_tile * x_blk_d : (next_tile + 1) * x_blk_d,
                                    ],
                                    w_smem[next_phase, :, :],
                                )

                            for sub_warp_id in T.serial(warps_per_head):
                                for lane_id in T.serial(warp_size):
                                    for i_sub in T.serial(x_sub_blks):
                                        sub_off = (
                                            i_sub * threads_per_head * x_vec_size
                                            + sub_warp_id * warp_size * x_vec_size
                                        )
                                        local_base = sub_off + lane_id * x_vec_size
                                        global_base = tile_id * x_blk_d + local_base

                                        for i_k in T.Parallel(x_vec_size):
                                            go_x_local[i_k] = go_smem[
                                                head_id, global_base + i_k
                                            ]
                                            x_local[i_k] = x_smem[
                                                cur_phase, head_id, local_base + i_k
                                            ]
                                            k_local[i_k] = k_smem[
                                                cur_phase, head_id, local_base + i_k
                                            ]
                                            w_fused_local[i_k] = w_smem[
                                                cur_phase, head_id, local_base + i_k
                                            ]

                                        for i_k in T.Parallel(x_vec_size):
                                            xw_val[i_k] = (
                                                x_local[i_k] * w_fused_local[i_k]
                                            )
                                            kdotk_val[i_k] = (
                                                k_local[i_k] * dot_k_local[0]
                                            )
                                            gx_val[i_k] = go_x_local[i_k] + dldg_r[
                                                0
                                            ] * (
                                                k_local[i_k] * w_fused_local[i_k]
                                                - x_local[i_k] * dot_x_local[0]
                                            )
                                            gk_val[i_k] = dldg_r[0] * (
                                                xw_val[i_k] - kdotk_val[i_k]
                                            )

                                            grad_x[i_s, head_id, global_base + i_k] = (
                                                gx_val[i_k]
                                            )
                                            grad_k[i_s, head_id, global_base + i_k] = (
                                                gk_val[i_k]
                                            )

            # === Pass 3: grad_w_partial ===
            for head_id in T.serial(hc_mult):
                for tile_id in T.serial(num_x_tiles):
                    T.clear(grad_w_tile)

                    for i_off in T.serial(per_block_static):
                        i_s = t_start + i_off

                        if i_s < t_end:
                            T.copy(
                                hidden_states[
                                    i_s, :, tile_id * x_blk_d : (tile_id + 1) * x_blk_d
                                ],
                                x_smem[0, :, :],
                            )
                            T.copy(
                                k[i_s, :, tile_id * x_blk_d : (tile_id + 1) * x_blk_d],
                                k_smem[0, :, :],
                            )

                            for sub_warp_id in T.serial(warps_per_head):
                                for lane_id in T.serial(warp_size):
                                    for i_sub in T.serial(x_sub_blks):
                                        sub_off = (
                                            i_sub * threads_per_head * x_vec_size
                                            + sub_warp_id * warp_size * x_vec_size
                                        )
                                        local_base = sub_off + lane_id * x_vec_size

                                        for i_k in T.Parallel(x_vec_size):
                                            x_local[i_k] = x_smem[
                                                0, head_id, local_base + i_k
                                            ]
                                            k_local[i_k] = k_smem[
                                                0, head_id, local_base + i_k
                                            ]

                                        for i_k in T.Parallel(x_vec_size):
                                            grad_w_tile[local_base + i_k] += (
                                                dldg_r_cache[i_off, head_id]
                                                * x_local[i_k]
                                                * k_local[i_k]
                                            )

                    for sub_warp_id in T.serial(warps_per_head):
                        for lane_id in T.serial(warp_size):
                            for i_sub in T.serial(x_sub_blks):
                                sub_off = (
                                    i_sub * threads_per_head * x_vec_size
                                    + sub_warp_id * warp_size * x_vec_size
                                )
                                local_base = sub_off + lane_id * x_vec_size
                                global_base = tile_id * x_blk_d + local_base

                                for i_k in T.Parallel(x_vec_size):
                                    grad_w_partial[
                                        pid_p, head_id, global_base + i_k
                                    ] = grad_w_tile[local_base + i_k]

    return engram_gate_bwd_kernel


def engram_gate_ref(
    hidden_states: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    weight_hidden: torch.Tensor,
    weight_embed: torch.Tensor,
    clamp_value: float,
    eps: float,
    save_for_backward: bool = False,
) -> (
    torch.Tensor
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
):
    """Pure PyTorch reference implementation of engram gate (vectorized, supports autograd).

    Computes: output = x + sigmoid(signed_sqrt(dot(RMSNorm(x, wh), RMSNorm(k, we)) * scalar)) * v

    Args:
        hidden_states: Input of shape (num_tokens, hc_mult, hidden_size), bfloat16.
        k: Key embeddings of shape (num_tokens, hc_mult, hidden_size), bfloat16.
        v: Value embeddings of shape (num_tokens, hidden_size), bfloat16.
        weight_hidden: RMSNorm weight for hidden states, shape (hc_mult, hidden_size), bfloat16.
        weight_embed: RMSNorm weight for key embeddings, shape (hc_mult, hidden_size), bfloat16.
        clamp_value: Clamp threshold for signed-sqrt gate activation.
        eps: Epsilon for RMSNorm numerical stability.
        save_for_backward: If True, also return (dot, gate_score, rstd_x, rstd_k).

    Returns:
        If save_for_backward is False: output tensor of shape (num_tokens, hc_mult, hidden_size), bfloat16.
        If save_for_backward is True: tuple of (output, dot, gate_score, rstd_x, rstd_k).
    """
    hidden_size = hidden_states.shape[-1]
    scalar = hidden_size**-0.5

    x = hidden_states.float()
    k_f = k.float()
    wh = weight_hidden.float().unsqueeze(0)
    we = weight_embed.float().unsqueeze(0)

    # RMSNorm
    rstd_x = torch.rsqrt(x.pow(2).mean(-1) + eps)
    rstd_k = torch.rsqrt(k_f.pow(2).mean(-1) + eps)

    # Dot -> sqrt-gate -> sigmoid
    # raw_dot is the unnormalized sum(x * wh * k * we), matching the kernel's dot_out
    raw_dot = torch.einsum("...d,...d->...", x * wh, k_f * we)
    dot = raw_dot * rstd_x * rstd_k * scalar
    signed_sqrt = dot.abs().clamp_min(clamp_value).sqrt() * dot.sign()
    gate_score = signed_sqrt.sigmoid()

    output = x + gate_score.unsqueeze(-1) * v.unsqueeze(-2)
    output = output.bfloat16()

    if save_for_backward:
        return output, raw_dot, gate_score, rstd_x, rstd_k
    return output


def engram_gate_bwd(
    grad_out: torch.Tensor,
    hidden_states: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    weight_fused: torch.Tensor,
    dot: torch.Tensor,
    gate_score: torch.Tensor,
    rstd_x: torch.Tensor,
    rstd_k: torch.Tensor,
    clamp_value: float,
    num_persistent_blocks: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Engram gate backward: computes grad_hidden_states, grad_k, grad_v, grad_w_partial.

    Args:
        grad_out: Gradient of output, shape (num_tokens, hc_mult, hidden_size), bfloat16.
        hidden_states: Original input from forward, shape (num_tokens, hc_mult, hidden_size), bfloat16.
        k: Original key embeddings from forward, shape (num_tokens, hc_mult, hidden_size), bfloat16.
        v: Original value embeddings from forward, shape (num_tokens, hidden_size), bfloat16.
        weight_fused: Fused RMSNorm weight (weight_hidden * weight_embed), shape (hc_mult, hidden_size), float32.
        dot: Saved scaled dot product from forward, shape (num_tokens, hc_mult), float32.
        gate_score: Saved gate scores from forward, shape (num_tokens, hc_mult), float32.
        rstd_x: Saved reciprocal std of x from forward, shape (num_tokens, hc_mult), float32.
        rstd_k: Saved reciprocal std of k from forward, shape (num_tokens, hc_mult), float32.
        clamp_value: Clamp threshold for signed-sqrt gate activation.

    Returns:
        tuple: (grad_hidden_states, grad_k, grad_v, grad_w_partial) where grad_w_partial
            has shape (num_persistent_blocks, hc_mult, hidden_size) and needs further reduction.
    """
    num_tokens, hc_mult, hidden_size = hidden_states.shape
    scalar = hidden_size**-0.5
    assert k.stride(-1) == 1
    assert v.stride(-1) == 1

    kernel = get_engram_gate_bwd_kernel(
        num_tokens, hidden_size, scalar, num_persistent_blocks, clamp_value, hc_mult
    )

    grad_hidden_states = torch.empty_like(hidden_states)
    grad_k = torch.empty_like(k)
    grad_v = torch.empty_like(v)
    grad_w_partial = torch.empty(
        (num_persistent_blocks, hc_mult, hidden_size),
        dtype=torch.float32,
        device=hidden_states.device,
    )

    kernel(
        grad_out,
        hidden_states,
        k,
        v,
        weight_fused,
        dot,
        gate_score,
        rstd_x,
        rstd_k,
        grad_hidden_states,
        grad_k,
        grad_v,
        grad_w_partial,
    )

    return grad_hidden_states, grad_k, grad_v, grad_w_partial


def calc_diff(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim if denominator != 0 else 0


def run_test():
    torch.manual_seed(0)

    device = "npu"

    num_tokens = 16
    hc_mult = 4
    hidden_size = 4096
    clamp_value = 1e-6
    num_persistent_blocks = 4
    eps = 1e-20

    x = torch.randn(
        (num_tokens, hc_mult, hidden_size),
        device=device,
        dtype=torch.bfloat16,
    )

    k = torch.randn(
        (num_tokens, hc_mult, hidden_size),
        device=device,
        dtype=torch.bfloat16,
    )

    v = torch.randn(
        (num_tokens, hidden_size),
        device=device,
        dtype=torch.bfloat16,
    )

    weight_hidden = torch.randn(
        (hc_mult, hidden_size),
        device=device,
        dtype=torch.bfloat16,
    )

    weight_embed = torch.randn(
        (hc_mult, hidden_size),
        device=device,
        dtype=torch.bfloat16,
    )

    weight_fused = (weight_hidden.float() * weight_embed.float()).contiguous()

    grad_out = torch.randn(
        (num_tokens, hc_mult, hidden_size),
        device=device,
        dtype=torch.bfloat16,
    )

    x_ref = x.clone().requires_grad_(True)
    k_ref = k.clone().requires_grad_(True)
    v_ref = v.clone().requires_grad_(True)
    # Cast to float32 so autograd produces fp32 gradients matching the kernel
    wh_ref = weight_hidden.float().requires_grad_(True)
    we_ref = weight_embed.float().requires_grad_(True)
    o_ref, dot_ref, gate_score_ref, rstd_x_ref, rstd_k_ref = engram_gate_ref(
        x_ref,
        k_ref,
        v_ref,
        wh_ref,
        we_ref,
        clamp_value,
        eps,
        save_for_backward=True,
    )
    o_ref.backward(grad_out)

    grad_x, grad_k, grad_v, grad_w_partial = engram_gate_bwd(
        grad_out,
        x,
        k,
        v,
        weight_fused,
        dot_ref,
        gate_score_ref,
        rstd_x_ref,
        rstd_k_ref,
        clamp_value,
        num_persistent_blocks,
    )

    grad_w_fused = grad_w_partial.sum(0)
    grad_wh = grad_w_fused * weight_embed.float()
    grad_we = grad_w_fused * weight_hidden.float()

    diff_x = calc_diff(grad_x, x_ref.grad)
    diff_k = calc_diff(grad_k, k_ref.grad)
    diff_v = calc_diff(grad_v, v_ref.grad)
    diff_wh = calc_diff(grad_wh, wh_ref.grad)
    diff_we = calc_diff(grad_we, we_ref.grad)

    diff_x = calc_diff(grad_x, x_ref.grad)
    assert diff_x < 1e-2, f"grad_x mismatch: {diff_x:.6e}"
    diff_k = calc_diff(grad_k, k_ref.grad)
    assert diff_k < 1e-2, f"grad_k mismatch: {diff_k:.6e}"
    diff_v = calc_diff(grad_v, v_ref.grad)
    assert diff_v < 1e-2, f"grad_v mismatch: {diff_v:.6e}"
    diff_wh = calc_diff(grad_wh, wh_ref.grad)
    assert diff_wh < 1e-2, f"grad_wh mismatch: {diff_wh:.6e}"
    diff_we = calc_diff(grad_we, we_ref.grad)
    assert diff_we < 1e-2, f"grad_we mismatch: {diff_we:.6e}"

    print("All check pass!")


if __name__ == "__main__":
    os.environ["TILELANG_ASCEND_MODE"] = "Dev"
    run_test()
