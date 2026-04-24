# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import os
import torch
import tilelang
import tilelang.language as T
import glob

from tilelang.carver.anneal.policy import AnnealTemplate, Annealparam

tilelang.cache.clear_cache()

block_n = 64
block_m = 64
block_k = 64
batch = 2
seq_len = 512
heads = 8
heads_kv = 2
dim = 64
is_causal = True
wl = -1
wr = -1
num_stages = 2
dtype = torch.float16

# PARAMS = [
#     ("batch, seq, heads, heads_kv, dim, is_causal, wl, wr, dtype", [
#         # ── Basic correctness ─────────────────────────────────────────────
#         pytest.param(2, 512,  8, 2,  64, True,  -1,  -1, torch.float16),
#         pytest.param(2, 512,  8, 2,  64, True,  128, -1, torch.float16),
#         pytest.param(2, 512,  8, 2,  64, False, -1,  -1, torch.float16),
#         pytest.param(2, 512,  8, 2,  64, False, 64,  64, torch.float16),
#         pytest.param(2, 128,  8, 1, 128, True,   1,  -1, torch.float16),
#         # ── dtype ─────────────────────────────────────────────────────────
#         pytest.param(2, 512,  8, 2,  64, True,  -1,  -1, torch.bfloat16),
#         pytest.param(2, 512,  8, 2,  64, False, 64,  64, torch.bfloat16),
#         # ── GQA ratio ─────────────────────────────────────────────────────
#         pytest.param(2, 512,  8, 8,  64, True,  -1,  -1, torch.float16),
#         pytest.param(2, 512, 16, 1,  64, True,  -1,  -1, torch.float16,),
#         # ── Non-power-of-2 sequence lengths ───────────────────────────────
#         pytest.param(2, 384,  8, 2,  64, True,  -1,  -1, torch.float16),
#         pytest.param(2, 768,  8, 2,  64, False, 256, -1, torch.float16),
#         # ── Large sequence ─────────────────────────────────────────────────
#         pytest.param(1, 2048, 8, 2,  64, True,  512, -1, torch.float16),
#         # ── Right window only ──────────────────────────────────────────────
#         pytest.param(2, 512,  8, 2,  64, False, -1,  64, torch.float16),
#         # ── wl=0 boundary: only current-position left context ─────────────
#         pytest.param(2, 256,  8, 2,  64, True,   0,  -1, torch.float16),
#     ]),
# ]


def get_config():
    annealparam = Annealparam(topk=40)
    anneal_template = AnnealTemplate(
        shape=[seq_len, seq_len, dim],
        annealparam=annealparam,
        use_template="FlashAttention",
    )

    hints = anneal_template.get_configs()

    configs = []
    for hint in hints:
        print(hint.kwargs)
        print(hint.value)
        configs.append(
            {
                "block_m": hint.kwargs[0],
                "block_n": hint.kwargs[1],
                "block_k": hint.kwargs[2],
            }
        )
    return configs


class GqaSlidingWindowFwdTest:
    def __init__(
        self,
        batch: int,
        seq: int,
        heads: int,
        heads_kv: int,
        dim: int,
        is_causal: bool,
        wl: int,
        wr: int,
        dtype: torch.dtype,
    ) -> None:
        self.batch = batch
        self.seq = seq
        self.heads = heads
        self.heads_kv = heads_kv
        self.dim = dim
        self.is_causal = is_causal
        self.wl = wl
        self.wr = wr
        self.dtype = dtype

    def gen_inputs(self):
        q = (
            torch.randn(
                (self.batch, self.seq, self.heads, self.dim), dtype=self.dtype
            ).npu()
            * 0.1
        )
        k = (
            torch.randn(
                (self.batch, self.seq, self.heads_kv, self.dim), dtype=self.dtype
            ).npu()
            * 0.1
        )
        v = (
            torch.randn(
                (self.batch, self.seq, self.heads_kv, self.dim), dtype=self.dtype
            ).npu()
            * 0.1
        )
        return {
            "q": q,
            "k": k,
            "v": v,
            "is_causal": self.is_causal,
            "wl": self.wl,
            "wr": self.wr,
        }

    def ref_program(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal, wl, wr
    ) -> torch.Tensor:
        """Pure-PyTorch reference: expand KV heads, compute masked softmax attention."""
        groups = self.heads // self.heads_kv
        scale = self.dim**-0.5

        # Expand KV to match Q heads: [B, S, H, D]
        k_exp = k.repeat_interleave(groups, dim=2).float()
        v_exp = v.repeat_interleave(groups, dim=2).float()

        # [B, H, S, S]
        scores = (
            torch.matmul(
                q.float().transpose(1, 2),
                k_exp.transpose(1, 2).transpose(-2, -1),
            )
            * scale
        )

        # Build attention mask
        S = self.seq
        q_pos = torch.arange(S, device=q.device).unsqueeze(1)  # [S, 1]
        k_pos = torch.arange(S, device=q.device).unsqueeze(0)  # [1, S]
        mask = torch.zeros(S, S, dtype=torch.bool, device=q.device)
        if self.is_causal:
            mask = mask | (q_pos < k_pos)
        if self.wl >= 0:
            mask = mask | (k_pos < q_pos - self.wl)
        if self.wr >= 0:
            mask = mask | (k_pos > q_pos + self.wr)

        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        output = torch.matmul(probs, v_exp.transpose(1, 2))  # [B, H, S, D]
        return output.transpose(1, 2).to(q.dtype)  # [B, S, H, D]


def generate_and_save_data(case_id, **kwargs):
    gqatest = GqaSlidingWindowFwdTest(**kwargs)
    inputs = gqatest.gen_inputs()
    outputs = gqatest.ref_program(**inputs)
    torch.save({"inputs": inputs, "outputs": outputs}, f"case_{case_id}.pt")


def generate_data():
    generate_and_save_data(
        case_id=0,
        batch=2,
        seq=512,
        heads=8,
        heads_kv=2,
        dim=64,
        is_causal=True,
        wl=-1,
        wr=-1,
        dtype=torch.float16,
    )

    generate_and_save_data(
        case_id=1,
        batch=2,
        seq=512,
        heads=8,
        heads_kv=2,
        dim=64,
        is_causal=True,
        wl=128,
        wr=-1,
        dtype=torch.float16,
    )

    generate_and_save_data(
        case_id=2,
        batch=2,
        seq=512,
        heads=8,
        heads_kv=2,
        dim=64,
        is_causal=False,
        wl=-1,
        wr=-1,
        dtype=torch.float16,
    )

    generate_and_save_data(
        case_id=3,
        batch=2,
        seq=512,
        heads=8,
        heads_kv=8,
        dim=64,
        is_causal=True,
        wl=-1,
        wr=-1,
        dtype=torch.float16,
    )

    generate_and_save_data(
        case_id=4,
        batch=2,
        seq=512,
        heads=16,
        heads_kv=1,
        dim=64,
        is_causal=False,
        wl=-1,
        wr=-1,
        dtype=torch.float16,
    )

    generate_and_save_data(
        case_id=5,
        batch=2,
        seq=512,
        heads=16,
        heads_kv=1,
        dim=64,
        is_causal=True,
        wl=-1,
        wr=-1,
        dtype=torch.float16,
    )

    generate_and_save_data(
        case_id=6,
        batch=2,
        seq=384,
        heads=8,
        heads_kv=2,
        dim=64,
        is_causal=True,
        wl=-1,
        wr=-1,
        dtype=torch.float16,
    )

    generate_and_save_data(
        case_id=7,
        batch=2,
        seq=768,
        heads=8,
        heads_kv=2,
        dim=64,
        is_causal=False,
        wl=256,
        wr=-1,
        dtype=torch.float16,
    )

    generate_and_save_data(
        case_id=8,
        batch=1,
        seq=2048,
        heads=8,
        heads_kv=2,
        dim=64,
        is_causal=True,
        wl=512,
        wr=-1,
        dtype=torch.float16,
    )

    generate_and_save_data(
        case_id=9,
        batch=2,
        seq=512,
        heads=8,
        heads_kv=2,
        dim=64,
        is_causal=True,
        wl=0,
        wr=-1,
        dtype=torch.float16,
    )


def ref_prog(q, k, v, w1, w2, w3):
    gqatest = GqaSlidingWindowFwdTest(
        batch, seq_len, heads, heads_kv, dim, is_causal, wl, wr, dtype
    )
    return gqatest.ref_program(q, k, v, is_causal, wl, wr)


def supply_prog(params, config):
    torch.manual_seed(0)
    num_blocks = (seq_len - 1) // config["block_n"] + 1

    return [
        torch.randn(batch, seq_len, heads, dim).half().npu(),
        torch.randn(batch, seq_len, heads_kv, dim).half().npu(),
        torch.randn(batch, seq_len, heads_kv, dim).half().npu(),
        torch.empty(batch, heads, seq_len, seq_len).half().npu(),
        torch.empty(batch, heads, seq_len, seq_len).half().npu(),
        torch.empty(batch, heads, seq_len, dim * num_blocks).half().npu(),
    ]


@tilelang.autotune(
    configs=get_config(),
    ref_prog=ref_prog,
    supply_prog=supply_prog,
    atol=1e-2,
    rtol=1e-2,
)
@tilelang.jit(target="npuir", out_idx=[3])
def _gqa_sw_fwd_kernel(
    batch: int,
    heads: int,
    heads_kv: int,
    seq_len: int,
    dim: int,
    is_causal: bool,
    window_size_left: int,  # -1 = unlimited
    window_size_right: int,  # -1 = unlimited
    block_m: int,
    block_n: int,
    block_k: int,
    num_stages: int,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):

    if heads % heads_kv != 0:
        raise ValueError("heads must be divisible by heads_kv")
    groups = heads // heads_kv
    has_window = window_size_left >= 0 or window_size_right >= 0

    scale = (1.0 / dim) ** 0.5
    shape2 = [batch, heads, seq_len, seq_len]

    q_shape = (batch, seq_len, heads, dim)
    kv_shape = (batch, seq_len, heads_kv, dim)

    block_m_half = (block_m + 1) // 2
    block_share = max(block_n, block_k)

    num_blocks = (seq_len - 1) // block_n + 1
    shape3 = [batch, heads, seq_len, dim * num_blocks]

    @T.prim_func
    def FlashAttnExp(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(kv_shape, dtype),
        V: T.Tensor(kv_shape, dtype),
        Output: T.Tensor(q_shape, dtype),
        workspace_1: T.Tensor(shape2, dtype),
        workspace_2: T.Tensor(shape2, dtype),
        workspace_3: T.Tensor(shape3, dtype),
    ):
        with T.Kernel(T.ceildiv(seq_len, block_m) * heads * batch, is_npu=True) as (
            cid,
            subid,
        ):
            bxc = cid // batch // heads
            by = cid // batch % heads
            bz = cid % batch

            tail_size_m = T.min(block_m, seq_len - bxc * block_m)
            acc_c_scale = scale
            offset_m = bxc * block_m

            # ── Loop range ──────────────────────────────────────────────
            k_end = T.ceildiv(seq_len, block_n)
            k_start = 0

            if is_causal:
                k_end = T.ceildiv(T.min(seq_len, (bxc + 1) * block_m), block_n)
            elif has_window and window_size_right >= 0:
                k_end = T.ceildiv(
                    T.min(seq_len, (bxc + 1) * block_m + window_size_right), block_n
                )
            else:
                k_end = T.ceildiv(seq_len, block_n)

            if has_window and window_size_left >= 0:
                k_start = T.max(0, bxc * block_m - window_size_left) // block_n
            else:
                k_start = 0

            loop_count = T.max(k_end - k_start, 0)

            with T.Scope("Cube"):
                l1_a = T.alloc_L1([block_m, block_share], dtype)
                l1_b = T.alloc_L1([block_n, block_k], dtype)

                l0_c = T.alloc_L0C([block_m, block_share], accum_dtype)

                for k_offset in T.Pipelined(loop_count, num_stages=num_stages):
                    k_idx = k_start + k_offset
                    offset_n = k_idx * block_n
                    tail_size_n = T.min(block_n, seq_len - offset_n)

                    for k in T.serial(T.ceildiv(dim, block_k)):
                        offset_k = k * block_k
                        tail_size_k = T.min(block_k, dim - offset_k)

                        T.copy(
                            Q[
                                bz,
                                offset_m : offset_m + tail_size_m,
                                by,
                                offset_k : offset_k + tail_size_k,
                            ],
                            l1_a[:tail_size_m, :tail_size_k],
                        )
                        T.copy(
                            K[
                                bz,
                                offset_n : offset_n + tail_size_n,
                                by // groups,
                                offset_k : offset_k + tail_size_k,
                            ],
                            l1_b[:tail_size_n, :tail_size_k],
                        )
                        T.gemm(
                            l1_a,
                            l1_b,
                            l0_c,
                            initC=(k == 0),
                            b_transpose=True,
                            size=[tail_size_m, tail_size_k, tail_size_n],
                        )

                    with T.rs("PIPE_FIX"):
                        T.copy(
                            l0_c[:tail_size_m, :tail_size_n],
                            workspace_1[
                                bz : bz + 1,
                                by : by + 1,
                                offset_m : offset_m + tail_size_m,
                                offset_n : offset_n + tail_size_n,
                            ],
                        )
                        T.sync_block_set(k_offset)

                for k_offset in T.Pipelined(loop_count, num_stages=num_stages):
                    k_idx = k_start + k_offset
                    offset_n = k_idx * block_n
                    tail_size_n = T.min(block_n, seq_len - offset_n)
                    with T.rs("PIPE_MTE2"):
                        T.sync_block_wait(k_offset)
                        T.copy(
                            workspace_2[
                                bz : bz + 1,
                                by : by + 1,
                                offset_m : offset_m + tail_size_m,
                                offset_n : offset_n + tail_size_n,
                            ],
                            l1_a[:tail_size_m, :tail_size_n],
                        )

                    for k in T.serial(T.ceildiv(dim, block_k)):
                        offset_k = k * block_k
                        tail_size_k = T.min(block_k, dim - offset_k)
                        offset_w3 = k_offset * dim + offset_k
                        T.copy(
                            V[
                                bz,
                                offset_n : offset_n + tail_size_n,
                                by // groups,
                                offset_k : offset_k + tail_size_k,
                            ],
                            l1_b[:tail_size_n, :tail_size_k],
                        )
                        T.gemm(
                            l1_a,
                            l1_b,
                            l0_c,
                            initC=True,
                            size=[tail_size_m, tail_size_n, tail_size_k],
                        )
                        T.copy(
                            l0_c[:tail_size_m, :tail_size_k],
                            workspace_3[
                                bz : bz + 1,
                                by : by + 1,
                                offset_m : offset_m + tail_size_m,
                                offset_w3 : offset_w3 + tail_size_k,
                            ],
                        )

                    with T.rs("PIPE_FIX"):
                        T.sync_block_set(k_offset)

            with T.Scope("Vector"):
                logsum = T.alloc_ub([block_m_half, 1], accum_dtype)
                scores_max = T.alloc_ub([block_m_half, 1], accum_dtype)
                scores_max_prev = T.alloc_ub([block_m_half, 1], accum_dtype)
                scores_scale = T.alloc_ub([block_m_half, 1], accum_dtype)

                scores_sum = T.alloc_ub([block_m_half, 1], accum_dtype)
                scales = T.alloc_ub(
                    [T.ceildiv(seq_len, block_n) * block_m_half, 1], accum_dtype
                )

                cross_kernel_f16_dim = T.alloc_ub([block_m_half, dim], dtype)
                cross_kernel_f16_N = T.alloc_ub([block_m_half, block_n], dtype)
                cross_kernel_f32_dim = T.alloc_ub([block_m_half, dim], accum_dtype)
                cross_kernel_f32_N = T.alloc_ub([block_m_half, block_n], accum_dtype)
                acc_o = T.alloc_ub([block_m_half, dim], accum_dtype)

                value_zero = 0
                value_min = -T.infinity(accum_dtype)
                finite_floor = T.cast(-1e38, accum_dtype)
                T.vbrc(value_zero, logsum)
                T.vbrc(value_zero, acc_o)
                T.vbrc(value_zero, scores_scale)
                T.vbrc(value_zero, scales)
                T.vbrc(value_min, scores_max)

                real_m = (tail_size_m + 1) // 2
                bx = bxc * block_m + subid * real_m
                real_m = real_m - (tail_size_m % 2) * subid

                for k_offset in T.Pipelined(loop_count, num_stages=num_stages):
                    k_idx = k_start + k_offset
                    offset_n = k_idx * block_n

                    tail_size_n = T.min(block_n, seq_len - offset_n)
                    T.copy(scores_max, scores_max_prev)
                    with T.rs("PIPE_MTE2"):
                        T.sync_block_wait(k_offset)
                        T.copy(
                            workspace_1[
                                bz : bz + 1,
                                by : by + 1,
                                bx : bx + real_m,
                                offset_n : offset_n + tail_size_n,
                            ],
                            cross_kernel_f16_N[0:real_m, 0:tail_size_n],
                        )
                        T.vcast(
                            cross_kernel_f16_N, cross_kernel_f32_N, round_mode="rint"
                        )

                    if is_causal and has_window:
                        for i, j in T.Parallel(real_m, tail_size_n):
                            causal_mask = bx + i < offset_n + j
                            left_mask = (window_size_left >= 0) and (
                                offset_n + j < bx + i - window_size_left
                            )
                            if causal_mask or left_mask:
                                cross_kernel_f32_N[i, j] = finite_floor
                    elif is_causal:
                        for i, j in T.Parallel(real_m, tail_size_n):
                            if bx + i < offset_n + j:
                                cross_kernel_f32_N[i, j] = finite_floor
                    elif has_window:
                        for i, j in T.Parallel(real_m, tail_size_n):
                            left_mask = (window_size_left >= 0) and (
                                offset_n + j < bx + i - window_size_left
                            )
                            right_mask = (window_size_right >= 0) and (
                                offset_n + j > bx + i + window_size_right
                            )
                            if left_mask or right_mask:
                                cross_kernel_f32_N[i, j] = finite_floor

                    T.vmul(cross_kernel_f32_N, acc_c_scale, cross_kernel_f32_N)
                    T.reduce(
                        cross_kernel_f32_N,
                        scores_max,
                        dims=[1],
                        reduce_mode="max",
                        size=[real_m, tail_size_n],
                    )
                    if k_offset != 0:
                        T.vmax(scores_max_prev, scores_max, scores_max)
                        T.vsub(scores_max_prev, scores_max, scores_scale)
                        T.vexp(scores_scale, scores_scale)

                        T.copy(
                            scores_scale,
                            scales[
                                k_offset * block_m_half : k_offset * block_m_half
                                + block_m_half,
                                :,
                            ],
                        )

                    T.vsub(cross_kernel_f32_N, scores_max, cross_kernel_f32_N)
                    T.vexp(cross_kernel_f32_N, cross_kernel_f32_N)
                    T.vcast(cross_kernel_f32_N, cross_kernel_f16_N, round_mode="rint")

                    with T.rs("PIPE_MTE3"):
                        T.copy(
                            cross_kernel_f16_N[0:real_m, 0:tail_size_n],
                            workspace_2[
                                bz : bz + 1,
                                by : by + 1,
                                bx : bx + real_m,
                                offset_n : offset_n + tail_size_n,
                            ],
                        )
                        T.sync_block_set(k_offset)

                    T.reduce(
                        cross_kernel_f32_N,
                        scores_sum,
                        dims=[1],
                        reduce_mode="sum",
                        size=[real_m, tail_size_n],
                    )
                    T.vmul(logsum, scores_scale, logsum)
                    T.vadd(logsum, scores_sum, logsum)

                for k_offset in T.Pipelined(loop_count, num_stages=num_stages):
                    k_idx = k_start + k_offset
                    offset_n = k_idx * block_n

                    with T.rs("PIPE_MTE2"):
                        T.sync_block_wait(k_offset)
                        T.copy(
                            workspace_3[
                                bz : bz + 1,
                                by : by + 1,
                                bx : bx + real_m,
                                k_offset * dim : k_offset * dim + dim,
                            ],
                            cross_kernel_f16_dim[0:real_m, 0:dim],
                        )
                    T.vcast(
                        cross_kernel_f16_dim, cross_kernel_f32_dim, round_mode="rint"
                    )
                    if k_offset != 0:
                        T.copy(
                            scales[
                                k_offset * block_m_half : k_offset * block_m_half
                                + block_m_half,
                                0:1,
                            ],
                            scores_scale,
                        )
                    T.vmul(acc_o, scores_scale, acc_o)
                    T.vadd(acc_o, cross_kernel_f32_dim, acc_o)

                T.vdiv(acc_o, logsum, acc_o)
                T.vcast(acc_o, cross_kernel_f16_dim, round_mode="rint")
                T.copy(
                    cross_kernel_f16_dim[0:real_m, 0:dim],
                    Output[bz, bx : bx + real_m, by, 0:dim],
                )

    return FlashAttnExp


def _gqa_sw_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal, wl, wr):
    batch, seq_len, heads, dim = q.size()
    _, _, heads_kv, _ = k.size()

    kernel = _gqa_sw_fwd_kernel(
        batch,
        heads,
        heads_kv,
        seq_len,
        dim,
        is_causal,
        wl,
        wr,
        block_m,
        block_n,
        block_k,
        num_stages,
    )

    num_blocks = (seq_len - 1) // block_n + 1
    shape2 = [batch, heads, seq_len, seq_len]
    shape3 = [batch, heads, seq_len, dim * num_blocks]

    w_1 = torch.empty(shape2, dtype=torch.float16).npu()
    w_2 = torch.empty(shape2, dtype=torch.float16).npu()
    w_3 = torch.empty(shape3, dtype=torch.float16).npu()
    output = kernel(q, k, v, w_1, w_2, w_3)

    return output


def run_test(verify_acc=True):
    """
    Traverse all case_*.pt files in the current directory for testing.
    The verify_acc parameter determines whether to perform accuracy check.
    """
    pattern = os.path.join(os.getcwd(), "case_*.pt")
    file_paths = sorted(glob.glob(pattern))
    if not file_paths:
        raise FileNotFoundError("No case_*.pt files found in current directory.")

    errors = []
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        try:
            data = torch.load(file_path, map_location=torch.device("npu"))
            output = _gqa_sw_fwd(**data["inputs"])
            print(data["outputs"])
            print(output)

            if verify_acc:
                torch.testing.assert_close(
                    data["outputs"], output, rtol=1e-2, atol=1e-2
                )
                print(f"{filename}: \033[92mPassed.\033[0m")
            else:
                assert output is not None
                print(f"{filename}: \033[92mFinished.\033[0m")
        except Exception as e:
            errors.append(f"{filename}: {str(e)}")
            print(f"{filename}: \033[91mFailed: {e}\033[0m")

    if errors:
        error_msg = "\n".join(errors)
        raise AssertionError(f"Some cases failed:\n{error_msg}")
    else:
        print("\033[92mAll checks passed.\033[0m")


if __name__ == "__main__":
    # Specifies which NPU device to use
    os.environ["TILELANG_ASCEND_MODE"] = "Expert"

    kernel = _gqa_sw_fwd_kernel(
        batch=batch,
        heads=heads,
        heads_kv=heads_kv,
        seq_len=seq_len,
        dim=dim,
        is_causal=is_causal,
        window_size_left=wl,
        window_size_right=wr,
        num_stages=num_stages,
    )

    print("Best Config:", kernel.get_tuner_result())
    print("Test Passed!")
    # Generate data and run tests
    # generate_data()
    # run_test(verify_acc=True)
