import functools

import tilelang
import tilelang.language as tl
import torch

tilelang.cache.clear_cache()

pass_configs = {tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True}


def prepare_chunk_indices(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    indices = []
    cu_seqlens_list = cu_seqlens.cpu().tolist()
    for seq_idx in range(len(cu_seqlens_list) - 1):
        seq_len = int(cu_seqlens_list[seq_idx + 1] - cu_seqlens_list[seq_idx])
        chunk_num = (seq_len + chunk_size - 1) // chunk_size
        for chunk_idx in range(chunk_num):
            indices.append([seq_idx, chunk_idx])
    return torch.tensor(indices, dtype=torch.int32, device=cu_seqlens.device)


def input_guard(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        contiguous_args = tuple(arg.contiguous() if isinstance(arg, torch.Tensor) else arg for arg in args)
        contiguous_kwargs = {key: value.contiguous() if isinstance(value, torch.Tensor) else value for key, value in kwargs.items()}
        return fn(*contiguous_args, **contiguous_kwargs)

    return wrapper


def _sequence_bounds(cu_seqlens: torch.Tensor) -> list[tuple[int, int]]:
    cu_seqlens_list = cu_seqlens.cpu().tolist()
    return [(int(cu_seqlens_list[i]), int(cu_seqlens_list[i + 1])) for i in range(len(cu_seqlens_list) - 1)]


def _canonicalize_chunk_indices(cu_seqlens: torch.Tensor, chunk_size: int, chunk_indices: torch.Tensor | None) -> torch.Tensor:
    expected = prepare_chunk_indices(cu_seqlens, chunk_size)
    if chunk_indices is None:
        return expected
    if tuple(chunk_indices.shape) != tuple(expected.shape):
        raise ValueError("chunk_indices must match the canonical shape derived from cu_seqlens")
    if not torch.equal(chunk_indices.to(torch.int64).cpu(), expected.to(torch.int64).cpu()):
        raise ValueError("Only canonical chunk_indices derived from cu_seqlens are supported")
    return chunk_indices


def _slice_sequence(tensor: torch.Tensor, bos: int, eos: int, head_first: bool) -> torch.Tensor:
    if head_first:
        return tensor[:, :, bos:eos, ...]
    return tensor[:, bos:eos, ...]


def _write_sequence(dst: torch.Tensor, src: torch.Tensor, bos: int, eos: int, head_first: bool) -> None:
    if head_first:
        dst[:, :, bos:eos, ...] = src
    else:
        dst[:, bos:eos, ...] = src


def _run_varlen_dense(
    dense_fn,
    s: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor,
    head_first: bool,
    output_dtype: torch.dtype,
    **kwargs,
) -> torch.Tensor:
    out = torch.empty_like(s, dtype=output_dtype or s.dtype)
    for bos, eos in _sequence_bounds(cu_seqlens):
        if bos == eos:
            continue
        seq = _slice_sequence(s, bos, eos, head_first).contiguous()
        seq_out = dense_fn(seq, head_first=head_first, output_dtype=output_dtype, **kwargs)
        _write_sequence(out, seq_out, bos, eos, head_first)
    return out


def _pad_scalar_head_first(s: torch.Tensor, block_t: int) -> tuple[torch.Tensor, int, int]:
    B, H, T = s.shape
    padded_h = H if H % 2 == 0 else H + 1
    padded_t = ((T + block_t - 1) // block_t) * block_t
    if padded_h == H and padded_t == T:
        return s, H, T
    s_padded = torch.zeros((B, padded_h, padded_t), dtype=s.dtype, device=s.device)
    s_padded[:, :H, :T] = s
    return s_padded, H, T


def _pad_vector_head_first(s: torch.Tensor, block_t: int, block_s: int) -> tuple[torch.Tensor, int, int, int]:
    B, H, T, S = s.shape
    padded_h = H if H % 2 == 0 else H + 1
    padded_t = ((T + block_t - 1) // block_t) * block_t
    padded_s = ((S + block_s - 1) // block_s) * block_s
    if padded_h == H and padded_t == T and padded_s == S:
        return s, H, T, S
    s_padded = torch.zeros((B, padded_h, padded_t, padded_s), dtype=s.dtype, device=s.device)
    s_padded[:, :H, :T, :S] = s
    return s_padded, H, T, S


@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def chunk_local_cumsum_scalar_kernel(B, H, SEQ_LEN, BT, reverse=False, head_first=True, dtype="float"):
    chunk_num = tl.ceildiv(SEQ_LEN, BT)
    VEC_NUM = 2
    h_block_num = H // VEC_NUM
    shape = (B, H, SEQ_LEN)

    @tl.prim_func
    def main(
        s: tl.Tensor(shape, dtype),
        o: tl.Tensor(shape, dtype),
    ):
        with tl.Kernel(chunk_num * B * h_block_num, is_npu=True) as (cid, vid):
            i_t = cid % chunk_num
            i_bh = cid // chunk_num
            i_b = i_bh // h_block_num
            i_h = (i_bh % h_block_num) * VEC_NUM + vid

            b_s = tl.alloc_ub([BT], dtype)
            b_o = tl.alloc_ub([BT], dtype)
            total_buf = tl.alloc_ub([1], dtype)

            with tl.Scope("V"):
                tl.tile.fill(b_o, 0.0)
                tl.copy(s[i_b, i_h, i_t * BT], b_s)

                for i in range(BT):
                    if i > 0:
                        b_o[i] = b_o[i - 1]
                    b_o[i] = b_o[i] + b_s[i]

                if reverse:
                    tl.tile.fill(total_buf, 0.0)
                    for i in range(BT):
                        total_buf[0] = total_buf[0] + b_s[i]
                    for i in range(BT):
                        b_o[i] = total_buf[0] - b_o[i] + b_s[i]

                tl.copy(b_o, o[i_b, i_h, i_t * BT])

    return main


@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def chunk_global_cumsum_scalar_kernel(B, H, SEQ_LEN, BT, reverse=False, head_first=True, dtype="float"):
    chunk_num = tl.ceildiv(SEQ_LEN, BT)
    VEC_NUM = 2
    h_block_num = H // VEC_NUM
    shape = (B, H, SEQ_LEN)

    @tl.prim_func
    def main(
        s: tl.Tensor(shape, dtype),
        o: tl.Tensor(shape, dtype),
    ):
        with tl.Kernel(B * h_block_num, is_npu=True) as (cid, vid):
            i_b = cid // h_block_num
            i_h = (cid % h_block_num) * VEC_NUM + vid

            b_s = tl.alloc_ub([BT], dtype)
            b_o = tl.alloc_ub([BT], dtype)
            carry = tl.alloc_ub([1], dtype)
            b_ss_buf = tl.alloc_ub([1], dtype)

            with tl.Scope("V"):
                tl.tile.fill(carry, 0.0)

                for k in range(chunk_num):
                    i_t = chunk_num - 1 - k if reverse else k

                    tl.tile.fill(b_o, 0.0)
                    tl.tile.fill(b_ss_buf, 0.0)
                    tl.copy(s[i_b, i_h, i_t * BT], b_s)

                    for i in range(BT):
                        if i > 0:
                            b_o[i] = b_o[i - 1]
                        b_o[i] = b_o[i] + b_s[i]

                    for i in range(BT):
                        b_ss_buf[0] = b_ss_buf[0] + b_s[i]

                    if reverse:
                        for i in range(BT):
                            b_o[i] = b_ss_buf[0] - b_o[i] + b_s[i]

                    for i in range(BT):
                        b_o[i] = b_o[i] + carry[0]

                    tl.copy(b_o, o[i_b, i_h, i_t * BT])
                    carry[0] = carry[0] + b_ss_buf[0]

    return main


@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def chunk_local_cumsum_vector_kernel(B, H, SEQ_LEN, S_DIM, BT, BS, reverse=False, head_first=True, dtype="float"):
    chunk_num = tl.ceildiv(SEQ_LEN, BT)
    s_block_num = tl.ceildiv(S_DIM, BS)
    VEC_NUM = 2
    h_block_num = H // VEC_NUM
    shape = (B, H, SEQ_LEN, S_DIM)

    @tl.prim_func
    def main(
        s: tl.Tensor(shape, dtype),
        o: tl.Tensor(shape, dtype),
    ):
        with tl.Kernel(s_block_num * chunk_num * B * h_block_num, is_npu=True) as (cid, vid):
            i_s = cid % s_block_num
            i_t = (cid // s_block_num) % chunk_num
            i_bh = cid // (s_block_num * chunk_num)
            i_b = i_bh // h_block_num
            i_h = (i_bh % h_block_num) * VEC_NUM + vid

            b_s = tl.alloc_ub([BT, BS], dtype)
            b_o = tl.alloc_ub([BT, BS], dtype)
            total_buf = tl.alloc_ub([BS], dtype)

            with tl.Scope("V"):
                tl.tile.fill(b_o, 0.0)
                tl.copy(s[i_b, i_h, i_t * BT, i_s * BS], b_s)

                for i in range(BT):
                    for j in range(BS):
                        if i > 0:
                            b_o[i, j] = b_o[i - 1, j]
                        b_o[i, j] = b_o[i, j] + b_s[i, j]

                if reverse:
                    tl.tile.fill(total_buf, 0.0)
                    for i in range(BT):
                        for j in range(BS):
                            total_buf[j] = total_buf[j] + b_s[i, j]
                    for i in range(BT):
                        for j in range(BS):
                            b_o[i, j] = total_buf[j] - b_o[i, j] + b_s[i, j]

                tl.copy(b_o, o[i_b, i_h, i_t * BT, i_s * BS])

    return main


@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def chunk_global_cumsum_vector_kernel(B, H, SEQ_LEN, S_DIM, BT, BS, reverse=False, head_first=True, dtype="float"):
    chunk_num = tl.ceildiv(SEQ_LEN, BT)
    s_block_num = tl.ceildiv(S_DIM, BS)
    VEC_NUM = 2
    h_block_num = H // VEC_NUM
    shape = (B, H, SEQ_LEN, S_DIM)

    @tl.prim_func
    def main(
        s: tl.Tensor(shape, dtype),
        o: tl.Tensor(shape, dtype),
    ):
        with tl.Kernel(s_block_num * B * h_block_num, is_npu=True) as (cid, vid):
            i_s = cid % s_block_num
            i_bh = cid // s_block_num
            i_b = i_bh // h_block_num
            i_h = (i_bh % h_block_num) * VEC_NUM + vid

            b_s = tl.alloc_ub([BT, BS], dtype)
            b_o = tl.alloc_ub([BT, BS], dtype)
            carry = tl.alloc_ub([BS], dtype)
            b_ss_buf = tl.alloc_ub([BS], dtype)

            with tl.Scope("V"):
                tl.tile.fill(carry, 0.0)

                for k in range(chunk_num):
                    i_t = chunk_num - 1 - k if reverse else k

                    tl.tile.fill(b_o, 0.0)
                    tl.tile.fill(b_ss_buf, 0.0)
                    tl.copy(s[i_b, i_h, i_t * BT, i_s * BS], b_s)

                    for i in range(BT):
                        for j in range(BS):
                            if i > 0:
                                b_o[i, j] = b_o[i - 1, j]
                            b_o[i, j] = b_o[i, j] + b_s[i, j]

                    for i in range(BT):
                        for j in range(BS):
                            b_ss_buf[j] = b_ss_buf[j] + b_s[i, j]

                    if reverse:
                        for i in range(BT):
                            for j in range(BS):
                                b_o[i, j] = b_ss_buf[j] - b_o[i, j] + b_s[i, j]

                    for i in range(BT):
                        for j in range(BS):
                            b_o[i, j] = b_o[i, j] + carry[j]

                    tl.copy(b_o, o[i_b, i_h, i_t * BT, i_s * BS])

                    for j in range(BS):
                        carry[j] = carry[j] + b_ss_buf[j]

    return main


def _chunk_local_cumsum_scalar_dense(s, chunk_size, reverse=False, scale=None, head_first=False, output_dtype=torch.float):
    if head_first:
        B, H, SEQ_LEN = s.shape
    else:
        B, SEQ_LEN, H = s.shape

    assert chunk_size == 2 ** (chunk_size.bit_length() - 1), "chunk_size must be a power of 2"

    if (not head_first) or (H % 2 != 0) or (SEQ_LEN % chunk_size != 0):
        return ref_chunk_local_cumsum_scalar(
            s,
            chunk_size,
            reverse=reverse,
            scale=scale,
            head_first=head_first,
        ).to(output_dtype)

    kernel = chunk_local_cumsum_scalar_kernel(
        B,
        H,
        SEQ_LEN,
        chunk_size,
        reverse=reverse,
        head_first=True,
        dtype="float",
    )
    o = kernel(s)

    if scale is not None:
        o = o * scale

    return o.to(output_dtype)


def _chunk_global_cumsum_scalar_dense(s, reverse=False, scale=None, head_first=False, output_dtype=torch.float):
    BT = 64
    if head_first:
        B, H, SEQ_LEN = s.shape
    else:
        B, SEQ_LEN, H = s.shape

    if (not head_first) or (H % 2 != 0) or (SEQ_LEN % BT != 0):
        return ref_chunk_global_cumsum_scalar(
            s,
            reverse=reverse,
            scale=scale,
            head_first=head_first,
        ).to(output_dtype)

    kernel = chunk_global_cumsum_scalar_kernel(
        B,
        H,
        SEQ_LEN,
        BT,
        reverse=reverse,
        head_first=True,
        dtype="float",
    )
    o = kernel(s)

    if scale is not None:
        o = o * scale

    return o.to(output_dtype)


def _chunk_local_cumsum_vector_dense(s, chunk_size, reverse=False, scale=None, head_first=False, output_dtype=torch.float):
    if head_first:
        B, H, SEQ_LEN, S_DIM = s.shape
    else:
        B, SEQ_LEN, H, S_DIM = s.shape

    assert chunk_size == 2 ** (chunk_size.bit_length() - 1), "chunk_size must be a power of 2"
    BS = min(32, 2 ** (S_DIM.bit_length() - 1)) if S_DIM > 0 else 1

    if (not head_first) or (H % 2 != 0) or (SEQ_LEN % chunk_size != 0) or (S_DIM % BS != 0):
        return ref_chunk_local_cumsum_vector(
            s,
            chunk_size,
            reverse=reverse,
            scale=scale,
            head_first=head_first,
        ).to(output_dtype)

    kernel = chunk_local_cumsum_vector_kernel(
        B,
        H,
        SEQ_LEN,
        S_DIM,
        chunk_size,
        BS,
        reverse=reverse,
        head_first=True,
        dtype="float",
    )
    o = kernel(s)

    if scale is not None:
        o = o * scale

    return o.to(output_dtype)


def _chunk_global_cumsum_vector_dense(s, reverse=False, scale=None, head_first=False, output_dtype=torch.float):
    BT = 64
    if head_first:
        B, H, SEQ_LEN, S_DIM = s.shape
    else:
        B, SEQ_LEN, H, S_DIM = s.shape

    BS = min(32, 2 ** (S_DIM.bit_length() - 1)) if S_DIM > 0 else 1

    if (not head_first) or (H % 2 != 0) or (SEQ_LEN % BT != 0) or (S_DIM % BS != 0):
        return ref_chunk_global_cumsum_vector(
            s,
            reverse=reverse,
            scale=scale,
            head_first=head_first,
        ).to(output_dtype)

    kernel = chunk_global_cumsum_vector_kernel(
        B,
        H,
        SEQ_LEN,
        S_DIM,
        BT,
        BS,
        reverse=reverse,
        head_first=True,
        dtype="float",
    )
    o = kernel(s)

    if scale is not None:
        o = o * scale

    return o.to(output_dtype)


@input_guard
def chunk_local_cumsum_scalar(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
    scale: float = None,
    cu_seqlens: torch.Tensor = None,
    head_first: bool = False,
    output_dtype: torch.dtype = torch.float,
    chunk_indices: torch.LongTensor = None,
) -> torch.Tensor:
    if cu_seqlens is None:
        return _chunk_local_cumsum_scalar_dense(
            g,
            chunk_size,
            reverse=reverse,
            scale=scale,
            head_first=head_first,
            output_dtype=output_dtype,
        )
    assert g.shape[0] == 1, "Only batch size 1 is supported when cu_seqlens are provided"
    _canonicalize_chunk_indices(cu_seqlens, chunk_size, chunk_indices)
    return _run_varlen_dense(
        _chunk_local_cumsum_scalar_dense,
        g,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        reverse=reverse,
        scale=scale,
        head_first=head_first,
        output_dtype=output_dtype,
    )


@input_guard
def chunk_global_cumsum_scalar(
    s: torch.Tensor,
    reverse: bool = False,
    cu_seqlens: torch.Tensor = None,
    scale: float = None,
    head_first: bool = False,
    output_dtype: torch.dtype = torch.float,
) -> torch.Tensor:
    if cu_seqlens is None:
        return _chunk_global_cumsum_scalar_dense(
            s,
            reverse=reverse,
            scale=scale,
            head_first=head_first,
            output_dtype=output_dtype,
        )
    assert s.shape[0] == 1, "Only batch size 1 is supported when cu_seqlens are provided"
    return _run_varlen_dense(
        _chunk_global_cumsum_scalar_dense,
        s,
        cu_seqlens=cu_seqlens,
        reverse=reverse,
        scale=scale,
        head_first=head_first,
        output_dtype=output_dtype,
    )


@input_guard
def chunk_local_cumsum_vector(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
    scale: float = None,
    cu_seqlens: torch.Tensor = None,
    head_first: bool = False,
    output_dtype: torch.dtype = torch.float,
    chunk_indices: torch.LongTensor = None,
) -> torch.Tensor:
    if cu_seqlens is None:
        return _chunk_local_cumsum_vector_dense(
            g,
            chunk_size,
            reverse=reverse,
            scale=scale,
            head_first=head_first,
            output_dtype=output_dtype,
        )
    assert g.shape[0] == 1, "Only batch size 1 is supported when cu_seqlens are provided"
    _canonicalize_chunk_indices(cu_seqlens, chunk_size, chunk_indices)
    return _run_varlen_dense(
        _chunk_local_cumsum_vector_dense,
        g,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        reverse=reverse,
        scale=scale,
        head_first=head_first,
        output_dtype=output_dtype,
    )


@input_guard
def chunk_global_cumsum_vector(
    s: torch.Tensor,
    reverse: bool = False,
    cu_seqlens: torch.Tensor = None,
    scale: float = None,
    head_first: bool = False,
    output_dtype: torch.dtype = torch.float,
) -> torch.Tensor:
    if cu_seqlens is None:
        return _chunk_global_cumsum_vector_dense(
            s,
            reverse=reverse,
            scale=scale,
            head_first=head_first,
            output_dtype=output_dtype,
        )
    assert s.shape[0] == 1, "Only batch size 1 is supported when cu_seqlens are provided"
    return _run_varlen_dense(
        _chunk_global_cumsum_vector_dense,
        s,
        cu_seqlens=cu_seqlens,
        reverse=reverse,
        scale=scale,
        head_first=head_first,
        output_dtype=output_dtype,
    )


@input_guard
def chunk_global_cumsum(
    s: torch.Tensor,
    reverse: bool = False,
    cu_seqlens: torch.Tensor = None,
    scale: float = None,
    head_first: bool = False,
    output_dtype: torch.dtype = torch.float,
) -> torch.Tensor:
    if cu_seqlens is not None:
        assert s.shape[0] == 1, "Only batch size 1 is supported when cu_seqlens are provided"
    if len(s.shape) == 3:
        return chunk_global_cumsum_scalar(
            s=s,
            reverse=reverse,
            cu_seqlens=cu_seqlens,
            scale=scale,
            head_first=head_first,
            output_dtype=output_dtype,
        )
    if len(s.shape) == 4:
        return chunk_global_cumsum_vector(
            s=s,
            reverse=reverse,
            cu_seqlens=cu_seqlens,
            scale=scale,
            head_first=head_first,
            output_dtype=output_dtype,
        )
    raise ValueError(
        f"Unsupported input shape {s.shape}, "
        f"which should be [B, T, H]/[B, T, H, D] if `head_first=False` "
        f"or [B, H, T]/[B, H, T, D] otherwise",
    )


@input_guard
def chunk_local_cumsum(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
    scale: float = None,
    cu_seqlens: torch.Tensor = None,
    head_first: bool = False,
    output_dtype: torch.dtype = torch.float,
    chunk_indices: torch.LongTensor = None,
    **kwargs,
) -> torch.Tensor:
    if cu_seqlens is not None:
        assert g.shape[0] == 1, "Only batch size 1 is supported when cu_seqlens are provided"
    if len(g.shape) == 3:
        return chunk_local_cumsum_scalar(
            g=g,
            chunk_size=chunk_size,
            reverse=reverse,
            scale=scale,
            cu_seqlens=cu_seqlens,
            head_first=head_first,
            output_dtype=output_dtype,
            chunk_indices=chunk_indices,
        )
    if len(g.shape) == 4:
        return chunk_local_cumsum_vector(
            g=g,
            chunk_size=chunk_size,
            reverse=reverse,
            scale=scale,
            cu_seqlens=cu_seqlens,
            head_first=head_first,
            output_dtype=output_dtype,
            chunk_indices=chunk_indices,
        )
    raise ValueError(
        f"Unsupported input shape {g.shape}, "
        f"which should be [B, T, H]/[B, T, H, D] if `head_first=False` "
        f"or [B, H, T]/[B, H, T, D] otherwise",
    )


def ref_chunk_local_cumsum_scalar(s, chunk_size, reverse=False, scale=None, head_first=False):
    if head_first:
        _, _, SEQ_LEN = s.shape
        result = torch.empty_like(s)
        for start in range(0, SEQ_LEN, chunk_size):
            end = min(start + chunk_size, SEQ_LEN)
            chunk = s[:, :, start:end]
            if reverse:
                result[:, :, start:end] = torch.flip(torch.cumsum(torch.flip(chunk, dims=[2]), dim=2), dims=[2])
            else:
                result[:, :, start:end] = torch.cumsum(chunk, dim=2)
    else:
        _, SEQ_LEN, _ = s.shape
        result = torch.empty_like(s)
        for start in range(0, SEQ_LEN, chunk_size):
            end = min(start + chunk_size, SEQ_LEN)
            chunk = s[:, start:end, :]
            if reverse:
                result[:, start:end, :] = torch.flip(torch.cumsum(torch.flip(chunk, dims=[1]), dim=1), dims=[1])
            else:
                result[:, start:end, :] = torch.cumsum(chunk, dim=1)

    if scale is not None:
        result = result * scale

    return result.to(torch.float32)


def ref_chunk_global_cumsum_scalar(s, reverse=False, scale=None, head_first=False):
    if head_first:
        if reverse:
            result = torch.flip(torch.cumsum(torch.flip(s, dims=[2]), dim=2), dims=[2])
        else:
            result = torch.cumsum(s, dim=2)
    else:
        if reverse:
            result = torch.flip(torch.cumsum(torch.flip(s, dims=[1]), dim=1), dims=[1])
        else:
            result = torch.cumsum(s, dim=1)

    if scale is not None:
        result = result * scale

    return result.to(torch.float32)


def ref_chunk_local_cumsum_vector(s, chunk_size, reverse=False, scale=None, head_first=False):
    if head_first:
        _, _, SEQ_LEN, _ = s.shape
        result = torch.empty_like(s)
        for start in range(0, SEQ_LEN, chunk_size):
            end = min(start + chunk_size, SEQ_LEN)
            chunk = s[:, :, start:end, :]
            if reverse:
                result[:, :, start:end, :] = torch.flip(torch.cumsum(torch.flip(chunk, dims=[2]), dim=2), dims=[2])
            else:
                result[:, :, start:end, :] = torch.cumsum(chunk, dim=2)
    else:
        _, SEQ_LEN, _, _ = s.shape
        result = torch.empty_like(s)
        for start in range(0, SEQ_LEN, chunk_size):
            end = min(start + chunk_size, SEQ_LEN)
            chunk = s[:, start:end, :, :]
            if reverse:
                result[:, start:end, :, :] = torch.flip(torch.cumsum(torch.flip(chunk, dims=[1]), dim=1), dims=[1])
            else:
                result[:, start:end, :, :] = torch.cumsum(chunk, dim=1)

    if scale is not None:
        result = result * scale

    return result.to(torch.float32)


def ref_chunk_global_cumsum_vector(s, reverse=False, scale=None, head_first=False):
    if head_first:
        if reverse:
            result = torch.flip(torch.cumsum(torch.flip(s, dims=[2]), dim=2), dims=[2])
        else:
            result = torch.cumsum(s, dim=2)
    else:
        if reverse:
            result = torch.flip(torch.cumsum(torch.flip(s, dims=[1]), dim=1), dims=[1])
        else:
            result = torch.cumsum(s, dim=1)

    if scale is not None:
        result = result * scale

    return result.to(torch.float32)


def ref_varlen_cumsum(ref_fn, s, cu_seqlens, head_first=False, **kwargs):
    result = torch.empty_like(s, dtype=torch.float32)
    for bos, eos in _sequence_bounds(cu_seqlens):
        if bos == eos:
            continue
        seq = _slice_sequence(s, bos, eos, head_first)
        seq_result = ref_fn(seq, head_first=head_first, **kwargs)
        _write_sequence(result, seq_result, bos, eos, head_first)
    return result


if __name__ == "__main__":
    tilelang.cache.clear_cache()
    torch.manual_seed(0)

    print("=== Testing chunk_local_cumsum_scalar ===")

    test_configs_local_scalar = [
        (1, 8, 128, 32, False, True),
        (1, 8, 128, 32, True, True),
        (1, 8, 130, 32, False, False),
        (1, 8, 130, 32, True, False),
        (1, 7, 130, 32, False, False),
        (1, 7, 130, 32, True, True),
        (2, 16, 256, 64, False, True),
        (2, 16, 256, 64, True, True),
    ]

    for B, H, SEQ_LEN, BT, reverse, head_first in test_configs_local_scalar:
        shape = (B, H, SEQ_LEN) if head_first else (B, SEQ_LEN, H)
        print(f"Testing B={B}, H={H}, SEQ_LEN={SEQ_LEN}, BT={BT}, reverse={reverse}, head_first={head_first}")
        s = torch.randn(shape).npu().to(torch.float)
        o = chunk_local_cumsum_scalar(s, BT, reverse=reverse, head_first=head_first)
        ref_o = ref_chunk_local_cumsum_scalar(s, BT, reverse=reverse, head_first=head_first)
        torch.testing.assert_close(o.cpu(), ref_o.cpu(), rtol=1e-5, atol=1e-5)
        print("  Passed!")

    print("\n=== Testing chunk_global_cumsum_scalar ===")

    test_configs_global_scalar = [
        (1, 8, 128, False, True),
        (1, 8, 130, True, False),
        (1, 7, 130, False, False),
        (1, 7, 130, True, True),
        (2, 16, 256, False, True),
        (2, 16, 256, True, True),
    ]

    for B, H, SEQ_LEN, reverse, head_first in test_configs_global_scalar:
        shape = (B, H, SEQ_LEN) if head_first else (B, SEQ_LEN, H)
        print(f"Testing B={B}, H={H}, SEQ_LEN={SEQ_LEN}, reverse={reverse}, head_first={head_first}")
        s = torch.randn(shape).npu().to(torch.float)
        o = chunk_global_cumsum_scalar(s, reverse=reverse, head_first=head_first)
        ref_o = ref_chunk_global_cumsum_scalar(s, reverse=reverse, head_first=head_first)
        torch.testing.assert_close(o.cpu(), ref_o.cpu(), rtol=1e-5, atol=1e-5)
        print("  Passed!")

    print("\n=== Testing chunk_local_cumsum_vector ===")

    test_configs_local_vector = [
        (1, 8, 128, 16, 32, False, True),
        (1, 8, 128, 16, 32, True, True),
        (1, 8, 130, 17, 32, False, False),
        (1, 8, 130, 17, 32, True, False),
        (1, 7, 130, 17, 32, False, False),
        (1, 7, 130, 17, 32, True, True),
        (2, 16, 256, 32, 64, False, True),
        (2, 16, 256, 32, 64, True, True),
    ]

    for B, H, SEQ_LEN, S_DIM, BT, reverse, head_first in test_configs_local_vector:
        shape = (B, H, SEQ_LEN, S_DIM) if head_first else (B, SEQ_LEN, H, S_DIM)
        print(f"Testing B={B}, H={H}, SEQ_LEN={SEQ_LEN}, S_DIM={S_DIM}, BT={BT}, reverse={reverse}, head_first={head_first}")
        s = torch.randn(shape).npu().to(torch.float)
        o = chunk_local_cumsum_vector(s, BT, reverse=reverse, head_first=head_first)
        ref_o = ref_chunk_local_cumsum_vector(s, BT, reverse=reverse, head_first=head_first)
        torch.testing.assert_close(o.cpu(), ref_o.cpu(), rtol=1e-5, atol=1e-5)
        print("  Passed!")

    print("\n=== Testing chunk_global_cumsum_vector ===")

    test_configs_global_vector = [
        (1, 8, 128, 16, False, True),
        (1, 8, 130, 17, True, False),
        (1, 7, 130, 17, False, False),
        (1, 7, 130, 17, True, True),
        (2, 16, 256, 32, False, True),
        (2, 16, 256, 32, True, True),
    ]

    for B, H, SEQ_LEN, S_DIM, reverse, head_first in test_configs_global_vector:
        shape = (B, H, SEQ_LEN, S_DIM) if head_first else (B, SEQ_LEN, H, S_DIM)
        print(f"Testing B={B}, H={H}, SEQ_LEN={SEQ_LEN}, S_DIM={S_DIM}, reverse={reverse}, head_first={head_first}")
        s = torch.randn(shape).npu().to(torch.float)
        o = chunk_global_cumsum_vector(s, reverse=reverse, head_first=head_first)
        ref_o = ref_chunk_global_cumsum_vector(s, reverse=reverse, head_first=head_first)
        torch.testing.assert_close(o.cpu(), ref_o.cpu(), rtol=1e-4, atol=1e-4)
        print("  Passed!")

    print("\n=== Testing with scale ===")

    s_scalar = torch.randn((2, 16, 256)).npu().to(torch.float)
    scale = 0.5

    o_local = chunk_local_cumsum_scalar(s_scalar, 64, scale=scale, head_first=True)
    ref_o_local = ref_chunk_local_cumsum_scalar(s_scalar, 64, scale=scale, head_first=True)
    torch.testing.assert_close(o_local.cpu(), ref_o_local.cpu(), rtol=1e-5, atol=1e-5)
    print("local scalar cumsum with scale: Passed!")

    o_global = chunk_global_cumsum_scalar(s_scalar, scale=scale, head_first=True)
    ref_o_global = ref_chunk_global_cumsum_scalar(s_scalar, scale=scale, head_first=True)
    torch.testing.assert_close(o_global.cpu(), ref_o_global.cpu(), rtol=1e-5, atol=1e-5)
    print("global scalar cumsum with scale: Passed!")

    s_vector = torch.randn((2, 16, 256, 32)).npu().to(torch.float)
    o_local_v = chunk_local_cumsum_vector(s_vector, 64, scale=scale, head_first=True)
    ref_o_local_v = ref_chunk_local_cumsum_vector(s_vector, 64, scale=scale, head_first=True)
    torch.testing.assert_close(o_local_v.cpu(), ref_o_local_v.cpu(), rtol=1e-5, atol=1e-5)
    print("local vector cumsum with scale: Passed!")

    o_global_v = chunk_global_cumsum_vector(s_vector, scale=scale, head_first=True)
    ref_o_global_v = ref_chunk_global_cumsum_vector(s_vector, scale=scale, head_first=True)
    torch.testing.assert_close(o_global_v.cpu(), ref_o_global_v.cpu(), rtol=1e-5, atol=1e-5)
    print("global vector cumsum with scale: Passed!")

    print("\n=== Testing varlen wrappers ===")

    cu_seqlens = torch.tensor([0, 45, 118, 160], dtype=torch.int32).npu()
    chunk_indices = prepare_chunk_indices(cu_seqlens, 32)

    s_varlen_scalar = torch.randn((1, 160, 7)).npu().to(torch.float)
    o_local_varlen = chunk_local_cumsum(
        s_varlen_scalar,
        32,
        reverse=True,
        cu_seqlens=cu_seqlens,
        head_first=False,
        chunk_indices=chunk_indices,
    )
    ref_o_local_varlen = ref_varlen_cumsum(
        ref_chunk_local_cumsum_scalar,
        s_varlen_scalar,
        cu_seqlens,
        chunk_size=32,
        reverse=True,
        head_first=False,
    )
    torch.testing.assert_close(o_local_varlen.cpu(), ref_o_local_varlen.cpu(), rtol=1e-5, atol=1e-5)
    print("varlen local scalar dispatcher: Passed!")

    o_global_varlen = chunk_global_cumsum(
        s_varlen_scalar,
        reverse=False,
        cu_seqlens=cu_seqlens,
        head_first=False,
    )
    ref_o_global_varlen = ref_varlen_cumsum(
        ref_chunk_global_cumsum_scalar,
        s_varlen_scalar,
        cu_seqlens,
        reverse=False,
        head_first=False,
    )
    torch.testing.assert_close(o_global_varlen.cpu(), ref_o_global_varlen.cpu(), rtol=1e-5, atol=1e-5)
    print("varlen global scalar dispatcher: Passed!")

    s_varlen_vector = torch.randn((1, 7, 160, 17)).npu().to(torch.float)
    o_local_varlen_v = chunk_local_cumsum(
        s_varlen_vector,
        32,
        reverse=False,
        cu_seqlens=cu_seqlens,
        head_first=True,
        chunk_indices=chunk_indices,
    )
    ref_o_local_varlen_v = ref_varlen_cumsum(
        ref_chunk_local_cumsum_vector,
        s_varlen_vector,
        cu_seqlens,
        chunk_size=32,
        reverse=False,
        head_first=True,
    )
    torch.testing.assert_close(o_local_varlen_v.cpu(), ref_o_local_varlen_v.cpu(), rtol=1e-4, atol=1e-4)
    print("varlen local vector dispatcher: Passed!")

    o_global_varlen_v = chunk_global_cumsum(
        s_varlen_vector,
        reverse=True,
        cu_seqlens=cu_seqlens,
        head_first=True,
    )
    ref_o_global_varlen_v = ref_varlen_cumsum(
        ref_chunk_global_cumsum_vector,
        s_varlen_vector,
        cu_seqlens,
        reverse=True,
        head_first=True,
    )
    torch.testing.assert_close(o_global_varlen_v.cpu(), ref_o_global_varlen_v.cpu(), rtol=1e-4, atol=1e-4)
    print("varlen global vector dispatcher: Passed!")

    print("\nKernel Output Match!")
