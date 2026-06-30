import torch
import torch.nn.functional as F

import tilelang
import tilelang.language as T

# UnsortedSegmentSum: y[i] = sum(data[j]) where segment_ids[j] == i
# Adaptive strategy: atomic_add (default) / seg_reduce (large D small N) / block_reduce (single seg)

_pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

_kernel_cache = {}
_seg_reduce_cache = {}
_block_reduce_cache = {}
VEC_NUM = 2


# Strategy 1: atomic_add kernel


@tilelang.jit(pass_configs=_pass_configs)
def atomic_add_kernel(N, D, num_segments, block_D, rows_per_core, dtype="float"):
    d_blocks = (D + block_D - 1) // block_D
    num_blocks = (N + rows_per_core - 1) // rows_per_core
    rows_per_vec = rows_per_core // VEC_NUM

    @T.prim_func
    def main(
        Data: T.Tensor((N, D), dtype),
        SegIds: T.Tensor((N,), "int32"),
        Output: T.Tensor((num_segments, D), dtype),
    ):
        with T.Kernel(num_blocks, is_npu=True) as (cid, vid):
            row_ub = T.alloc_ub([1, block_D], dtype)
            seg_ub = T.alloc_ub([1], "int32")

            for j in T.serial(rows_per_vec):
                row = cid * rows_per_core + j * VEC_NUM + vid
                if row < N:
                    T.copy(SegIds[row : row + 1], seg_ub)

                    for d_blk in T.serial(d_blocks):
                        T.copy(
                            Data[row : row + 1, d_blk * block_D : (d_blk + 1) * block_D],
                            row_ub,
                        )
                        T.tile.atomic_add(Output[seg_ub[0], d_blk * block_D], row_ub)

    return main


# Strategy 2: seg_reduce kernel (large D, small N)


@tilelang.jit(out_idx=[2], pass_configs=_pass_configs)
def seg_reduce_kernel(N, D, num_segments, block_D, max_seg_len, dtype="float"):
    d_blocks = (D + block_D - 1) // block_D

    @T.prim_func
    def main(
        SortedData: T.Tensor((N, D), dtype),
        Offsets: T.Tensor((num_segments + 1,), "int32"),
        Output: T.Tensor((num_segments, D), dtype),
    ):
        with T.Kernel(num_segments * d_blocks, is_npu=True) as (cid, vid):
            seg = cid // d_blocks
            d_blk = cid % d_blocks
            start = Offsets[seg]
            end = Offsets[seg + 1]

            acc_ub = T.alloc_ub([block_D], dtype)
            T.tile.fill(acc_ub, 0.0)
            row_ub = T.alloc_ub([block_D], dtype)

            for j in T.serial(max_seg_len):
                row = start + j
                if row < end:
                    T.copy(SortedData[row, d_blk * block_D], row_ub)
                    T.tile.add(acc_ub, acc_ub, row_ub)

            T.copy(acc_ub, Output[seg, d_blk * block_D])

    return main


# Strategy 3: block_reduce kernel (small D, large N, single segment)


@tilelang.jit(out_idx=[1], pass_configs=_pass_configs)
def block_reduce_kernel(N, D, block_N, dtype="float"):
    num_blocks = N // block_N

    @T.prim_func
    def main(
        Data: T.Tensor((N, D), dtype),
        Partial: T.Tensor((num_blocks, D), dtype),
    ):
        with T.Kernel(num_blocks, is_npu=True) as (cid, vid):
            if vid == 0:
                data_ub = T.alloc_ub([block_N, D], dtype)
                acc_ub = T.alloc_ub([D], dtype)
                T.tile.fill(acc_ub, 0.0)
                T.copy(Data[cid * block_N, 0], data_ub)
                T.reduce_sum(data_ub, acc_ub, dim=0, clear=True)
                T.copy(acc_ub, Partial[cid, 0])

    return main


# Helpers


def _choose_block_D(D):
    if D <= 1:
        return 1
    for bd in [32768, 16384, 8192, 4096, 2048, 1024, 512, 256, 128, 64]:
        if bd <= D:
            return bd
    return 64


def _choose_rows_per_core(N, target_blocks=32):
    if target_blocks * VEC_NUM >= N:
        return VEC_NUM
    rpc = N // target_blocks
    rpc = max(VEC_NUM, ((rpc + VEC_NUM - 1) // VEC_NUM) * VEC_NUM)
    return rpc


def _torch_dtype_to_tl_local(dtype):
    if dtype == torch.float32:
        return "float"
    elif dtype == torch.float16:
        return "float16"
    elif dtype == torch.bfloat16:
        return "bfloat16"
    elif dtype == torch.int32:
        return "int32"
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def _use_seg_reduce(N, D, num_segments):
    # Only worthwhile when N >> num_seg (host sort overhead ~1ms)
    return D > 4096 and N < 2048 and 4 * num_segments < N


def _use_block_reduce(N, D, num_segments):
    # Pure reduce for single-segment large-N case; D<=4 needs pad (reduce_sum bug)
    return num_segments == 1 and N > 65536 and D <= 4


# Main interface


def unsorted_segment_sum(data: torch.Tensor, segment_ids: torch.Tensor, num_segments: int) -> torch.Tensor:
    """UnsortedSegmentSum: y[i] = sum(data[j]) where segment_ids[j] == i

    Adaptive strategy:
      - seg_reduce: D > 4096, N < 2048, N > 4 * num_segments
      - block_reduce: num_segments == 1, N > 65536, D <= 4
      - atomic_add: default

    Args:
        data: input tensor, shape (N, *feature_dims)
        segment_ids: 1D int tensor, length N
        num_segments: number of segments

    Returns:
        output tensor, shape (num_segments, *data.shape[1:])
    """
    orig_shape = data.shape
    orig_dtype = data.dtype

    data_2d = data.reshape(data.shape[0], -1).contiguous()
    N, D = data_2d.shape

    # int64 not supported by atomic_add, cast to int32
    if orig_dtype == torch.int64:
        data_2d = data_2d.to(torch.int32)

    # fp16/bf16 accumulate in fp32 for precision
    compute_dtype = data_2d.dtype
    if orig_dtype in (torch.float16, torch.bfloat16):
        data_2d = data_2d.to(torch.float32)
        compute_dtype = torch.float32

    seg_ids_32 = segment_ids.to(torch.int32) if segment_ids.dtype != torch.int32 else segment_ids

    if _use_block_reduce(N, D, num_segments):
        return _block_reduce_impl(data_2d, num_segments, orig_shape, orig_dtype, D, compute_dtype)

    if _use_seg_reduce(N, D, num_segments):
        return _seg_reduce_impl(data_2d, seg_ids_32, num_segments, orig_shape, orig_dtype, D)

    return _atomic_impl(data_2d, seg_ids_32, num_segments, orig_shape, orig_dtype, D, compute_dtype)


def _block_reduce_impl(data_2d, num_segments, orig_shape, orig_dtype, D, compute_dtype):
    # Pad D to >= 4 (reduce_sum bug on D <= 2)
    D2 = max(D, 4)
    if D2 != D:
        data_pad = F.pad(data_2d, (0, D2 - D))
    else:
        data_pad = data_2d

    block_N = 256
    N = data_pad.shape[0]
    N_pad = ((N + block_N - 1) // block_N) * block_N
    if N_pad != N:
        data_pad = F.pad(data_pad, (0, 0, 0, N_pad - N), value=0.0)

    tl_dtype = _torch_dtype_to_tl_local(compute_dtype)

    key = (N_pad, D2, block_N, tl_dtype)
    if key not in _block_reduce_cache:
        _block_reduce_cache[key] = block_reduce_kernel(N_pad, D2, block_N, dtype=tl_dtype)
    kernel = _block_reduce_cache[key]

    partial = kernel(data_pad)
    output = partial.sum(0).unsqueeze(0)  # [1, D2]
    if D2 != D:
        output = output[:, :D]
    if orig_dtype in (torch.float16, torch.bfloat16) or orig_dtype == torch.int64:
        output = output.to(orig_dtype)

    output_shape = (num_segments,) + orig_shape[1:]
    return output.reshape(output_shape)


def _atomic_impl(data_2d, seg_ids_32, num_segments, orig_shape, orig_dtype, D, compute_dtype):
    block_D = _choose_block_D(D)
    D_padded = ((D + block_D - 1) // block_D) * block_D
    if D_padded != D:
        data_2d = F.pad(data_2d, (0, D_padded - D))

    rows_per_core = _choose_rows_per_core(data_2d.shape[0])
    tl_dtype = _torch_dtype_to_tl_local(compute_dtype)
    N = data_2d.shape[0]

    key = (N, D_padded, num_segments, block_D, rows_per_core, tl_dtype)
    if key not in _kernel_cache:
        _kernel_cache[key] = atomic_add_kernel(N, D_padded, num_segments, block_D, rows_per_core, dtype=tl_dtype)
    kernel = _kernel_cache[key]

    output = torch.zeros(num_segments, D_padded, dtype=compute_dtype, device=data_2d.device)
    kernel(data_2d, seg_ids_32, output)

    if D_padded != D:
        output = output[:, :D]
    if orig_dtype in (torch.float16, torch.bfloat16) or orig_dtype == torch.int64:
        output = output.to(orig_dtype)

    output_shape = (num_segments,) + orig_shape[1:]
    return output.reshape(output_shape)


def _seg_reduce_impl(data_2d, seg_ids_32, num_segments, orig_shape, orig_dtype, D):
    N = data_2d.shape[0]

    sorted_ids, sort_idx = torch.sort(seg_ids_32)
    sorted_data = data_2d[sort_idx]

    counts = torch.zeros(num_segments, dtype=torch.int32, device=data_2d.device)
    counts.scatter_add_(0, seg_ids_32, torch.ones(N, dtype=torch.int32, device=data_2d.device))
    offsets = torch.zeros(num_segments + 1, dtype=torch.int32, device=data_2d.device)
    offsets[1:] = counts.cumsum(0)
    max_seg_len = counts.max().item()

    # acc_ub + row_ub share UB, cap block_D at 16384
    block_D = min(_choose_block_D(D), 16384)
    D_padded = ((D + block_D - 1) // block_D) * block_D
    if D_padded != D:
        sorted_data = F.pad(sorted_data, (0, D_padded - D))

    tl_dtype = _torch_dtype_to_tl_local(data_2d.dtype)

    key = (N, D_padded, num_segments, block_D, max_seg_len, tl_dtype)
    if key not in _seg_reduce_cache:
        _seg_reduce_cache[key] = seg_reduce_kernel(N, D_padded, num_segments, block_D, max_seg_len, dtype=tl_dtype)
    kernel = _seg_reduce_cache[key]

    output = kernel(sorted_data, offsets)

    if D_padded != D:
        output = output[:, :D]
    if orig_dtype in (torch.float16, torch.bfloat16) or orig_dtype == torch.int64:
        output = output.to(orig_dtype)

    output_shape = (num_segments,) + orig_shape[1:]
    return output.reshape(output_shape)


# Reference & Test


def ref_unsorted_segment_sum(data, segment_ids, num_segments):
    output_shape = (num_segments,) + data.shape[1:]
    if data.dtype in (torch.float16, torch.bfloat16):
        y_fp32 = torch.zeros(output_shape, dtype=torch.float32, device=data.device)
        y_fp32.index_add_(0, segment_ids, data.to(torch.float32))
        y = y_fp32.to(data.dtype)
    else:
        y = torch.zeros(output_shape, dtype=data.dtype, device=data.device)
        y.index_add_(0, segment_ids, data)
    return y


def _test(N, D, num_segments, dtype_str):
    dt = getattr(torch, dtype_str)
    torch.manual_seed(0)

    if dtype_str in ("int32", "int64"):
        data = torch.randint(-1000, 1000, (N, D), dtype=dt).npu()
    else:
        data = torch.randn(N, D, dtype=dt).npu()
    ids = torch.randint(0, num_segments, (N,), dtype=torch.int32).npu()

    out = unsorted_segment_sum(data, ids, num_segments)
    ref = ref_unsorted_segment_sum(data, ids, num_segments)

    if dtype_str in ("int32", "int64"):
        assert torch.equal(out, ref)
    else:
        tol = {"float16": (1e-3, 1e-3), "bfloat16": (1e-2, 1e-2), "float32": (1e-3, 1e-3)}[dtype_str]
        torch.testing.assert_close(out, ref, rtol=tol[0], atol=tol[1])

    print(f"Test passed: N={N}, D={D}, num_segments={num_segments}, dtype={dtype_str}")


def _test_3d(N, D2, num_segments, dtype_str):
    dt = getattr(torch, dtype_str)
    torch.manual_seed(0)
    data = torch.randn(N, D2, 32, dtype=dt).npu()
    ids = torch.randint(0, num_segments, (N,), dtype=torch.int32).npu()

    out = unsorted_segment_sum(data, ids, num_segments)
    ref = ref_unsorted_segment_sum(data, ids, num_segments)

    tol = {"float16": (1e-3, 1e-3), "bfloat16": (1e-2, 1e-2), "float32": (1e-3, 1e-3)}[dtype_str]
    torch.testing.assert_close(out, ref, rtol=tol[0], atol=tol[1])

    print(f"Test passed: N={N}, shape=({N},{D2},32), num_segments={num_segments}, dtype={dtype_str}")


if __name__ == "__main__":
    import tilelang

    tilelang.disable_cache()

    _test(1024, 512, 256, "float16")
    _test(512, 256, 128, "bfloat16")
    _test(2048, 256, 512, "float32")
    _test(1024, 128, 256, "int32")
    _test(4096, 64, 64, "int64")
    _test_3d(128, 64, 32, "float16")

    print("Kernel Output Match!")
