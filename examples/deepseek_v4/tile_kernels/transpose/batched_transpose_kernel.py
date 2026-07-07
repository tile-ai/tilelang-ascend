import tilelang
import tilelang.language as T
import torch

_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}
_DTYPE_TO_STR = {
    torch.float32: "float32",
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
    torch.uint8: "uint8",
    torch.float8_e4m3fn: "float8_e4m3fn",
}

VEC_NUM = 2
TILE = 16
MAX_NPU_BLOCKS = 65535
GRID_THRESHOLD = 2048  # 超过此块数阈值自动激活网格收缩策略
_BLOCK_LARGE = 128
_BLOCK_SMALL = 64
_CORE_NUM = 24  # NPU 物理 AI Core 数 (A2/A3)


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def _align_up(x: int, y: int) -> int:
    return _ceil_div(x, y) * y


# ==================== 1. FP32 经典内核 (借道 FP16 硬件转置管道) ====================
@tilelang.jit(out_idx=[-1], pass_configs=_PASS_CONFIGS)
def _batched_transpose_kernel_fp32_vector(
    num_batches: int,
    shape_x: int,
    shape_y: int,
    block_M: int,
    block_N: int,
    dtype: str = "float32",
):
    m_blocks = shape_x // block_M
    n_blocks = shape_y // block_N
    total_blocks = num_batches * m_blocks * n_blocks
    num_tiles_m = block_M // TILE
    num_tiles_n = block_N // TILE
    total_iters = num_tiles_m * num_tiles_n
    assert total_iters % 2 == 0, "Total tiles within a block must be even to split across 2 AIV threads."
    iters_per_thread = total_iters // 2

    core_num = min(total_blocks, _CORE_NUM)
    blocks_per_core = _ceil_div(total_blocks, core_num)

    @T.prim_func
    def transpose_fp32_vector_kernel(
        x: T.Tensor((num_batches, shape_x, shape_y), dtype),
        out: T.Tensor((num_batches, shape_y, shape_x), dtype),
    ):
        with T.Kernel(core_num, is_npu=True) as (cid, vid):
            src_ping = T.alloc_ub((TILE, TILE), dtype)
            src_pong = T.alloc_ub((TILE, TILE), dtype)
            dst_ping = T.alloc_ub((TILE, TILE), dtype)
            dst_pong = T.alloc_ub((TILE, TILE), dtype)
            src_ping_16 = T.alloc_ub((TILE, TILE), "float16")
            src_pong_16 = T.alloc_ub((TILE, TILE), "float16")
            dst_ping_16 = T.alloc_ub((TILE, TILE), "float16")
            dst_pong_16 = T.alloc_ub((TILE, TILE), "float16")

            idx_ping = vid * 2 + 0
            idx_pong = vid * 2 + 1

            for blk in T.serial(blocks_per_core):
                global_blk = cid * blocks_per_core + blk
                if global_blk < total_blocks:
                    bid = global_blk // (m_blocks * n_blocks)
                    rem = global_blk % (m_blocks * n_blocks)
                    bx = rem // n_blocks
                    by = rem % n_blocks

                    T.set_flag("MTE3", "V", idx_ping)
                    T.set_flag("MTE3", "V", idx_pong)

                    for i in T.serial(iters_per_thread + 2):
                        if i >= 2:
                            idx_m3 = i - 2
                            tile_idx_m3 = vid * iters_per_thread + idx_m3
                            t_m_m3 = tile_idx_m3 % num_tiles_m
                            t_n_m3 = tile_idx_m3 // num_tiles_m
                            dst_r = by * block_N + t_n_m3 * TILE
                            dst_c = bx * block_M + t_m_m3 * TILE
                            if idx_m3 % 2 == 0:
                                T.wait_flag("V", "MTE3", idx_ping)
                                T.copy(dst_ping, out[bid, dst_r, dst_c])
                                T.set_flag("MTE3", "V", idx_ping)
                            else:
                                T.wait_flag("V", "MTE3", idx_pong)
                                T.copy(dst_pong, out[bid, dst_r, dst_c])
                                T.set_flag("MTE3", "V", idx_pong)

                        if i >= 1 and i < (iters_per_thread + 1):
                            idx_v = i - 1
                            if idx_v % 2 == 0:
                                T.wait_flag("MTE2", "V", idx_ping)
                                T.wait_flag("MTE3", "V", idx_ping)
                                T.copy(src_ping, src_ping_16)
                                T.tile.transpose(dst_ping_16, src_ping_16)
                                T.copy(dst_ping_16, dst_ping)
                                T.set_flag("V", "MTE2", idx_ping)
                                T.set_flag("V", "MTE3", idx_ping)
                            else:
                                T.wait_flag("MTE2", "V", idx_pong)
                                T.wait_flag("MTE3", "V", idx_pong)
                                T.copy(src_pong, src_pong_16)
                                T.tile.transpose(dst_pong_16, src_pong_16)
                                T.copy(dst_pong_16, dst_pong)
                                T.set_flag("V", "MTE2", idx_pong)
                                T.set_flag("V", "MTE3", idx_pong)

                        if i < iters_per_thread:
                            idx_m2 = i
                            tile_idx_m2 = vid * iters_per_thread + idx_m2
                            t_m_m2 = tile_idx_m2 % num_tiles_m
                            t_n_m2 = tile_idx_m2 // num_tiles_m
                            src_r = bx * block_M + t_m_m2 * TILE
                            src_c = by * block_N + t_n_m2 * TILE
                            if idx_m2 % 2 == 0:
                                if idx_m2 >= 2:
                                    T.wait_flag("V", "MTE2", idx_ping)
                                T.copy(x[bid, src_r, src_c], src_ping)
                                T.set_flag("MTE2", "V", idx_ping)
                            else:
                                if idx_m2 >= 2:
                                    T.wait_flag("V", "MTE2", idx_pong)
                                T.copy(x[bid, src_r, src_c], src_pong)
                                T.set_flag("MTE2", "V", idx_pong)

                    T.wait_flag("V", "MTE2", idx_ping)
                    T.wait_flag("V", "MTE2", idx_pong)
                    T.wait_flag("MTE3", "V", idx_ping)
                    T.wait_flag("MTE3", "V", idx_pong)
    return transpose_fp32_vector_kernel


# ==================== 2. FP32 约束网格内核 (一维平坦化长流水 + 借道 FP16) ====================
@tilelang.jit(out_idx=[-1], pass_configs=_PASS_CONFIGS)
def _batched_transpose_kernel_fp32_serial(
    num_batches: int,
    shape_x: int,
    shape_y: int,
    block_M: int,
    block_N: int,
    dtype: str = "float32",
):
    m_blocks = shape_x // block_M
    n_blocks = shape_y // block_N
    grid_blocks = num_batches * m_blocks
    sub_M = block_M // VEC_NUM
    num_tiles_m = sub_M // TILE
    num_tiles_n = block_N // TILE
    iters_per_thread = n_blocks * num_tiles_m * num_tiles_n

    core_num = min(grid_blocks, _CORE_NUM)
    blocks_per_core = _ceil_div(grid_blocks, core_num)

    @T.prim_func
    def transpose_fp32_serial_kernel(
        x: T.Tensor((num_batches, shape_x, shape_y), dtype),
        out: T.Tensor((num_batches, shape_y, shape_x), dtype),
    ):
        with T.Kernel(core_num, is_npu=True) as (cid, vid):
            src_ping = T.alloc_ub((TILE, TILE), dtype)
            src_pong = T.alloc_ub((TILE, TILE), dtype)
            dst_ping = T.alloc_ub((TILE, TILE), dtype)
            dst_pong = T.alloc_ub((TILE, TILE), dtype)
            
            src_ping_16 = T.alloc_ub((TILE, TILE), "float16")
            src_pong_16 = T.alloc_ub((TILE, TILE), "float16")
            dst_ping_16 = T.alloc_ub((TILE, TILE), "float16")
            dst_pong_16 = T.alloc_ub((TILE, TILE), "float16")

            idx_ping = vid * 2 + 0
            idx_pong = vid * 2 + 1

            for blk in T.serial(blocks_per_core):
                global_blk = cid * blocks_per_core + blk
                if global_blk < grid_blocks:
                    bid = global_blk // m_blocks
                    bx = global_blk % m_blocks

                    T.set_flag("MTE3", "V", idx_ping)
                    T.set_flag("MTE3", "V", idx_pong)

                    for i in T.serial(iters_per_thread + 2):
                        if i >= 2:
                            idx_m3 = i - 2
                            tj_m3 = idx_m3 % num_tiles_n
                            rem_m3 = idx_m3 // num_tiles_n
                            ti_m3 = rem_m3 % num_tiles_m
                            by_m3 = rem_m3 // num_tiles_m
                            row_m3 = vid * sub_M + ti_m3 * TILE
                            dst_r = by_m3 * block_N + tj_m3 * TILE
                            dst_c = bx * block_M + row_m3
                            if idx_m3 % 2 == 0:
                                T.wait_flag("V", "MTE3", idx_ping)
                                T.copy(dst_ping, out[bid, dst_r, dst_c])
                                T.set_flag("MTE3", "V", idx_ping)
                            else:
                                T.wait_flag("V", "MTE3", idx_pong)
                                T.copy(dst_pong, out[bid, dst_r, dst_c])
                                T.set_flag("MTE3", "V", idx_pong)

                        if i >= 1 and i < (iters_per_thread + 1):
                            idx_v = i - 1
                            if idx_v % 2 == 0:
                                T.wait_flag("MTE2", "V", idx_ping)
                                T.wait_flag("MTE3", "V", idx_ping)
                                T.copy(src_ping, src_ping_16)
                                T.tile.transpose(dst_ping_16, src_ping_16)
                                T.copy(dst_ping_16, dst_ping)
                                T.set_flag("V", "MTE2", idx_ping)
                                T.set_flag("V", "MTE3", idx_ping)
                            else:
                                T.wait_flag("MTE2", "V", idx_pong)
                                T.wait_flag("MTE3", "V", idx_pong)
                                T.copy(src_pong, src_pong_16)
                                T.tile.transpose(dst_pong_16, src_pong_16)
                                T.copy(dst_pong_16, dst_pong)
                                T.set_flag("V", "MTE2", idx_pong)
                                T.set_flag("V", "MTE3", idx_pong)

                        if i < iters_per_thread:
                            idx_m2 = i
                            tj_m2 = idx_m2 % num_tiles_n
                            rem_m2 = idx_m2 // num_tiles_n
                            ti_m2 = rem_m2 % num_tiles_m
                            by_m2 = rem_m2 // num_tiles_m
                            row_m2 = vid * sub_M + ti_m2 * TILE
                            src_r = bx * block_M + row_m2
                            src_c = by_m2 * block_N + tj_m2 * TILE
                            if idx_m2 % 2 == 0:
                                if idx_m2 >= 2:
                                    T.wait_flag("V", "MTE2", idx_ping)
                                T.copy(x[bid, src_r, src_c], src_ping)
                                T.set_flag("MTE2", "V", idx_ping)
                            else:
                                if idx_m2 >= 2:
                                    T.wait_flag("V", "MTE2", idx_pong)
                                T.copy(x[bid, src_r, src_c], src_pong)
                                T.set_flag("MTE2", "V", idx_pong)

                    T.wait_flag("V", "MTE2", idx_ping)
                    T.wait_flag("V", "MTE2", idx_pong)
                    T.wait_flag("MTE3", "V", idx_ping)
                    T.wait_flag("MTE3", "V", idx_pong)
    return transpose_fp32_serial_kernel


# ==================== 3. BF16/FP16 经典与约束内核,大shape防止OOM ====================
@tilelang.jit(out_idx=[-1], pass_configs=_PASS_CONFIGS)
def _batched_transpose_kernel_db(
    num_batches: int,
    shape_x: int,
    shape_y: int,
    block_M: int,
    block_N: int,
    dtype: str = "bfloat16",
):
    m_blocks = shape_x // block_M
    n_blocks = shape_y // block_N
    total_blocks = num_batches * m_blocks * n_blocks
    num_tiles_m = block_M // TILE
    num_tiles_n = block_N // TILE
    total_iters = num_tiles_m * num_tiles_n
    assert total_iters % 2 == 0, "Total tiles within a block must be even."
    iters_per_thread = total_iters // 2
    core_num = min(total_blocks, _CORE_NUM)
    blocks_per_core = _ceil_div(total_blocks, core_num)

    @T.prim_func
    def transpose_manual_pipelined_kernel(
        x: T.Tensor((num_batches, shape_x, shape_y), dtype),
        out: T.Tensor((num_batches, shape_y, shape_x), dtype),
    ):
        with T.Kernel(core_num, is_npu=True) as (cid, vid):
            src_ping = T.alloc_ub((TILE, TILE), dtype)
            src_pong = T.alloc_ub((TILE, TILE), dtype)
            dst_ping = T.alloc_ub((TILE, TILE), dtype)
            dst_pong = T.alloc_ub((TILE, TILE), dtype)
            idx_ping = vid * 2 + 0
            idx_pong = vid * 2 + 1

            for blk in T.serial(blocks_per_core):
                global_blk = cid * blocks_per_core + blk
                if global_blk < total_blocks:
                    bid = global_blk // (m_blocks * n_blocks)
                    rem = global_blk % (m_blocks * n_blocks)
                    bx = rem // n_blocks
                    by = rem % n_blocks

                    T.set_flag("MTE3", "V", idx_ping)
                    T.set_flag("MTE3", "V", idx_pong)

                    for i in T.serial(iters_per_thread + 2):
                        if i >= 2:
                            idx_m3 = i - 2
                            tile_idx_m3 = vid * iters_per_thread + idx_m3
                            t_m_m3 = tile_idx_m3 % num_tiles_m
                            t_n_m3 = tile_idx_m3 // num_tiles_m
                            dst_r = by * block_N + t_n_m3 * TILE
                            dst_c = bx * block_M + t_m_m3 * TILE
                            if idx_m3 % 2 == 0:
                                T.wait_flag("V", "MTE3", idx_ping)
                                T.copy(dst_ping, out[bid, dst_r, dst_c])
                                T.set_flag("MTE3", "V", idx_ping)
                            else:
                                T.wait_flag("V", "MTE3", idx_pong)
                                T.copy(dst_pong, out[bid, dst_r, dst_c])
                                T.set_flag("MTE3", "V", idx_pong)

                        if i >= 1 and i < (iters_per_thread + 1):
                            idx_v = i - 1
                            if idx_v % 2 == 0:
                                T.wait_flag("MTE2", "V", idx_ping)
                                T.wait_flag("MTE3", "V", idx_ping)
                                T.tile.transpose(dst_ping, src_ping)
                                T.set_flag("V", "MTE2", idx_ping)
                                T.set_flag("V", "MTE3", idx_ping)
                            else:
                                T.wait_flag("MTE2", "V", idx_pong)
                                T.wait_flag("MTE3", "V", idx_pong)
                                T.tile.transpose(dst_pong, src_pong)
                                T.set_flag("V", "MTE2", idx_pong)
                                T.set_flag("V", "MTE3", idx_pong)

                        if i < iters_per_thread:
                            idx_m2 = i
                            tile_idx_m2 = vid * iters_per_thread + idx_m2
                            t_m_m2 = tile_idx_m2 % num_tiles_m
                            t_n_m2 = tile_idx_m2 // num_tiles_m
                            src_r = bx * block_M + t_m_m2 * TILE
                            src_c = by * block_N + t_n_m2 * TILE
                            if idx_m2 % 2 == 0:
                                if idx_m2 >= 2:
                                    T.wait_flag("V", "MTE2", idx_ping)
                                T.copy(x[bid, src_r, src_c], src_ping)
                                T.set_flag("MTE2", "V", idx_ping)
                            else:
                                if idx_m2 >= 2:
                                    T.wait_flag("V", "MTE2", idx_pong)
                                T.copy(x[bid, src_r, src_c], src_pong)
                                T.set_flag("MTE2", "V", idx_pong)

                    T.wait_flag("V", "MTE2", idx_ping)
                    T.wait_flag("V", "MTE2", idx_pong)
                    T.wait_flag("MTE3", "V", idx_ping)
                    T.wait_flag("MTE3", "V", idx_pong)
    return transpose_manual_pipelined_kernel


@tilelang.jit(out_idx=[-1], pass_configs=_PASS_CONFIGS)
def _batched_transpose_kernel_db_serial(
    num_batches: int,
    shape_x: int,
    shape_y: int,
    block_M: int,
    block_N: int,
    dtype: str = "bfloat16",
):
    m_blocks = shape_x // block_M
    n_blocks = shape_y // block_N
    grid_blocks = num_batches * m_blocks
    sub_M = block_M // VEC_NUM
    num_tiles_m = sub_M // TILE
    num_tiles_n = block_N // TILE
    iters_per_thread = n_blocks * num_tiles_m * num_tiles_n
    core_num = min(grid_blocks, _CORE_NUM)
    blocks_per_core = _ceil_div(grid_blocks, core_num)

    @T.prim_func
    def transpose_db_serial_kernel(
        x: T.Tensor((num_batches, shape_x, shape_y), dtype),
        out: T.Tensor((num_batches, shape_y, shape_x), dtype),
    ):
        with T.Kernel(core_num, is_npu=True) as (cid, vid):
            src_ping = T.alloc_ub((TILE, TILE), dtype)
            src_pong = T.alloc_ub((TILE, TILE), dtype)
            dst_ping = T.alloc_ub((TILE, TILE), dtype)
            dst_pong = T.alloc_ub((TILE, TILE), dtype)
            idx_ping = vid * 2 + 0
            idx_pong = vid * 2 + 1

            for blk in T.serial(blocks_per_core):
                global_blk = cid * blocks_per_core + blk
                if global_blk < grid_blocks:
                    bid = global_blk // m_blocks
                    bx = global_blk % m_blocks

                    T.set_flag("MTE3", "V", idx_ping)
                    T.set_flag("MTE3", "V", idx_pong)

                    for i in T.serial(iters_per_thread + 2):
                        if i >= 2:
                            idx_m3 = i - 2
                            tj_m3 = idx_m3 % num_tiles_n
                            rem_m3 = idx_m3 // num_tiles_n
                            ti_m3 = rem_m3 % num_tiles_m
                            by_m3 = rem_m3 // num_tiles_m
                            row_m3 = vid * sub_M + ti_m3 * TILE
                            dst_r = by_m3 * block_N + tj_m3 * TILE
                            dst_c = bx * block_M + row_m3
                            if idx_m3 % 2 == 0:
                                T.wait_flag("V", "MTE3", idx_ping)
                                T.copy(dst_ping, out[bid, dst_r, dst_c])
                                T.set_flag("MTE3", "V", idx_ping)
                            else:
                                T.wait_flag("V", "MTE3", idx_pong)
                                T.copy(dst_pong, out[bid, dst_r, dst_c])
                                T.set_flag("MTE3", "V", idx_pong)

                        if i >= 1 and i < (iters_per_thread + 1):
                            idx_v = i - 1
                            if idx_v % 2 == 0:
                                T.wait_flag("MTE2", "V", idx_ping)
                                T.wait_flag("MTE3", "V", idx_ping)
                                T.tile.transpose(dst_ping, src_ping)
                                T.set_flag("V", "MTE2", idx_ping)
                                T.set_flag("V", "MTE3", idx_ping)
                            else:
                                T.wait_flag("MTE2", "V", idx_pong)
                                T.wait_flag("MTE3", "V", idx_pong)
                                T.tile.transpose(dst_pong, src_pong)
                                T.set_flag("V", "MTE2", idx_pong)
                                T.set_flag("V", "MTE3", idx_pong)

                        if i < iters_per_thread:
                            idx_m2 = i
                            tj_m2 = idx_m2 % num_tiles_n
                            rem_m2 = idx_m2 // num_tiles_n
                            ti_m2 = rem_m2 % num_tiles_m
                            by_m2 = rem_m2 // num_tiles_m
                            row_m2 = vid * sub_M + ti_m2 * TILE
                            src_r = bx * block_M + row_m2
                            src_c = by_m2 * block_N + tj_m2 * TILE
                            if idx_m2 % 2 == 0:
                                if idx_m2 >= 2:
                                    T.wait_flag("V", "MTE2", idx_ping)
                                T.copy(x[bid, src_r, src_c], src_ping)
                                T.set_flag("MTE2", "V", idx_ping)
                            else:
                                if idx_m2 >= 2:
                                    T.wait_flag("V", "MTE2", idx_pong)
                                T.copy(x[bid, src_r, src_c], src_pong)
                                T.set_flag("MTE2", "V", idx_pong)

                    T.wait_flag("V", "MTE2", idx_ping)
                    T.wait_flag("V", "MTE2", idx_pong)
                    T.wait_flag("MTE3", "V", idx_ping)
                    T.wait_flag("MTE3", "V", idx_pong)
    return transpose_db_serial_kernel


# ==================== 4. 8-Bit 内核 (保持原样) ====================
@tilelang.jit(out_idx=[-1], pass_configs=_PASS_CONFIGS)
def _batched_transpose_kernel_8bit(
    num_batches: int,
    shape_x: int,
    shape_y: int,
    block_M: int,
    block_N: int,
    dtype: str,
):
    m_blocks = shape_x // block_M
    n_blocks = shape_y // block_N
    total_blocks = num_batches * m_blocks * n_blocks
    sub_M = block_M // VEC_NUM
    core_num = min(total_blocks, _CORE_NUM)
    blocks_per_core = _ceil_div(total_blocks, core_num)

    @T.prim_func
    def transpose_8bit_kernel(
        x: T.Tensor((num_batches, shape_x, shape_y), dtype),
        out: T.Tensor((num_batches, shape_y, shape_x), dtype),
    ):
        with T.Kernel(core_num, is_npu=True) as (cid, vid):
            src_buf = T.alloc_ub((sub_M, block_N), dtype)
            dst_buf = T.alloc_ub((block_N, sub_M), dtype)
            with T.Scope("V"):
                for blk in T.serial(blocks_per_core):
                    global_blk = cid * blocks_per_core + blk
                    if global_blk < total_blocks:
                        bid = global_blk // (m_blocks * n_blocks)
                        rem = global_blk % (m_blocks * n_blocks)
                        bx = rem // n_blocks
                        by = rem % n_blocks
                        row_base = bx * block_M + vid * sub_M
                        col_base = by * block_N
                        T.copy(x[bid, row_base, col_base], src_buf)
                        for i in T.serial(sub_M):
                            for j in T.serial(block_N):
                                dst_buf[j, i] = src_buf[i, j]
                        T.copy(dst_buf, out[bid, col_base, row_base])
    return transpose_8bit_kernel


# ==================== 5. 主调度与智能缓存层 ====================
def _transpose_via_pytorch(x: torch.Tensor) -> torch.Tensor:
    return torch.transpose(x, 1, 2).contiguous()


def _select_block_size(shape_x: int, shape_y: int):
    block_M = _BLOCK_LARGE if shape_x % _BLOCK_LARGE == 0 else _BLOCK_SMALL
    block_N = _BLOCK_LARGE if shape_y % _BLOCK_LARGE == 0 else _BLOCK_SMALL
    return block_M, block_N


def _run_kernel(x_contig: torch.Tensor, shape_x: int, shape_y: int) -> torch.Tensor:
    num_batches = x_contig.shape[0]
    block_M, block_N = _select_block_size(shape_x, shape_y)
    padded_x = _align_up(shape_x, block_M)
    padded_y = _align_up(shape_y, block_N)
    m_blocks = padded_x // block_M
    n_blocks = padded_y // block_N
    total_blocks = num_batches * m_blocks * n_blocks
    grid_blocks = num_batches * m_blocks

    use_serial_grid = total_blocks > GRID_THRESHOLD
    limit_exceeded = (use_serial_grid and grid_blocks > MAX_NPU_BLOCKS) or (not use_serial_grid and total_blocks > MAX_NPU_BLOCKS)

    # 1. 物理网格超限切分策略
    if limit_exceeded:
        max_batches_per_launch = max(1, MAX_NPU_BLOCKS // (m_blocks * (1 if use_serial_grid else n_blocks)))
        if max_batches_per_launch < num_batches:
            try:
                result = torch.empty((num_batches, shape_y, shape_x), dtype=x_contig.dtype, device=x_contig.device)
            except (torch.OutOfMemoryError, RuntimeError):
                try:
                    import torch_npu
                    torch.npu.empty_cache()
                    result = torch.empty((num_batches, shape_y, shape_x), dtype=x_contig.dtype, device=x_contig.device)
                except (torch.OutOfMemoryError, RuntimeError, ImportError):
                    return _transpose_via_pytorch(x_contig)

            for start in range(0, num_batches, max_batches_per_launch):
                end = min(start + max_batches_per_launch, num_batches)
                chunk = x_contig[start:end]
                chunk_result = _run_kernel_single(chunk, shape_x, shape_y, block_M, block_N, padded_x, padded_y)
                result[start:end] = chunk_result
                del chunk_result
            return result

    # 2. 防止OOM
    try:
        return _run_kernel_single(x_contig, shape_x, shape_y, block_M, block_N, padded_x, padded_y)
    except (torch.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" in str(e).lower() or isinstance(e, torch.OutOfMemoryError):
            try:
                import torch_npu
                torch.npu.empty_cache()
            except:
                pass
            try:
                return _run_kernel_single(x_contig, shape_x, shape_y, block_M, block_N, padded_x, padded_y)
            except (torch.OutOfMemoryError, RuntimeError):
                # 内存彻底见底的终极兜底方案
                return _transpose_via_pytorch(x_contig)
        raise e


def _run_kernel_single(
    x_contig: torch.Tensor, shape_x: int, shape_y: int,
    block_M: int, block_N: int, padded_x: int, padded_y: int,
) -> torch.Tensor:
    num_batches = x_contig.shape[0]
    dtype_str = _DTYPE_TO_STR[x_contig.dtype]
    need_pad = padded_x != shape_x or padded_y != shape_y
    if need_pad:
        x_padded = torch.zeros((num_batches, padded_x, padded_y), dtype=x_contig.dtype, device=x_contig.device)
        x_padded[:, :shape_x, :shape_y] = x_contig
        x_contig = x_padded

    m_blocks = padded_x // block_M
    n_blocks = padded_y // block_N
    total_blocks = num_batches * m_blocks * n_blocks
    use_serial_grid = total_blocks > GRID_THRESHOLD

    # 核心修改：利用 _get_compiled_kernel 从全局缓存中拉取已编译好的 Module，阻断内存膨胀
    if x_contig.dtype in (torch.bfloat16, torch.float16):
        if use_serial_grid:
            kernel = _batched_transpose_kernel_db_serial(num_batches, padded_x, padded_y, block_M, block_N, dtype_str)
        else:
            kernel = _batched_transpose_kernel_db(num_batches, padded_x, padded_y, block_M, block_N, dtype_str)
    elif x_contig.dtype == torch.float32:
        if use_serial_grid:
            kernel = _batched_transpose_kernel_fp32_serial(num_batches, padded_x, padded_y, block_M, block_N, dtype_str)
        else:
            kernel = _batched_transpose_kernel_fp32_vector(num_batches, padded_x, padded_y, block_M, block_N, dtype_str)
    elif x_contig.dtype in (torch.float8_e4m3fn, torch.uint8):
        kernel = _batched_transpose_kernel_8bit(num_batches, padded_x, padded_y, block_M, block_N, dtype_str)
    else:
        raise NotImplementedError(f"Unsupported data type: {x_contig.dtype}")

    result = kernel(x_contig)
    if need_pad:
        result = result[:, :shape_y, :shape_x].contiguous()
    return result


def transpose(x: torch.Tensor) -> torch.Tensor:
    x = x.unsqueeze(0)
    out = batched_transpose(x)
    return out.squeeze(0)


def batched_transpose(x: torch.Tensor) -> torch.Tensor:
    assert x.dim() == 3, f"Expected 3D tensor, got {x.dim()}D"
    num_batches, shape_x, shape_y = x.shape
    orig_dtype = x.dtype
    if num_batches == 0 or shape_x == 0 or shape_y == 0:
        return torch.empty((num_batches, shape_y, shape_x), dtype=orig_dtype, device=x.device)
    x_contig = x.contiguous()
    return _run_kernel(x_contig, shape_x, shape_y)

def _main_assert_equal(actual: torch.Tensor, expected: torch.Tensor, case_name: str) -> None:
    actual_cpu = actual.detach().cpu()
    if actual_cpu.shape != expected.shape:
        raise AssertionError(
            f"{case_name}: shape mismatch actual={tuple(actual_cpu.shape)} expected={tuple(expected.shape)}"
        )
    if actual_cpu.dtype != expected.dtype:
        raise AssertionError(
            f"{case_name}: dtype mismatch actual={actual_cpu.dtype} expected={expected.dtype}"
        )
    if actual_cpu.dtype == torch.float32:
        try:
            torch.testing.assert_close(actual_cpu, expected, atol=1e-6, rtol=1e-6)
            return
        except AssertionError as err:
            diff = (actual_cpu - expected).abs()
            max_diff = diff.max().item() if diff.numel() else 0.0
            mean_diff = diff.mean().item() if diff.numel() else 0.0
            flat_idx = int(diff.reshape(-1).argmax().item()) if diff.numel() else 0
            max_idx = tuple(int(v) for v in torch.unravel_index(torch.tensor(flat_idx), diff.shape)) if diff.numel() else None
            raise AssertionError(
                f"{case_name}: value mismatch max_diff={max_diff:.8e} "
                f"mean_diff={mean_diff:.8e} max_idx={max_idx}; {err}"
            ) from err

    if not torch.equal(actual_cpu, expected):
        diff = (actual_cpu.to(torch.float32) - expected.to(torch.float32)).abs()
        max_diff = diff.max().item() if diff.numel() else 0.0
        mean_diff = diff.mean().item() if diff.numel() else 0.0
        flat_idx = int(diff.reshape(-1).argmax().item()) if diff.numel() else 0
        max_idx = tuple(int(v) for v in torch.unravel_index(torch.tensor(flat_idx), diff.shape)) if diff.numel() else None
        raise AssertionError(
            f"{case_name}: value mismatch max_diff={max_diff:.8e} "
            f"mean_diff={mean_diff:.8e} max_idx={max_idx}"
        )


if __name__ == "__main__":
    import os as _os

    _os.environ.setdefault("TK_DEVICE", "npu")
    _os.environ.setdefault("TILELANG_PRINT_ON_COMPILATION", "0")

    _cases = [
        (torch.bfloat16, 576, 8, 8064),
        (torch.float32, 2048, 32, 8064),
        (torch.bfloat16, 3072, 8, 8064),
    ]

    _device_id = int(_os.environ.get("ASCEND_DEVICE_ID", "3"))
    _device = f"npu:{_device_id}"
    if hasattr(torch, "npu"):
        torch.npu.set_device(_device_id)

    torch.manual_seed(42)

    for _dtype, _hidden, _experts, _num_tokens in _cases:
        _case_name = (
            f"dtype={_dtype},hidden={_hidden},experts={_experts},"
            f"num_tokens={_num_tokens}"
        )
        print(f"[batched_transpose __main__] running {_case_name}", flush=True)
        _x = torch.randn((_experts, _num_tokens, _hidden), dtype=torch.bfloat16, device=_device)
        if _dtype != torch.bfloat16:
            _x = _x.to(_dtype)

        _y = batched_transpose(_x)
        _y_ref = torch.transpose(_x.cpu(), 1, 2).contiguous()
        _main_assert_equal(_y, _y_ref, _case_name)

        if hasattr(torch, "npu"):
            torch.npu.synchronize()
        print(f"[batched_transpose __main__] test case passed", flush=True)

    print("test PASSED!", flush=True)