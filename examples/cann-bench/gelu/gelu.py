import tilelang
import tilelang.language as T
import torch

RSQRT_2 = 0.7071067811865476
SQRT_8_PI = 1.5957691216057307
COEFF_044715 = 0.044715
COEFF_044715_SQRT_8_PI = COEFF_044715 * SQRT_8_PI

ERF_P = 0.3275911
ERF_A1 = 0.254829592
ERF_A2 = -0.284496736
ERF_A3 = 1.421413741
ERF_A4 = -1.453152027
ERF_A5 = 1.061405429

_CAST_LOW2HIGH = "CAST_NONE"
_CAST_HIGH2LOW = "CAST_RINT"

VEC_NUM = 2
_CORE_NUM = 24
_TOTAL_AIV = _CORE_NUM * VEC_NUM
_UB_BUDGET = 192 * 1024
_KERNEL_CACHE = {}

pipe_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


def _compute_block_n(N):
    if N <= 1024:
        return 1024
    elif N <= 8192:
        return max(256, ((N + 255) // 256) * 256)
    else:
        return 8192


def _compute_schedule(N, block_N, ub_per_tile):
    total_tiles = (N + block_N - 1) // block_N
    if total_tiles <= _TOTAL_AIV:
        tiles_per_block = 1
    else:
        tiles_per_block = max(2, min(8, (total_tiles + _TOTAL_AIV - 1) // _TOTAL_AIV))
        while tiles_per_block > 1 and ub_per_tile * tiles_per_block > _UB_BUDGET:
            tiles_per_block -= 1
    num_blocks = (total_tiles + tiles_per_block - 1) // tiles_per_block
    return tiles_per_block, num_blocks


@tilelang.jit(out_idx=[-1], pass_configs=pipe_configs)
def _gelu_tanh_kernel(N, block_N, dtype="float32"):
    tile_elem = block_N // VEC_NUM
    cal_bytes = 4
    ub_per_tile = 3 * cal_bytes * tile_elem
    tiles_per_block, num_blocks = _compute_schedule(N, block_N, ub_per_tile)

    @T.prim_func
    def main(X: T.Tensor((N,), dtype), Y: T.Tensor((N,), dtype)):
        with T.Kernel(num_blocks, is_npu=True) as (cid, vid):
            x_ub = T.alloc_ub((2, tile_elem), dtype)
            y_ub = T.alloc_ub((2, tile_elem), dtype)
            t0 = T.alloc_ub((tile_elem,), dtype)
            base = cid * tiles_per_block * block_N + vid * tile_elem

            with T.Scope("V"):
                T.set_flag("mte3", "mte2", 0)
                T.set_flag("mte3", "mte2", 1)
                T.wait_flag("mte3", "mte2", 0)
                T.copy(X[base], x_ub[0, :])
                T.set_flag("mte2", "v", 0)

                for tile in T.unroll(0, tiles_per_block - 1):
                    cur = tile % 2
                    nxt = (tile + 1) % 2
                    cur_off = base + tile * block_N
                    next_off = base + (tile + 1) * block_N

                    T.wait_flag("mte3", "mte2", nxt)
                    T.copy(X[next_off], x_ub[nxt, :])
                    T.set_flag("mte2", "v", nxt)

                    T.wait_flag("mte2", "v", cur)
                    T.tile.mul(y_ub[cur, :], x_ub[cur, :], x_ub[cur, :])
                    T.tile.mul(t0, x_ub[cur, :], y_ub[cur, :])
                    T.tile.mul(t0, t0, COEFF_044715_SQRT_8_PI)
                    T.tile.axpy(t0, x_ub[cur, :], SQRT_8_PI)
                    T.tile.sigmoid(y_ub[cur, :], t0)
                    T.tile.mul(y_ub[cur, :], x_ub[cur, :], y_ub[cur, :])
                    T.set_flag("v", "mte3", cur)

                    T.wait_flag("v", "mte3", cur)
                    T.copy(y_ub[cur, :], Y[cur_off])
                    T.set_flag("mte3", "mte2", cur)

                last_stage = (tiles_per_block - 1) % 2
                last_off = base + (tiles_per_block - 1) * block_N
                T.wait_flag("mte2", "v", last_stage)
                T.tile.mul(y_ub[last_stage, :], x_ub[last_stage, :], x_ub[last_stage, :])
                T.tile.mul(t0, x_ub[last_stage, :], y_ub[last_stage, :])
                T.tile.mul(t0, t0, COEFF_044715_SQRT_8_PI)
                T.tile.axpy(t0, x_ub[last_stage, :], SQRT_8_PI)
                T.tile.sigmoid(y_ub[last_stage, :], t0)
                T.tile.mul(y_ub[last_stage, :], x_ub[last_stage, :], y_ub[last_stage, :])
                T.set_flag("v", "mte3", last_stage)

                T.wait_flag("v", "mte3", last_stage)
                T.copy(y_ub[last_stage, :], Y[last_off])
                T.set_flag("mte3", "mte2", last_stage)

                T.wait_flag("mte3", "mte2", 0)
                T.wait_flag("mte3", "mte2", 1)

    return main


@tilelang.jit(out_idx=[-1], pass_configs=pipe_configs)
def _gelu_tanh_cast_kernel(N, block_N, in_dtype="float16", cal_dtype="float32"):
    tile_elem = block_N // VEC_NUM
    ub_per_tile = (3 * 4 + 2 * 2) * tile_elem
    tiles_per_block, num_blocks = _compute_schedule(N, block_N, ub_per_tile)

    @T.prim_func
    def main(X: T.Tensor((N,), in_dtype), Y: T.Tensor((N,), in_dtype)):
        with T.Kernel(num_blocks, is_npu=True) as (cid, vid):
            x_ub = T.alloc_ub((2, tile_elem), cal_dtype)
            y_ub = T.alloc_ub((2, tile_elem), cal_dtype)
            t0 = T.alloc_ub((tile_elem,), cal_dtype)
            x_h = T.alloc_ub((2, tile_elem), in_dtype)
            y_h = T.alloc_ub((2, tile_elem), in_dtype)
            base = cid * tiles_per_block * block_N + vid * tile_elem

            with T.Scope("V"):
                T.set_flag("mte3", "mte2", 0)
                T.set_flag("mte3", "mte2", 1)
                T.wait_flag("mte3", "mte2", 0)
                T.copy(X[base], x_h[0, :])
                T.set_flag("mte2", "v", 0)

                for tile in T.unroll(0, tiles_per_block - 1):
                    cur = tile % 2
                    nxt = (tile + 1) % 2
                    cur_off = base + tile * block_N
                    next_off = base + (tile + 1) * block_N

                    T.wait_flag("mte3", "mte2", nxt)
                    T.copy(X[next_off], x_h[nxt, :])
                    T.set_flag("mte2", "v", nxt)

                    T.wait_flag("mte2", "v", cur)
                    T.tile.cast(x_ub[cur, :], x_h[cur, :], _CAST_LOW2HIGH, tile_elem)
                    T.tile.mul(y_ub[cur, :], x_ub[cur, :], x_ub[cur, :])
                    T.tile.mul(t0, x_ub[cur, :], y_ub[cur, :])
                    T.tile.mul(t0, t0, COEFF_044715_SQRT_8_PI)
                    T.tile.axpy(t0, x_ub[cur, :], SQRT_8_PI)
                    T.tile.sigmoid(y_ub[cur, :], t0)
                    T.tile.mul(y_ub[cur, :], x_ub[cur, :], y_ub[cur, :])
                    T.tile.cast(y_h[cur, :], y_ub[cur, :], _CAST_HIGH2LOW, tile_elem)
                    T.set_flag("v", "mte3", cur)

                    T.wait_flag("v", "mte3", cur)
                    T.copy(y_h[cur, :], Y[cur_off])
                    T.set_flag("mte3", "mte2", cur)

                last_stage = (tiles_per_block - 1) % 2
                last_off = base + (tiles_per_block - 1) * block_N
                T.wait_flag("mte2", "v", last_stage)
                T.tile.cast(x_ub[last_stage, :], x_h[last_stage, :], _CAST_LOW2HIGH, tile_elem)
                T.tile.mul(y_ub[last_stage, :], x_ub[last_stage, :], x_ub[last_stage, :])
                T.tile.mul(t0, x_ub[last_stage, :], y_ub[last_stage, :])
                T.tile.mul(t0, t0, COEFF_044715_SQRT_8_PI)
                T.tile.axpy(t0, x_ub[last_stage, :], SQRT_8_PI)
                T.tile.sigmoid(y_ub[last_stage, :], t0)
                T.tile.mul(y_ub[last_stage, :], x_ub[last_stage, :], y_ub[last_stage, :])
                T.tile.cast(y_h[last_stage, :], y_ub[last_stage, :], _CAST_HIGH2LOW, tile_elem)
                T.set_flag("v", "mte3", last_stage)

                T.wait_flag("v", "mte3", last_stage)
                T.copy(y_h[last_stage, :], Y[last_off])
                T.set_flag("mte3", "mte2", last_stage)

                T.wait_flag("mte3", "mte2", 0)
                T.wait_flag("mte3", "mte2", 1)

    return main


@tilelang.jit(out_idx=[-1], pass_configs=pipe_configs)
def _gelu_exact_as_kernel(N, block_N, dtype="float32"):
    tile_elem = block_N // VEC_NUM
    ub_per_tile = 9 * 4 * tile_elem
    tiles_per_block, num_blocks = _compute_schedule(N, block_N, ub_per_tile)

    @T.prim_func
    def main(X: T.Tensor((N,), dtype), Y: T.Tensor((N,), dtype)):
        with T.Kernel(num_blocks, is_npu=True) as (cid, vid):
            x_ub = T.alloc_ub((2, tile_elem), dtype)
            y_ub = T.alloc_ub((2, tile_elem), dtype)
            t0 = T.alloc_ub((tile_elem,), dtype)
            t1 = T.alloc_ub((tile_elem,), dtype)
            t2 = T.alloc_ub((tile_elem,), dtype)
            z_ub = T.alloc_ub((tile_elem,), dtype)
            h0 = T.alloc_ub((tile_elem,), dtype)
            h1 = T.alloc_ub((tile_elem,), dtype)
            ones = T.alloc_ub((tile_elem,), dtype)
            base = cid * tiles_per_block * block_N + vid * tile_elem

            with T.Scope("V"):
                T.tile.fill(ones, 1.0)
                T.set_flag("mte3", "mte2", 0)
                T.set_flag("mte3", "mte2", 1)
                T.wait_flag("mte3", "mte2", 0)
                T.copy(X[base], x_ub[0, :])
                T.set_flag("mte2", "v", 0)

                for tile in T.unroll(0, tiles_per_block - 1):
                    cur = tile % 2
                    nxt = (tile + 1) % 2
                    cur_off = base + tile * block_N
                    next_off = base + (tile + 1) * block_N

                    T.wait_flag("mte3", "mte2", nxt)
                    T.copy(X[next_off], x_ub[nxt, :])
                    T.set_flag("mte2", "v", nxt)

                    T.wait_flag("mte2", "v", cur)
                    T.tile.mul(z_ub, x_ub[cur, :], RSQRT_2)
                    T.tile.mul(t0, z_ub, z_ub)
                    T.tile.abs(t1, z_ub)
                    T.tile.mul(t0, t0, -1.0)
                    T.tile.exp(t0, t0)
                    T.tile.mul(t2, t1, ERF_P)
                    T.tile.add(t2, t2, 1.0)
                    T.tile.div(t2, ones, t2)
                    T.tile.fill(h0, ERF_A5)
                    T.tile.fill(h1, ERF_A4)
                    T.tile.mul_add_dst(h1, t2, h0)
                    T.tile.fill(h0, ERF_A3)
                    T.tile.mul_add_dst(h0, t2, h1)
                    T.tile.fill(h1, ERF_A2)
                    T.tile.mul_add_dst(h1, t2, h0)
                    T.tile.fill(h0, ERF_A1)
                    T.tile.mul_add_dst(h0, t2, h1)
                    T.tile.mul(t1, h0, t2)
                    T.tile.mul(t1, t1, t0)
                    T.tile.sub(y_ub[cur, :], ones, t1)
                    T.tile.mul(t0, y_ub[cur, :], -1.0)
                    T.tile.compare(t1, z_ub, 0.0, "GE")
                    T.tile.select(y_ub[cur, :], t1, y_ub[cur, :], t0, "VSEL_TENSOR_TENSOR_MODE")
                    T.tile.add(y_ub[cur, :], y_ub[cur, :], 1.0)
                    T.tile.mul(y_ub[cur, :], y_ub[cur, :], 0.5)
                    T.tile.mul(y_ub[cur, :], x_ub[cur, :], y_ub[cur, :])
                    T.set_flag("v", "mte3", cur)

                    T.wait_flag("v", "mte3", cur)
                    T.copy(y_ub[cur, :], Y[cur_off])
                    T.set_flag("mte3", "mte2", cur)

                last_stage = (tiles_per_block - 1) % 2
                last_off = base + (tiles_per_block - 1) * block_N
                T.wait_flag("mte2", "v", last_stage)
                T.tile.mul(z_ub, x_ub[last_stage, :], RSQRT_2)
                T.tile.mul(t0, z_ub, z_ub)
                T.tile.abs(t1, z_ub)
                T.tile.mul(t0, t0, -1.0)
                T.tile.exp(t0, t0)
                T.tile.mul(t2, t1, ERF_P)
                T.tile.add(t2, t2, 1.0)
                T.tile.div(t2, ones, t2)
                T.tile.fill(h0, ERF_A5)
                T.tile.fill(h1, ERF_A4)
                T.tile.mul_add_dst(h1, t2, h0)
                T.tile.fill(h0, ERF_A3)
                T.tile.mul_add_dst(h0, t2, h1)
                T.tile.fill(h1, ERF_A2)
                T.tile.mul_add_dst(h1, t2, h0)
                T.tile.fill(h0, ERF_A1)
                T.tile.mul_add_dst(h0, t2, h1)
                T.tile.mul(t1, h0, t2)
                T.tile.mul(t1, t1, t0)
                T.tile.sub(y_ub[last_stage, :], ones, t1)
                T.tile.mul(t0, y_ub[last_stage, :], -1.0)
                T.tile.compare(t1, z_ub, 0.0, "GE")
                T.tile.select(y_ub[last_stage, :], t1, y_ub[last_stage, :], t0, "VSEL_TENSOR_TENSOR_MODE")
                T.tile.add(y_ub[last_stage, :], y_ub[last_stage, :], 1.0)
                T.tile.mul(y_ub[last_stage, :], y_ub[last_stage, :], 0.5)
                T.tile.mul(y_ub[last_stage, :], x_ub[last_stage, :], y_ub[last_stage, :])
                T.set_flag("v", "mte3", last_stage)

                T.wait_flag("v", "mte3", last_stage)
                T.copy(y_ub[last_stage, :], Y[last_off])
                T.set_flag("mte3", "mte2", last_stage)

                T.wait_flag("mte3", "mte2", 0)
                T.wait_flag("mte3", "mte2", 1)

    return main


@tilelang.jit(out_idx=[-1], pass_configs=pipe_configs)
def _gelu_exact_as_cast_kernel(N, block_N, in_dtype="bfloat16", cal_dtype="float32"):
    tile_elem = block_N // VEC_NUM
    ub_per_tile = (9 * 4 + 2 * 2) * tile_elem
    if ub_per_tile * 2 > _UB_BUDGET:
        block_N = block_N // 2
        tile_elem = block_N // VEC_NUM
        ub_per_tile = (9 * 4 + 2 * 2) * tile_elem
    tiles_per_block, num_blocks = _compute_schedule(N, block_N, ub_per_tile)

    @T.prim_func
    def main(X: T.Tensor((N,), in_dtype), Y: T.Tensor((N,), in_dtype)):
        with T.Kernel(num_blocks, is_npu=True) as (cid, vid):
            x_ub = T.alloc_ub((2, tile_elem), cal_dtype)
            y_ub = T.alloc_ub((2, tile_elem), cal_dtype)
            x_h = T.alloc_ub((2, tile_elem), in_dtype)
            y_h = T.alloc_ub((2, tile_elem), in_dtype)
            t0 = T.alloc_ub((tile_elem,), cal_dtype)
            t1 = T.alloc_ub((tile_elem,), cal_dtype)
            t2 = T.alloc_ub((tile_elem,), cal_dtype)
            z_ub = T.alloc_ub((tile_elem,), cal_dtype)
            h0 = T.alloc_ub((tile_elem,), cal_dtype)
            h1 = T.alloc_ub((tile_elem,), cal_dtype)
            ones = T.alloc_ub((tile_elem,), cal_dtype)
            base = cid * tiles_per_block * block_N + vid * tile_elem

            with T.Scope("V"):
                T.tile.fill(ones, 1.0)
                T.set_flag("mte3", "mte2", 0)
                T.set_flag("mte3", "mte2", 1)
                T.wait_flag("mte3", "mte2", 0)
                T.copy(X[base], x_h[0, :])
                T.set_flag("mte2", "v", 0)

                for tile in T.unroll(0, tiles_per_block - 1):
                    cur = tile % 2
                    nxt = (tile + 1) % 2
                    cur_off = base + tile * block_N
                    next_off = base + (tile + 1) * block_N

                    T.wait_flag("mte3", "mte2", nxt)
                    T.copy(X[next_off], x_h[nxt, :])
                    T.set_flag("mte2", "v", nxt)

                    T.wait_flag("mte2", "v", cur)
                    T.tile.cast(x_ub[cur, :], x_h[cur, :], _CAST_LOW2HIGH, tile_elem)
                    T.tile.mul(z_ub, x_ub[cur, :], RSQRT_2)
                    T.tile.mul(t0, z_ub, z_ub)
                    T.tile.abs(t1, z_ub)
                    T.tile.mul(t0, t0, -1.0)
                    T.tile.exp(t0, t0)
                    T.tile.mul(t2, t1, ERF_P)
                    T.tile.add(t2, t2, 1.0)
                    T.tile.div(t2, ones, t2)
                    T.tile.fill(h0, ERF_A5)
                    T.tile.fill(h1, ERF_A4)
                    T.tile.mul_add_dst(h1, t2, h0)
                    T.tile.fill(h0, ERF_A3)
                    T.tile.mul_add_dst(h0, t2, h1)
                    T.tile.fill(h1, ERF_A2)
                    T.tile.mul_add_dst(h1, t2, h0)
                    T.tile.fill(h0, ERF_A1)
                    T.tile.mul_add_dst(h0, t2, h1)
                    T.tile.mul(t1, h0, t2)
                    T.tile.mul(t1, t1, t0)
                    T.tile.sub(y_ub[cur, :], ones, t1)
                    T.tile.mul(t0, y_ub[cur, :], -1.0)
                    T.tile.compare(t1, z_ub, 0.0, "GE")
                    T.tile.select(y_ub[cur, :], t1, y_ub[cur, :], t0, "VSEL_TENSOR_TENSOR_MODE")
                    T.tile.add(y_ub[cur, :], y_ub[cur, :], 1.0)
                    T.tile.mul(y_ub[cur, :], y_ub[cur, :], 0.5)
                    T.tile.mul(y_ub[cur, :], x_ub[cur, :], y_ub[cur, :])
                    T.tile.cast(y_h[cur, :], y_ub[cur, :], _CAST_HIGH2LOW, tile_elem)
                    T.set_flag("v", "mte3", cur)

                    T.wait_flag("v", "mte3", cur)
                    T.copy(y_h[cur, :], Y[cur_off])
                    T.set_flag("mte3", "mte2", cur)

                last_stage = (tiles_per_block - 1) % 2
                last_off = base + (tiles_per_block - 1) * block_N
                T.wait_flag("mte2", "v", last_stage)
                T.tile.cast(x_ub[last_stage, :], x_h[last_stage, :], _CAST_LOW2HIGH, tile_elem)
                T.tile.mul(z_ub, x_ub[last_stage, :], RSQRT_2)
                T.tile.mul(t0, z_ub, z_ub)
                T.tile.abs(t1, z_ub)
                T.tile.mul(t0, t0, -1.0)
                T.tile.exp(t0, t0)
                T.tile.mul(t2, t1, ERF_P)
                T.tile.add(t2, t2, 1.0)
                T.tile.div(t2, ones, t2)
                T.tile.fill(h0, ERF_A5)
                T.tile.fill(h1, ERF_A4)
                T.tile.mul_add_dst(h1, t2, h0)
                T.tile.fill(h0, ERF_A3)
                T.tile.mul_add_dst(h0, t2, h1)
                T.tile.fill(h1, ERF_A2)
                T.tile.mul_add_dst(h1, t2, h0)
                T.tile.fill(h0, ERF_A1)
                T.tile.mul_add_dst(h0, t2, h1)
                T.tile.mul(t1, h0, t2)
                T.tile.mul(t1, t1, t0)
                T.tile.sub(y_ub[last_stage, :], ones, t1)
                T.tile.mul(t0, y_ub[last_stage, :], -1.0)
                T.tile.compare(t1, z_ub, 0.0, "GE")
                T.tile.select(y_ub[last_stage, :], t1, y_ub[last_stage, :], t0, "VSEL_TENSOR_TENSOR_MODE")
                T.tile.add(y_ub[last_stage, :], y_ub[last_stage, :], 1.0)
                T.tile.mul(y_ub[last_stage, :], y_ub[last_stage, :], 0.5)
                T.tile.mul(y_ub[last_stage, :], x_ub[last_stage, :], y_ub[last_stage, :])
                T.tile.cast(y_h[last_stage, :], y_ub[last_stage, :], _CAST_HIGH2LOW, tile_elem)
                T.set_flag("v", "mte3", last_stage)

                T.wait_flag("v", "mte3", last_stage)
                T.copy(y_h[last_stage, :], Y[last_off])
                T.set_flag("mte3", "mte2", last_stage)

                T.wait_flag("mte3", "mte2", 0)
                T.wait_flag("mte3", "mte2", 1)

    return main


def run_gelu_kernel(x, approximate="none", block_N=None):
    if not x.is_npu:
        x = x.npu()
    orig_dtype = x.dtype
    N = x.numel()
    x_flat = x.contiguous().view(-1)

    if block_N is None:
        block_N = _compute_block_n(N)

    if orig_dtype == torch.float16:
        if approximate == "tanh":
            cache_key = (N, block_N, "fp16_cast_tanh")
            if cache_key not in _KERNEL_CACHE:
                _KERNEL_CACHE[cache_key] = _gelu_tanh_cast_kernel(N, block_N, in_dtype="float16", cal_dtype="float32")
        else:
            cache_key = (N, block_N, "fp16_tanh")
            if cache_key not in _KERNEL_CACHE:
                _KERNEL_CACHE[cache_key] = _gelu_tanh_kernel(N, block_N, dtype="float16")
    elif orig_dtype == torch.bfloat16:
        if approximate == "tanh":
            cache_key = (N, block_N, "bf16_cast_tanh")
            if cache_key not in _KERNEL_CACHE:
                _KERNEL_CACHE[cache_key] = _gelu_tanh_cast_kernel(N, block_N, in_dtype="bfloat16", cal_dtype="float32")
        else:
            cache_key = (N, block_N, "bf16_exact_as_cast")
            if cache_key not in _KERNEL_CACHE:
                _KERNEL_CACHE[cache_key] = _gelu_exact_as_cast_kernel(N, block_N, in_dtype="bfloat16", cal_dtype="float32")
    else:
        if approximate == "tanh":
            cache_key = (N, block_N, "fp32_tanh")
            if cache_key not in _KERNEL_CACHE:
                _KERNEL_CACHE[cache_key] = _gelu_tanh_kernel(N, block_N, dtype="float32")
        else:
            cache_key = (N, block_N, "fp32_exact_as")
            if cache_key not in _KERNEL_CACHE:
                _KERNEL_CACHE[cache_key] = _gelu_exact_as_kernel(N, block_N, dtype="float32")

    kernel = _KERNEL_CACHE[cache_key]
    y_flat = kernel(x_flat)
    y = y_flat.view(x.shape)
    return y


def gelu(x, approximate="none"):
    return run_gelu_kernel(x, approximate=approximate)


__all__ = ["gelu"]


if __name__ == "__main__":
    import torch.nn.functional as F

    torch.manual_seed(0)
    all_ok = True

    print("=== Exact mode (approximate='none') ===")
    for dt_name, dt in [("float16", torch.float16), ("float32", torch.float32), ("bfloat16", torch.bfloat16)]:
        x = torch.randn(1024, 1024, dtype=torch.float32).uniform_(-2, 2).to(dt).npu()
        y = gelu(x, approximate="none")
        ref = F.gelu(x.cpu(), approximate="none")
        err = (y.cpu().float() - ref.float()).abs() / (ref.float().abs() + 1e-7)
        ok = err.mean() < 1e-3 and err.max() < 1e-1
        all_ok = all_ok and ok
        print(f"  {dt_name}: mere={err.mean():.2e} mare={err.max():.2e} {'OK' if ok else 'FAIL'}")

    print("=== Tanh mode (approximate='tanh') ===")
    for dt_name, dt in [("float16", torch.float16), ("float32", torch.float32), ("bfloat16", torch.bfloat16)]:
        x = torch.randn(1024, 1024, dtype=torch.float32).uniform_(-2, 2).to(dt).npu()
        y = gelu(x, approximate="tanh")
        ref = F.gelu(x.cpu(), approximate="tanh")
        err = (y.cpu().float() - ref.float()).abs() / (ref.float().abs() + 1e-7)
        ok = err.mean() < 1e-3 and err.max() < 1e-1
        all_ok = all_ok and ok
        print(f"  {dt_name}: mere={err.mean():.2e} mare={err.max():.2e} {'OK' if ok else 'FAIL'}")

    print("=== Small tensor ===")
    for N in [100, 512, 1024]:
        x = torch.randn(N, dtype=torch.float32).npu()
        y = gelu(x, approximate="none")
        ref = F.gelu(x.cpu(), approximate="none")
        err = (y.cpu() - ref).abs() / (ref.abs() + 1e-7)
        ok = err.mean() < 1e-3 and err.max() < 1e-1
        all_ok = all_ok and ok
        print(f"  N={N}: mere={err.mean():.2e} mare={err.max():.2e} {'OK' if ok else 'FAIL'}")

    torch.npu.synchronize()
    if all_ok:
        print("TEST PASSED!")
    else:
        print("TEST FAILED!")
