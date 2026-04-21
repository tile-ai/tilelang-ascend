import tilelang
import tilelang.language as tl
import torch

tilelang.cache.clear_cache()

pass_configs = {tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True}


@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def chunk_local_cumsum_scalar_kernel(B, H, SEQ_LEN, BT, reverse=False, head_first=False, dtype="float"):
    chunk_num = tl.ceildiv(SEQ_LEN, BT)
    VEC_NUM = 2

    shape = (B, H, SEQ_LEN) if head_first else (B, SEQ_LEN, H)

    @tl.prim_func
    def main(
        s: tl.Tensor(shape, dtype),
        o: tl.Tensor(shape, dtype),
    ):
        with tl.Kernel(chunk_num * B * (H // VEC_NUM), is_npu=True) as (cid, vid):
            i_t = cid % chunk_num
            i_bh = cid // chunk_num
            i_b = i_bh // (H // VEC_NUM)
            i_h = (i_bh % (H // VEC_NUM)) * VEC_NUM + vid

            b_s = tl.alloc_ub([BT], dtype)
            b_o = tl.alloc_ub([BT], dtype)
            total_buf = tl.alloc_ub([1], dtype)

            with tl.Scope("V"):
                tl.tile.fill(b_o, 0.0)

                if head_first:
                    tl.copy(s[i_b, i_h, i_t * BT], b_s)
                else:
                    tl.copy(s[i_b, i_t * BT, i_h], b_s)

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

                if head_first:
                    tl.copy(b_o, o[i_b, i_h, i_t * BT])
                else:
                    tl.copy(b_o, o[i_b, i_t * BT, i_h])

    return main


@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def chunk_global_cumsum_scalar_kernel(B, H, SEQ_LEN, BT, reverse=False, head_first=False, dtype="float"):
    chunk_num = tl.ceildiv(SEQ_LEN, BT)

    shape = (B, H, SEQ_LEN) if head_first else (B, SEQ_LEN, H)

    @tl.prim_func
    def main(
        s: tl.Tensor(shape, dtype),
        o: tl.Tensor(shape, dtype),
    ):
        with tl.Kernel(B * H, is_npu=True) as (cid, vid):
            i_b = cid // H
            i_h = cid % H

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

                    if head_first:
                        tl.copy(s[i_b, i_h, i_t * BT], b_s)
                    else:
                        tl.copy(s[i_b, i_t * BT, i_h], b_s)

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

                    if head_first:
                        tl.copy(b_o, o[i_b, i_h, i_t * BT])
                    else:
                        tl.copy(b_o, o[i_b, i_t * BT, i_h])

                    carry[0] = carry[0] + b_ss_buf[0]

    return main


@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def chunk_local_cumsum_vector_kernel(B, H, SEQ_LEN, S_DIM, BT, BS, reverse=False, head_first=False, dtype="float"):
    chunk_num = tl.ceildiv(SEQ_LEN, BT)
    s_block_num = tl.ceildiv(S_DIM, BS)
    VEC_NUM = 2

    shape = (B, H, SEQ_LEN, S_DIM) if head_first else (B, SEQ_LEN, H, S_DIM)

    @tl.prim_func
    def main(
        s: tl.Tensor(shape, dtype),
        o: tl.Tensor(shape, dtype),
    ):
        with tl.Kernel(s_block_num * chunk_num * B * (H // VEC_NUM), is_npu=True) as (cid, vid):
            i_s = cid % s_block_num
            i_t = (cid // s_block_num) % chunk_num
            i_bh = cid // (s_block_num * chunk_num)
            i_b = i_bh // (H // VEC_NUM)
            i_h = (i_bh % (H // VEC_NUM)) * VEC_NUM + vid

            b_s = tl.alloc_ub([BT, BS], dtype)
            b_o = tl.alloc_ub([BT, BS], dtype)
            total_buf = tl.alloc_ub([BS], dtype)

            with tl.Scope("V"):
                tl.tile.fill(b_o, 0.0)

                if head_first:
                    tl.copy(s[i_b, i_h, i_t * BT, i_s * BS], b_s)
                else:
                    tl.copy(s[i_b, i_t * BT, i_h, i_s * BS], b_s)

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

                if head_first:
                    tl.copy(b_o, o[i_b, i_h, i_t * BT, i_s * BS])
                else:
                    tl.copy(b_o, o[i_b, i_t * BT, i_h, i_s * BS])

    return main


@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def chunk_global_cumsum_vector_kernel(B, H, SEQ_LEN, S_DIM, BT, BS, reverse=False, head_first=False, dtype="float"):
    chunk_num = tl.ceildiv(SEQ_LEN, BT)
    s_block_num = tl.ceildiv(S_DIM, BS)

    shape = (B, H, SEQ_LEN, S_DIM) if head_first else (B, SEQ_LEN, H, S_DIM)

    @tl.prim_func
    def main(
        s: tl.Tensor(shape, dtype),
        o: tl.Tensor(shape, dtype),
    ):
        with tl.Kernel(s_block_num * B * H, is_npu=True) as (cid, vid):
            i_s = cid % s_block_num
            i_nh = cid // s_block_num
            i_b = i_nh // H
            i_h = i_nh % H

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

                    if head_first:
                        tl.copy(s[i_b, i_h, i_t * BT, i_s * BS], b_s)
                    else:
                        tl.copy(s[i_b, i_t * BT, i_h, i_s * BS], b_s)

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

                    if head_first:
                        tl.copy(b_o, o[i_b, i_h, i_t * BT, i_s * BS])
                    else:
                        tl.copy(b_o, o[i_b, i_t * BT, i_h, i_s * BS])

                    for j in range(BS):
                        carry[j] = carry[j] + b_ss_buf[j]

    return main


def chunk_local_cumsum_scalar(s, chunk_size, reverse=False, scale=None, head_first=False, output_dtype=torch.float32):
    if head_first:
        B, H, SEQ_LEN = s.shape
    else:
        B, SEQ_LEN, H = s.shape

    assert chunk_size == 2 ** (chunk_size.bit_length() - 1), "chunk_size must be a power of 2"

    kernel = chunk_local_cumsum_scalar_kernel(B, H, SEQ_LEN, chunk_size, reverse=reverse, head_first=head_first, dtype="float")
    o = kernel(s)

    if scale is not None:
        o = o * scale

    return o.to(output_dtype)


def chunk_global_cumsum_scalar(s, reverse=False, scale=None, head_first=False, output_dtype=torch.float32):
    if head_first:
        B, H, SEQ_LEN = s.shape
    else:
        B, SEQ_LEN, H = s.shape

    BT = 64

    kernel = chunk_global_cumsum_scalar_kernel(B, H, SEQ_LEN, BT, reverse=reverse, head_first=head_first, dtype="float")
    o = kernel(s)

    if scale is not None:
        o = o * scale

    return o.to(output_dtype)


def chunk_local_cumsum_vector(s, chunk_size, reverse=False, scale=None, head_first=False, output_dtype=torch.float32):
    if head_first:
        B, H, SEQ_LEN, S_DIM = s.shape
    else:
        B, SEQ_LEN, H, S_DIM = s.shape

    assert chunk_size == 2 ** (chunk_size.bit_length() - 1), "chunk_size must be a power of 2"

    BS = min(32, 2 ** (S_DIM.bit_length() - 1)) if S_DIM > 0 else 1

    kernel = chunk_local_cumsum_vector_kernel(B, H, SEQ_LEN, S_DIM, chunk_size, BS, reverse=reverse, head_first=head_first, dtype="float")
    o = kernel(s)

    if scale is not None:
        o = o * scale

    return o.to(output_dtype)


def chunk_global_cumsum_vector(s, reverse=False, scale=None, head_first=False, output_dtype=torch.float32):
    if head_first:
        B, H, SEQ_LEN, S_DIM = s.shape
    else:
        B, SEQ_LEN, H, S_DIM = s.shape

    BT = 64
    BS = min(32, 2 ** (S_DIM.bit_length() - 1)) if S_DIM > 0 else 1

    kernel = chunk_global_cumsum_vector_kernel(B, H, SEQ_LEN, S_DIM, BT, BS, reverse=reverse, head_first=head_first, dtype="float")
    o = kernel(s)

    if scale is not None:
        o = o * scale

    return o.to(output_dtype)


def ref_chunk_local_cumsum_scalar(s, chunk_size, reverse=False, scale=None, head_first=False):
    if head_first:
        B, H, SEQ_LEN = s.shape
    else:
        B, SEQ_LEN, H = s.shape

    chunk_num = (SEQ_LEN + chunk_size - 1) // chunk_size

    if head_first:
        s_reshaped = s.view(B, H, chunk_num, chunk_size)
        if reverse:
            result = torch.flip(torch.cumsum(torch.flip(s_reshaped, dims=[3]), dim=3), dims=[3])
        else:
            result = torch.cumsum(s_reshaped, dim=3)
        result = result.view(B, H, SEQ_LEN)
    else:
        s_reshaped = s.view(B, chunk_num, chunk_size, H)
        if reverse:
            result = torch.flip(torch.cumsum(torch.flip(s_reshaped, dims=[2]), dim=2), dims=[2])
        else:
            result = torch.cumsum(s_reshaped, dim=2)
        result = result.view(B, SEQ_LEN, H)

    if scale is not None:
        result = result * scale

    return result.to(torch.float32)


def ref_chunk_global_cumsum_scalar(s, reverse=False, scale=None, head_first=False):
    if head_first:
        B, H, SEQ_LEN = s.shape
        if reverse:
            result = torch.flip(torch.cumsum(torch.flip(s, dims=[2]), dim=2), dims=[2])
        else:
            result = torch.cumsum(s, dim=2)
    else:
        B, SEQ_LEN, H = s.shape
        if reverse:
            result = torch.flip(torch.cumsum(torch.flip(s, dims=[1]), dim=1), dims=[1])
        else:
            result = torch.cumsum(s, dim=1)

    if scale is not None:
        result = result * scale

    return result.to(torch.float32)


def ref_chunk_local_cumsum_vector(s, chunk_size, reverse=False, scale=None, head_first=False):
    if head_first:
        B, H, SEQ_LEN, S_DIM = s.shape
        chunk_num = (SEQ_LEN + chunk_size - 1) // chunk_size
        s_reshaped = s.view(B, H, chunk_num, chunk_size, S_DIM)
        if reverse:
            result = torch.flip(torch.cumsum(torch.flip(s_reshaped, dims=[3]), dim=3), dims=[3])
        else:
            result = torch.cumsum(s_reshaped, dim=3)
        result = result.view(B, H, SEQ_LEN, S_DIM)
    else:
        B, SEQ_LEN, H, S_DIM = s.shape
        chunk_num = (SEQ_LEN + chunk_size - 1) // chunk_size
        s_reshaped = s.view(B, chunk_num, chunk_size, H, S_DIM)
        if reverse:
            result = torch.flip(torch.cumsum(torch.flip(s_reshaped, dims=[2]), dim=2), dims=[2])
        else:
            result = torch.cumsum(s_reshaped, dim=2)
        result = result.view(B, SEQ_LEN, H, S_DIM)

    if scale is not None:
        result = result * scale

    return result.to(torch.float32)


def ref_chunk_global_cumsum_vector(s, reverse=False, scale=None, head_first=False):
    if head_first:
        B, H, SEQ_LEN, S_DIM = s.shape
        if reverse:
            result = torch.flip(torch.cumsum(torch.flip(s, dims=[2]), dim=2), dims=[2])
        else:
            result = torch.cumsum(s, dim=2)
    else:
        B, SEQ_LEN, H, S_DIM = s.shape
        if reverse:
            result = torch.flip(torch.cumsum(torch.flip(s, dims=[1]), dim=1), dims=[1])
        else:
            result = torch.cumsum(s, dim=1)

    if scale is not None:
        result = result * scale

    return result.to(torch.float32)


if __name__ == "__main__":
    tilelang.cache.clear_cache()
    torch.manual_seed(0)

    print("=== Testing chunk_local_cumsum_scalar ===")

    test_configs_local_scalar = [
        (1, 8, 128, 32, False, True),
        (1, 8, 128, 32, True, True),
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

    print("\n=== Testing chunk_local_cumsum_vector (P1) ===")

    test_configs_local_vector = [
        (1, 8, 128, 16, 32, False, True),
        (1, 8, 128, 16, 32, True, True),
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

    print("\n=== Testing chunk_global_cumsum_vector (P1) ===")

    test_configs_global_vector = [
        (1, 8, 128, 16, False, True),
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

    print("\nKernel Output Match!")
