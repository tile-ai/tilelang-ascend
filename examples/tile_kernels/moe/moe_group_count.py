import os

import torch
import tilelang
import tilelang.language as T


tilelang.cache.clear_cache()
os.environ["TILELANG_PRINT_ON_COMPILATION"] = "0"

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(pass_configs=pass_configs)
def get_group_count_kernel(num_tokens: int, num_topk: int, num_groups: int):
    num_cores = 24
    # Keep one hardware wave without rounding rows up: 24 blocks expose all
    # 48 vector tasks for the common 4001-token workload.
    rows_for_one_wave = (num_tokens + 2 * num_cores - 1) // (2 * num_cores)
    rows_per_vec = min(128, max(32, rows_for_one_wave))
    tokens_per_block = rows_per_vec * 2
    num_token_blocks = (num_tokens + tokens_per_block - 1) // tokens_per_block
    metadata_count = rows_per_vec * num_topk
    metadata_aligned_count = ((metadata_count + 63) // 64) * 64
    tail_rows = num_tokens % rows_per_vec
    # Use a non-zero compile-time extent when there is no partial vector tile;
    # the corresponding runtime branch is then unreachable.
    tail_metadata_count = (tail_rows or rows_per_vec) * num_topk
    # Small group counts also amortize vector compare/reduce well, even for K=2.
    use_vector_histogram = num_topk >= 6 or num_groups <= 9
    use_group_parallel = num_topk == 2 and num_groups == 32
    META_ADDR_READY = 0

    if use_group_parallel:
        # Split 32 groups into 8-wide atomic tiles. Twelve route chunks per
        # group tile produce exactly 48 vector tasks (one hardware wave).
        group_tile_size = 8
        num_group_tiles = num_groups // group_tile_size
        chunks_per_group_tile = (2 * num_cores) // num_group_tiles
        total_metadata_count = num_tokens * num_topk
        metadata_per_group_chunk = (total_metadata_count + chunks_per_group_tile - 1) // chunks_per_group_tile
        tail_group_chunk_count = total_metadata_count - (chunks_per_group_tile - 1) * metadata_per_group_chunk
        group_chunk_aligned_count = ((metadata_per_group_chunk + 63) // 64) * 64

        @T.prim_func
        def group_parallel_kernel(
            group_idx_flat: T.Tensor((total_metadata_count,), "int64"),
            out: T.Tensor((num_groups,), "int32"),
        ):
            with T.Kernel(num_cores, is_npu=True) as (cid, vid), T.Scope("V"):
                group_idx_ub = T.alloc_ub((1, group_chunk_aligned_count), "int64")
                group_idx_i32_ub = T.alloc_ub((1, group_chunk_aligned_count), "int32")
                compare_mask_ub = T.alloc_ub((1, group_chunk_aligned_count), "uint8")
                one_ub = T.alloc_ub((1, group_chunk_aligned_count), "float")
                hit_ub = T.alloc_ub((1, group_chunk_aligned_count), "float")
                count_one_f32 = T.alloc_ub((1,), "float")
                count_local = T.alloc_ub((group_tile_size,), "int32")

                task_id = T.alloc_var("int32", init=0)
                group_tile_id = T.alloc_var("int32", init=0)
                route_chunk_id = T.alloc_var("int32", init=0)
                group_base = T.alloc_var("int32", init=0)
                metadata_offset = T.alloc_var("int32", init=0)

                task_id = cid * 2 + vid
                group_tile_id = task_id // chunks_per_group_tile
                route_chunk_id = task_id % chunks_per_group_tile
                group_base = group_tile_id * group_tile_size
                metadata_offset = route_chunk_id * metadata_per_group_chunk

                T.tile.fill(group_idx_i32_ub, -1)
                T.tile.fill(one_ub, 1.0)
                T.tile.fill(count_local, 0)

                T.set_flag("s", "mte2", META_ADDR_READY)
                T.wait_flag("s", "mte2", META_ADDR_READY)
                if route_chunk_id < chunks_per_group_tile - 1:
                    T.copy(
                        group_idx_flat[metadata_offset : metadata_offset + metadata_per_group_chunk],
                        group_idx_ub[0, 0:metadata_per_group_chunk],
                        pad_value=-1,
                    )
                    T.tile.cast(group_idx_i32_ub, group_idx_ub, "CAST_NONE", metadata_per_group_chunk)
                else:
                    T.copy(
                        group_idx_flat[metadata_offset : metadata_offset + tail_group_chunk_count],
                        group_idx_ub[0, 0:tail_group_chunk_count],
                        pad_value=-1,
                    )
                    T.tile.cast(group_idx_i32_ub, group_idx_ub, "CAST_NONE", tail_group_chunk_count)

                for local_group in T.serial(group_tile_size):
                    T.tile.compare(compare_mask_ub, group_idx_i32_ub, T.int32(group_base + local_group), "EQ")
                    T.tile.select(hit_ub, compare_mask_ub, one_ub, 0.0, "VSEL_TENSOR_SCALAR_MODE")
                    T.reduce_sum(hit_ub, count_one_f32, dim=-1, real_shape=[1, group_chunk_aligned_count])
                    count_local[local_group] = T.int32(count_one_f32[0])

                # Eight int32 values form one aligned 32-byte DMA atomic.
                T.tile.atomic_add(out[group_base], count_local)

        return group_parallel_kernel

    @T.prim_func
    def group_count_kernel(
        group_idx_flat: T.Tensor((num_tokens * num_topk,), "int64"),
        out: T.Tensor((num_groups,), "int32"),
    ):
        with T.Kernel(num_token_blocks, is_npu=True) as (cid, vid), T.Scope("V"):
            group_idx_ub = T.alloc_ub((1, metadata_aligned_count), "int64")
            group_idx_i32_ub = T.alloc_ub((1, metadata_aligned_count), "int32")
            compare_mask_ub = T.alloc_ub((1, metadata_aligned_count), "uint8")
            one_ub = T.alloc_ub((1, metadata_aligned_count), "float")
            hit_ub = T.alloc_ub((1, metadata_aligned_count), "float")
            count_one_f32 = T.alloc_ub((1,), "float")
            count_local = T.alloc_ub((num_groups,), "int32")

            token_base = T.alloc_var("int32", init=0)
            metadata_offset = T.alloc_var("int32", init=0)
            expert_idx = T.alloc_var("int32", init=-1)

            token_base = cid * tokens_per_block + vid * rows_per_vec
            if token_base < num_tokens:
                T.tile.fill(count_local, 0)
                if use_vector_histogram:
                    T.tile.fill(group_idx_i32_ub, -1)
                    T.tile.fill(one_ub, 1.0)
                metadata_offset = token_base * num_topk

                T.set_flag("s", "mte2", META_ADDR_READY)
                T.wait_flag("s", "mte2", META_ADDR_READY)
                if token_base + rows_per_vec <= num_tokens:
                    if use_vector_histogram:
                        T.copy(
                            group_idx_flat[metadata_offset : metadata_offset + metadata_count],
                            group_idx_ub[0, 0:metadata_count],
                            pad_value=-1,
                        )
                        T.tile.cast(group_idx_i32_ub, group_idx_ub, "CAST_NONE", metadata_count)
                    else:
                        T.copy(
                            group_idx_flat[metadata_offset : metadata_offset + metadata_count],
                            group_idx_ub[0, 0:metadata_count],
                            pad_value=-1,
                        )
                        for route in T.serial(metadata_count):
                            expert_idx = T.int32(group_idx_ub[0, route])
                            if expert_idx >= 0 and expert_idx < num_groups:
                                count_local[expert_idx] += 1
                else:
                    T.copy(
                        group_idx_flat[metadata_offset : metadata_offset + tail_metadata_count],
                        group_idx_ub[0, 0:tail_metadata_count],
                        pad_value=-1,
                    )
                    if use_vector_histogram:
                        T.tile.cast(group_idx_i32_ub, group_idx_ub, "CAST_NONE", tail_metadata_count)
                    else:
                        for route in T.serial(tail_metadata_count):
                            expert_idx = T.int32(group_idx_ub[0, route])
                            if expert_idx >= 0 and expert_idx < num_groups:
                                count_local[expert_idx] += 1

                if use_vector_histogram:
                    # Both full and partial tiles use the same dense vector
                    # scan. Unused tail lanes remain -1 and never match.
                    for group_id in T.serial(num_groups):
                        T.tile.compare(compare_mask_ub, group_idx_i32_ub, T.int32(group_id), "EQ")
                        T.tile.select(hit_ub, compare_mask_ub, one_ub, 0.0, "VSEL_TENSOR_SCALAR_MODE")
                        T.reduce_sum(hit_ub, count_one_f32, dim=-1, real_shape=[1, metadata_count])
                        count_local[group_id] = T.int32(count_one_f32[0])

                # Ascend cross-AIV accumulation is a local-tile to GM DMA
                # atomic operation. The scalar T.atomic_add helper does not
                # provide the required inter-AIV writeback semantics here.
                T.tile.atomic_add(out[0], count_local)

    return group_count_kernel


def ascend_group_count(group_idx: torch.Tensor, num_groups: int) -> torch.Tensor:
    """Count the number of tokens assigned to each group/expert on Ascend."""
    assert group_idx.dim() == 2 and group_idx.is_contiguous()
    assert group_idx.dtype == torch.int64

    num_tokens, num_topk = group_idx.shape
    out = torch.zeros(
        (num_groups,),
        dtype=torch.int32,
        device=group_idx.device,
    )
    if num_tokens == 0:
        return out

    kernel = get_group_count_kernel(num_tokens, num_topk, num_groups)
    kernel(group_idx.contiguous().view(-1), out)
    return out


def ref_group_count(group_idx: torch.Tensor, num_groups: int) -> torch.Tensor:
    valid_group_idx = group_idx[group_idx >= 0].to(torch.int64).cpu()
    out = torch.bincount(valid_group_idx, minlength=num_groups)[:num_groups]
    return out.to(dtype=torch.int32, device=group_idx.device)


def main():
    try:
        import torch_npu

        has_npu = torch_npu is not None
    except ImportError:
        has_npu = False

    device = "npu" if has_npu and torch.npu.is_available() else "cuda"

    test_configs = [
        {"num_tokens": 7970, "num_topk": 2, "num_groups": 9},
        {"num_tokens": 8989, "num_topk": 2, "num_groups": 32},
        {"num_tokens": 9857, "num_topk": 2, "num_groups": 4},
        {"num_tokens": 18483, "num_topk": 6, "num_groups": 9},
        {"num_tokens": 18690, "num_topk": 6, "num_groups": 32},
        {"num_tokens": 25105, "num_topk": 6, "num_groups": 4},
        {"num_tokens": 21862, "num_topk": 8, "num_groups": 9},
        {"num_tokens": 21925, "num_topk": 8, "num_groups": 32},
        {"num_tokens": 32556, "num_topk": 8, "num_groups": 4},
        {"num_tokens": 23176, "num_topk": 9, "num_groups": 9},
        {"num_tokens": 23139, "num_topk": 9, "num_groups": 32},
        {"num_tokens": 36168, "num_topk": 9, "num_groups": 4},
    ]

    for config in test_configs:
        num_tokens = config["num_tokens"]
        num_topk = config["num_topk"]
        num_groups = config["num_groups"]

        # Keep data generation off the NPU. Under msprof, aclnnInplaceRandom
        # may fail to load its dynamic random/cast kernels before our custom
        # operator is launched.
        group_idx_cpu = torch.randint(0, num_groups, (num_tokens, num_topk), dtype=torch.int64, device="cpu")
        invalid_mask_cpu = torch.rand((num_tokens, num_topk), device="cpu") < 0.2
        group_idx_cpu[invalid_mask_cpu] = -1
        group_idx = group_idx_cpu.to(device).contiguous()

        if device == "npu":
            torch.npu.synchronize()
        out = ascend_group_count(group_idx, num_groups)
        if device == "npu":
            torch.npu.synchronize()

        ref_out_cpu = ref_group_count(group_idx_cpu, num_groups)
        out_cpu = out.cpu()
        assert torch.equal(out_cpu, ref_out_cpu), f"Group count mismatch!\nout={out_cpu.tolist()}\nref={ref_out_cpu.tolist()}"

        case = f"ascend_group_count T={num_tokens} K={num_topk} G={num_groups}"
        print(f"pass {case}")

    print("TEST PASSED!")


if __name__ == "__main__":
    main()
