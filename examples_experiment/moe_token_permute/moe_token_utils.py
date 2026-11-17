import torch


def is_fp32_dtype(dtype: str) -> bool:
    return dtype in ("float32", "float")


def auto_tile_h(hidden_size: int) -> int:
    return hidden_size


def auto_tile_t(total: int, num_cores: int, large_candidates=None) -> int:
    default_candidates = [64, 32, 16, 8, 4, 2, 1]
    if total < num_cores:
        for candidate in default_candidates:
            if candidate <= total and total % candidate == 0:
                return candidate
        return max(1, total)
    if large_candidates is None:
        large_candidates = default_candidates
    for candidate in large_candidates:
        if total // candidate >= num_cores:
            return candidate
    return max(1, total // num_cores)


def pad_first_dim(tensor: torch.Tensor, target_rows: int) -> torch.Tensor:
    if tensor.shape[0] >= target_rows:
        return tensor
    out = torch.zeros((target_rows, *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device)
    out[: tensor.shape[0]] = tensor
    return out


def pad_last_dim(tensor: torch.Tensor, target_cols: int) -> torch.Tensor:
    if tensor.shape[-1] >= target_cols:
        return tensor
    out = torch.zeros((*tensor.shape[:-1], target_cols), dtype=tensor.dtype, device=tensor.device)
    out[..., : tensor.shape[-1]] = tensor
    return out


if __name__ == "__main__":
    assert is_fp32_dtype("float32") and is_fp32_dtype("float")
    assert not is_fp32_dtype("float16") and not is_fp32_dtype("bfloat16")

    assert auto_tile_h(7168) == 7168
    assert auto_tile_h(384) == 384

    assert auto_tile_t(8192, 24) == 64
    assert auto_tile_t(16, 24) == 16
    assert auto_tile_t(8192, 24, large_candidates=[128, 64, 32]) == 128

    a = torch.zeros(3, 5)
    assert pad_first_dim(a, 3).shape == (3, 5)
    assert pad_first_dim(a, 8).shape == (8, 5)
    assert pad_last_dim(a, 5).shape == (3, 5)
    assert pad_last_dim(a, 9).shape == (3, 9)

    print("Kernel Output Match")
