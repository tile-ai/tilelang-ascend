import torch


def is_fp32_dtype(dtype: str) -> bool:
    return dtype in ("float32", "float")


def auto_tile_h(hidden_size: int, dtype: str) -> int:
    dtype_scale = 2 if is_fp32_dtype(dtype) else 1
    max_tile_h = 4096 // dtype_scale
    for candidate in [hidden_size, max_tile_h, 2048 // dtype_scale, 1024, 512, 256]:
        if candidate > 0 and hidden_size % candidate == 0:
            return candidate
    return 256


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
