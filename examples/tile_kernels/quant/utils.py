from dataclasses import dataclass, replace
from typing import Optional

import torch

__all__ = [
    "BaseCastConfig",
    "CastInputConfig",
    "CastOutputConfig",
    "ceil_div",
    "align_up",
    "is_power_of_two",
    "get_sf_shape",
    "alloc_scaling_factors",
    "cast_epilogue",
    "get_cast_input_and_config",
    "get_cast_output_config",
]


@dataclass(frozen=True)
class BaseCastConfig:
    torch_dtype: torch.dtype = torch.float32
    sf_block: tuple[int, int] = (1, 1)
    use_tma_aligned_col_major_sf: bool = False
    use_packed_ue8m0: bool = False

    @property
    def dtype(self) -> str:
        return str(self.torch_dtype).replace("torch.", "")

    @property
    def sf_torch_dtype(self) -> torch.dtype:
        return torch.uint8 if self.use_packed_ue8m0 else torch.float32

    @property
    def sf_dtype(self) -> str:
        return str(self.sf_torch_dtype).replace("torch.", "")


@dataclass(frozen=True)
class CastInputConfig(BaseCastConfig):
    torch_dtype: torch.dtype = torch.bfloat16
    with_sf: bool = True


@dataclass(frozen=True)
class CastOutputConfig(BaseCastConfig):
    torch_dtype: torch.dtype = torch.float32
    round_sf: bool = False
    custom_clamp_min_value: Optional[float] = None

    @property
    def clamp_min_value(self) -> float:
        if self.custom_clamp_min_value is not None:
            return self.custom_clamp_min_value
        if self.dtype == "float32":
            return torch.finfo(torch.float32).tiny
        if self.dtype == "bfloat16":
            return torch.finfo(torch.bfloat16).tiny
        if self.dtype == "int8":
            return 6.0 * 2 ** (-126)
        raise ValueError(f"Unsupported dtype {self.dtype}")


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def align_up(x: int, y: int) -> int:
    return ceil_div(x, y) * y


def is_power_of_two(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


def get_sf_shape(shape: tuple[int, int], config: BaseCastConfig) -> tuple[int, int]:
    num_block_m = ceil_div(shape[0], config.sf_block[0])
    num_block_k = ceil_div(shape[1], config.sf_block[1])
    if config.use_packed_ue8m0:
        num_block_m *= 4
        num_block_k = ceil_div(num_block_k, 4)
    if config.use_tma_aligned_col_major_sf:
        num_block_m = align_up(num_block_m, 16 if config.use_packed_ue8m0 else 4)
        return (num_block_k, num_block_m)
    return (num_block_m, num_block_k)


def alloc_scaling_factors(
    shape: tuple[int, int],
    out_config: BaseCastConfig,
    device: torch.device,
) -> torch.Tensor:
    if out_config.use_packed_ue8m0:
        assert out_config.use_tma_aligned_col_major_sf, "packed UE8M0 scaling factors require TMA-aligned col-major layout"
    sf_shape = get_sf_shape(shape, out_config)
    return torch.empty(
        size=sf_shape,
        dtype=out_config.sf_torch_dtype,
        device=device,
    )


def cast_epilogue(
    out_sf: torch.Tensor,
    num_tokens: int,
    hidden: int,
    config: BaseCastConfig,
) -> torch.Tensor:
    if config.use_packed_ue8m0:
        if num_tokens == 0:
            out_sf = torch.empty(
                (out_sf.shape[0], out_sf.shape[1] // 4),
                dtype=torch.int32,
                device=out_sf.device,
            )
        else:
            out_sf = out_sf.view(dtype=torch.int32)
    out_sf = out_sf.T if config.use_tma_aligned_col_major_sf else out_sf
    return out_sf[
        : ceil_div(num_tokens, config.sf_block[0]),
        : ceil_div(hidden, config.sf_block[1]),
    ]


def get_cast_input_and_config(
    x: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    sf_block: tuple[int, int],
    use_tma_aligned_col_major_sf: bool | None = None,
    round_sf: bool | None = None,
    use_packed_ue8m0: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, CastInputConfig]:
    _ = round_sf
    if isinstance(x, tuple):
        x_data, x_sf = x
        config = CastInputConfig(torch_dtype=x_data.dtype, sf_block=sf_block, with_sf=True)
        assert x_data.dtype in (torch.bfloat16, torch.float32)
        assert isinstance(x_sf, torch.Tensor)
        if use_tma_aligned_col_major_sf is None:
            use_tma_aligned_col_major_sf = x_sf.stride(0) == 1
        if use_packed_ue8m0 is None:
            use_packed_ue8m0 = x_sf.dtype == torch.int32
        config = replace(
            config,
            use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
            use_packed_ue8m0=use_packed_ue8m0,
        )
        if config.use_tma_aligned_col_major_sf:
            x_sf = x_sf.T
            if config.use_packed_ue8m0:
                assert x_sf.dtype == torch.int32
                x_sf = x_sf.contiguous().view(torch.uint8)
            else:
                assert x_sf.dtype == torch.float32
        else:
            assert x_sf.stride(1) == 1
            assert x_sf.dtype == torch.float32
        return x_data, x_sf, config

    assert x.dtype in (torch.bfloat16, torch.float32)
    sf_block = (1, 1) if sf_block is None else sf_block
    return (
        x,
        None,
        CastInputConfig(
            torch_dtype=x.dtype,
            sf_block=sf_block,
            with_sf=False,
            use_tma_aligned_col_major_sf=bool(use_tma_aligned_col_major_sf),
            use_packed_ue8m0=bool(use_packed_ue8m0),
        ),
    )


def get_cast_output_config(
    fmt: str,
    sf_block: tuple[int, int],
    use_tma_aligned_col_major_sf: bool = False,
    round_sf: bool = False,
    use_packed_ue8m0: bool = False,
    custom_clamp_min_value: Optional[float] = None,
) -> CastOutputConfig:
    assert fmt in ("fp32", "float32", "e5m6", "fp8", "fp4")
    mapping = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "e5m6": torch.uint32,
        "fp8": torch.int8,
        "fp4": torch.int8,
    }
    if custom_clamp_min_value is None and fmt == "fp8":
        custom_clamp_min_value = 1e-4
    if custom_clamp_min_value is None and fmt == "fp4":
        custom_clamp_min_value = 6.0 * 2 ** (-126)
    return CastOutputConfig(
        torch_dtype=mapping[fmt],
        sf_block=sf_block,
        use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
        round_sf=round_sf,
        use_packed_ue8m0=use_packed_ue8m0,
        custom_clamp_min_value=custom_clamp_min_value,
    )
