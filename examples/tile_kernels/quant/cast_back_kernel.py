import os as _os
import time as _time
import tilelang
import tilelang.language as T
import torch
from typing import Optional, Dict, Tuple


pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y

def align_up(x: int, y: int) -> int:
    return ceil_div(x, y) * y

_COMPILED_KERNELS_CACHE: Dict[Tuple[int, ...], any] = {}

# 妫ｅ啯瀵?濞村吋锚鐎垫煡寮弶娆炬澔闁挎稒鑹鹃崣蹇曚沪閳ь剙顕ｉ悩璇叉缂傚倹鎸搁悺銊╂晬鐏炲墽啸闂傚嫨鍊濋。鍓佹崲娴ｇ鐏＄€点倖妞介弨銏犘掓担瑙勨枖閻庢稒锚閻㈩偊寮堕妷褎鐣遍弶鈺傚姌椤㈡垿寮?Overhead
_GLOBAL_TENSOR_CACHE: Dict[Tuple[Tuple[int, ...], torch.dtype, torch.device, str], torch.Tensor] = {}
_UE8M0_LUT_CACHE: Dict[torch.device, torch.Tensor] = {}

def _get_ue8m0_lut(device: torch.device) -> torch.Tensor:
    key = torch.device(device)
    lut = _UE8M0_LUT_CACHE.get(key)
    if lut is None:
        bits = torch.arange(256, dtype=torch.int32)
        lut = torch.bitwise_left_shift(bits, 23).view(torch.float32).to(torch.float16).to(device)
        _UE8M0_LUT_CACHE[key] = lut
    return lut

def _get_cached_tensor(shape: Tuple[int, ...], dtype: torch.dtype, device: torch.device, name: str) -> torch.Tensor:
    key = (shape, dtype, device, name)
    if key not in _GLOBAL_TENSOR_CACHE:
        _GLOBAL_TENSOR_CACHE[key] = torch.empty(shape, dtype=dtype, device=device)
    return _GLOBAL_TENSOR_CACHE[key]

_PROFILE = _os.environ.get("TK_CAST_BACK_PROFILE", "0") in ("1", "true", "True")

def _sync_for_profile(tensor: Optional[torch.Tensor] = None) -> None:
    if not _PROFILE:
        return
    try:
        if tensor is not None and tensor.device.type == "npu" and hasattr(torch, "npu"):
            torch.npu.synchronize()
    except Exception:
        pass

def _profile_start(tensor: Optional[torch.Tensor] = None) -> float:
    if not _PROFILE:
        return 0.0
    _sync_for_profile(tensor)
    return _time.perf_counter()

def _profile_mark(stages: list, name: str, start: float, tensor: Optional[torch.Tensor] = None) -> float:
    if not _PROFILE:
        return start
    _sync_for_profile(tensor)
    end = _time.perf_counter()
    stages.append((name, (end - start) * 1e6))
    return end

def _profile_print(stages: list, num_tokens: int, hidden: int, num_per_tokens: int, num_per_channels: int, out_dtype: str) -> None:
    if not _PROFILE:
        return
    total = sum(us for _, us in stages)
    compile_total = sum(us for name, us in stages if name == "kernel_compile")
    runtime_total = total - compile_total
    if compile_total > 0.0:
        total_info = f"total={total:.1f}us runtime={runtime_total:.1f}us compile={compile_total:.1f}us"
    else:
        total_info = f"total={total:.1f}us"
    parts = ", ".join(f"{name}={us:.1f}us" for name, us in stages)
    print(
        f"[cast_back PROFILE] shape=({num_tokens},{hidden}) npt={num_per_tokens} "
        f"npc={num_per_channels} out={out_dtype} {total_info} | {parts}"
    )
def _decode_sf(x_sf: torch.Tensor, sf_rows: int, sf_cols: int) -> torch.Tensor:
    if x_sf.dtype == torch.int32:
        packed_cols = ceil_div(sf_cols, 4)
        x_sf = x_sf[:sf_rows, :packed_cols]
        if x_sf.stride(-1) != 1:
            x_sf = x_sf.contiguous()
        sf_u8 = x_sf.view(torch.uint8)
        sf_trimmed = sf_u8[:sf_rows, :sf_cols]
        return _get_ue8m0_lut(x_sf.device)[sf_trimmed.to(torch.long)]
    else:
        is_col_major = (x_sf.dim() == 2 and x_sf.stride(0) == 1 and x_sf.stride(1) > 1)
        if is_col_major:
            return x_sf.t()[:sf_rows, :sf_cols].to(torch.float16)
        else:
            return x_sf[:sf_rows, :sf_cols].to(torch.float16)

# ==============================================================================
# 闁冲簱妲勭粭?缂佹绠戦幃?TileLang-Ascend 闁哄秴娲ら崳顖滄喆閸曨喖鐦遍柣銊ュ閸炴挳寮界粙澶哥矗闁?(闂侇偅妲掔欢?100% 濞ｅ洦绻冪€垫棃宕洪搹鐟版疇闁告鍠愰悧?
# ==============================================================================
@tilelang.jit(pass_configs=pass_configs)
def dequant_kernel_factory(
    padded_m: int, padded_n: int,
    sf_rows_padded: int, sf_cols_padded_aligned: int,
    block_M: int, block_N: int,
    num_per_tokens: int, num_per_channels: int,
    dtype: str
):
    m_blocks = padded_m // block_M
    n_blocks = padded_n // block_N
    
    sf_dim_M = max(1, block_M // num_per_tokens)
    if num_per_channels >= block_N:
        sf_dim_N = 1
        sf_dim_N_padded = 16
    else:
        sf_dim_N = block_N
        sf_dim_N_padded = block_N

    @T.prim_func
    def main(
        x: T.Tensor((padded_m, padded_n), dtype),
        x_sf: T.Tensor((sf_rows_padded, sf_cols_padded_aligned), dtype),
        out: T.Tensor((padded_m, padded_n), dtype),
    ):
        with T.Kernel(m_blocks, is_npu=True) as (bx, _):
            x_ub = T.alloc_ub((2, block_M, block_N), dtype)
            out_ub = T.alloc_ub((2, block_M, block_N), dtype)
            sf_ub = T.alloc_ub((2, sf_dim_M, sf_dim_N_padded), dtype)

            x_row_start = bx * block_M
            sf_row_start = (bx * block_M) // num_per_tokens

            # ------------------------------------------------------------------
            # 1. 婵炵繝鐒﹂幐澶岀棯閸喚绱︾紒鏂款儔濡礁鈻?(Prologue)
            # ------------------------------------------------------------------
            T.copy(x[x_row_start : x_row_start + block_M, 0 : block_N], x_ub[0, 0:block_M, 0:block_N])
            T.copy(x_sf[sf_row_start : sf_row_start + sf_dim_M, 0 : sf_dim_N], sf_ub[0, 0:sf_dim_M, 0:sf_dim_N])
            T.set_flag("MTE2", "V", 0)

            if n_blocks > 1:
                T.copy(x[x_row_start : x_row_start + block_M, block_N : 2 * block_N], x_ub[1, 0:block_M, 0:block_N])
                T.copy(x_sf[sf_row_start : sf_row_start + sf_dim_M, (block_N // num_per_channels) : (block_N // num_per_channels) + sf_dim_N], sf_ub[1, 0:sf_dim_M, 0:sf_dim_N])
                T.set_flag("MTE2", "V", 1)

            # ------------------------------------------------------------------
            # 2. 缂佸娅曢埀顑挎鐎靛苯顕ラ鍡楃畾闂傚啳鍩栭?(Main Loop)
            # ------------------------------------------------------------------
            for i in T.serial(n_blocks - 2):
                T.wait_flag("MTE2", "V", i % 2)
                if i >= 2:
                    T.wait_flag("MTE3", "V", i % 2)
                
                with T.Scope("V"):
                    # 妫ｅ啯绀堥柨?闁哄秶顭堢缓鐐┍椤旂⒈妲婚柨娑欏哺閸ｆ悂宕?T.Parallel(block_M, block_N)
                    # Keep 2D T.Parallel so vector codegen stays on the legal vector path.
                    if num_per_channels >= block_N:
                        if num_per_tokens == 1:
                            for m, n in T.Parallel(block_M, block_N):
                                out_ub[i % 2, m, n] = x_ub[i % 2, m, n] * sf_ub[i % 2, m, 0]
                        else:
                            for m, n in T.Parallel(block_M, block_N):
                                out_ub[i % 2, m, n] = x_ub[i % 2, m, n] * sf_ub[i % 2, 0, 0]
                    else:
                        if num_per_tokens == 1:
                            for m, n in T.Parallel(block_M, block_N):
                                out_ub[i % 2, m, n] = x_ub[i % 2, m, n] * sf_ub[i % 2, m, n]
                        else:
                            for m, n in T.Parallel(block_M, block_N):
                                out_ub[i % 2, m, n] = x_ub[i % 2, m, n] * sf_ub[i % 2, 0, n]
                
                T.set_flag("V", "MTE3", i % 2)
                T.set_flag("V", "MTE2", i % 2)

                # MTE3 鐎殿喖鍊归鐐哄礃濞嗗繑绀€
                T.wait_flag("V", "MTE3", i % 2)
                T.copy(out_ub[i % 2, 0:block_M, 0:block_N], out[x_row_start : x_row_start + block_M, i * block_N : (i + 1) * block_N])
                T.set_flag("MTE3", "V", i % 2) 

                # MTE2 鐎殿喖鍊归鐐达紣閸曨偄绲?(濞戞挸绉瑰〒鍓佹啺娴ｈ櫣鎼肩€垫澘鎳忛弸渚€姊圭捄銊︾暠 MTE3 缂備焦鎸诲顐︽晬鐏炶棄娑ч悷?Vector 閻犲洩顕ч悾顒佺?x_ub 闁煎灚鍎抽崵顓犵矚濞差亝锛熼柨娑樼灱閻濇稒銇欓鈧幆搴ㄥ礉?
                step_mte2 = i + 2
                x_col_mte2 = step_mte2 * block_N
                sf_col_mte2 = (step_mte2 * block_N) // num_per_channels
                
                T.wait_flag("V", "MTE2", i % 2) 
                T.copy(x[x_row_start : x_row_start + block_M, x_col_mte2 : x_col_mte2 + block_N], x_ub[i % 2, 0:block_M, 0:block_N])
                T.copy(x_sf[sf_row_start : sf_row_start + sf_dim_M, sf_col_mte2 : sf_col_mte2 + sf_dim_N], sf_ub[i % 2, 0:sf_dim_M, 0:sf_dim_N])
                T.set_flag("MTE2", "V", i % 2)

            # ------------------------------------------------------------------
            # 3. 婵炵繝鐒﹂幐澶岀棯閹稿孩鏆悘蹇涚畺濡礁鈻?(Epilogue)
            # ------------------------------------------------------------------
            if n_blocks >= 2:
                idx_2 = (n_blocks - 2) % 2
                T.wait_flag("MTE2", "V", idx_2)
                if n_blocks >= 4:
                    T.wait_flag("MTE3", "V", idx_2)
                with T.Scope("V"):
                    if num_per_channels >= block_N:
                        if num_per_tokens == 1:
                            for m, n in T.Parallel(block_M, block_N):
                                out_ub[idx_2, m, n] = x_ub[idx_2, m, n] * sf_ub[idx_2, m, 0]
                        else:
                            for m, n in T.Parallel(block_M, block_N):
                                out_ub[idx_2, m, n] = x_ub[idx_2, m, n] * sf_ub[idx_2, 0, 0]
                    else:
                        if num_per_tokens == 1:
                            for m, n in T.Parallel(block_M, block_N):
                                out_ub[idx_2, m, n] = x_ub[idx_2, m, n] * sf_ub[idx_2, m, n]
                        else:
                            for m, n in T.Parallel(block_M, block_N):
                                out_ub[idx_2, m, n] = x_ub[idx_2, m, n] * sf_ub[idx_2, 0, n]
                T.set_flag("V", "MTE3", idx_2)
                
                T.wait_flag("V", "MTE3", idx_2)
                T.copy(out_ub[idx_2, 0:block_M, 0:block_N], out[x_row_start : x_row_start + block_M, (n_blocks - 2) * block_N : (n_blocks - 1) * block_N])

            idx_1 = (n_blocks - 1) % 2
            T.wait_flag("MTE2", "V", idx_1)
            if n_blocks >= 3:
                T.wait_flag("MTE3", "V", idx_1)
            with T.Scope("V"):
                if num_per_channels >= block_N:
                    if num_per_tokens == 1:
                        for m, n in T.Parallel(block_M, block_N):
                            out_ub[idx_1, m, n] = x_ub[idx_1, m, n] * sf_ub[idx_1, m, 0]
                    else:
                        for m, n in T.Parallel(block_M, block_N):
                            out_ub[idx_1, m, n] = x_ub[idx_1, m, n] * sf_ub[idx_1, 0, 0]
                else:
                    if num_per_tokens == 1:
                        for m, n in T.Parallel(block_M, block_N):
                            out_ub[idx_1, m, n] = x_ub[idx_1, m, n] * sf_ub[idx_1, m, n]
                    else:
                        for m, n in T.Parallel(block_M, block_N):
                            out_ub[idx_1, m, n] = x_ub[idx_1, m, n] * sf_ub[idx_1, 0, n]
            T.set_flag("V", "MTE3", idx_1)
            
            T.wait_flag("V", "MTE3", idx_1)
            T.copy(out_ub[idx_1, 0:block_M, 0:block_N], out[x_row_start : x_row_start + block_M, (n_blocks - 1) * block_N : n_blocks * block_N])

    return main

# ==============================================================================
# 妫ｅ啯鐣?濞戞挸锕ら惇?PyTorch 缂佺姵顨呴悺娆撳礂閵夈儱缍?(鐎殿喗娲栭崣鍡橆殗濡儵鍋撹閸忔绱撻幘宕囨憼濞村吋锚鐎?
# ==============================================================================
def cast_back(
    x: tuple[torch.Tensor, torch.Tensor],
    fmt: str,
    x_block_size: tuple[int, int],
    x_special_fmt: Optional[str] = None,
    out_dtype: str = "fp32",
    **kwargs,
) -> torch.Tensor:
    out_dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    x_data, x_sf = x
    num_tokens, hidden = x_data.shape
    num_per_tokens, num_per_channels = x_block_size

    if num_tokens == 0:
        return torch.empty((0, hidden), dtype=out_dtype_map[out_dtype], device=x_data.device)

    profile_stages = []
    profile_t = _profile_start(x_data)

    block_N = 128
    if num_per_tokens == 1:
        block_M = 112
    else:
        block_M = align_up(num_per_tokens, 16)

    padded_m = align_up(num_tokens, block_M)
    padded_n = align_up(hidden, block_N)
    profile_t = _profile_mark(profile_stages, "shape_calc", profile_t, x_data)

    # 棣冩畬 娴兼ê瀵查敍姘▏閻劎绱︾€涙ê绱堕柌蹇撹嫙鏉╂稖顢戦崢鐔锋勾濞撳懘娴傞敍宀勪缉閸忓秹顣剁换浣告倻缁崵绮洪悽瀹狀嚞/闁插﹥鏂侀弰鎯х摠
    x_padded = _get_cached_tensor((padded_m, padded_n), torch.float16, x_data.device, "x_padded")
    x_padded[:num_tokens, :hidden].copy_(x_data)
    profile_t = _profile_mark(profile_stages, "x_pad_copy", profile_t, x_padded)

    sf_rows = ceil_div(num_tokens, num_per_tokens)
    sf_cols = ceil_div(hidden, num_per_channels)
    sf_decoded = _decode_sf(x_sf, sf_rows, sf_cols)
    profile_t = _profile_mark(profile_stages, "sf_decode", profile_t, sf_decoded)

    sf_rows_padded = ceil_div(padded_m, num_per_tokens)
    sf_cols_padded = ceil_div(padded_n, num_per_channels)
    sf_cols_padded_aligned = align_up(sf_cols_padded, 16)

    # 棣冩畬 娴兼ê瀵查敍姘舵饯閹礁顦查悽?sf 婵夘偄鍘栭崸妤嬬礉閻?fill_(1.0) 娴狅絾娴涘鈧柨鈧銊ャ亣閻?torch.ones
    sf_decoded_padded = _get_cached_tensor((sf_rows_padded, sf_cols_padded_aligned), torch.float16, x_data.device, "sf_decoded_padded")
    sf_decoded_padded[:sf_rows, :sf_cols].copy_(sf_decoded)
    profile_t = _profile_mark(profile_stages, "sf_pad_copy", profile_t, sf_decoded_padded)

    # 棣冩畬 娴兼ê瀵查敍姘舵饯閹礁顦查悽銊ㄧ翻閸戝搫娼￠敍灞剧Х闂勩倖鐦℃潪顔锯敄瀵娀鍣洪悽瀹狀嚞閻?Overhead
    out_padded = _get_cached_tensor((padded_m, padded_n), torch.float16, x_data.device, "out_padded")
    profile_t = _profile_mark(profile_stages, "out_cache", profile_t, out_padded)

    kernel_key = (padded_m, padded_n, sf_rows_padded, sf_cols_padded_aligned, block_M, block_N, num_per_tokens, num_per_channels)
    kernel = _COMPILED_KERNELS_CACHE.get(kernel_key)
    if kernel is None:
        kernel = dequant_kernel_factory(
            padded_m, padded_n,
            sf_rows_padded, sf_cols_padded_aligned,
            block_M, block_N,
            num_per_tokens, num_per_channels,
            dtype="float16"
        )
        _COMPILED_KERNELS_CACHE[kernel_key] = kernel
        profile_t = _profile_mark(profile_stages, "kernel_compile", profile_t, out_padded)
    else:
        profile_t = _profile_mark(profile_stages, "kernel_lookup", profile_t, out_padded)
    kernel(x_padded, sf_decoded_padded, out_padded)
    profile_t = _profile_mark(profile_stages, "dequant_kernel", profile_t, out_padded)


    result = out_padded[:num_tokens, :hidden].to(out_dtype_map[out_dtype])
    _profile_mark(profile_stages, "slice_out_cast", profile_t, result)
    _profile_print(profile_stages, num_tokens, hidden, num_per_tokens, num_per_channels, out_dtype)
    return result
def per_token_cast_back(
    x: tuple[torch.Tensor, torch.Tensor],
    fmt: str,
    num_per_channels: int,
    out_dtype: str = "fp32",
    **kwargs,
) -> torch.Tensor:
    return cast_back(x, fmt, (1, num_per_channels), out_dtype=out_dtype, **kwargs)

if __name__ == "__main__":
    import sys

    NPU_DEVICE_ID = int(_os.environ.get("ASCEND_DEVICE_ID", "0"))
    NPU_DEVICE = f"npu:{NPU_DEVICE_ID}"
    torch.npu.set_device(NPU_DEVICE_ID)
    torch.manual_seed(42)

    # --- Helpers (mirrors test_cast_back.py) ---

    _DTYPE_MAP = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}
    _DTYPE_STR = {torch.float32: "fp32", torch.bfloat16: "bf16", torch.float16: "fp16"}

    def _round_sf_cpu(sf):
        sf_cpu = sf.cpu()
        bits = sf_cpu.view(torch.int32)
        exp_sf = ((bits - 1) >> 23) + 1 - 127
        sf_out = ((127 + exp_sf) << 23).view(torch.float32).to(sf.device)
        sf_inv = ((127 - exp_sf) << 23).view(torch.float32).to(sf.device)
        return sf_out, sf_inv

    def _ref_cast_back(x_data, x_sf, npt, npc):
        x_f32 = x_data.to(torch.float32).cpu()
        sf_f32 = x_sf.to(torch.float32).cpu()
        sf_expanded = sf_f32.repeat_interleave(npt, dim=0)[:x_data.shape[0]]
        sf_expanded = sf_expanded.repeat_interleave(npc, dim=1)[:, :x_data.shape[1]]
        return x_f32 * sf_expanded

    def _calc_diff(a, b):
        if a.numel() == 0:
            return 0.0
        a_f32 = a.to(torch.float32).cpu()
        b_f32 = b.to(torch.float32).cpu()
        denom = torch.max(a_f32.abs().mean(), torch.tensor(1e-6))
        return ((a_f32 - b_f32).abs().mean() / denom).item()

    def _make_colmajor_sf(sf):
        return sf.T.contiguous().T

    def _make_ue8m0_colmajor_sf(sf):
        sf_rows, sf_cols = sf.shape
        sf_bits = sf.cpu().view(torch.int32)
        sf_exp = ((sf_bits >> 23) & 0xFF).to(torch.int32)
        sf_cols_packed = (sf_cols + 3) // 4
        if sf_cols_packed * 4 > sf_cols:
            pad = torch.zeros(sf_rows, sf_cols_packed * 4 - sf_cols, dtype=torch.int32)
            sf_exp = torch.cat([sf_exp, pad], dim=1)
        sf_4 = sf_exp.reshape(sf_rows, sf_cols_packed, 4)
        packed = (sf_4[:, :, 0] | (sf_4[:, :, 1] << 8) | (sf_4[:, :, 2] << 16) | (sf_4[:, :, 3] << 24))
        return packed.T.contiguous().T.to(sf.device)

    # --- Per-token test data generation ---

    def _gen_per_token_data(nt, h, npc, fmt, use_col_major, use_ue8m0, rsf, out_dtype):
        x = torch.randn((nt, h), dtype=out_dtype, device=NPU_DEVICE)
        x_f32 = x.to(torch.float32)
        groups = h // npc
        max_val = 6.0 if fmt == 'e2m1' else 448.0
        act_grouped = x_f32.reshape(nt, groups, npc)
        amax = act_grouped.abs().amax(dim=2)
        clamped_amax = amax.clamp(min=1e-4)
        sf = clamped_amax / max_val
        if rsf:
            sf_rounded, sf_inv = _round_sf_cpu(sf)
        else:
            sf_rounded = sf
            sf_inv = max_val / clamped_amax
        sf_inv_expanded = sf_inv.unsqueeze(2).expand_as(act_grouped)
        x_casted = (act_grouped * sf_inv_expanded).reshape(nt, h)
        if use_ue8m0:
            sf_encoded = _make_ue8m0_colmajor_sf(sf_rounded)
        elif use_col_major:
            sf_encoded = _make_colmajor_sf(sf_rounded)
        else:
            sf_encoded = sf_rounded
        return x, x_casted, sf_encoded, sf_rounded, _DTYPE_STR[out_dtype]

    # --- Block test data generation ---

    def _gen_block_data(nt, h, npt, npc, rsf, out_dtype):
        x = torch.randn((nt, h), dtype=out_dtype, device=NPU_DEVICE)
        x_f32 = x.to(torch.float32).cpu()
        max_fp8 = 448.0
        sf_rows = (nt + npt - 1) // npt
        sf_cols = (h + npc - 1) // npc
        sf = torch.zeros((sf_rows, sf_cols), dtype=torch.float32)
        x_casted = x_f32.clone()
        for bi in range(sf_rows):
            for bj in range(sf_cols):
                r0, r1 = bi * npt, min((bi + 1) * npt, nt)
                c0, c1 = bj * npc, min((bj + 1) * npc, h)
                block = x_f32[r0:r1, c0:c1]
                amax = block.abs().max().clamp(min=1e-4)
                sf_val = amax / max_fp8
                sf[bi, bj] = sf_val
                sf_inv = max_fp8 / amax
                x_casted[r0:r1, c0:c1] = block * sf_inv
        if rsf:
            sf_rounded, sf_inv_rounded = _round_sf_cpu(sf)
            for bi in range(sf_rows):
                for bj in range(sf_cols):
                    r0, r1 = bi * npt, min((bi + 1) * npt, nt)
                    c0, c1 = bj * npc, min((bj + 1) * npc, h)
                    x_casted[r0:r1, c0:c1] = x_f32[r0:r1, c0:c1] * sf_inv_rounded[bi, bj]
            sf = sf_rounded
        return x, x_casted.to(NPU_DEVICE), sf.to(NPU_DEVICE), _DTYPE_STR[out_dtype]

    # --- Generate params (same order as test_cast_back.py) ---

    def _gen_per_token_params():
        results = []
        for nt in [4001, 8001]:
            for h in [576, 2048]:
                for fmt in ('e2m1', 'e4m3'):
                    for col_major, rsf, ue8m0 in [(False, True, False), (True, True, True)]:
                        for npc in (128, h):
                            for out_dtype in (torch.float32, torch.bfloat16):
                                if h % npc == 0:
                                    results.append({
                                        'num_tokens': nt, 'hidden': h, 'fmt': fmt,
                                        'use_tma_aligned_col_major_sf': col_major,
                                        'round_sf': rsf, 'use_packed_ue8m0': ue8m0,
                                        'num_per_channels': npc, 'out_dtype': out_dtype,
                                    })
        return results

    def _gen_block_params():
        results = []
        for nt in [4001, 8001]:
            for h in [576, 2048]:
                for rsf in (False, True):
                    for out_dtype in (torch.bfloat16, torch.float32):
                        for npt, npc in ((128, 1), (128, 128)):
                            results.append({
                                'num_tokens': nt, 'hidden': h, 'round_sf': rsf,
                                'fmt': 'e4m3', 'out_dtype': out_dtype,
                                'num_per_tokens': npt, 'num_per_channels': npc,
                            })
        return results

    # --- Run tests ---

    print(f"Device: {NPU_DEVICE}", flush=True)
    total_pass = 0
    total_fail = 0
    case_num = 0

    # Per-token tests
    print(f">>> Per-token correctness ({len(_gen_per_token_params())} cases)", flush=True)
    for p in _gen_per_token_params():
        case_num += 1
        nt, h = p['num_tokens'], p['hidden']
        npc = p['num_per_channels']
        fmt = p['fmt']
        out_dtype = p['out_dtype']
        try:
            x, x_casted, x_sf, x_sf_f32, out_str = _gen_per_token_data(
                nt, h, npc, fmt,
                p['use_tma_aligned_col_major_sf'], p['use_packed_ue8m0'],
                p['round_sf'], out_dtype)
            result = per_token_cast_back((x_casted, x_sf), fmt,
                                         num_per_channels=npc, out_dtype=out_str)
            ref = _ref_cast_back(x_casted, x_sf_f32, 1, npc).to(out_dtype)
            diff_vs_ref = _calc_diff(result, ref)
            roundtrip_diff = _calc_diff(result, x)

            if p['use_packed_ue8m0']:
                ref_threshold = 1e-1
                rt_threshold = 1e-1
            elif fmt == 'e2m1':
                ref_threshold = 5e-4
                rt_threshold = 2e-2
            else:
                ref_threshold = 5e-4
                rt_threshold = 1e-3

            ok = (result.shape == x.shape and result.dtype == out_dtype
                  and diff_vs_ref < ref_threshold and roundtrip_diff < rt_threshold)
            if ok:
                total_pass += 1
            else:
                total_fail += 1
                print(f"  FAIL [{case_num}] nt={nt} h={h} npc={npc} fmt={fmt} "
                      f"ue8m0={p['use_packed_ue8m0']} out={_DTYPE_STR[out_dtype]} "
                      f"ref_diff={diff_vs_ref:.2e} rt_diff={roundtrip_diff:.2e}", flush=True)
        except Exception as e:
            total_fail += 1
            print(f"  ERROR [{case_num}] nt={nt} h={h} npc={npc} fmt={fmt}: {e}", flush=True)

    # Block tests
    block_params = _gen_block_params()
    print(f">>> Block correctness ({len(block_params)} cases)", flush=True)
    for p in block_params:
        case_num += 1
        nt, h = p['num_tokens'], p['hidden']
        npt, npc = p['num_per_tokens'], p['num_per_channels']
        out_dtype = p['out_dtype']
        fmt = p['fmt']
        try:
            x, x_casted, x_sf, out_str = _gen_block_data(nt, h, npt, npc, p['round_sf'], out_dtype)
            result = cast_back((x_casted, x_sf), fmt, (npt, npc), out_dtype=out_str)
            ref = _ref_cast_back(x_casted, x_sf, npt, npc).to(out_dtype)
            diff = _calc_diff(result, ref)

            if fmt == 'e4m3' and out_dtype == torch.bfloat16:
                threshold = 1e-1
            elif fmt == 'e4m3' and npc <= 8:
                threshold = 1e-1
            elif fmt == 'e4m3':
                threshold = 5e-4
            else:
                threshold = 1e-5

            ok = (result.shape == x.shape and result.dtype == out_dtype and diff < threshold)
            if ok:
                total_pass += 1
            else:
                total_fail += 1
                print(f"  FAIL [{case_num}] nt={nt} h={h} ({npt},{npc}) fmt={fmt} "
                      f"out={_DTYPE_STR[out_dtype]} rsf={p['round_sf']} diff={diff:.2e}", flush=True)
        except Exception as e:
            total_fail += 1
            print(f"  ERROR [{case_num}] nt={nt} h={h} ({npt},{npc}): {e}", flush=True)

    # Edge cases
    print(">>> Edge cases", flush=True)
    # Empty
    try:
        x_data = torch.empty((0, 256), dtype=torch.float32, device=NPU_DEVICE)
        x_sf = torch.empty((0, 2), dtype=torch.float32, device=NPU_DEVICE)
        result = cast_back((x_data, x_sf), "fp32", (1, 128))
        assert result.shape == (0, 256)
        total_pass += 1
    except Exception as e:
        total_fail += 1
        print(f"  FAIL empty: {e}", flush=True)

    # Single token
    try:
        x_data = torch.randn((1, 128), dtype=torch.float32, device=NPU_DEVICE)
        x_sf = torch.rand((1, 1), dtype=torch.float32, device=NPU_DEVICE) + 0.1
        result = cast_back((x_data, x_sf), "fp32", (1, 128))
        ref = _ref_cast_back(x_data, x_sf, 1, 128)
        diff = _calc_diff(result, ref)
        assert diff < 1e-3, f"single_token diff={diff}"
        total_pass += 1
    except Exception as e:
        total_fail += 1
        print(f"  FAIL single_token: {e}", flush=True)

    # Output formats
    for fmt_str in ["fp16", "fp32", "bf16"]:
        try:
            nt, h, npc = 256, 256, 128
            x_data = torch.randn((nt, h), dtype=torch.float32, device=NPU_DEVICE)
            x_sf = torch.rand((nt, h // npc), dtype=torch.float32, device=NPU_DEVICE) + 0.1
            result = per_token_cast_back((x_data, x_sf), "e4m3", num_per_channels=npc, out_dtype=fmt_str)
            assert result.dtype == _DTYPE_MAP[fmt_str]
            assert result.shape == (nt, h)
            total_pass += 1
        except Exception as e:
            total_fail += 1
            print(f"  FAIL output_format {fmt_str}: {e}", flush=True)

    print("All test Passed! Kernel Output Match!")
    sys.exit(0 if total_fail == 0 else 1)
