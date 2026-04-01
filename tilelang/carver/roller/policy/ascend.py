# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""
Policy for Ascend NPU architecture.

Ascend NPU uses SIMD (Single Instruction Multiple Data) architecture,
which is fundamentally different from GPU's SIMT (Single Instruction Multiple Thread).

Key differences:
- SIMT (GPU): Multiple threads execute same instruction, thread-level parallelism
- SIMD (Ascend): Single instruction operates on vector data, data-level parallelism

This module provides:
- AscendDefaultPolicy: Base policy for Ascend SIMD architecture (replaces DefaultPolicy)
- AscendCubePolicy: CUBE unit optimization (similar to TensorCorePolicy)
"""

from collections.abc import Iterable
from queue import PriorityQueue

from ..hint import Hint, Stride, TileDict, IntrinInfo
from ..node import PrimFuncNode
from ...arch import TileDevice
from .default import DefaultPolicy
from .common import coalesced_factor, get_all_factors
from ..rasterization import NoRasterization
import tvm
import numpy as np
import logging

logger = logging.getLogger(__name__)


class AscendDefaultPolicy(DefaultPolicy):
    """
    Default Policy for Ascend NPU SIMD architecture.

    This policy adapts the GPU-oriented DefaultPolicy for Ascend's SIMD architecture:
    - No thread/warp concept - uses AI Core level parallelism
    - SIMD vector operations instead of thread-level parallelism
    - Memory hierarchy: L0A, L0B, L0C, L1, UB (instead of shared memory)
    - DMA alignment requirements (32 bytes)
    - Vector alignment for SIMD operations

    Key adaptations:
    1. block_size -> 1 (no thread parallelism, AI Core executes sequentially)
    2. thread config -> vector config (SIMD width)
    3. shared memory -> L0/UB buffer
    4. coalesced access -> DMA aligned access
    """

    def __init__(self, arch: TileDevice, tags: dict | None = None) -> None:
        assert arch.platform == "ascend", (
            "AscendDefaultPolicy requires Ascend architecture."
        )
        super().__init__(arch, tags)

        # Integrate all Ascend hardware parameters from arch into Policy
        # SIMD and DMA parameters (fixed for Ascend architecture)
        self.simd_width = 32  # Unified Buffer alignment (32 bytes)
        self.dma_alignment = 32  # DMA requires 32-byte alignment

        # AI Core configuration - use actual hardware value
        self.num_ai_cores = getattr(arch, "compute_max_core", 32)

        # Memory hierarchy capacities (bytes) - prioritize arch values
        self.l0a_capacity = getattr(arch, "l0a_cap", 64 * 1024)
        self.l0b_capacity = getattr(arch, "l0b_cap", 64 * 1024)
        self.l0c_capacity = getattr(arch, "l0c_cap", 256 * 1024)
        self.l1_capacity = getattr(arch, "l1_cap", 1024 * 1024)
        self.ub_capacity = getattr(arch, "ub_cap", 256 * 1024)

        # L2 cache and additional parameters
        self.l2_cache_size = getattr(arch, "l2_cache_size_bytes", 16 * 1024 * 1024)

        # CUBE unit specification [M, N, K]
        self.cube_shape = getattr(arch, "cube_spec", [16, 16, 16])
        self.cube_dim = self.cube_shape[-1]  # K dimension
        self.fractal_shape = (self.cube_shape[0], self.cube_shape[1])  # (M, N)

        # Chip identification
        self.chip_name = getattr(arch, "chip_name", "Ascend910A")

        # L0 buffer alignment requirements (bytes)
        # ref: https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/opdevg/Ascendcopdevg/atlas_ascendc_10_0010.html
        # These are architecture constants that depend on the CUBE shape and data format
        # L0A/L0B: 512-byte alignment for DMA transfer (fractal format)
        # L0C: Aligned to CUBE output block size (16x16 elements * dtype_size)
        self.l0a_alignment = 512  # L0A: 512-byte aligned (input A)
        self.l0b_alignment = 512  # L0B: 512-byte aligned (input B)
        self.l0c_alignment = 64  # L0C: 64-byte aligned (16x16 fp16 block)

        # DMA transaction parameters
        self.transaction_size = getattr(arch, "transaction_size", [32, 32])
        self.bandwidth = getattr(arch, "bandwidth", [900000, 900000])

        logger.debug(
            f"AscendDefaultPolicy initialized for {self.chip_name}:\n"
            f"  AI Cores: {self.num_ai_cores}\n"
            f"  Memory: L0A={self.l0a_capacity // 1024}KB, L0B={self.l0b_capacity // 1024}KB, "
            f"L0C={self.l0c_capacity // 1024}KB, L1={self.l1_capacity // 1024}KB, UB={self.ub_capacity // 1024}KB\n"
            f"  CUBE: {self.cube_shape}, Fractal: {self.fractal_shape}"
        )

    def _get_buffer_capacity(self) -> int:
        """Get total buffer capacity (analogous to smem_cap on GPU)."""
        # For general operations, use UB as the primary buffer
        return self.ub_capacity

    def _get_tile_buffer_capacity(self) -> int:
        """Get buffer capacity for tile validation.

        For AscendDefaultPolicy (vector operations): use UB capacity.
        Subclasses (e.g., AscendCubePolicy) can override to use different capacity.
        """
        buffer_cap = self._get_buffer_capacity()
        logger.debug(f"Using UB buffer capacity: {buffer_cap} bytes")
        return buffer_cap

    def _get_total_l0_capacity(self) -> int:
        """Get total L0 buffer capacity (sum of L0A + L0B + L0C)."""
        return self.l0a_capacity + self.l0b_capacity + self.l0c_capacity

    def _get_l0_capacities(self) -> tuple[int, int, int]:
        """Get individual L0 buffer capacities (L0A, L0B, L0C)."""
        return self.l0a_capacity, self.l0b_capacity, self.l0c_capacity

    def _check_l0_usage(
        self, a_size: int, b_size: int, c_size: int, pipeline_stage: int = 1
    ) -> bool:
        """
        Check if L0 buffer usage is within limits.

        Args:
            a_size: Size of matrix A tile in bytes
            b_size: Size of matrix B tile in bytes
            c_size: Size of output C tile in bytes
            pipeline_stage: Pipeline depth multiplier for double buffering

        Returns:
            True if usage is within L0 capacity limits
        """
        # L0A/L0B need double buffering for pipeline
        a_usage = a_size * pipeline_stage
        b_usage = b_size * pipeline_stage
        # L0C typically doesn't need double buffering (accumulator)
        c_usage = c_size

        return (
            a_usage <= self.l0a_capacity
            and b_usage <= self.l0b_capacity
            and c_usage <= self.l0c_capacity
        )

    def _align_to_l0(self, size: int, buffer_type: str = "a") -> int:
        """
        Align size to L0 buffer alignment requirement.

        Args:
            size: Size in bytes
            buffer_type: "a" for L0A, "b" for L0B, "c" for L0C

        Returns:
            Aligned size in bytes
        """
        if buffer_type == "a":
            alignment = self.l0a_alignment
        elif buffer_type == "b":
            alignment = self.l0b_alignment
        else:  # "c"
            alignment = self.l0c_alignment

        return ((size + alignment - 1) // alignment) * alignment

    def calc_numel_threshold(self, steps):
        dtype_bytes = (self.output_nodes[0].get_dtype().bits + 7) // 8
        print(dtype_bytes)
        self.max_numel_threshold = self.ub_capacity // dtype_bytes

        self.max_total_numel = 1
        for _steps in steps:
            self.max_total_numel *= _steps[len(_steps) - 1]

        self.tiny_kernel = (
            self.max_total_numel < 128 * 1024
        )  # Threshold of tiny kernel, block shape product  < 128 * 1024
        self.stop_numel = (
            min(1024 // dtype_bytes, self.max_total_numel // (self.num_ai_cores * 2))
            if self.tiny_kernel
            else 1024 // dtype_bytes
        )

    def calculate_tile_numel(self, hint):
        tile_numel = 1
        for axis in range(len(hint.block)):
            tile_numel *= hint.block[axis]

        return tile_numel

    # ==================== Tile Search (adapted for SIMD) ====================

    def dfs_smem_tile(self, init_tile, rstep_map) -> Iterable[TileDict]:
        """
        DFS search for tile candidates.

        For SIMD architecture:
        - Prefer tiles aligned to SIMD width
        - Consider DMA alignment for memory efficiency
        - Optimize for AI Core utilization
        """
        _steps = [get_all_factors(n) for n in self.output_nodes[0].get_space_dim()]
        steps = [
            step[step.index(t) :] for step, t in zip(_steps, init_tile, strict=True)
        ]
        logger.debug(f"Initial steps: {steps}")

        self.calc_numel_threshold(steps)

        # Add SIMD-friendly tile sizes (multiples of common SIMD widths)
        simd_friendly_sizes = [16, 32, 64, 128, 256, 512]
        for i in range(len(steps)):
            added = list(
                filter(
                    lambda s: (
                        s < steps[i][-1] * 2 and s > steps[i][0] and s not in steps[i]
                    ),
                    simd_friendly_sizes,
                )
            )
            steps[i].extend(added)
            steps[i] = sorted(steps[i])

        visited_tiles = {}
        queue = PriorityQueue()

        def prio(td: TileDict):
            """Priority function for SIMD architecture."""
            # For SIMD, prioritize:
            # 1. Lower memory traffic
            # 2. Better AI Core utilization
            # 3. SIMD-aligned tiles
            traffic_score = td.traffic + 1
            wave_score = td.num_wave

            # Bonus for SIMD-aligned tiles
            alignment_bonus = 1.0
            for t in td.output_tile:
                if t % 16 == 0:
                    alignment_bonus *= 0.1  # Better alignment = lower priority value

            return traffic_score * wave_score * alignment_bonus

        def add_to_queue(tile):
            if tuple(tile) in visited_tiles:
                return
            td = self.compute_tile_dict(tile, rstep_map)
            visited_tiles[tuple(tile)] = td
            if td.valid:
                queue.put([prio(td), tile])

        add_to_queue(init_tile)
        while not (queue.empty() or len(visited_tiles) > 2000):
            _, tile = queue.get()
            dim_ids = [step.index(t) for step, t in zip(steps, tile, strict=True)]
            for i in reversed(range(len(dim_ids))):
                if dim_ids[i] + 1 < len(steps[i]):
                    new_tile = tile.copy()
                    new_tile[i] = steps[i][dim_ids[i] + 1]
                    add_to_queue(new_tile)

        visited_tiles = filter(lambda td: td.valid, visited_tiles.values())
        sorted_tiles = sorted(visited_tiles, key=lambda td: prio(td))
        logger.debug(
            f"Found {len(sorted_tiles)} valid tile candidates for Ascend target."
        )
        return sorted_tiles

    def compute_tile_dict(self, output_tile: list[int], rstep_map) -> TileDict:
        """
        Compute TileDict for Ascend architecture.

        Key points:
        - Use _get_tile_buffer_capacity() to determine memory constraint
        - Use AI Core count instead of SM count
        """
        td = TileDict(output_tile)
        td.rstep_map = rstep_map
        td.traffic, td.tile_map = self._compute_memory_traffic(output_tile)
        logger.debug(
            f"corresponding to the output tile {output_tile}, "
            f"the traffic is {td.traffic}, tile_map is {td.tile_map}"
        )

        # Compute buffer usage
        td.smem_cost, td.cached_tensors_map = self._compute_shared_memory_usage(td)
        logger.debug(
            f"the shared memory cost is {td.smem_cost},"
            f" cached tensors map is {td.cached_tensors_map}"
        )

        # Get buffer capacity (subclass can override to use different capacity)
        buffer_cap = self._get_tile_buffer_capacity()

        if td.smem_cost > buffer_cap:
            logger.debug(
                f"Tile invalid: smem cost={td.smem_cost} > buffer cap={buffer_cap}"
            )
            td.valid = False
            return td

        output_shape = self.output_nodes[0].get_space_dim()
        td.grid_size = int(
            np.prod(
                [
                    (y + x - 1) // x
                    for x, y in zip(output_tile, output_shape, strict=True)
                ]
            )
        )

        # AI Core parallel execution
        td.block_per_SM = 1
        td.num_wave = int(np.ceil(td.grid_size / self.num_ai_cores))

        return td

    # ==================== Block/Thread Assignment (adapted for SIMD) ====================

    def recommend_block_size(self, td: TileDict):
        """
        For SIMD architecture, block_size concept doesn't apply.
        Return [1] as Ascend uses AI Core level parallelism, not thread parallelism.
        """
        # SIMD: no thread-level parallelism within a tile
        # The "block_size" is effectively 1 (single instruction stream per AI Core)
        return [1]

    def _assign_block_size(self, node: PrimFuncNode, td: TileDict, block_size: int):
        """
        Assign configuration for SIMD execution.

        For SIMD architecture:
        - No thread decomposition (thread = [1, 1, ...])
        - Vectorization is implicit in SIMD operations
        - Focus on DMA-aligned data access
        """
        tile, rsteps = td.get_tile(node), td.get_rstep(node)

        codegen_dict = Hint()
        codegen_dict.block = tile

        # SIMD: no thread-level parallelism
        # thread = [1] means single execution stream per AI Core
        codegen_dict.thread = [1 for _ in tile]

        codegen_dict.rstep = [rsteps[ax.var.name] for ax in node.raxis]
        codegen_dict.reduce_thread = [1 for _ in node.raxis]  # No thread reduction
        codegen_dict.cached_tensors = td.cached_tensors_map.get(node, [])
        codegen_dict.rasterization_plan = NoRasterization()

        # Plan vectorize for SIMD
        codegen_dict.vectorize = self._plan_vectorize_simd(node, td)
        codegen_dict.arch = self.arch
        codegen_dict.opt_shapes = node.get_tag("opt_shapes")

        return codegen_dict

    def _plan_vectorize_simd(self, node: PrimFuncNode, td: TileDict):
        """
        Plan vectorization for SIMD architecture.

        For Ascend SIMD:
        - Vectorization is based on SIMD width (256 bits)
        - Must respect DMA alignment (32 bytes)
        - Different from GPU where vectorization is per-thread
        """

        def is_aligned(shape, factor):
            """Check if shape is aligned to factor."""
            return int(np.prod(shape)) % factor == 0

        def is_dma_aligned(shape, dtype_bytes, vec):
            """Check if access is DMA aligned (32 bytes)."""
            access_bytes = dtype_bytes * vec
            return (
                access_bytes % self.dma_alignment == 0
                or self.dma_alignment % access_bytes == 0
            )

        def is_type_allowed(dtype, vec):
            """Check if vectorization fits SIMD width."""
            return dtype.bits * vec <= self.simd_width

        # SIMD-friendly vector sizes
        vectorize_sizes = [16, 8, 4, 2, 1]
        dtypes = node.get_reduce_inputs_dtype()
        shapes = node.propagate_reduction_inputs(td.get_tile(node), td.get_rstep(node))
        vectorize_result = {}

        for tensor, shape in shapes.items():
            dtype_bytes = (dtypes[tensor].bits + 7) // 8
            for v in vectorize_sizes:
                if (
                    is_aligned(shape, v)
                    and is_dma_aligned(shape, dtype_bytes, v)
                    and is_type_allowed(dtypes[tensor], v)
                ):
                    vectorize_result[tensor] = v
                    break
            if tensor not in vectorize_result:
                vectorize_result[tensor] = 1

        return vectorize_result

    # ==================== Reduce Step Assignment (adapted for SIMD) ====================

    def _assign_reduce_step(self, node):
        """
        Assign reduce step for SIMD architecture.

        For SIMD:
        - Optimize for vector operations
        - Consider DMA alignment for reduction inputs
        """
        if node.reduction_block is None:
            return {}

        raxis = node.raxis
        if len(raxis) == 0:
            return {}

        tile = [1] * len(node.get_space_dim())
        # get possible steps for all nodes in the graph, padding with power of 2 if axis length is not divisible
        all_steps = self.get_node_reduce_step_candidates(node)

        def _score(rstep_id):
            """Score function optimized for SIMD memory access."""
            rstep = {k: all_steps[k][rstep_id[k]] for k in rstep_id}
            score = 0
            shape = node.propagate_inputs(tile, rstep=rstep)

            for i, input_buffer in enumerate(node.input_buffers):
                dtype_bytes = (node.get_buffer_dtype(input_buffer).bits + 7) // 8

                # prefer DMA-aligned access, any other size would not be optimal
                innermost = shape[i][-1] if len(shape[i]) > 0 else 1
                dma_score = (
                    1.0 if (innermost * dtype_bytes) % self.dma_alignment == 0 else 0.5
                )

                # Prefer larger contiguous access
                contiguous_score = coalesced_factor(shape[i], input_buffer.shape)

                score += dma_score * contiguous_score

            return score

        def _enlarge(rstep_id):
            candidates = []
            candidates.append((rstep_id, _score(rstep_id)))
            for ax in rstep_id:
                if rstep_id[ax] + 1 == len(all_steps[ax]):
                    continue
                r = rstep_id.copy()
                r[ax] += 1
                candidates.append((r, _score(r)))
            best = max(candidates, key=lambda x: x[1])
            return best

        cur_rstep_id = {ax.var.name: 0 for ax in raxis}
        cur_score = _score(cur_rstep_id)

        max_iterations = 100
        for _ in range(max_iterations):
            new_rstep, new_score = _enlarge(cur_rstep_id)
            if new_score <= cur_score:
                break
            cur_rstep_id, cur_score = new_rstep, new_score

        rstep = {k: all_steps[k][cur_rstep_id[k]] for k in cur_rstep_id}
        return rstep

    def _expand_reduce_axis(self, td: TileDict):
        """
        Expand reduce axis for SIMD architecture.
        Uses UB capacity instead of smem.
        """
        buffer_limit = self._get_buffer_capacity()
        rstep_map = td.rstep_map.copy()

        def _optimize(node, rstep):
            all_steps = self.get_node_reduce_step_candidates(node)
            for k in all_steps:
                all_steps[k] = list(filter(lambda x: x % rstep[k] == 0, all_steps[k]))

            if any([v == [] for v in all_steps.values()]):
                return rstep

            def _score(rstep_id):
                rstep = {
                    k.var.name: all_steps[k.var.name][rstep_id[k.var.name]]
                    for k in node.raxis
                }
                score = 0
                shape = node.propagate_inputs(td.get_tile(node), rstep=rstep)
                for i, input_buffer in enumerate(node.input_buffers):
                    score += coalesced_factor(shape[i], input_buffer.shape)
                return score

            def _enlarge(rstep_id):
                candidates = []
                for ax in rstep_id:
                    if rstep_id[ax] + 1 == len(all_steps[ax]):
                        continue
                    r = rstep_id.copy()
                    r[ax] += 1
                    candidates.append((r, _score(r)))
                if len(candidates) == 0:
                    return None
                return max(candidates, key=lambda x: x[1])[0]

            cur_rstep_id = {
                k.var.name: all_steps[k.var.name].index(rstep[k.var.name])
                for k in node.raxis
            }
            new_rstep_map = rstep_map.copy()

            while True:
                new_rstep_id = _enlarge(cur_rstep_id)
                if new_rstep_id is None:
                    break
                new_rstep_map[node] = {
                    k.var.name: all_steps[k.var.name][new_rstep_id[k.var.name]]
                    for k in node.raxis
                }
                old_rstep_map = td.rstep_map
                td.rstep_map = new_rstep_map
                buffer_usage, _ = self._compute_shared_memory_usage(td)
                td.rstep_map = old_rstep_map

                if buffer_usage > buffer_limit:
                    break
                else:
                    cur_rstep_id = new_rstep_id

            rstep = {
                k.var.name: all_steps[k.var.name][cur_rstep_id[k.var.name]]
                for k in node.raxis
            }
            return rstep

        for node in self.ordered_nodes:
            if len(node.raxis) > 0:
                rstep = _optimize(node, rstep_map.get(node, {}))
                rstep_map[node] = rstep

        td.rstep_map = rstep_map
        td.smem_cost, td.cached_tensors_map = self._compute_shared_memory_usage(td)

    def check_tile_shape_isvalid(self, td: TileDict):
        """
        Check if tile shape is valid for SIMD architecture.

        For Ascend SIMD:
        - Check buffer capacity (UB/L0)
        - Prefer SIMD-aligned tiles (not strictly required)
        """
        # Check buffer capacity
        buffer_cap = self._get_buffer_capacity()
        if td.smem_cost > buffer_cap:
            logger.debug(
                f"Tile invalid: smem cost={td.smem_cost} > buffer cap={buffer_cap}"
            )
            return False

        # Check that tiles can produce meaningful work
        for node in self.ordered_nodes:
            tile = td.get_tile(node)
            # Ensure positive tile dimensions
            if any(t <= 0 for t in tile):
                logger.debug(
                    f"Tile invalid: non-positive tile dimension in {tile} for node {node.name}"
                )
                return False

        return True

    def plan_rasterization(self, td: TileDict):
        """Ascend doesn't need GPU-style rasterization."""
        return NoRasterization()


# ============================================================================
# AscendCubePolicy: CUBE unit optimization (similar to TensorCorePolicy)
# ============================================================================


class AscendCubePolicy(AscendDefaultPolicy):
    """
    Policy for Ascend NPU with CUBE unit optimization.

    CUBE unit is analogous to TensorCore on NVIDIA GPU:
    - 16x16x16 matrix operation in fractal format
    - Optimized for matmul and similar operations
    - Uses L0A/L0B for inputs, L0C for accumulator

    This policy inherits from AscendDefaultPolicy (SIMD-adapted)
    and adds CUBE-specific optimizations similar to TensorCorePolicy.
    """

    # CUBE unit parameters
    pipeline_stage: int = 2
    use_async_copy: bool = False
    block_reduction_depth: int | None = None

    @property
    def cube_k(self) -> int:
        return self.cube_dim

    def _init_with_prim_func(
        self,
        func: tvm.tir.PrimFunc,
        name: str = "PrimFuncNode",
        custom_mem_mul: float = 1,
    ):
        super()._init_with_prim_func(func, name, custom_mem_mul)
        self._legalize_cube_info()
        return self

    def _legalize_cube_info(self):
        """Legalize CUBE configuration from tags."""
        pipeline_stage = self.prim_func_node.get_tag("pipeline_stage")
        if pipeline_stage:
            self.pipeline_stage = int(pipeline_stage)

        block_reduction_depth = self.prim_func_node.get_tag("block_reduction_depth")
        if block_reduction_depth:
            self.block_reduction_depth = int(block_reduction_depth)

    def _compute_cube_strides(
        self,
        node: PrimFuncNode,
        tile: list[int],
        rstep: dict[str, int] | None = None,
    ) -> tuple[Stride, Stride, Stride]:
        """Compute strides for CUBE operation (similar to TensorCore strides)."""
        if rstep is None:
            rstep = {}

        shapes = node.propagate_reduction_inputs(tile, rstep)
        AS_shape, BS_shape = shapes.values()
        CS_shape = tile
        A_ax_m, A_ax_k, B_ax_k, B_ax_n, C_ax_m, C_ax_n = node.infer_tensorcore_axis()

        # For Ascend's 32-byte DMA alignment
        offset = 8
        A_high_ax = min(A_ax_m, A_ax_k)
        B_high_ax = min(B_ax_n, B_ax_k)
        C_high_ax = min(C_ax_m, C_ax_n)
        A_stride = Stride(
            stride=np.prod(AS_shape[A_high_ax + 1 :]) + offset, ax=A_high_ax
        )
        B_stride = Stride(
            stride=np.prod(BS_shape[B_high_ax + 1 :]) + offset, ax=B_high_ax
        )
        C_stride = Stride(
            stride=np.prod(CS_shape[C_high_ax + 1 :]) + offset, ax=C_high_ax
        )
        return A_stride, B_stride, C_stride

    def infer_node_smem_usage(self, td: TileDict, node: PrimFuncNode):
        """Override to account for CUBE pipeline stages."""
        value, cached_tensors = super().infer_node_smem_usage(td, node)
        value *= self.pipeline_stage
        return value, cached_tensors

    def _get_tile_buffer_capacity(self) -> int:
        """Override to use L1 capacity for CUBE operations.

        CUBE operations use L0A/L0B/L0C buffers, which are fed from L1.
        Therefore, L1 capacity is the actual constraint for CUBE tile sizes.
        """
        buffer_cap = self.l1_capacity
        logger.debug(
            f"Using L1 buffer capacity for CUBE operations: {buffer_cap} bytes"
        )
        return buffer_cap

    def _assign_reduce_step(self, node):
        """Assign reduce step for CUBE operations."""
        if not node.get_tag("tensorcore_config"):
            logger.debug(
                "Ascend cube computation doesn't find necessary tensorcore_config tag, use default reduce steps."
            )
            return super()._assign_reduce_step(node)

        # For CUBE nodes, align to cube_k
        target_transaction = 512  # 512 bytes optimal for Ascend L0x DMA
        reduce_input_dtype = node.get_buffer_dtype(
            node.block_analyzer.get_input_buffers(node.reduction_block)[0]
        )
        basic = (target_transaction * 8) // reduce_input_dtype.bits

        result = {}
        for iter_info in node.raxis:
            iter_name = iter_info.var.name
            iter_dom = iter_info.dom.extent
            if iter_dom % self.cube_dim > 0:
                result[iter_name] = self.cube_dim if iter_dom < basic else basic
            elif iter_dom % basic == 0:
                result[iter_name] = basic
            else:
                return super()._assign_reduce_step(node)
        return result

    def get_node_reduce_step_candidates(self, node):
        """Get reduce step candidates for CUBE operations."""
        if not node.get_tag("tensorcore_config"):
            return super().get_node_reduce_step_candidates(node)
        else:
            return {
                k.var.name: [
                    x * self.cube_k
                    for x in get_all_factors(int(k.dom.extent) // self.cube_k)
                ]
                for k in node.raxis
            }

    def _check_node_l0_capacity(self, node: PrimFuncNode, td: TileDict) -> bool:
        """
        Check if a CUBE node's tile fits in L0 buffers.

        Args:
            node: The PrimFuncNode to check (must have tensorcore_config)
            td: The TileDict containing tile configuration

        Returns:
            True if L0A/L0B/L0C usage is within capacity
        """
        tile = td.get_tile(node)
        rstep = td.get_rstep(node)
        shapes = node.propagate_reduction_inputs(tile, rstep)

        if len(shapes) < 2:
            return True

        # Get tile shapes for A, B, C
        input_shapes = list(shapes.values())
        a_shape, b_shape = input_shapes[0], input_shapes[1]
        c_shape = tile

        # Get actual dtypes from buffers
        input_buffers = node.block_analyzer.get_input_buffers(node.reduction_block)
        output_buffers = node.block_analyzer.get_output_buffers(node.reduction_block)

        a_dtype = node.get_buffer_dtype(input_buffers[0])
        b_dtype = (
            node.get_buffer_dtype(input_buffers[1])
            if len(input_buffers) > 1
            else a_dtype
        )
        c_dtype = (
            node.get_buffer_dtype(output_buffers[0]) if output_buffers else a_dtype
        )

        a_dtype_bytes = (a_dtype.bits + 7) // 8
        b_dtype_bytes = (b_dtype.bits + 7) // 8
        c_dtype_bytes = (c_dtype.bits + 7) // 8

        # Calculate buffer sizes
        a_size = int(np.prod(a_shape)) * a_dtype_bytes
        b_size = int(np.prod(b_shape)) * b_dtype_bytes
        c_size = int(np.prod(c_shape)) * c_dtype_bytes

        # Check with alignment and pipeline stage
        a_aligned = self._align_to_l0(a_size, "a")
        b_aligned = self._align_to_l0(b_size, "b")
        c_aligned = self._align_to_l0(c_size, "c")

        if not self._check_l0_usage(
            a_aligned, b_aligned, c_aligned, self.pipeline_stage
        ):
            logger.debug(
                f"L0 usage exceeds capacity for node {node.name}: "
                f"L0A={a_aligned * self.pipeline_stage}/{self.l0a_capacity}, "
                f"L0B={b_aligned * self.pipeline_stage}/{self.l0b_capacity}, "
                f"L0C={c_aligned}/{self.l0c_capacity}"
            )
            return False

        return True

    def check_tile_shape_isvalid(self, td: TileDict):
        """Check if tile shape is valid for CUBE operations."""
        # Check CUBE tile shape constraints
        for node in self.ordered_nodes:
            if node.get_tag("tensorcore_config"):
                ax_m, ax_n = node.get_tag("tensorcore_config")
                # Check if indices are within bounds of the tile map
                if ax_m >= len(td.tile_map[node]) or ax_n >= len(td.tile_map[node]):
                    logger.debug(
                        f"Tile invalid for CUBE: ax_m={ax_m} or ax_n={ax_n} out of bounds"
                    )
                    return False
                block_m, block_n = (
                    td.tile_map[node][ax_m],
                    td.tile_map[node][ax_n],
                )
                # CUBE requires minimum tiles
                if block_m < self.fractal_shape[0] or block_n < self.fractal_shape[1]:
                    logger.debug(
                        f"Tile invalid for CUBE: block_m={block_m} < {self.fractal_shape[0]}"
                        f" or block_n={block_n} < {self.fractal_shape[1]}"
                    )
                    return False
                if any(
                    [
                        y % x
                        for x, y in zip(
                            td.tile_map[node], node.get_space_dim(), strict=True
                        )
                    ]
                ):
                    logger.debug(
                        f"Tile invalid for CUBE: tile {td.tile_map[node]} "
                        f"not divisible by shape {node.get_space_dim()}"
                    )
                    return False

        # Check L0 buffer capacity for each CUBE node (必须单独检查每个node)
        for node in self.ordered_nodes:
            if not isinstance(node, PrimFuncNode):
                continue
            if not node.get_tag("tensorcore_config"):
                continue

            if not self._check_node_l0_capacity(node, td):
                return False

        return super().check_tile_shape_isvalid(td)

    def _assign_block_size(self, node: PrimFuncNode, td: TileDict, block_size: int):
        """Assign configuration for CUBE operations."""
        if not node.get_tag("tensorcore_config"):
            return super()._assign_block_size(node, td, block_size)

        ax_m, ax_n = node.get_tag("tensorcore_config")
        tile, rsteps = td.get_tile(node), td.get_rstep(node)
        ndim = len(tile)

        # CUBE tile size is cube_dim x cube_dim
        cube_m, cube_n = self.fractal_shape[0], self.fractal_shape[1]
        cube_tile = [1 for _ in range(ndim)]
        cube_tile[ax_m] = cube_m
        cube_tile[ax_n] = cube_n

        if tile[ax_m] < cube_m or tile[ax_n] < cube_n:
            logger.debug(
                f"Tile invalid for CUBE assignment: tile {tile} smaller than cube size "
            )
            return None

        codegen_dict = Hint()
        codegen_dict.block = tile
        codegen_dict.warp = cube_tile
        codegen_dict.thread = [1 for _ in tile]  # SIMD: no threads
        codegen_dict.use_tc = True
        codegen_dict.pipeline_stage = self.pipeline_stage
        codegen_dict.block_reduction_depth = self.block_reduction_depth
        codegen_dict.use_async = self.use_async_copy
        codegen_dict.rstep = [int(rsteps[ax.var.name]) for ax in node.raxis]
        codegen_dict.reduce_thread = [1 for _ in node.raxis]
        codegen_dict.cached_tensors = td.cached_tensors_map.get(node, [])
        codegen_dict.rasterization_plan = NoRasterization()

        intrin_info = node.get_tag("intrin_info")
        if intrin_info:
            codegen_dict.intrin_info = IntrinInfo(**intrin_info)

        codegen_dict.complete_config(node)
        # CUBE operations don't need vectorization, skip vectorization planning
        codegen_dict.vectorize = {}
        codegen_dict.arch = self.arch
        codegen_dict.opt_shapes = node.get_tag("opt_shapes")

        if hasattr(codegen_dict, "tensorcore_legalization"):
            codegen_dict.tensorcore_legalization()

        return codegen_dict

    def _expand_reduce_axis(self, td: TileDict):
        """Expand reduce axis for CUBE operations."""

        def _check_small_tile(td: TileDict):
            minimal_threshold = 32
            for node in self.ordered_nodes:
                tile = td.get_tile(node)
                if any([t <= minimal_threshold for t in tile]):
                    return True
            return False

        if _check_small_tile(td):
            total_l0 = self._get_total_l0_capacity()
            l0_limit = total_l0  # No SM partitioning on Ascend
            rstep_map = td.rstep_map.copy()

            def _optimize(node, rstep):
                all_steps = self.get_node_reduce_step_candidates(node)
                for k in all_steps:
                    all_steps[k] = list(
                        filter(lambda x: x % rstep[k] == 0, all_steps[k])
                    )
                if any([v == [] for v in all_steps.values()]):
                    return rstep

                def _score(rstep_id):
                    rstep = {
                        k.var.name: all_steps[k.var.name][rstep_id[k.var.name]]
                        for k in node.raxis
                    }
                    score = 0
                    shape = node.propagate_inputs_on_reduction(
                        td.get_tile(node), rstep=rstep
                    )
                    input_buffers = node.block_analyzer.get_input_buffers(
                        node.reduction_block
                    )
                    for i, input_buffer in enumerate(input_buffers):
                        score += coalesced_factor(shape[i], input_buffer.shape)
                    return score

                def _enlarge(rstep_id):
                    candidates = []
                    for ax in rstep_id:
                        if rstep_id[ax] + 1 == len(all_steps[ax]):
                            continue
                        r = rstep_id.copy()
                        r[ax] += 1
                        candidates.append((r, _score(r)))
                    if len(candidates) == 0:
                        return None
                    return max(candidates, key=lambda x: x[1])[0]

                cur_rstep_id = {
                    k.var.name: all_steps[k.var.name].index(rstep[k.var.name])
                    for k in node.raxis
                }

                while True:
                    new_rstep_id = _enlarge(cur_rstep_id)
                    if new_rstep_id is None:
                        break
                    new_rstep = {
                        k.var.name: all_steps[k.var.name][new_rstep_id[k.var.name]]
                        for k in node.raxis
                    }
                    old_rstep = td.rstep_map
                    td.rstep_map = {node: new_rstep}
                    l0_usage, _ = self.infer_node_smem_usage(td, node)
                    td.rstep_map = old_rstep

                    if l0_usage > l0_limit:
                        break
                    else:
                        cur_rstep_id = new_rstep_id

                rstep = {
                    k.var.name: all_steps[k.var.name][cur_rstep_id[k.var.name]]
                    for k in node.raxis
                }
                return rstep

            for node in self.ordered_nodes:
                if len(node.raxis) > 0:
                    rstep = _optimize(node, rstep_map.get(node, {}))
                    rstep_map[node] = rstep

            td.rstep_map = rstep_map
            td.smem_cost, td.cached_tensors_map = self._compute_shared_memory_usage(td)

        if self.block_reduction_depth is not None:

            def _expand_with_tags(rstep):
                return {k: v * self.block_reduction_depth for k, v in rstep.items()}

            rstep_map = td.rstep_map.copy()
            for node in self.ordered_nodes:
                if len(node.raxis) > 0 and node in rstep_map:
                    rstep_map[node] = _expand_with_tags(rstep_map[node])
            td.rstep_map = rstep_map
