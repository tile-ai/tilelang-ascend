# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

from typing import List, Optional, Union
from tvm import tir, IRModule
from tvm.tir import PrimFunc
from .arch import TileDevice
from .roller.policy import (
    TensorCorePolicy,
    DefaultPolicy,
    AscendCubePolicy,
    AscendDefaultPolicy,
)
from .roller.hint import Hint
from .roller.node import OutputNode
from .matmul_analysis import get_tensorized_func_and_tags
import logging

logger = logging.getLogger(__name__)


def _detect_operator_type(func: tir.PrimFunc) -> str:
    """Detect the operator type from PrimFunc attributes and structure.

    Returns:
        'conv2d': Convolution-like operators
        'batch_matmul': Batched matrix multiplication
        'matmul': Standard matrix multiplication
        'unknown': Cannot determine
    """
    # Check function attributes for operator hints
    if func.attrs is not None:
        # Check for explicit operator type annotation
        if "operator_name" in func.attrs:
            op_name = str(func.attrs["operator_name"]).lower()
            if "conv" in op_name:
                return "conv2d"
            elif "batch" in op_name and "matmul" in op_name:
                return "batch_matmul"
            elif "matmul" in op_name or "gemm" in op_name:
                return "matmul"

        # Check for transform hints (conv2d often has im2col transforms)
        if (
            "input_transform_kind" in func.attrs
            or "weight_transform_kind" in func.attrs
        ):
            return "conv2d"

    # Analyze buffer shapes and access patterns
    try:
        # Get the main block
        sch = tir.Schedule(func)
        blocks = sch.get_child_blocks(sch.get_block("root"))

        if len(blocks) > 0:
            main_block_stmt = sch.get(blocks[0])

            # Conv2d typically has > 4 dimensions or complex spatial patterns
            if len(main_block_stmt.iter_vars) > 4:
                # Check if there are multiple spatial dimensions (H, W, etc.)
                spatial_dims = sum(
                    1
                    for iv in main_block_stmt.iter_vars
                    if iv.iter_type == tir.IterVar.DataPar
                )
                if spatial_dims > 3:  # Batch + OutH + OutW + Channel = 4+
                    return "conv2d"

            # BatchMatmul typically has exactly 4 dimensions (Batch, M, N, K)
            # with first dimension appearing in all tensors
            if (
                len(main_block_stmt.iter_vars) == 4
                and len(main_block_stmt.reads) == 2
                and len(main_block_stmt.writes) == 1
            ):
                # Check if first dimension is batch (appears in A, B, C)
                return "batch_matmul"
    except Exception:
        pass

    return "matmul"  # Default to standard matmul


def _is_already_normalized_matmul(func: tir.PrimFunc) -> bool:
    """Check if the function is already in normalized matmul form.

    A normalized matmul should have:
    1. Exactly 3-4 dimensions (Batch optional, M, N, K)
    2. Simple affine access patterns (no complex indexing like h+r, w+s in convolution)
    3. Standard GEMM structure: C[...i, j] += A[...i, k] * B[...j, k] or B[...k, j]

    Returns:
        True if already normalized, False if needs normalization
    """
    try:
        sch = tir.Schedule(func)
        blocks = sch.get_child_blocks(sch.get_block("root"))

        if len(blocks) != 1:
            return False

        main_block_stmt = sch.get(blocks[0])

        # Check basic structure: 2 reads, 1 write
        if len(main_block_stmt.reads) != 2 or len(main_block_stmt.writes) != 1:
            return False

        # Check number of dimensions: should be 3 or 4 (with optional batch)
        num_iters = len(main_block_stmt.iter_vars)
        if num_iters < 3 or num_iters > 4:
            return False

        # Check if access patterns are simple (no complex expressions like h+r)
        def has_simple_access(region) -> bool:
            """Check if all accesses are simple variable references"""
            return all(isinstance(r.min, tir.Var) for r in region)

        for read in main_block_stmt.reads:
            if not has_simple_access(read.region):
                return False

        # If all checks pass, it's likely already normalized
        return has_simple_access(main_block_stmt.writes[0].region)

    except Exception:
        return False


def get_optimal_layout_for_ascend(func: tir.PrimFunc) -> List[str]:
    """Get optimal layout configuration for Ascend Cube based on operator type.

    Args:
        func: The PrimFunc to analyze

    Returns:
        Layout specification [A_layout, B_layout, C_layout]
        - 'n': Normal layout [S, I, K] or [S, K, J] or [S, I, J]
        - 't': Transpose layout (swap last two dimensions)
    """
    op_type = _detect_operator_type(func)

    if op_type == "conv2d":
        # Conv2d similar: Weight is often pre-transformed to Fractal format
        # Use normal layout to avoid redundant transpose
        # L1->L0B will still use hardware transpose if needed
        logger.debug("Detected Conv2d operator, using layout ['n', 'n', 'n']")
        return ["n", "n", "n"]
    else:
        # Standard Matmul, Batch_Matmul and unknown ops: Use NT mode (most common)
        # This matches cuBLAS and most BLAS libraries convention
        logger.debug(
            f"Detected {op_type} operator, using default layout ['n', 't', 'n']"
        )
        return ["n", "t", "n"]


def get_rasterization_code(pannel_width: int = 8) -> str:
    return f"""
        const int MAX_BLOCK_N = {pannel_width};
        const auto baseBlockIdx = blockIdx.x + gridDim.x *blockIdx.y;
        const auto totalPanel = (gridDim.x * gridDim.y +MAX_BLOCK_N * gridDim.x - 1) / (MAX_BLOCK_N * gridDim.x);
        const auto totalBlock = gridDim.x * gridDim.y;
        const auto panelIdx = baseBlockIdx / (MAX_BLOCK_N *gridDim.x);
        const auto strideLd = panelIdx + 1 < totalPanel ?MAX_BLOCK_N : (totalBlock - panelIdx * (MAX_BLOCK_N *gridDim.x)) / gridDim.x;
        const auto bx = (panelIdx & 1) ? gridDim.x -(baseBlockIdx - panelIdx * MAX_BLOCK_N * gridDim.x) /strideLd - 1 : (baseBlockIdx - panelIdx * MAX_BLOCK_N *gridDim.x) / strideLd;
        const auto by = (baseBlockIdx - panelIdx * MAX_BLOCK_N *gridDim.x) % strideLd + panelIdx * MAX_BLOCK_N;
        const auto bz = blockIdx.z;
        const dim3 blockIdx(bx, by, bz);
    """


def get_roller_hints_from_func(
    func_or_module: Union[tir.PrimFunc, IRModule],
    arch: TileDevice,
    topk: int = 10,
    tensorcore_only: bool = False,
    allow_gemv: bool = False,
    custom_mem_mul: float = 1,
) -> Optional[List[Hint]]:
    func = None
    if isinstance(func_or_module, tir.PrimFunc):
        func = func_or_module
    elif isinstance(func_or_module, IRModule):
        func = retrieve_func_from_module(func_or_module)
    else:
        raise ValueError("Not supported type: ", type(func_or_module))

    assert func is not None, "The function should not be None"

    roller_hints = None

    # Ascend platform handling
    if arch.platform == "ascend":
        tensorized_func = None
        tags = None
        try:
            # Get optimal layout based on operator type
            layout = get_optimal_layout_for_ascend(func)

            # Check if already normalized - can skip expensive reindex operations
            is_normalized = _is_already_normalized_matmul(func)
            if is_normalized:
                logger.debug("Detected already normalized matmul, using fast path")

            tensorized_func, tags = get_tensorized_func_and_tags(
                func,
                arch.target,
                layout=layout,
                skip_normalize=is_normalized,
                allow_gemv=allow_gemv,
            )
        except Exception as e_msg:
            logger.debug(f"Get tensorized func and tags failed: {e_msg}")
            tags = None

        if tags and tensorized_func:
            # Use AscendPolicy for CUBE optimization
            policy = AscendCubePolicy.from_prim_func(
                func=tensorized_func,
                arch=arch,
                tags=tags,
                custom_mem_mul=custom_mem_mul,
            )
            roller_hints = policy.emit_config(topk)
        else:
            # Fallback to AscendDefaultPolicy for non-CUBE operations
            policy = AscendDefaultPolicy.from_prim_func(
                func=func, arch=arch, custom_mem_mul=custom_mem_mul
            )
            roller_hints = policy.emit_config(topk)
        return roller_hints

    # GPU/Other platforms handling
    if tensorcore_only:
        try:
            tensorized_func, tags = get_tensorized_func_and_tags(
                func, arch.target, allow_gemv=allow_gemv
            )
        except Exception as e_msg:
            logger.debug("Get tensorized func and tags failed: ", e_msg)
            tags = None
        if tags and tensorized_func:
            policy = TensorCorePolicy(func=tensorized_func, arch=arch, tags=tags)
            roller_hints = policy.emit_config(topk)
        else:
            roller_hints = None
    else:
        policy = DefaultPolicy.from_prim_func(func=func, arch=arch)
        tensorized_func = None
        try:
            tensorized_func, tags = get_tensorized_func_and_tags(
                func, arch.target, allow_gemv=allow_gemv
            )
        except Exception as e_msg:
            logger.debug("Get tensorized func and tags failed: ", e_msg)
            tags = None
        if tags and tensorized_func:
            policy = TensorCorePolicy.from_prim_func(
                func=tensorized_func, arch=arch, tags=tags
            )
        roller_hints = policy.emit_config(topk)
    return roller_hints


def get_roller_hints_from_output_nodes(
    output_nodes: List[OutputNode],
    arch: TileDevice,
    topk: int = 10,
    extra_tags: Optional[List[str]] = None,
) -> Optional[List[Hint]]:
    assert isinstance(output_nodes, list), "The input should be a list of functions."

    lints = []

    # Ascend platform handling
    if arch.platform == "ascend":
        try:
            # Try to use AscendCubePolicy for CUBE-optimized operations
            policy = AscendCubePolicy.from_output_nodes(
                output_nodes, arch=arch, tags=None
            )
            lints = policy.emit_config(topk)
        except Exception as e_msg:
            logger.debug(
                f"AscendCubePolicy failed: {e_msg}, fallback to AscendDefaultPolicy"
            )

        if len(lints) == 0:
            # Fallback to default policy for non-CUBE operations
            policy = AscendDefaultPolicy.from_output_nodes(
                output_nodes, arch=arch, tags=None
            )
            lints = policy.emit_config(topk)
        return lints

    # GPU/Other platforms handling
    try:
        policy = TensorCorePolicy.from_output_nodes(output_nodes, arch=arch, tags=None)
        lints = policy.emit_config(topk)
    except Exception as e_msg:
        logger.debug(
            f"Generate hints from output nodes failed: {e_msg}, "
            "fallback to default policy"
        )

    if len(lints) == 0:
        policy = DefaultPolicy.from_output_nodes(output_nodes, arch=arch, tags=None)
        lints = policy.emit_config(topk)
    return lints


def retrieve_func_from_module(ir_module: IRModule) -> PrimFunc:
    if not isinstance(ir_module, IRModule):
        raise ValueError("Not supported type: ", type(ir_module))
    assert len(ir_module.get_global_vars()) == 1, (
        "The optimized module should only have one global variable for default schedule."
    )
    func = list(ir_module.functions.values())[0]
    return func
