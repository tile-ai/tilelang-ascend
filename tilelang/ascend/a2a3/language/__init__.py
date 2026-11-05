"""TileLang-Ascend A2/A3 dialect on the common eager frontend."""

from tilelang.language.common import *  # noqa: F401,F403
from .kernel import *  # noqa: F401,F403
from .allocate import *  # noqa: F401,F403
from .annotations import annotate_address  # noqa: F401
from .constant import allocate_const, decl_buffer  # noqa: F401
from .warpgroup import *  # noqa: F401,F403
from ..layout import Layout  # noqa: F401
from .copy_op import c2d_im2col, npu_copy_v2 as copy  # noqa: F401
from .customize import (  # noqa: F401
    atomic_add,
    atomic_addx2,
    atomic_addx4,
    dp4a,
    clamp,
    reshape,
    view,
    npu_gemm as mma,
    loop_break,
)
from .ascend import *  # noqa: F401,F403
from .ascend import _src_code  # noqa: F401
from .reduce_ascend import *  # noqa: F401,F403
from . import ascend_tile as tile  # noqa: F401

from . import pto as pto  # noqa: F401

from .pipeline import Pipelined  # noqa: F401
from .swizzle import npu_use_swizzle, use_swizzle  # noqa: F401
from .persistent import Persistent  # noqa: F401
