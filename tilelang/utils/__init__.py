# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""The profiler and convert to torch utils"""

from .target import determine_target  # noqa: F401
from .tensor import TensorSupplyType, torch_assert_close, map_torch_type  # noqa: F401
from .language import (
    is_global,  # noqa: F401
    is_shared,  # noqa: F401
    is_shared_dynamic,  # noqa: F401
    is_fragment,  # noqa: F401
    is_local,  # noqa: F401
    array_reduce,  # noqa: F401
)
from .deprecated import deprecated  # noqa: F401
from .ascend_npu import (
    NPUUtils,
    get_ascend_path,
    get_cxx,
    get_npucompiler_path,
    get_npucompiler_opt_path,
    get_bisheng_path
)  # noqa: F401
