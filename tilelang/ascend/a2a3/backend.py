"""Compiler manifests for the A2/A3 AscendC and PTO variants."""

from tilelang.backend.module import BackendModule, register_backend

from . import codegen, execution_backend, pipeline
from .target import target_is_ascend_a2a3, target_is_pto_a2a3


BACKEND = register_backend(
    BackendModule(
        name="ascend_a2a3",
        target_kinds=("ascend",),
        supports_target=target_is_ascend_a2a3,
        pipelines={"ascend": pipeline.ASCENDC_PIPELINE},
        device_codegens={"ascend": codegen.ASCENDC_CODEGEN},
        execution_backends=execution_backend.EXECUTION_BACKENDS,
    )
)

PTO_BACKEND = register_backend(
    BackendModule(
        name="pto_a2a3",
        target_kinds=("ascend",),
        supports_target=target_is_pto_a2a3,
        pipelines={"ascend": pipeline.PTO_PIPELINE},
        device_codegens={"ascend": codegen.PTO_CODEGEN},
        execution_backends=execution_backend.EXECUTION_BACKENDS,
    )
)
