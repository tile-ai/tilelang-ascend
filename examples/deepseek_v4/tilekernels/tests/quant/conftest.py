# Ascend NPU test conftest
# Override root conftest to avoid loading GPU-dependent benchmark plugin
# which triggers `tile_kernels` full import (incompatible with NPU tilelang).
#
# This file MUST exist to prevent pytest from loading the root conftest's
# pytest_plugins (pytest_benchmark_plugin → tile_kernels → engram → error).

collect_ignore_glob = []

# Register the 'benchmark' marker so pytest doesn't warn about unknown markers
def pytest_configure(config):
    config.addinivalue_line("markers", "benchmark: mark test as performance benchmark")
