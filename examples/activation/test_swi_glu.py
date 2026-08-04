import os
import subprocess
import sys

import pytest
import torch


@pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="requires an Ascend NPU",
)
def test_swi_glu_run():
    here = os.path.dirname(os.path.abspath(__file__))
    r = subprocess.run(
        [sys.executable, os.path.join(here, "swi_glu.py")],
        capture_output=True,
        text=True,
        timeout=600,
        env={**os.environ, "TILELANG_CLEAR_CACHE": "1"},
    )
    assert r.returncode == 0, (
        f"swi_glu.py exited with {r.returncode}\n--- stdout (tail) ---\n{r.stdout[-1000:]}\n--- stderr (tail) ---\n{r.stderr[-2000:]}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
