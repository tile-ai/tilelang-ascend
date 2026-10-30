"""Repeat an existing device regression without recompiling its kernel."""


def repeat_launch(kernel, arguments, outputs, check, repeat=1, reset=None):
    """Poison every output and apply the caller's original golden each launch."""
    import torch

    if repeat < 1:
        raise ValueError("repeat must be positive")
    for _ in range(repeat):
        for output in outputs:
            output.fill_(float("nan") if output.is_floating_point() else -99)
        if reset is not None:
            reset()
        torch.npu.synchronize()
        kernel(*arguments)
        torch.npu.synchronize()
        check()
