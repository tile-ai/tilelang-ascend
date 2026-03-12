# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""
Common utilities for TileLangIR passes (transforms and conversion).
Provides create_pass_runner, pass_fn, and run logic for str/Module.
"""


def _format_pass_option_value(value) -> str:
    """Format a Python value for MLIR pipeline option (e.g. bool -> 'true'/'false')."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _pass_spec(pass_name: str, **options) -> str:
    """Build pipeline spec: builtin.module(pass-name) or builtin.module(pass-name{k=v,...})."""
    if not options:
        return f"builtin.module({pass_name})"
    opts = ",".join(
        f"{k.replace('_', '-')}={_format_pass_option_value(v)}" for k, v in options.items()
    )
    return f"builtin.module({pass_name}{{{opts}}})"


def create_pass_runner(pass_name: str, **options):
    """
    Create a pass runner via tladapter native (libtilelangir). The returned callable has
    pass_name attached (from pybind) for dump/debug. Accepts only str input.
    """
    try:
        import tilelang.tladapter as _tladapter
    except ImportError as e:
        raise ImportError(
            "TileLang adapter passes require native module (libtilelangir). "
            "Build with USE_NPUIR and set PYTHONPATH to tilelangir build dir."
        ) from e
    _native = getattr(_tladapter, "_native", None)
    if _native is None or not hasattr(_native, "create_pass_runner"):
        raise RuntimeError(
            "tladapter native module or create_pass_runner not found; rebuild tilelangir."
        )
    spec = _pass_spec(pass_name, **options)
    return _native.create_pass_runner(pass_name, spec)


def run_pass(x, pass_name: str, **options):
    """Run pass using create_pass_runner. Handles str and Module."""
    runner = create_pass_runner(pass_name, **options)
    if isinstance(x, str):
        return runner(x)
    s = runner(str(x))
    ctx = getattr(x, "context", None)
    if ctx is None:
        raise RuntimeError("module has no context attribute")
    parse = type(x).parse
    try:
        return parse(s, ctx)
    except TypeError:
        return parse(ctx, s)


class _PassWithNameFromPybind:
    """Callable with pass_name from pybind. Created via pass_fn()."""

    def __init__(self, pass_name, **default_options):
        self._pass_name = pass_name
        self._default_options = default_options
        self._cached_runner = None

    @property
    def pass_name(self):
        if self._cached_runner is None:
            self._cached_runner = create_pass_runner(
                self._pass_name, **self._default_options
            )
        return self._cached_runner.pass_name

    def __call__(self, *args, **options):
        if not args:
            return create_pass_runner(self._pass_name, **{**self._default_options, **options})
        merged = {**self._default_options, **options}
        return run_pass(args[0], self._pass_name, **merged)


def pass_fn(pass_name: str, **options):
    """Create a pass function with pass_name from pybind. One line per pass."""
    return _PassWithNameFromPybind(pass_name, **options)
