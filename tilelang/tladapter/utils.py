# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""
Common utilities for TileLangIR passes (transforms and conversion).

Provides Pipeline for batched pass execution, and pass_fn for
defining individual passes that can be composed into a Pipeline.
"""


def _format_pass_option_value(value) -> str:
    """Format a Python value for MLIR pipeline option (e.g. bool -> 'true'/'false')."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _build_pipeline_text(pass_name: str, anchor: str | None = None, **options) -> str:
    """Build inner MLIR pipeline text for a single pass.

    Returns the text to be executed under a builtin.module root PassManager,
    e.g. ``"pass-name"`` or ``"func.func(pass-name{k=v,...})"``.
    """
    inner = pass_name
    if options:
        opts = ",".join(
            f"{k.replace('_', '-')}={_format_pass_option_value(v)}"
            for k, v in options.items()
        )
        inner = f"{pass_name}{{{opts}}}"
    if anchor:
        inner = f"{anchor}({inner})"
    return inner


def _get_native_module():
    """Import and return the native tladapter pybind module."""
    try:
        import tilelang.tladapter as _tladapter
    except ImportError as e:
        raise ImportError(
            "TileLang adapter passes require native module (libtilelangir). "
            "Build with USE_NPUIR and set PYTHONPATH to tilelangir build dir."
        ) from e
    _native = getattr(_tladapter, "_native", None)
    if _native is None or not hasattr(_native, "PassPipeline"):
        raise RuntimeError(
            "tladapter native module or PassPipeline not found; rebuild tilelangir."
        )
    return _native


# ---------------------------------------------------------------------------
# Pipeline: batched pass execution
# ---------------------------------------------------------------------------


class Pipeline:
    """Batched MLIR pass pipeline.

    Add passes via ``add()`` using standard MLIR textual pipeline format
    , then call ``run()`` once.  All passes share a single parse/serialize cycle.

    Debug:
      - ``enable_ir_printing()``  - print IR to stderr after each pass.
      - ``enable_ir_printing_to_file_tree(dir)`` - write per-pass IR to files.
    """

    def __init__(self):
        self._pp = _get_native_module().PassPipeline()

    def add(self, pass_or_text, **options) -> "Pipeline":
        """Add a pass to the pipeline.
        Example: ``"func.func(my-pass{k=v})"``.
        """
        if isinstance(pass_or_text, str):
            if options:
                raise ValueError(
                    "options not supported with raw pipeline text; "
                    "embed them in the string, e.g. 'my-pass{k=v}'"
                )
            self._pp.add(pass_or_text)
        elif isinstance(pass_or_text, _PassDescriptor):
            self._pp.add(pass_or_text._make_pipeline_text(**options))
        else:
            raise TypeError(f"expected str or pass_fn object, got {type(pass_or_text)}")
        return self

    def enable_ir_printing(self) -> "Pipeline":
        """Enable per-pass IR printing to stderr."""
        self._pp.enable_ir_printing()
        return self

    def enable_ir_printing_to_file_tree(
        self, dir: str = ".pass_manager_output"
    ) -> "Pipeline":
        """Enable per-pass IR printing to a directory tree."""
        self._pp.enable_ir_printing_to_file_tree(dir)
        return self

    def run(self, mlir_str: str) -> str:
        """Execute all passes. Returns the result MLIR string."""
        return self._pp.run(mlir_str)

    def __str__(self):
        return str(self._pp)

    def __repr__(self):
        return repr(self._pp)


# ---------------------------------------------------------------------------
# pass_fn: define individual passes for use with Pipeline
# ---------------------------------------------------------------------------


class _PassDescriptor:
    """Describes a single MLIR pass. Created via pass_fn().

    Can be passed directly to ``Pipeline.add()`` or called standalone.
    """

    def __init__(self, pass_name: str, anchor: str | None = None, **default_options):
        self._pass_name = pass_name
        self._anchor = anchor
        self._default_options = default_options

    def _make_pipeline_text(self, **extra_options):
        merged = {**self._default_options, **extra_options}
        return _build_pipeline_text(self._pass_name, self._anchor, **merged)

    @property
    def pipeline_text(self):
        """The inner MLIR pipeline text for this pass."""
        return self._make_pipeline_text()

    @property
    def pass_name(self):
        return self._pass_name

    def __call__(self, *args, **options):
        """Run this pass standalone via a single-element Pipeline."""
        text = self._make_pipeline_text(**options)
        pp = _get_native_module().PassPipeline()
        pp.add(text)
        if not args:
            return pp
        x = args[0]
        result_str = pp.run(str(x) if not isinstance(x, str) else x)
        if isinstance(x, str):
            return result_str
        ctx = getattr(x, "context", None)
        if ctx is None:
            raise RuntimeError("module has no context attribute")
        parse = type(x).parse
        try:
            return parse(result_str, ctx)
        except TypeError:
            return parse(ctx, result_str)


def pass_fn(pass_name: str, anchor: str | None = None, **options):
    """Create a pass descriptor. One line per pass."""
    return _PassDescriptor(pass_name, anchor=anchor, **options)
