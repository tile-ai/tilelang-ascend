"""Keep parser error regressions in bounded children so failures cannot dump core."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("case", ["macro", "nested_macro", "inline", "valid", "already_reported"])
def test_parser_diagnostic(case):
    env = dict(os.environ, PYTHONFAULTHANDLER="1")
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--case", case],
        env=env,
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CASE_PASSED " + case in result.stdout
    if case in ("macro", "nested_macro", "inline"):
        assert "no_such_operation" in result.stdout + result.stderr


def _run_case(case):
    global T
    # Set the limit in the child before importing the compiler. Resource is a
    # POSIX facility; unsupported platforms skip only the core-limit operation.
    try:
        import resource
    except ImportError:
        pass
    else:
        resource.setrlimit(resource.RLIMIT_CORE, (0, resource.getrlimit(resource.RLIMIT_CORE)[1]))

    import tilelang.language as T
    import tvm
    from tvm.script.parser.core.parser import Parser

    @T.macro
    def invalid(A):
        T.no_such_operation(A)

    @T.macro
    def nested(A):
        invalid(A)

    @T.macro
    def assign(A):
        A[0] = T.float32(1)

    if case == "already_reported":

        class UnusedDiagnostic:
            def error(self, node, message):
                raise AssertionError("An existing diagnostic must not be rendered again")

        parser = Parser.__new__(Parser)
        parser.diag = UnusedDiagnostic()
        original = tvm.error.DiagnosticError("already emitted")
        try:
            parser.report_error(None, original)
        except tvm.error.DiagnosticError as err:
            assert err is original
        else:
            raise AssertionError("DiagnosticError must propagate")
    elif case == "valid":

        @T.prim_func
        def valid(A: T.Tensor((4,), "float32")):
            assign(A)

        assert isinstance(valid, tvm.tir.PrimFunc)
    else:
        try:
            if case == "macro":

                @T.prim_func
                def broken(A: T.Tensor((4,), "float32")):
                    invalid(A)
            elif case == "nested_macro":

                @T.prim_func
                def broken(A: T.Tensor((4,), "float32")):
                    nested(A)
            elif case == "inline":

                @T.prim_func
                def broken(A: T.Tensor((4,), "float32")):
                    T.no_such_operation(A)
            else:
                raise AssertionError("Unknown case")
        except tvm.error.DiagnosticError:
            pass
        else:
            raise AssertionError("Invalid TVMScript must report DiagnosticError")
    print("CASE_PASSED " + case, flush=True)


if __name__ == "__main__":
    assert len(sys.argv) == 3 and sys.argv[1] == "--case"
    _run_case(sys.argv[2])
