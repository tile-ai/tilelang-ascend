"""Actual A2/A3 emitters, executed by host C++ against Python integer arithmetic."""

import ctypes
import random
import re
import shutil
import subprocess

import pytest
from tilelang import tvm
from tilelang.env import TILELANG_TEMPLATE_PATH

from test_a2a3_codegen import _call, _make_func

T = tvm.tirx


def _emit(body, variables=(), buffers=(), backend="ascend"):
    marked = T.SeqStmt([_call("tl.ascend_src_code", "// FLOOR_HOST_BEGIN"), body, _call("tl.ascend_src_code", "// FLOOR_HOST_END")])
    f = _make_func(marked, buffers)
    f = T.PrimFunc([*f.params, *variables], f.body, buffer_map=f.buffer_map, attrs=f.attrs)
    builder = tvm.ffi.get_global_func(f"target.build.tilelang_{backend}_a2a3")
    target = tvm.target.Target({"kind": "ascend", "arch": "dav-2201"})
    return builder(tvm.IRModule.from_expr(f), target, "A3").inspect_source()


def _result(expr):
    out = T.decl_buffer((1,), expr.dtype, name="floor_result", scope="local.var")
    return T.SeqStmt([T.AllocBuffer(out), T.BufferStore(out, expr, [0])])


def _math_source(source):
    return source.split("// FLOOR_HOST_BEGIN", 1)[1].split("// FLOOR_HOST_END", 1)[0]


def _host(tmp_path, source, dtype, setup="", extra=""):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    body = _math_source(source)
    aliases = "using uchar = unsigned char; using ushort = unsigned short; using uint = unsigned int;"
    signed = dtype.startswith("int")
    bits = int(re.search(r"\d+", dtype).group())
    abi = "int64_t" if signed else "uint64_t"
    ctype = ("int" if signed else "uint") + str(bits) + "_t"
    parameters = f"{abi} input_a, {abi} input_b"
    setup = setup or f"{ctype} a = static_cast<{ctype}>(input_a); {ctype} b = static_cast<{ctype}>(input_b);"
    path = tmp_path / "floor.cpp"
    path.write_text(
        '#include <cstdint>\n#include "ascend/a2a3/tl_templates/integer_math.h"\n'
        + "static_assert(::tl::ascend::floor_div(-7, 3) == -3);\n"
        + "static_assert(::tl::ascend::floor_mod(-7, 3) == 2);\n"
        + "template <int N> struct FloorExtent { char values[N]; };\n"
        + "static_assert(sizeof(FloorExtent<::tl::ascend::floor_mod(-7, 3)>) == 2);\n"
        + aliases
        + "\n"
        + extra
        + f'\nextern "C" {abi} run({parameters}) {{\n'
        + setup
        + body
        + "\nreturn floor_result;\n}\n"
    )
    lib = tmp_path / "floor.so"
    subprocess.run(
        [compiler, "-std=c++17", "-shared", "-fPIC", "-O2", "-I", TILELANG_TEMPLATE_PATH, str(path), "-o", str(lib)],
        check=True,
        capture_output=True,
        text=True,
    )
    module = ctypes.CDLL(str(lib))
    fn = module.run
    fn.argtypes = [ctypes.c_int64 if signed else ctypes.c_uint64] * 2
    fn.restype = ctypes.c_int64 if signed else ctypes.c_uint64
    return fn, module


@pytest.mark.parametrize("backend", ["ascend", "pto"])
@pytest.mark.parametrize("dtype", ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"])
def test_native_floor_matches_independent_python_bigint(tmp_path, backend, dtype):
    bits, signed = int(re.search(r"\d+", dtype).group()), dtype.startswith("int")
    low, high = (-(1 << (bits - 1)), (1 << (bits - 1)) - 1) if signed else (0, (1 << bits) - 1)
    if dtype == "int8":
        cases = [(a, b) for a in range(low, high + 1) for b in range(low, high + 1) if b]
    else:
        boundary = {low, low + 1, high - 1, high, 0, 1, 7, 9, 16, 17}
        if signed:
            boundary.update([-17, -16, -9, -7, -1])
        boundary = sorted(v for v in boundary if low <= v <= high)
        cases = [(a, b) for a in boundary for b in boundary if b]
        rng = random.Random(71)
        cases += [
            (rng.randint(low, high), rng.choice([-1, 1]) * rng.randint(1, high))
            if signed
            else (rng.randint(low, high), rng.randint(1, high))
            for _ in range(128)
        ]
    a, b = T.Var("a", dtype), T.Var("b", dtype)
    for name, constructor in [("div", T.FloorDiv), ("mod", T.FloorMod)]:
        folder = tmp_path / name
        folder.mkdir()
        text = _emit(_result(constructor(a, b)), [a, b], backend=backend)
        fn, keep_alive = _host(folder, text, dtype)
        assert f"::tl::ascend::floor_{name}(" in text
        for lhs, rhs in cases:
            expected = lhs // rhs if name == "div" else lhs % rhs
            # A non-representable quotient is outside the original TIR domain;
            # modulo MIN/-1 is representable and must still execute safely.
            if name == "div" and not low <= expected <= high:
                continue
            actual = fn(lhs, rhs)
            assert actual == expected, (backend, dtype, name, lhs, rhs, actual, expected)
            if name == "mod":
                assert (lhs // rhs) * rhs + actual == lhs
                assert actual == 0 or (actual > 0) == (rhs > 0)


@pytest.mark.parametrize("backend", ["ascend", "pto"])
def test_select_keeps_division_lazy(tmp_path, backend):
    a, b = T.Var("a", "int32"), T.Var("b", "int32")
    expr = T.Select(a != 0, T.FloorDiv(b, a), T.IntImm("int32", 7))
    text = _emit(_result(expr), [a, b], backend=backend)
    fn, keep_alive = _host(tmp_path, text, "int32")
    assert fn(0, 99) == 7
    assert fn(3, -7) == -7 // 3


@pytest.mark.parametrize("backend", ["ascend", "pto"])
def test_buffer_operands_evaluated_once(tmp_path, backend):
    inp = T.decl_buffer((2,), "int32", name="input", scope="global")
    text = _emit(_result(T.FloorMod(inp[0], inp[1])), buffers=[inp], backend=backend)
    extra = "struct HostTensor { int32_t values[2]; int calls = 0; int32_t GetValue(int i) { ++calls; return values[i]; } int32_t* operator+(int i) { ++calls; return values + i; } };"
    receiver = "input" if backend == "ascend" else "input_handle"
    setup = f"HostTensor {receiver}{{{{int32_t(input_a), int32_t(input_b)}}}};"
    # The generated expression is unchanged. The host tensor only replaces the
    # SDK GlobalTensor receiver, and counts the actual generated memory reads.
    body = _math_source(text)
    assert body.count("input.GetValue" if backend == "ascend" else "*(input_handle") == 2
    text = text.replace("// FLOOR_HOST_END", f"if ({receiver}.calls != 2) __builtin_trap();\n// FLOOR_HOST_END")
    fn, keep_alive = _host(tmp_path, text, "int32", setup=setup, extra=extra)
    assert fn(-7, 3) == -7 % 3


@pytest.mark.parametrize("backend", ["ascend", "pto"])
def test_int32_ir_shape_keeps_actual_int64_abi_arithmetic(tmp_path, backend):
    a, b = T.Var("a", "int32"), T.Var("b", "int32")
    inp = T.decl_buffer((a, 64), "float32", name="input", scope="global")
    for name, constructor in [("div", T.FloorDiv), ("mod", T.FloorMod)]:
        folder = tmp_path / name
        folder.mkdir()
        text = _emit(_result(constructor(a, b)), [b], [inp], backend)
        assert "int64_t a" in text
        fn, keep_alive = _host(folder, text, "int32", setup="int64_t a = input_a; int32_t b = int32_t(input_b);")
        for lhs, rhs in [(-(1 << 40) + 7, (1 << 30) - 3), ((1 << 40) + 7, -(1 << 30) + 3), (-(1 << 40) + 7, -(1 << 30) + 3)]:
            expected = lhs // rhs if name == "div" else lhs % rhs
            assert fn(lhs, rhs) == expected


@pytest.mark.parametrize("backend", ["ascend", "pto"])
def test_unsigned_actual_promoted_domain_does_not_treat_max_as_minus_one(tmp_path, backend):
    # A real thread-index terminal is emitted as auto/uint32 even though its IR
    # Var is int32. Keep those original usual arithmetic conversions.
    index, b = T.Var("index", "int32"), T.Var("b", "int32")
    thread = T.IterVar(tvm.ir.Range(0, 4), index, T.IterVar.ThreadIndex, "blockIdx.x")
    body = T.AttrStmt(thread, "thread_extent", 4, _result(T.FloorMod(index, b)))
    text = _emit(body, [b], backend=backend)
    assert "::tl::ascend::floor_mod(" in text
    # The actual generated body obtains the uint32 hardware index. Replace just
    # that SDK terminal with a host input; leave the helper and operation intact.
    text = re.sub(r"auto index = (?:AscendC::GetBlockIdx\(\)|get_block_idx\(\));", "uint32_t index = uint32_t(input_a);", text)
    text = text.replace("index = index / 2;", "index = index;")
    text = text.replace("if ASCEND_IS_AIV", "if (false)")
    fn, keep_alive = _host(
        tmp_path,
        text,
        "int32",
        setup="int32_t b = int32_t(input_b); uint64_t ffts_Addr = 0;",
        extra="inline void set_ffts_base_addr(uint64_t) {}",
    )
    assert fn(3, -1) == 3 % ((1 << 32) - 1)


@pytest.mark.parametrize("backend", ["ascend", "pto"])
def test_packed_integer_keeps_existing_whole_value_interface(tmp_path, backend):
    # PrintType maps int8x4 to one packed C++ int. This checks that interface,
    # and deliberately makes no claim about four independent lane results.
    a, b = T.Var("a", "int8x4"), T.Var("b", "int8x4")
    text = _emit(T.Bind(T.Var("floor_result", "int8x4"), T.FloorMod(a, b)), [a, b], backend=backend)
    assert "int floor_result" in text
    fn, keep_alive = _host(tmp_path, text, "int32")
    assert fn(-7, 3) == -7 % 3
