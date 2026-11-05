# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Normalize literal declarations before source pass ordering and C/V cloning."""

import re

from tilelang import tvm
from tilelang.ir import get_stmt_span
from tvm import tirx

_OP_NAME = "tl.a2a3_const_data"
# A literal name is printed verbatim because raw source does not substitute args.
_CPP_KEYWORDS = set(
    [
        "alignas",
        "alignof",
        "and",
        "and_eq",
        "asm",
        "auto",
        "bitand",
        "bitor",
        "bool",
        "break",
        "case",
        "catch",
        "char",
        "char8_t",
        "char16_t",
        "char32_t",
        "class",
        "compl",
        "concept",
        "const",
        "consteval",
        "constexpr",
        "constinit",
        "const_cast",
        "continue",
        "co_await",
        "co_return",
        "co_yield",
        "decltype",
        "default",
        "delete",
        "do",
        "double",
        "dynamic_cast",
        "else",
        "enum",
        "explicit",
        "export",
        "extern",
        "false",
        "float",
        "for",
        "friend",
        "goto",
        "if",
        "inline",
        "int",
        "long",
        "mutable",
        "namespace",
        "new",
        "noexcept",
        "not",
        "not_eq",
        "nullptr",
        "operator",
        "or",
        "or_eq",
        "private",
        "protected",
        "public",
        "register",
        "reinterpret_cast",
        "requires",
        "return",
        "short",
        "signed",
        "sizeof",
        "static",
        "static_assert",
        "static_cast",
        "struct",
        "switch",
        "template",
        "this",
        "thread_local",
        "throw",
        "true",
        "try",
        "typedef",
        "typeid",
        "typename",
        "union",
        "unsigned",
        "using",
        "virtual",
        "void",
        "volatile",
        "wchar_t",
        "while",
        "xor",
        "xor_eq",
    ]
)


def _is_constant(call):
    return isinstance(call, tirx.Call) and isinstance(call.op, tvm.ir.Op) and call.op.name == _OP_NAME


def normalize_constants(func):
    """Assign serialized declaration IDs; reject old uncompileable accesses."""
    declarations = {}
    aliases = {}
    names = set()
    bindings = set()
    redefined = set()

    def collect(node):
        if isinstance(node, tirx.Bind):
            if node.var in bindings:
                redefined.add(node.var)
            bindings.add(node.var)
        if isinstance(node, tirx.Bind) and _is_constant(node.value):
            call = node.value
            if len(call.args) < 5:
                raise ValueError("Malformed A2/A3 constant declaration.")
            name = node.var.name
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) or name in _CPP_KEYWORDS:
                raise ValueError(f"A2/A3 allocate_const needs a safe C++ literal name, got {name!r}.")
            if name in names:
                raise ValueError(f"A2/A3 constant name {name!r} is ambiguous in raw source; give distinct declarations distinct names.")
            names.add(name)
            args = list(call.args)
            args[2:4] = [tirx.StringImm(f"c{len(declarations)}"), tirx.StringImm(name)]
            declarations[node.var] = tirx.Call(call.dtype, call.op, args, getattr(call, "span", None))
            aliases[node.var] = node.var
        elif isinstance(node, tirx.Bind) and isinstance(node.value, tirx.Var) and node.value in aliases:
            aliases[node.var] = aliases[node.value]

    tirx.stmt_functor.post_order_visit(func.body, collect)
    if not declarations:
        return func
    if any(var in aliases for var in redefined):
        raise ValueError("A2/A3 constant declarations require unambiguous SSA bindings.")

    def references(expr):
        found = set()
        tirx.stmt_functor.post_order_visit(expr, lambda node: found.add(node) if isinstance(node, tirx.Var) and node in aliases else None)
        return found

    def check(node):
        if isinstance(node, (tirx.BufferLoad, tirx.BufferStore)) and node.buffer.data in aliases:
            raise ValueError(
                "A2/A3 allocate_const BufferLoad/BufferStore cannot compile in the TA Ascend source path; use T._src_code(code, pointer) with an explicit literal array access."
            )
        if isinstance(node, tirx.Call):
            refs = references(node)
            if refs:
                direct = {arg for arg in node.args if isinstance(arg, tirx.Var) and arg in aliases}
                if not isinstance(node.op, tvm.ir.Op) or node.op.name != "tl.ascend_src_code" or refs != direct:
                    raise ValueError(
                        "A2/A3 constant pointers may only be direct dependency operands of T._src_code; pointer escape is unsupported."
                    )
        elif isinstance(node, tirx.Bind) and not _is_constant(node.value) and references(node.value) and node.var not in aliases:
            raise ValueError(
                "A2/A3 constant pointer escape is unsupported; only direct Var aliases and T._src_code dependencies are allowed."
            )
        elif isinstance(node, tirx.Evaluate) and isinstance(node.value, tirx.Var) and node.value in aliases:
            raise ValueError("A2/A3 constant pointer must be consumed by T._src_code.")

    tirx.stmt_functor.post_order_visit(func.body, check)

    def rewrite(node):
        if isinstance(node, tirx.Bind) and node.var in declarations:
            return tirx.Bind(node.var, declarations[node.var], get_stmt_span(node))
        return None

    return func.with_body(tirx.stmt_functor.ir_transform(func.body, None, rewrite, ["tirx.Bind"]))
