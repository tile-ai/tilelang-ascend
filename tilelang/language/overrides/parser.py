"""TVMScript parser overrides tailored for TileLang."""

from functools import partial

from typing import Any

from tvm.script.ir_builder import tir as T
from tvm.script.ir_builder.base import IRBuilder
from tvm.script.ir_builder.base import IRBuilderFrame as Frame
from tvm.script.parser._core import dispatch, doc, Parser
from tvm.tir import Buffer,BufferLoad, Var, IterVar, PrimExpr

from tvm.script.parser.tir import parser as tvm_tir_parser

import tvm
from functools import partial

# Original implementation located at
# 3rdparty/tvm/python/tvm/script/parser/tir/parser.py (tilelang_bind_assign_value).
def tilelang_bind_assign_value(self: Parser, node: doc.expr, var_name: str, value: Any) -> Any:
    if isinstance(value, T.meta_var):
        return value.value
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            tilelang_bind_assign_value(self, node, f"{var_name}_{i}", v)
        return value
    elif isinstance(value, Frame):
        value.add_callback(partial(value.__exit__, None, None, None))
        res = value.__enter__()
        IRBuilder.name(var_name, res)
        return res
    # bind local.var Buffer to BufferLoad
    elif isinstance(value, Buffer) and value.scope() == "local.var":
        IRBuilder.name(var_name, value)
        return BufferLoad(value, indices=[0])
    elif isinstance(value, (Buffer, IterVar)) or (
        isinstance(value, Var) and not self.var_table.exist(value)
    ):
        IRBuilder.name(var_name, value)
        return value
    else:
        value = tvm.runtime.convert(value)
        frame = T.LetStmt(value)
        var = frame.var
        IRBuilder.name(var_name, var)
        frame.add_callback(partial(frame.__exit__, None, None, None))
        frame.__enter__()
        return var



def _get_node_span(node: doc.AST) -> tuple[int, int, int, int]:
    """Return the span (lineno, col, end_lineno, end_col) for a doc node."""
    return (node.lineno, node.col_offset, node.end_lineno, node.end_col_offset)


# Original implementation located at
# 3rdparty/tvm/python/tvm/script/parser/tir/parser.py (visit_assign).
@dispatch.register(token="tir", type_name="Assign")
def tilelang_visit_assign(self, node: doc.Assign) -> None:  # pylint: disable=unused-argument
    """Override `Assign` to support chained writes and `local.var` buffers."""
    if not node.targets:
        self.report_error(node, "Assignment must have at least one target.")

    if isinstance(node.value, doc.Subscript):
        check_slices = []
        if isinstance(node.value.slice, doc.Slice):
            check_slices = [node.value.slice]
        elif isinstance(node.value.slice, doc.Tuple):
            for part in node.value.slice.elts:
                if isinstance(part, doc.Slice):
                    check_slices.append(part)
        for slice_node in check_slices:
            if not slice_node.step and slice_node.upper and slice_node.lower:
                slice_node.step = doc.Constant(
                    1,
                    None,
                    1,
                    1,
                    slice_node.upper.lineno,
                    slice_node.upper.end_col_offset + 1,
                    slice_node.upper.lineno,
                    slice_node.upper.end_col_offset + 2,
                )

    rhs = self.eval_expr(node.value)
    for lhs in node.targets:
        if isinstance(lhs, doc.Subscript):
            if isinstance(lhs.slice, doc.Tuple):
                indices = [self.eval_expr(index) for index in lhs.slice.elts]
            else:
                indices = self.eval_expr(lhs.slice)
            T.buffer_store(self.eval_expr(lhs.value), rhs, indices)
            continue

        if isinstance(lhs, doc.Name) and lhs.id in self.var_table.get():
            span = _get_node_span(lhs)
            load_ctx = doc.Load(*span)
            store_ctx = doc.Store(*span)
            lhs.ctx = load_ctx
            lhs_value = self.eval_expr(lhs)
            lhs.ctx = store_ctx
            if (
                isinstance(lhs_value, BufferLoad)
                and lhs_value.buffer.scope() == "local.var"
                and len(lhs_value.indices) == 1
                and lhs_value.indices[0] == 0
            ):
                T.buffer_store(lhs_value.buffer, rhs, indices=[0])
                continue
        self.eval_assign(target=lhs, source=rhs, bind_value=tilelang_bind_assign_value)


# Original implementation located at
# 3rdparty/tvm/python/tvm/script/parser/tir/parser.py (visit_aug_assign).
@dispatch.register(token="tir", type_name="AugAssign")
def tilelang_visit_aug_assign(self, node: doc.AugAssign) -> None:  # pylint: disable=unused-argument
    """Override `AugAssign` to support writes into `local.var` buffers."""
    lhs_pos = _get_node_span(node.target)
    rhs_pos = _get_node_span(node.value)

    node.target.ctx = doc.Load(*lhs_pos)
    with self.var_table.with_frame():
        lhs_name = "__tvm_tmp_value_aug_assign_lhs"
        rhs_name = "__tvm_tmp_value_aug_assign_rhs"
        lhs_expr = self.eval_expr(node.target)
        rhs_expr = self.eval_expr(node.value)
        self.var_table.add(lhs_name, lhs_expr)
        self.var_table.add(rhs_name, rhs_expr)
        op = doc.BinOp(
            doc.Name(lhs_name, doc.Load(*lhs_pos), *lhs_pos),
            node.op,
            doc.Name(rhs_name, doc.Load(*rhs_pos), *rhs_pos),
            *lhs_pos,
        )
        rhs = self.eval_expr(op)

    lhs = node.target
    lhs.ctx = doc.Store(*lhs_pos)
    if isinstance(lhs, doc.Subscript):
        if isinstance(lhs.slice, doc.Tuple):
            indices = [self.eval_expr(index) for index in lhs.slice.elts]
        else:
            indices = [self.eval_expr(lhs.slice)]
        T.buffer_store(self.eval_expr(lhs.value), rhs, indices)
        return

    if isinstance(lhs, doc.Name) and lhs.id in self.var_table.get():
        span = _get_node_span(lhs)
        load_ctx = doc.Load(*span)
        store_ctx = doc.Store(*span)
        lhs.ctx = load_ctx
        lhs_value = self.eval_expr(lhs)
        lhs.ctx = store_ctx
        if (
            isinstance(lhs_value, BufferLoad)
            and lhs_value.buffer.scope() == "local.var"
            and len(lhs_value.indices) == 1
            and lhs_value.indices[0] == 0
        ):
            T.buffer_store(lhs_value.buffer, rhs, indices=[0])
            return

    self.eval_assign(target=lhs, source=rhs, bind_value=tilelang_bind_assign_value)


# Original implementation located at
# 3rdparty/tvm/python/tvm/script/parser/tir/parser.py (visit_ann_assign).
@dispatch.register(token="tir", type_name="AnnAssign")
def tilelang_visit_ann_assign(self, node: doc.AnnAssign) -> None:  # pylint: disable=unused-argument
    """Override `AnnAssign` to support writes into `local.var` buffers."""
    lhs = node.target
    rhs = self.eval_expr(node.value)
    ann_var = self.visit_tvm_annotation(node.annotation)
    if not isinstance(ann_var, Var):
        self.report_error(node.annotation, "Annotation should be Var")

    if isinstance(lhs, doc.Name) and lhs.id in self.var_table.get():
        span = _get_node_span(lhs)
        load_ctx = doc.Load(*span)
        store_ctx = doc.Store(*span)
        lhs.ctx = load_ctx
        lhs_value = self.eval_expr(lhs)
        lhs.ctx = store_ctx
        if (
            isinstance(lhs_value, BufferLoad)
            and lhs_value.buffer.scope() == "local.var"
            and len(lhs_value.indices) == 1
            and lhs_value.indices[0] == 0
        ):
            T.buffer_store(lhs_value.buffer, rhs, indices=[0])
            return

    self.eval_assign(target=lhs, source=ann_var, bind_value=tilelang_bind_assign_value)
    frame = T.LetStmt(rhs, var=ann_var)
    frame.add_callback(partial(frame.__exit__, None, None, None))
    frame.__enter__()