"""Original Flash manual addresses survive all native consumers."""

import ast
from functools import lru_cache
import hashlib
import importlib.util
from pathlib import Path
import re

import pytest

import tilelang
from tilelang import tvm
from tilelang.ascend.a2a3 import compiler

TARGET = {"kind": "ascend", "arch": "dav-2201", "model": "A3", "keys": ["ascendc", "ascend"]}
ADDRESSES = {
    "q_l1": 0,
    "k_l1": 65536,
    "acc_s_l1": 65536,
    "v_l1": 73728,
    "acc_s_l0c": 0,
    "acc_o_l0c": 0,
    "acc_o": 0,
    "sumexp": 65536,
    "m_i": 65664,
    "acc_s_ub": 66048,
    "m_i_prev": 74240,
    "acc_s_ub_": 74368,
    "sumexp_i_ub": 98944,
    "acc_s_half": 98944,
    "acc_o_ub": 98944,
    "acc_o_half": 98944,
}


@lru_cache(maxsize=1)
def _flash():
    path = Path(__file__).parent / "fixtures" / "a2a3_manual_address_flash.py"
    spec = importlib.util.spec_from_file_location("a2a3_manual_address_flash", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.flash_attention_fwd._get_tir_with_target(tvm.target.Target(TARGET), 1, 128, 1, 512)


def _frontend_addresses(func):
    maps = []
    tvm.tirx.stmt_functor.post_order_visit(
        func.body,
        lambda node: (
            maps.append(node.annotations["address_map"])
            if isinstance(node, tvm.tirx.SBlock) and "address_map" in node.annotations
            else None
        ),
    )
    assert len(maps) == 1
    return maps[0]


def test_original_factory_ast_preserves_full_kernel_and_decorator():
    path = Path(__file__).parent / "fixtures" / "a2a3_manual_address_flash.py"
    factory = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef))
    digest = hashlib.sha256(ast.dump(factory, include_attributes=False).encode()).hexdigest()
    assert digest == "ab3a79bc7cdde8290fc5398eba436048e3809bf894cc3a32900163472ed43712"


def test_full_flash_address_identity_aliases_and_codegen_offsets():
    func = _flash()
    original = tvm.ir.save_json(func)
    addresses = _frontend_addresses(func)
    assert {var.name: int(value) for var, value in addresses.items()} == ADDRESSES
    allocated = []

    def collect(node):
        if isinstance(node, tvm.tirx.AllocBuffer):
            allocated.append(node.buffer.data)
        elif isinstance(node, tvm.tirx.SBlock):
            allocated.extend(buffer.data for buffer in node.alloc_buffers)

    tvm.tirx.stmt_functor.post_order_visit(func.body, collect)
    assert all(any(var.same_as(data) for data in allocated) for var in addresses)
    assert all(isinstance(value, tvm.tirx.IntImm) and value.dtype == "int32" for value in addresses.values())
    artifact = tilelang.lower(func, target=TARGET)
    final = next(iter(artifact.device_mod.functions.values()))
    final_addresses = {var.name: int(value) for var, value in final.attrs["address_map"].items()}
    assert {name: final_addresses[name] for name in ADDRESSES} == ADDRESSES
    declarations = dict(re.findall(r"auto (\w+) = \w+\.GetWithOffset<[^>]+>\(\d+, (\d+)\);", artifact.kernel_source))
    assert {name: int(declarations[name]) for name in ADDRESSES} == ADDRESSES
    assert tvm.ir.save_json(func) == original
    restored = tvm.ir.load_json(original)
    assert tilelang.lower(restored, target=TARGET).kernel_source == artifact.kernel_source


def test_real_bisheng_compiles_full_original_flash():
    try:
        config = compiler.resolve_compile_config(tvm.target.Target(TARGET))
    except FileNotFoundError as error:
        pytest.skip(str(error))
    artifact = tilelang.lower(_flash(), target=TARGET)
    library = compiler.LibraryGenerator(config)
    library.update_lib_code(artifact.kernel_source)
    library.compile_lib(timeout=120)
    assert Path(library.get_lib_path()).is_file()
