"""Check Carver A3 resource admission without TVM or an NPU runtime."""

import ast
import configparser
from pathlib import Path
import re
import runpy
import tempfile
from types import SimpleNamespace
from typing import Optional
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[3]
SPECS = runpy.run_path(str(
    ROOT / "tilelang/carver/arch/ascend_specs.py"))

# Resource fields from CANN 9.1.0's Ascend910_9392.ini, file SHA256
# 7cc774afae581e223352d433076c4a2c4547385792ba71b4e568ac6831c51dc2.
SDK_9392 = """[version]
SoC_version=Ascend910_9392
NpuArch=2201
[SoCInfo]
cube_core_cnt=24
vector_core_cnt=48
l2_size=201326592
support_bf16=1
[AICoreSpec]
l0_a_size=65536
l0_b_size=65536
l0_c_size=131072
l1_size=524288
ub_size=196608
cube_m_size=16
cube_n_size=16
cube_k_size=16
"""


def source_function(path, name, class_name=None):
    module = ast.parse(path.read_text())
    if class_name:
        module = next(node for node in module.body
                      if isinstance(node, ast.ClassDef)
                      and node.name == class_name)
    function = next(node for node in module.body
                    if isinstance(node, ast.FunctionDef) and node.name == name)
    return ast.fix_missing_locations(ast.Module(
        body=[function], type_ignores=[]))


class AscendSpecsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.config = Path(self.temporary.name) / "Ascend910_9392.ini"
        self.config.write_text(SDK_9392)

    def resources(self, text=SDK_9392, sku="Ascend910_9392"):
        self.config.write_text(text)
        return SPECS["a3_resources"](sku, self.config)

    def test_exact_sku_recognition(self):
        recognize = SPECS["a3_sku"]
        self.assertEqual(
            [recognize(name) for name in (
                "Ascend910_9392", "ascend910_9382", "Ascend910B",
                "Ascend910C", "Ascend910_93", "Ascend910_93920", None,
            )],
            ["Ascend910_9392", "Ascend910_9382", None, None, None, None, None],
        )

    def test_raw_and_backend_budgets_are_distinct(self):
        profile = self.resources()
        self.assertEqual(
            (profile["raw_memory_capacities"],
             profile["usable_memory_capacities"]),
            ({"UB": 196608, "L1": 524288, "L0A": 65536,
              "L0B": 65536, "L0C": 131072}, SPECS["BACKEND_CAPACITIES"]),
        )

    def test_sdk_counts_and_bf16_are_read(self):
        profile = self.resources()
        self.assertEqual(
            (profile["cores"], profile["vector_cores"], profile["L2"],
             profile["cube"], profile["supports_bf16"]),
            (24, 48, 201326592, [16, 16, 16], True),
        )

    def test_actual_offline_descriptor_uses_exact_sdk_and_budgets(self):
        # Execute the actual descriptor with an explicit target/config input;
        # this checks its Python data flow, not TVM or device integration.
        path = ROOT / "tilelang/carver/arch/ascend.py"
        module = ast.parse(path.read_text())
        definitions = [node for node in module.body if (
            isinstance(node, ast.ClassDef) and node.name == "Ascend") or (
            isinstance(node, ast.Assign) and any(
                isinstance(name, ast.Name) and name.id in (
                    "CHIP_SPECS", "DEFAULT_CHIP") for name in node.targets))]
        namespace = {"Target": object, "TileDevice": object,
                     "Optional": Optional, "List": list,
                     "_TORCH_NPU_AVAILABLE": False, **SPECS}
        exec(compile(ast.fix_missing_locations(ast.Module(
            body=definitions, type_ignores=[])), str(path), "exec"), namespace)
        descriptor = namespace["Ascend"](
            SimpleNamespace(mcpu="ascend910_9392"),
            platform_config=str(self.config))
        self.assertEqual(
            (descriptor.chip_name, descriptor.compute_max_core,
             descriptor.compute_max_vector_core, descriptor.l0c_cap,
             descriptor.l1_cap, descriptor.ub_cap,
             descriptor.raw_memory_capacities["UB"]),
            ("Ascend910_9392", 24, 48, 131072, 524032, 196352, 196608),
        )

    def test_9382_reads_its_own_data(self):
        # Deliberately different synthetic fields; no 9382 device claim.
        content = SDK_9392.replace("9392", "9382").replace(
            "cube_core_cnt=24", "cube_core_cnt=20").replace(
            "l2_size=201326592", "l2_size=100663296")
        profile = self.resources(content, "Ascend910_9382")
        self.assertEqual((profile["sku"], profile["cores"], profile["L2"]),
                         ("Ascend910_9382", 20, 100663296))

    def test_wrong_sku_never_substitutes_9392(self):
        with self.assertRaisesRegex(ValueError, "does not describe"):
            self.resources(sku="Ascend910_9382")

    def test_wrong_architecture_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "dav-2201"):
            self.resources(SDK_9392.replace("NpuArch=2201", "NpuArch=3510"))

    def test_missing_file_is_rejected(self):
        with self.assertRaises(FileNotFoundError):
            SPECS["a3_resources"](
                "Ascend910_9392", self.config.with_suffix(".missing"))

    def test_zero_core_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            self.resources(SDK_9392.replace(
                "cube_core_cnt=24", "cube_core_cnt=0"))

    def test_missing_memory_field_is_rejected(self):
        with self.assertRaises(configparser.NoOptionError):
            self.resources(SDK_9392.replace("l0_c_size=131072\n", ""))

    def test_selected_sdk_root_resolves_exact_file(self):
        root = Path(self.temporary.name)
        directory = root / "data/platform_config"
        directory.mkdir(parents=True)
        self.config.rename(directory / self.config.name)
        with patch.dict("os.environ", {"CANN_ROOT": str(root)}):
            profile = SPECS["a3_resources"]("Ascend910_9392")
        self.assertEqual(Path(profile["path"]),
                         directory / "Ascend910_9392.ini")

    def test_missing_selected_sdk_does_not_mix_another_version(self):
        other = (Path(self.temporary.name) / "other-sdk"
                 / "data/platform_config")
        other.mkdir(parents=True)
        (other / "Ascend910_9392.ini").write_text(SDK_9392)
        with patch.dict("os.environ", {
            "CANN_ROOT": str(Path(self.temporary.name) / "selected-sdk"),
            "ASCEND_HOME_PATH": str(Path(self.temporary.name) / "other-sdk"),
        }):
            with self.assertRaises(FileNotFoundError):
                SPECS["a3_resources"]("Ascend910_9392")

    def test_backend_limits_match_both_actual_backend_consumers(self):
        planner = (ROOT / "src/transform"
                   / "ascend_memory_planning.cc").read_text()
        codegen = (ROOT / "src/target/codegen_ascend.cc").read_text()
        planner_names = {
            "UB": "ASCEND_SHARED_MEM_SIZE", "L1": "ASCEND_SHARED_DYN_MEM_SIZE",
            "L0A": "ASCEND_WMMA_MATRIX_A_MEM_SIZE",
            "L0B": "ASCEND_WMMA_MATRIX_B_MEM_SIZE",
            "L0C": "ASCEND_WMMA_ACCUMULATOR_MEM_SIZE",
        }
        for name, expected in SPECS["BACKEND_CAPACITIES"].items():
            with self.subTest(memory=name):
                planner_value = int(re.search(
                    rf"#define {planner_names[name]} (\d+)", planner).group(1))
                codegen_value = int(re.search(
                    rf"#define ASCEND_A2A3_{name}_SIZE \((\d+)\)",
                    codegen).group(1))
                self.assertEqual((planner_value, codegen_value),
                                 (expected, expected))

    def test_actual_l0_predicate_rejects_oversized_accumulator(self):
        policy = ROOT / "tilelang/carver/roller/policy/ascend.py"
        namespace = {}
        function = source_function(policy, "_check_l0_usage",
                                   "AscendDefaultPolicy")
        exec(compile(function, str(policy), "exec"), namespace)
        predicate = namespace["_check_l0_usage"]
        capacities = self.resources()["usable_memory_capacities"]
        arch = SimpleNamespace(l0a_capacity=capacities["L0A"],
                               l0b_capacity=capacities["L0B"],
                               l0c_capacity=capacities["L0C"])
        results = [predicate(arch, m * 64 * 2, 64 * n * 2, m * n * 4, 2)
                   for m, n in ((256, 256), (128, 256), (256, 128))]
        self.assertEqual(results, [False, True, True])

    def test_bf16_admission_uses_actual_sdk_feature(self):
        path = ROOT / "tilelang/carver/arch/ascend.py"
        namespace = {"Ascend": SimpleNamespace, "TileDevice": object,
                     "A3_SKUS": SPECS["A3_SKUS"], "CHIP_SPECS": {}}
        exec(compile(source_function(path, "is_cube_supported_precision"),
                     str(path), "exec"), namespace)
        predicate = namespace["is_cube_supported_precision"]
        arch = SimpleNamespace(chip_name="Ascend910_9382",
                               sdk_resources={"supports_bf16": False})
        self.assertEqual(
            [predicate(source, target, arch) for source, target in (
                ("float16", "float32"), ("bfloat16", "float32"),
                ("float16", "bfloat16"), ("int8", "float32"),
            )], [True, False, False, False],
        )


if __name__ == "__main__":
    unittest.main()
