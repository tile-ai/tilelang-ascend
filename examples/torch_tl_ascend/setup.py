import sys
from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

import torch_npu

HERE = Path(__file__).parent
SRC = HERE / "src"
TORCH_NPU_PATH = Path(torch_npu.__file__).parent
PACKAGE_PATH = SRC / "torch_tl_ascend"
VERSION_FILE = PACKAGE_PATH / "VERSION"

sys.path.append(HERE.as_posix())
from compile_tl_op import flash_attention
flash_attention.update_package_files()

setup(
    name="torch_tl_ascend",
    version=VERSION_FILE.read_text(encoding="utf-8").strip(),
    description="Torch TL Ascend",
    packages=[PACKAGE_PATH.name],  # torch_tl_ascend
    package_dir={"": SRC.as_posix()},  # src/**
    package_data={PACKAGE_PATH.name: ["*.so"]},  # libop.so
    ext_modules=[
        CppExtension(
            "torch_tl_ascend._inner",
            [cpp_path.as_posix() for cpp_path in SRC.glob("*.cpp")],  # _inner.cpp
            include_dirs=[
                PACKAGE_PATH.as_posix(),
                (TORCH_NPU_PATH / "include").as_posix()
            ],
            library_dirs=[(TORCH_NPU_PATH / "lib").as_posix()],
            libraries=["torch_npu"] + [
                so_path.stem.removeprefix("lib") for so_path in PACKAGE_PATH.glob("*.so")
            ],  # "libop.so" => "op"
            extra_link_args=["-Wl,-rpath,$ORIGIN"],  # search .so around _inner.so ($ORIGIN)
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
