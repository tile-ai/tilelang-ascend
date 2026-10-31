# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Shared-library compilation with a frozen A2/A3 compiler identity."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, fields
import logging
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any
from collections.abc import Mapping, Sequence

from tilelang.env import TILELANG_TEMPLATE_PATH, env

TILELANG_PACKAGE_PATH = env.TILELANG_PACKAGE_PATH

logger = logging.getLogger(__name__)
_DEFAULT_COMPILER_OPTIONS = {"cce_auto_sync": True, "opt_level": 2, "pto_debug": False}
_SCHEMA_VERSION = 1
_COMPILER_ARCHES = {"ascendc": "dav-2201", "pto": "dav-c220"}


def _coerce_opt_level(value) -> int:
    text = str(value).strip()
    if text not in ("0", "1", "2", "3"):
        raise ValueError(f"Invalid opt_level={value!r}, expected one of: 0, 1, 2, 3")
    return int(text)


def _normalize_options(pass_configs, environ: Mapping[str, str]) -> dict:
    resolved = dict(_DEFAULT_COMPILER_OPTIONS)
    if pass_configs and pass_configs.get("tl.ascend_auto_sync_vs", False):
        resolved.update(cce_auto_sync=False, opt_level=3)
    env_auto_sync = environ.get("TL_CCE_AUTO_SYNC")
    if env_auto_sync is not None:
        resolved["cce_auto_sync"] = env_auto_sync.strip().lower() != "off"
    env_opt_level = environ.get("TL_CCE_OPT_LEVEL")
    if env_opt_level is not None:
        resolved["opt_level"] = _coerce_opt_level(env_opt_level)
    env_pto_debug = environ.get("TL_PTO_DEBUG")
    if env_pto_debug is not None:
        resolved["pto_debug"] = env_pto_debug.strip() == "1"
    return resolved


def normalize_compiler_options(pass_configs=None) -> dict:
    """Derive the original per-kernel defaults and read-only TL_CCE fallbacks.

    Only 'off' disables auto-sync and only '1' enables PTO debug. The original
    opt-level parser, unlike its old docstring, raises for values outside 0..3.
    Caller compile flags are applied after these defaults.
    """
    return _normalize_options(pass_configs, os.environ)


def _flags_from_options(target: str, options: Mapping, compile_flags) -> list[str]:
    flags = [f"-O{options['opt_level']}"]
    if target in ("ascendc", "auto") and not options["cce_auto_sync"]:
        flags.append("--cce-auto-sync=off")
    if target == "pto" and options["pto_debug"]:
        flags += ["-D_DEBUG", "--cce-enable-print"]
    if compile_flags:
        flags += [compile_flags] if isinstance(compile_flags, str) else list(compile_flags)
    return flags


def _configured_flags(pass_configs) -> list[str]:
    # This entry exists in current mainline, but not in the TA source. Keep its
    # existing shell-like parsing without changing legacy explicit flag parsing.
    from tilelang.contrib.bisheng import normalize_options

    return normalize_options((pass_configs or {}).get("tl.device_compile_flags"))


def resolve_compile_flags(target: str, pass_configs=None, compile_flags: Sequence[str] | str | None = None) -> list[str]:
    """Resolve original derived defaults first and explicit caller flags last."""
    options = normalize_compiler_options(pass_configs)
    flags = _flags_from_options(target, options, None)
    flags += _configured_flags(pass_configs)
    if compile_flags:
        flags += [compile_flags] if isinstance(compile_flags, str) else list(compile_flags)
    return flags


def _get_tl_root(third_party_name="3rdparty", *, environ=None) -> str:
    environ = os.environ if environ is None else environ
    tl_root = environ.get("TL_ROOT")
    if tl_root is not None:
        if not (Path(tl_root) / third_party_name).exists():
            raise ValueError(f"3rdparty dependencies not found. TL_ROOT is {tl_root!r}")
        return tl_root
    for path in [Path(TILELANG_PACKAGE_PATH), Path(TILELANG_PACKAGE_PATH).parent]:
        if (path / third_party_name).exists():
            return str(path)
    raise ValueError(
        f"TL_ROOT is not set and {third_party_name}/ directory not found in "
        f"{TILELANG_PACKAGE_PATH} or {Path(TILELANG_PACKAGE_PATH).parent}. "
        "Please set the TL_ROOT environment variable."
    )


def _get_ascend_home_path(*, environ=None) -> str:
    environ = os.environ if environ is None else environ
    ascend_home = environ.get("ASCEND_HOME_PATH") or environ.get("ASCEND_HOME")
    if ascend_home is None:
        for path in ["/usr/local/Ascend/ascend-toolkit/latest"]:
            if os.path.exists(path):
                ascend_home = path
                break
    if ascend_home is None:
        raise ValueError("ASCEND_HOME_PATH or ASCEND_HOME is not set. Please set the environment variable.")
    return ascend_home


def _get_simulator_lib_path(ascend_home: str, platform: str) -> str:
    """Keep the original A2/A3 candidate order and lib/config directory choice."""
    if platform not in ("A2", "A3"):
        raise ValueError("The A2/A3 simulator driver requires platform A2 or A3.")
    sim_base = os.path.join(ascend_home, "tools", "simulator")
    if not os.path.isdir(sim_base):
        sim_base = os.path.join(ascend_home, "simulator")
    for soc in ["Ascend910B1", "Ascend910_9599"]:
        soc_dir = os.path.join(sim_base, soc)
        if not os.path.isdir(soc_dir):
            continue
        candidate = os.path.join(soc_dir, "lib")
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(
        f"Simulator library not found for platform {platform}. Searched: {sim_base}/<soc>/lib and {sim_base}/<soc>/camodel"
    )


def _command_options(
    ascend_home: str, package_root: str, template_path: str, simulator_lib_path: str | None, *, dialect: str = "ascendc"
) -> tuple[str, ...]:
    if dialect == "pto":
        return _pto_command_options(ascend_home, package_root, template_path, simulator_lib_path)
    # Preserve the complete original AscendC command ordering. No random source
    # or output path belongs in the compiler/cache identity.
    command = [
        "--npu-arch=dav-2201",
        "-std=c++17",
        "-xasc",
        f"-I{ascend_home}/include",
        f"-I{ascend_home}/include/experiment/msprof",
        f"-I{ascend_home}/include/experiment/runtime",
        f"-I{ascend_home}/pkg_inc",
        f"-I{ascend_home}/pkg_inc/runtime",
        f"-I{ascend_home}/pkg_inc/profiling",
        f"-I{package_root}/3rdparty/catlass/include",
        f"-I{package_root}/3rdparty/shmem/include",
        f"-I{package_root}/3rdparty/shmem/src/device",
        "-DBACKEND_HYBM",
        "-I" + template_path,
        f"-L{ascend_home}/lib64",
        "-Wno-macro-redefined",
        "-Wno-ignored-attributes",
        "-Wno-non-c-typedef-for-linkage",
        "-lruntime",
        "-lascendcl",
        "-lm",
        "-ltiling_api",
        "-lplatform",
        "-lc_sec",
        "-ldl",
        "-fPIC",
        "--shared",
    ]
    if simulator_lib_path is not None:
        ascend_lib_idx = command.index(f"-L{ascend_home}/lib64")
        command[ascend_lib_idx:ascend_lib_idx] = [
            f"-L{simulator_lib_path}",
            f"-Wl,-rpath,{simulator_lib_path}",
            "-Wl,--disable-new-dtags",
        ]
        command[command.index("-lruntime")] = "-lruntime_camodel"
    # The original appends the source before this tail. Keep caller flags in a
    # separate frozen field so compile_lib can place that tail after the source.
    return tuple(command)


def _pto_command_options(ascend_home: str, package_root: str, template_path: str, simulator_lib_path: str | None) -> tuple[str, ...]:
    # TA 0deb8b93's entire A2/A3 PTO branch. Keep the repeated LLVM option
    # introducers and relative include in their original positions; the frozen
    # working_directory retains the latter's meaning. A5 is a separate backend.
    command = [
        "--cce-aicore-arch=dav-c220",
        "-DMEMORY_BASE",
        "-std=gnu++17",
        "-xcce",
        "-mllvm",
        "-cce-aicore-stack-size=0x8000",
        "-mllvm",
        "-cce-aicore-function-stack-size=0x8000",
        "-mllvm",
        "-cce-aicore-record-overflow=true",
        "-mllvm",
        "-cce-aicore-addr-transform",
        "-mllvm",
        "-cce-aicore-dcci-insert-for-scalar=false",
        "-DL2_CACHE_HINT",
        "-I../../src/",
        f"-I{package_root}/3rdparty/pto-isa/include",
        f"-I{ascend_home}/include",
        f"-I{ascend_home}/include/experiment/msprof",
        f"-I{ascend_home}/include/experiment/runtime",
        "-I/usr/local/Ascend/driver/kernel/inc",
        f"-I{ascend_home}/pkg_inc",
        f"-I{ascend_home}/pkg_inc/runtime",
        f"-I{ascend_home}/pkg_inc/profiling",
        f"-L{ascend_home}/lib64",
        "-I" + template_path,
        "-Wno-macro-redefined",
        "-Wno-ignored-attributes",
        "-lruntime",
        "-lstdc++",
        "-lascendcl",
        "-lm",
        "-ltiling_api",
        "-lplatform",
        "-lc_sec",
        "-ldl",
        "-fPIC",
        "--shared",
    ]
    if simulator_lib_path is not None:
        ascend_lib_idx = command.index(f"-L{ascend_home}/lib64")
        command[ascend_lib_idx:ascend_lib_idx] = [
            f"-L{simulator_lib_path}",
            f"-Wl,-rpath,{simulator_lib_path}",
            "-Wl,--disable-new-dtags",
        ]
        command[command.index("-lruntime")] = "-lruntime_camodel"
    return tuple(command)


@dataclass(frozen=True)
class CompileConfig:
    """Serializable immutable identity, resolved once before any cache lookup."""

    dialect: str
    platform: str
    run_mode: str
    arch: str
    compiler_path: str
    ascend_home: str
    package_root: str
    template_path: str
    working_directory: str
    simulator_lib_path: str | None
    flags: tuple[str, ...]
    command_options: tuple[str, ...]

    def to_metadata(self) -> dict[str, Any]:
        result = {field.name: getattr(self, field.name) for field in fields(self)}
        result["flags"] = list(self.flags)
        result["command_options"] = list(self.command_options)
        return {"schema_version": _SCHEMA_VERSION, **result}

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any]) -> CompileConfig:
        """Rehydrate saved data without resolving paths, options or environment."""
        expected = {field.name for field in fields(cls)} | {"schema_version"}
        if set(metadata) != expected or metadata.get("schema_version") != _SCHEMA_VERSION:
            raise ValueError("Unsupported or incomplete A2/A3 compiler metadata.")
        values = {key: value for key, value in metadata.items() if key != "schema_version"}
        for key in ("flags", "command_options"):
            value = values[key]
            if not isinstance(value, (list, tuple)) or not all(isinstance(item, str) for item in value):
                raise ValueError(f"Invalid A2/A3 compiler metadata field: {key}")
            values[key] = tuple(value)
        for key in expected - {"schema_version", "flags", "command_options", "simulator_lib_path"}:
            if not isinstance(values[key], str):
                raise ValueError(f"Invalid A2/A3 compiler metadata field: {key}")
        if values["simulator_lib_path"] is not None and not isinstance(values["simulator_lib_path"], str):
            raise ValueError("Invalid A2/A3 simulator path metadata.")
        config = cls(**values)
        if config.dialect not in _COMPILER_ARCHES or config.platform not in ("A2", "A3") or config.arch != _COMPILER_ARCHES[config.dialect]:
            raise ValueError("Compiler metadata does not describe an A2/A3 compilation.")
        if config.run_mode not in ("npu", "sim") or (config.run_mode == "sim") != (config.simulator_lib_path is not None):
            raise ValueError("Compiler metadata has an inconsistent runtime/simulator identity.")
        base = _command_options(
            config.ascend_home, config.package_root, config.template_path, config.simulator_lib_path, dialect=config.dialect
        )
        if config.command_options != (*base, *config.flags):
            raise ValueError("Compiler metadata command options do not match its frozen paths and flags.")
        return config

    @property
    def cache_identity(self) -> tuple:
        return (_SCHEMA_VERSION, *(getattr(self, field.name) for field in fields(self)))


def with_compile_options(config: CompileConfig, pass_configs=None, compile_flags=None, *, environ: Mapping[str, str]) -> CompileConfig:
    """Derive candidate flags without resolving the frozen toolchain again.

    Callers supply complete candidate pass configs, original explicit flags,
    and a captured environment. Paths, platform, dialect, runtime mode and
    working directory remain frozen. Legacy PTO explicit flags retain their
    original filtering boundary; mainline configured LLVM pairs stay intact.
    """
    config = CompileConfig.from_metadata(config.to_metadata())
    options = _normalize_options(pass_configs, environ)
    derived = _flags_from_options(config.dialect, options, None)
    configured = _configured_flags(pass_configs)
    explicit = [compile_flags] if isinstance(compile_flags, str) else (compile_flags or ())
    legacy = tuple(item for flag in explicit for item in flag.split())
    base = _command_options(
        config.ascend_home, config.package_root, config.template_path, config.simulator_lib_path, dialect=config.dialect
    )
    excluded = (config.compiler_path, "bisheng", *base)
    if config.dialect == "pto":
        flags = (*derived, *configured, *(flag for flag in legacy if flag not in excluded))
    else:
        flags = tuple(flag for flag in (*derived, *configured, *legacy) if flag not in excluded)
    metadata = config.to_metadata()
    metadata.update(flags=list(flags), command_options=[*base, *flags])
    return CompileConfig.from_metadata(metadata)


def resolve_compile_config(target, pass_configs=None, compile_flags=None, *, platform=None, run_mode=None) -> CompileConfig:
    """Freeze platform, toolchain paths, runtime mode and actual compiler flags."""
    from tilelang.ascend import target as ascend_target
    from .platform import _known_platform, resolve_a2a3_platform

    if not ascend_target.is_a2a3_target(target):
        raise ValueError("The A2/A3 compiler requires an ascend target with arch=dav-2201.")
    dialect = "pto" if "pto" in target.keys else "ascendc"
    known_model = _known_platform(target.attrs.get("model"))
    if platform is not None:
        if platform not in ("A2", "A3"):
            raise ValueError("Compiler platform must be A2 or A3.")
        if known_model is not None and known_model != platform:
            raise ValueError(f"Compiler platform={platform} conflicts with frozen target.model={known_model}.")
    else:
        platform = resolve_a2a3_platform(target)
    environ = dict(os.environ)
    working_directory = os.getcwd()
    options = _normalize_options(pass_configs, environ)
    derived = _flags_from_options(dialect, options, None)
    configured = _configured_flags(pass_configs)
    explicit = [compile_flags] if isinstance(compile_flags, str) else (compile_flags or ())
    # Match the source's legacy whitespace splitting. Filtering against the
    # original command happens below, before freezing the cache identity.
    legacy = tuple(item for flag in explicit for item in flag.split())
    runtime = environ.get("TL_RUN_MODE", "npu") if run_mode is None else run_mode
    mode = "sim" if runtime == "sim" else "npu"
    ascend_home = os.path.abspath(_get_ascend_home_path(environ=environ))
    package_root = os.path.abspath(_get_tl_root(environ=environ))
    template_path = os.path.abspath(TILELANG_TEMPLATE_PATH)
    simulator_lib_path = _get_simulator_lib_path(ascend_home, platform) if mode == "sim" else None
    compiler_path = shutil.which("bisheng", path=environ.get("PATH"))
    if compiler_path is None:
        raise FileNotFoundError("Cannot find bisheng on PATH for A2/A3 compilation.")
    base = _command_options(ascend_home, package_root, template_path, simulator_lib_path, dialect=dialect)
    # Original filtering tests every token against the base command, not against
    # previously appended tokens. PTO already has -mllvm in its base command.
    excluded = (compiler_path, "bisheng", *base)
    if dialect == "pto":
        # TL_DEVICE_COMPILE_FLAGS is a mainline entry absent from TA. Its
        # normalized option/value pairs must stay intact, including -mllvm.
        # Legacy explicit flags retain TA's filter against the base command;
        # in PTO that old filter can discard the LLVM option introducer.
        flags = (*derived, *configured, *(flag for flag in legacy if flag not in excluded))
    else:
        # Preserve the already accepted AscendC identity/command byte order.
        flags = tuple(flag for flag in (*derived, *configured, *legacy) if flag not in excluded)
    return CompileConfig(
        dialect,
        platform,
        mode,
        _COMPILER_ARCHES[dialect],
        os.path.abspath(compiler_path),
        ascend_home,
        package_root,
        template_path,
        working_directory,
        simulator_lib_path,
        flags,
        (*base, *flags),
    )


class LibraryGenerator:
    """The original library protocol, consuming an already frozen config."""

    def __init__(self, config: CompileConfig, verbose: bool = False):
        self._config = CompileConfig.from_metadata(config.to_metadata())
        self.verbose = verbose
        self.srcpath: str | None = None
        self.libpath: str | None = None
        self.lib_code: str | None = None

    @property
    def config(self):
        return self._config

    @property
    def target(self):
        return self.config.dialect

    @property
    def platform(self):
        return self.config.platform

    @property
    def compile_flags(self):
        return self.config.flags

    def update_lib_code(self, lib_code: str):
        self.lib_code = lib_code

    def set_compile_flags(self, compile_flags):
        requested = (compile_flags,) if isinstance(compile_flags, str) else (compile_flags or ())
        if tuple(requested) == self.config.flags:
            return
        tokens = tuple(item for flag in requested for item in flag.split())
        if tokens != self.config.flags:
            raise ValueError("Cannot change the frozen A2/A3 compiler flags after cache identity resolution.")

    assign_compile_flags = set_compile_flags

    def load_lib(self, lib_path: str | None = None):
        if lib_path is None:
            lib_path = self.libpath
        else:
            self.libpath = lib_path
        if self.config.run_mode == "sim":
            sim_lib_path = self.config.simulator_lib_path
            ld_path = os.environ.get("LD_LIBRARY_PATH", "")
            if sim_lib_path not in ld_path.split(":"):
                os.environ["LD_LIBRARY_PATH"] = f"{sim_lib_path}:{ld_path}"
        return ctypes.CDLL(lib_path)

    def compile_lib(self, timeout: float = None):
        src = tempfile.NamedTemporaryFile(mode="w", suffix=".cpp", delete=False)  # noqa: SIM115
        libpath = src.name.replace(".cpp", ".so")
        split = len(self.config.command_options) - len(self.config.flags)
        command = [
            self.config.compiler_path,
            *self.config.command_options[:split],
            src.name,
            *self.config.command_options[split:],
            "-o",
            libpath,
        ]
        if self.config.run_mode == "sim":
            logger.info("camodel sim mode: using %s", self.config.simulator_lib_path)
        src.write(self.lib_code)
        src.flush()
        try:
            ret = subprocess.run(command, timeout=timeout, cwd=self.config.working_directory)
        except Exception as e:
            raise RuntimeError(f"Compile kernel failed because of {e}") from e
        finally:
            src.close()
        if ret.returncode != 0:
            hint = ""
            if self.config.dialect == "pto" and any(
                flag.startswith("-cce-") and (index == 0 or self.config.flags[index - 1] != "-mllvm")
                for index, flag in enumerate(self.config.flags)
            ):
                hint = (
                    " PTO LLVM options require a preceding -mllvm; TA's legacy compile_flags filter can remove it. "
                    "Use tl.device_compile_flags for complete option/value pairs."
                )
            raise RuntimeError(f"Compilation Failed! {command}{hint}")
        self.srcpath = src.name
        self.libpath = libpath

    def remove_lib(self):
        if self.libpath:
            os.remove(self.libpath)
        self.libpath = None

    def get_source_path(self):
        return self.srcpath

    def get_lib_path(self):
        return self.libpath

    def set_lib_path(self, libpath):
        self.libpath = libpath

    def set_src_path(self, srcpath):
        self.srcpath = srcpath
