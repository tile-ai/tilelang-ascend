"""A2/A3 compiler identity, source command ordering and simulator contracts."""

from dataclasses import FrozenInstanceError
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from tilelang import tvm
from tilelang.ascend.a2a3 import compiler


@pytest.fixture
def toolchain(monkeypatch, tmp_path):
    for key in ["TL_CCE_AUTO_SYNC", "TL_CCE_OPT_LEVEL", "TL_PTO_DEBUG", "TL_RUN_MODE", "TL_ROOT", "ASCEND_HOME_PATH", "ASCEND_HOME"]:
        monkeypatch.delenv(key, raising=False)
    package = tmp_path / "package"
    (package / "tilelang/src").mkdir(parents=True)
    for directory in ["catlass/include", "shmem/include", "shmem/src/device", "pto-isa/include"]:
        (package / "3rdparty" / directory).mkdir(parents=True)
    monkeypatch.setattr(compiler, "TILELANG_PACKAGE_PATH", package / "tilelang")
    monkeypatch.setattr(compiler, "TILELANG_TEMPLATE_PATH", str(package / "tilelang/src"))
    sdk = tmp_path / "cann"
    (sdk / "tools/simulator/Ascend910B1/lib").mkdir(parents=True)
    monkeypatch.setenv("ASCEND_HOME_PATH", str(sdk))
    bindir = tmp_path / "bin"
    bindir.mkdir()
    binary = bindir / "bisheng"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    monkeypatch.setenv("PATH", str(bindir))
    work = tmp_path / "project"
    work.mkdir()
    (work / "local/include").mkdir(parents=True)
    monkeypatch.chdir(work)
    return SimpleNamespace(package=package, sdk=sdk, binary=binary, work=work)


def _target(platform="A2", *, keys=None, arch="dav-2201"):
    return tvm.target.Target({"kind": "ascend", "arch": arch, "model": platform, "keys": keys or ["ascendc", "ascend"]})


def _config(toolchain, **kwargs):
    return compiler.resolve_compile_config(_target(), **kwargs)


def _capture_compile(monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        Path(command[-1]).write_bytes(b"shared")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(compiler.subprocess, "run", run)
    return calls


def test_original_defaults_vs_pairing_and_read_only_environment(toolchain):
    before = dict(os.environ)
    assert compiler.normalize_compiler_options() == {"cce_auto_sync": True, "opt_level": 2, "pto_debug": False}
    assert compiler.normalize_compiler_options({"tl.ascend_auto_sync_vs": True}) == {
        "cce_auto_sync": False,
        "opt_level": 3,
        "pto_debug": False,
    }
    assert compiler.resolve_compile_flags("ascendc", {"tl.ascend_auto_sync_vs": True}, ["-O0", "--cce-auto-sync=on"]) == [
        "-O3",
        "--cce-auto-sync=off",
        "-O0",
        "--cce-auto-sync=on",
    ]
    assert os.environ == before


@pytest.mark.parametrize("value,enabled", [("off", False), (" OFF ", False), ("on", True), ("false", True), ("0", True), ("", True)])
def test_auto_sync_environment_keeps_source_exact_off_rule(toolchain, monkeypatch, value, enabled):
    monkeypatch.setenv("TL_CCE_AUTO_SYNC", value)
    assert compiler.normalize_compiler_options()["cce_auto_sync"] is enabled


@pytest.mark.parametrize("value,enabled", [("1", True), (" 1 ", True), ("true", False), ("0", False), ("", False)])
def test_pto_debug_environment_keeps_source_exact_one_rule(toolchain, monkeypatch, value, enabled):
    monkeypatch.setenv("TL_PTO_DEBUG", value)
    assert compiler.normalize_compiler_options()["pto_debug"] is enabled
    assert compiler.resolve_compile_flags("pto") == (["-O2", "-D_DEBUG", "--cce-enable-print"] if enabled else ["-O2"])


@pytest.mark.parametrize("value,level", [("0", 0), (" 1 ", 1), ("2", 2), ("3", 3)])
def test_optimization_environment_matches_original_parser(toolchain, monkeypatch, value, level):
    monkeypatch.setenv("TL_CCE_OPT_LEVEL", value)
    assert compiler.normalize_compiler_options()["opt_level"] == level


@pytest.mark.parametrize("value", ["4", "-1", "", "fast", "2.0"])
def test_invalid_optimization_is_source_value_error(toolchain, monkeypatch, value):
    monkeypatch.setenv("TL_CCE_OPT_LEVEL", value)
    with pytest.raises(ValueError, match="Invalid opt_level"):
        compiler.normalize_compiler_options()


def test_environment_overrides_derived_vs_defaults_before_caller_flags(toolchain, monkeypatch):
    monkeypatch.setenv("TL_CCE_AUTO_SYNC", "on")
    monkeypatch.setenv("TL_CCE_OPT_LEVEL", "1")
    assert compiler.resolve_compile_flags("ascendc", {"tl.ascend_auto_sync_vs": True}, "-O0 --cce-auto-sync=off") == [
        "-O1",
        "-O0 --cce-auto-sync=off",
    ]
    config = _config(toolchain, pass_configs={"tl.ascend_auto_sync_vs": True}, compile_flags="-O0 --cce-auto-sync=off")
    assert config.flags == ("-O1", "-O0", "--cce-auto-sync=off")


def test_config_is_immutable_json_stable_and_rehydrates_without_environment(toolchain, monkeypatch):
    config = _config(toolchain, compile_flags=["-O0"])
    metadata = json.loads(json.dumps(config.to_metadata(), sort_keys=True))
    assert config.dialect == "ascendc"
    assert metadata["dialect"] == "ascendc"
    assert "execution_backend" not in metadata
    assert compiler.LibraryGenerator(config).target == config.dialect
    identity = json.dumps(config.cache_identity)
    assert hash(config.cache_identity)
    with pytest.raises(FrozenInstanceError):
        config.platform = "A3"
    monkeypatch.setenv("TL_CCE_OPT_LEVEL", "invalid-after-freeze")
    monkeypatch.setenv("TL_RUN_MODE", "sim")
    for name in ["_get_tl_root", "_get_ascend_home_path", "_get_simulator_lib_path", "_normalize_options"]:
        monkeypatch.setattr(compiler, name, lambda *args, **kwargs: pytest.fail("metadata re-resolved compiler identity"))
    restored = compiler.CompileConfig.from_metadata(metadata)
    assert restored == config
    assert json.dumps(restored.cache_identity) == identity


def test_platform_mode_paths_and_cwd_are_distinct_cache_identities(toolchain, monkeypatch, tmp_path):
    first = _config(toolchain)
    other_platform = compiler.resolve_compile_config(_target("A3"))
    sim = _config(toolchain, run_mode="sim")
    moved = tmp_path / "other-project"
    moved.mkdir()
    monkeypatch.chdir(moved)
    other_cwd = _config(toolchain)
    assert len({cfg.cache_identity for cfg in [first, other_platform, sim, other_cwd]}) == 4


def test_complete_command_preserves_headers_link_order_source_and_last_flags(toolchain, monkeypatch):
    config = _config(toolchain, compile_flags=["-O0", "-mllvm", "one", "-mllvm", "two", "-std=c++17"])
    calls = _capture_compile(monkeypatch)
    library = compiler.LibraryGenerator(config)
    library.update_lib_code("source text")
    library.compile_lib(timeout=7)
    [(command, kwargs)] = calls
    assert command[0] == str(toolchain.binary)
    assert command[1:4] == ["--npu-arch=dav-2201", "-std=c++17", "-xasc"]
    for relative in ["catlass/include", "shmem/include", "shmem/src/device"]:
        assert f"-I{toolchain.package}/3rdparty/{relative}" in command
    assert "-DBACKEND_HYBM" in command
    assert command.index("--shared") < command.index(library.srcpath)
    assert command[command.index(library.srcpath) + 1 : -2] == ["-O2", "-O0", "-mllvm", "one", "-mllvm", "two"]
    assert command.count("-std=c++17") == 1
    assert command[-2:] == ["-o", library.libpath]
    assert kwargs == {"timeout": 7, "cwd": str(toolchain.work)}
    assert Path(library.get_source_path()).read_text() == "source text"
    assert Path(library.get_lib_path()).read_bytes() == b"shared"
    library.remove_lib()
    assert library.libpath is None


def test_compile_and_flag_assignment_use_snapshot_after_environment_and_cwd_change(toolchain, monkeypatch, tmp_path):
    config = _config(toolchain, compile_flags=["-Ilocal/include", "-O0"])
    monkeypatch.setenv("TL_CCE_OPT_LEVEL", "invalid")
    monkeypatch.setenv("TL_ROOT", "/missing")
    monkeypatch.setenv("ASCEND_HOME_PATH", "/another-sdk")
    monkeypatch.setenv("TL_RUN_MODE", "sim")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    calls = _capture_compile(monkeypatch)
    library = compiler.LibraryGenerator(compiler.CompileConfig.from_metadata(config.to_metadata()))
    library.assign_compile_flags(config.flags)
    with pytest.raises(ValueError, match="frozen"):
        library.set_compile_flags(["-O3"])
    library.update_lib_code("frozen")
    library.compile_lib()
    [(command, kwargs)] = calls
    assert kwargs["cwd"] == str(toolchain.work)
    assert Path(kwargs["cwd"], "local/include").is_dir()
    assert "-Ilocal/include" in command
    assert f"-L{toolchain.sdk}/lib64" in command
    assert "-lruntime" in command and "-lruntime_camodel" not in command


@pytest.mark.parametrize("platform", ["A2", "A3"])
def test_simulator_command_keeps_original_transitive_link_order(toolchain, monkeypatch, platform):
    config = compiler.resolve_compile_config(_target(platform), run_mode="sim")
    calls = _capture_compile(monkeypatch)
    library = compiler.LibraryGenerator(config)
    library.update_lib_code("sim")
    library.compile_lib()
    [(command, _)] = calls
    simdir = str(toolchain.sdk / "tools/simulator/Ascend910B1/lib")
    begin = command.index(f"-L{simdir}")
    assert command[begin : begin + 4] == [f"-L{simdir}", f"-Wl,-rpath,{simdir}", "-Wl,--disable-new-dtags", f"-L{toolchain.sdk}/lib64"]
    assert "-lruntime_camodel" in command and "-lruntime" not in command


def test_simulator_discovery_uses_lib_configuration_not_camodel(toolchain):
    first = toolchain.sdk / "tools/simulator/Ascend910B1/lib"
    first.rmdir()
    (first.parent / "camodel").mkdir()
    with pytest.raises(FileNotFoundError, match="Simulator library not found"):
        compiler._get_simulator_lib_path(str(toolchain.sdk), "A2")
    fallback = toolchain.sdk / "tools/simulator/Ascend910_9599/lib"
    fallback.mkdir(parents=True)
    assert compiler._get_simulator_lib_path(str(toolchain.sdk), "A3") == str(fallback)


@pytest.mark.parametrize("dialect", ["ascendc", "pto"])
def test_sim_load_only_prepends_frozen_library_path_and_deduplicates(toolchain, monkeypatch, dialect):
    config = compiler.resolve_compile_config(_target(keys=[dialect, "ascend"]), run_mode="sim")
    monkeypatch.setenv("TL_RUN_MODE", "npu")
    monkeypatch.setenv("ASCEND_HOME_PATH", "/changed")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/existing")
    loaded = []
    monkeypatch.setattr(compiler.ctypes, "CDLL", lambda path: loaded.append(path) or object())
    library = compiler.LibraryGenerator(config)
    library.set_lib_path("cached.so")
    library.load_lib()
    assert loaded == ["cached.so"]
    assert os.environ["LD_LIBRARY_PATH"] == config.simulator_lib_path + ":/existing"
    library.load_lib()
    assert os.environ["LD_LIBRARY_PATH"] == config.simulator_lib_path + ":/existing"


@pytest.mark.parametrize("target", [_target(arch="dav-3510")])
def test_target_boundaries_are_strict_before_toolchain_resolution(toolchain, target):
    with pytest.raises((ValueError, NotImplementedError), match="A2/A3|PTO"):
        compiler.resolve_compile_config(target)


def test_explicit_platform_cannot_change_frozen_model(toolchain):
    with pytest.raises(ValueError, match="conflicts with frozen"):
        compiler.resolve_compile_config(_target("A2"), platform="A3")


@pytest.mark.parametrize("change", ["schema", "command", "runtime", "flags_type"])
def test_metadata_rejects_incomplete_or_inconsistent_snapshots(toolchain, change):
    metadata = _config(toolchain).to_metadata()
    if change == "schema":
        del metadata["working_directory"]
    elif change == "command":
        metadata["command_options"].append("--changed")
    elif change == "runtime":
        metadata["run_mode"] = "sim"
    else:
        metadata["flags"] = "-O2"
    with pytest.raises(ValueError):
        compiler.CompileConfig.from_metadata(metadata)


def test_compile_failure_and_timeout_preserve_original_error_contract(toolchain, monkeypatch):
    library = compiler.LibraryGenerator(_config(toolchain))
    library.update_lib_code("bad source")
    monkeypatch.setattr(compiler.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=3))
    with pytest.raises(RuntimeError, match="Compilation Failed"):
        library.compile_lib()

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 0.1)

    monkeypatch.setattr(compiler.subprocess, "run", timeout)
    with pytest.raises(RuntimeError, match="Compile kernel failed"):
        library.compile_lib(timeout=0.1)


@pytest.mark.parametrize(
    "configured", ["-O1 --cce-auto-sync=off -mllvm one -mllvm two", ["-O1", "--cce-auto-sync=off", "-mllvm one", "-mllvm two"]]
)
def test_mainline_configured_flags_precede_explicit_flags_without_losing_pairs(toolchain, monkeypatch, configured):
    config = _config(
        toolchain,
        pass_configs={"tl.ascend_auto_sync_vs": True, "tl.device_compile_flags": configured},
        compile_flags=["-O0", "--cce-auto-sync=on", "-mllvm", "three"],
    )
    assert config.flags == (
        "-O3",
        "--cce-auto-sync=off",
        "-O1",
        "--cce-auto-sync=off",
        "-mllvm",
        "one",
        "-mllvm",
        "two",
        "-O0",
        "--cce-auto-sync=on",
        "-mllvm",
        "three",
    )
    calls = _capture_compile(monkeypatch)
    library = compiler.LibraryGenerator(config)
    library.update_lib_code("options")
    library.compile_lib()
    [(command, _)] = calls
    tail = command[command.index(library.srcpath) + 1 : -2]
    assert tail == list(config.flags)
    assert [flag for flag in tail if flag.startswith("-O")] == ["-O3", "-O1", "-O0"]
    assert [flag for flag in tail if flag.startswith("--cce-auto-sync=")] == [
        "--cce-auto-sync=off",
        "--cce-auto-sync=off",
        "--cce-auto-sync=on",
    ]


def test_mainline_configured_quoted_option_is_one_argv_token(toolchain, monkeypatch):
    config = _config(toolchain, pass_configs={"tl.device_compile_flags": '-I"relative include"'}, compile_flags=["-O0"])
    assert config.flags == ("-O2", "-Irelative include", "-O0")
    calls = _capture_compile(monkeypatch)
    library = compiler.LibraryGenerator(config)
    library.assign_compile_flags(config.flags)
    library.update_lib_code("quoted")
    library.compile_lib()
    assert "-Irelative include" in calls[0][0]
    assert "-Irelative" not in calls[0][0]


def test_explicit_load_path_is_saved_for_mainline_cache_export_protocol(toolchain, monkeypatch, tmp_path):
    path = tmp_path / "restored.so"
    path.write_bytes(b"restored")
    loaded = []
    monkeypatch.setattr(compiler.ctypes, "CDLL", lambda value: loaded.append(value) or object())
    library = compiler.LibraryGenerator(_config(toolchain))
    library.load_lib(str(path))
    assert loaded == [str(path)]
    assert library.get_lib_path() == str(path)
    library.remove_lib()
    assert library.libpath is None
    assert not path.exists()


@pytest.mark.parametrize("platform", ["A2", "A3"])
@pytest.mark.parametrize("mode", ["npu", "sim"])
def test_pto_command_preserves_full_llvm_pairs_relative_include_and_link_order(toolchain, monkeypatch, platform, mode):
    target = _target(platform, keys=["pto", "ascend"])
    config = compiler.resolve_compile_config(target, run_mode=mode, compile_flags=["-O0", "-DUSER=1"])
    assert target.attrs["arch"] == "dav-2201"
    assert config.arch == "dav-c220"
    assert config.dialect == "pto"
    calls = _capture_compile(monkeypatch)
    library = compiler.LibraryGenerator(config)
    library.update_lib_code("PTO source")
    library.compile_lib()
    [(command, kwargs)] = calls
    assert command[1:5] == ["--cce-aicore-arch=dav-c220", "-DMEMORY_BASE", "-std=gnu++17", "-xcce"]
    llvm_options = [
        "-cce-aicore-stack-size=0x8000",
        "-cce-aicore-function-stack-size=0x8000",
        "-cce-aicore-record-overflow=true",
        "-cce-aicore-addr-transform",
        "-cce-aicore-dcci-insert-for-scalar=false",
    ]
    assert command[5:15] == [token for option in llvm_options for token in ("-mllvm", option)]
    assert command[15:18] == ["-DL2_CACHE_HINT", "-I../../src/", f"-I{toolchain.package}/3rdparty/pto-isa/include"]
    assert not any("catlass" in item or "shmem" in item or "REGISTER_BASE" in item for item in command)
    assert command[command.index("-lstdc++") : command.index("--shared") + 1] == [
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
    assert command[command.index(library.srcpath) + 1 : -2] == ["-O2", "-O0", "-DUSER=1"]
    assert kwargs["cwd"] == str(toolchain.work)
    if mode == "sim":
        simdir = str(toolchain.sdk / "tools/simulator/Ascend910B1/lib")
        start = command.index(f"-L{simdir}")
        assert command[start : start + 4] == [f"-L{simdir}", f"-Wl,-rpath,{simdir}", "-Wl,--disable-new-dtags", f"-L{toolchain.sdk}/lib64"]
        assert "-lruntime_camodel" in command and "-lruntime" not in command
    else:
        assert "-lruntime" in command and "-lruntime_camodel" not in command


def test_pto_public_vs_default_and_direct_fallback_are_separate(toolchain):
    from tilelang.ascend.a2a3.pass_config import normalize_a2a3_pass_configs

    target = _target(keys=["pto", "ascend"])
    public_configs = normalize_a2a3_pass_configs(target, None)
    assert public_configs["tl.ascend_auto_sync_vs"] is True
    assert compiler.resolve_compile_config(target, pass_configs=public_configs).flags == ("-O3",)
    assert compiler.resolve_compile_config(target).flags == ("-O2",)


@pytest.mark.parametrize("debug", ["1", "true"])
def test_pto_debug_options_only_follow_source_exact_one_rule(toolchain, monkeypatch, debug):
    monkeypatch.setenv("TL_PTO_DEBUG", debug)
    config = compiler.resolve_compile_config(_target(keys=["pto", "ascend"]), compile_flags=["-O0"])
    expected = ("-O2", "-D_DEBUG", "--cce-enable-print", "-O0") if debug == "1" else ("-O2", "-O0")
    assert config.flags == expected


@pytest.mark.parametrize("configured", ["-O1 -mllvm one -mllvm two", ["-O1", "-mllvm one", "-mllvm two"]])
def test_pto_mainline_configured_pairs_survive_legacy_base_filter(toolchain, monkeypatch, configured):
    config = compiler.resolve_compile_config(
        _target(keys=["pto", "ascend"]),
        pass_configs={"tl.ascend_auto_sync_vs": True, "tl.device_compile_flags": configured},
        compile_flags=["-O0", "-mllvm", "three", "-mllvm", "four", "-std=gnu++17"],
    )
    # The mainline key is new. The legacy explicit filter drops tokens already
    # present in TA's PTO base, including -mllvm and -std=gnu++17.
    assert config.flags == ("-O3", "-O1", "-mllvm", "one", "-mllvm", "two", "-O0", "three", "four")
    calls = _capture_compile(monkeypatch)
    library = compiler.LibraryGenerator(config)
    library.update_lib_code("options")
    library.compile_lib()
    assert calls[0][0][calls[0][0].index(library.srcpath) + 1 : -2] == list(config.flags)


def test_pto_legacy_llvm_filter_failure_explains_complete_pair_entry(toolchain, monkeypatch):
    config = compiler.resolve_compile_config(_target(keys=["pto", "ascend"]), compile_flags=["-mllvm", "-cce-aicore-stack-size=0x4000"])
    assert config.flags == ("-O2", "-cce-aicore-stack-size=0x4000")
    library = compiler.LibraryGenerator(config)
    library.update_lib_code("source")
    monkeypatch.setattr(compiler.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=1))
    with pytest.raises(RuntimeError, match="legacy compile_flags filter.*tl.device_compile_flags"):
        library.compile_lib()


@pytest.mark.parametrize("mode", ["npu", "sim"])
def test_pto_json_restore_keeps_schema_and_frozen_paths_options_and_mode(toolchain, monkeypatch, mode):
    ascendc_metadata = _config(toolchain).to_metadata()
    config = compiler.resolve_compile_config(_target("A3", keys=["pto", "ascend"]), run_mode=mode)
    metadata = json.loads(json.dumps(config.to_metadata()))
    assert metadata.keys() == ascendc_metadata.keys()
    monkeypatch.setenv("TL_CCE_OPT_LEVEL", "invalid-after-freeze")
    monkeypatch.setenv("TL_ROOT", "/missing")
    monkeypatch.setenv("TL_RUN_MODE", "sim" if mode == "npu" else "npu")
    for name in ["_get_tl_root", "_get_ascend_home_path", "_get_simulator_lib_path", "_normalize_options"]:
        monkeypatch.setattr(compiler, name, lambda *args, **kwargs: pytest.fail("PTO restore re-resolved identity"))
    restored = compiler.CompileConfig.from_metadata(metadata)
    assert restored == config
    assert restored.cache_identity == config.cache_identity
    assert compiler.LibraryGenerator(restored).target == "pto"
    metadata["arch"] = "dav-2201"
    with pytest.raises(ValueError, match="A2/A3 compilation"):
        compiler.CompileConfig.from_metadata(metadata)


@pytest.mark.parametrize("dialect", ["ascendc", "pto"])
@pytest.mark.parametrize("vs", [False, True])
@pytest.mark.parametrize("option_env", [{}, {"TL_CCE_OPT_LEVEL": "1", "TL_CCE_AUTO_SYNC": "off", "TL_PTO_DEBUG": "1"}])
def test_candidate_options_match_initial_resolution(toolchain, monkeypatch, dialect, vs, option_env):
    target = _target(keys=[dialect, "ascend"])
    base = compiler.resolve_compile_config(target)
    pass_configs = {"tl.ascend_auto_sync_vs": vs, "tl.device_compile_flags": ["-mllvm", "configured", '-DNAME="two words"']}
    explicit = ["-O0", "-mllvm", "legacy", "-std=c++17"]
    for key, value in option_env.items():
        monkeypatch.setenv(key, value)
    expected = compiler.resolve_compile_config(target, pass_configs, explicit)
    actual = compiler.with_compile_options(base, pass_configs, explicit, environ=option_env)
    assert actual == expected
    assert actual.arch == ("dav-c220" if dialect == "pto" else "dav-2201")
    if dialect == "pto":
        assert actual.flags.count("-mllvm") == 1
        assert actual.flags[actual.flags.index("-mllvm") + 1] == "configured"
        assert "legacy" in actual.flags
    else:
        assert actual.flags.count("-mllvm") == 2
    assert "-DNAME=two words" in actual.flags


@pytest.mark.parametrize("dialect", ["ascendc", "pto"])
def test_candidate_options_are_pure_and_preserve_frozen_identity(toolchain, monkeypatch, dialect):
    config = compiler.resolve_compile_config(_target(keys=[dialect, "ascend"]))

    def forbidden(*args, **kwargs):
        pytest.fail("candidate options re-resolved environment, cwd, SDK or platform")

    class NoEnvironment(dict):
        def get(self, *args):
            return forbidden()

    monkeypatch.setattr(compiler, "os", SimpleNamespace(environ=NoEnvironment(), getcwd=forbidden, path=os.path))
    for owner, name in [(compiler.shutil, "which"), (compiler, "_get_tl_root"), (compiler, "_get_ascend_home_path")]:
        monkeypatch.setattr(owner, name, forbidden)
    result = compiler.with_compile_options(config, {"tl.ascend_auto_sync_vs": True}, environ={})
    assert result.flags[0] == "-O3"
    for key, value in config.to_metadata().items():
        if key not in {"flags", "command_options"}:
            assert result.to_metadata()[key] == value
    with pytest.raises(ValueError, match="Invalid opt_level"):
        compiler.with_compile_options(config, environ={"TL_CCE_OPT_LEVEL": "invalid"})
