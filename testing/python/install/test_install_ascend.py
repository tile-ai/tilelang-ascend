# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
"""Exercise installer control flow with fake build and package commands only.

Run without importing TileLang: python3 -m unittest discover -s testing/python/install -v
"""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class InstallAscendTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "repo with spaces"
        self.root.mkdir()
        self.shmem = self.root / "3rdparty/shmem"
        self.python_dir = self.shmem / "src/python"
        self.python_dir.mkdir(parents=True)
        self.bin_dir = Path(self.temp.name) / "bin"
        self.bin_dir.mkdir()
        self.log = Path(self.temp.name) / "commands.log"
        source = Path(__file__).resolve().parents[3] / "install_ascend.sh"
        shutil.copyfile(source, self.root / "install_ascend.sh")
        config = self.root / "3rdparty/tvm/cmake/config.cmake"
        config.parent.mkdir(parents=True)
        config.touch()
        self.write_script(self.root / "3rdparty/patches/apply_tvm_patches.sh", "exit 0")
        self.write_script(
            self.shmem / "scripts/build.sh",
            'echo "native-build:$PWD" >> "$FAKE_COMMAND_LOG"\nexit "${FAKE_BUILD_STATUS:-0}"',
        )
        self.env_file = self.shmem / "install/set_env.sh"
        # Reproduce SHMEM's unquoted dirname substitution, including its behavior
        # when the source name contains spaces (pinned revision 157eecf7d210).
        self.write_script(
            self.env_file,
            'shmem_path=$(cd $(dirname "${BASH_SOURCE[0]}"); pwd)\n'
            'echo "source-env:$shmem_path" >> "$FAKE_COMMAND_LOG"\n'
            'return "${FAKE_ENV_STATUS:-0}"',
        )
        for name in ("git", "cmake", "make"):
            self.write_script(self.bin_dir / name, "exit 0")
        self.write_script(self.bin_dir / "nproc", "echo 2")
        # The installer currently requires GNU grep -P for Python version parsing.
        self.write_script(self.bin_dir / "grep", "echo 3.12")
        self.write_script(self.bin_dir / "python3", 'echo "Python 3.12.0"')
        self.write_script(
            self.bin_dir / "pip",
            "\n".join(
                [
                    'echo "pip:$*" >> "$FAKE_COMMAND_LOG"',
                    'if [[ "$1" == uninstall ]]; then',
                    '    exit "${FAKE_UNINSTALL_STATUS:-0}"',
                    "fi",
                    "exit 0",
                ]
            ),
        )
        self.write_script(
            self.bin_dir / "pip3",
            'echo "pip3:$*" >> "$FAKE_COMMAND_LOG"\nexit "${FAKE_FALLBACK_STATUS:-0}"',
        )
        self.write_script(
            self.bin_dir / "python",
            'echo "python:$*" >> "$FAKE_COMMAND_LOG"\n'
            'if [[ "$*" == "setup.py bdist_wheel" ]]; then\n'
            '    if [[ "${FAKE_WHEEL_STATUS:-0}" != 0 ]]; then\n'
            '        exit "$FAKE_WHEEL_STATUS"\n'
            "    fi\n"
            '    if [[ "${FAKE_NO_WHEEL:-0}" == 0 ]]; then\n'
            "        mkdir -p dist\n"
            "        touch dist/shmem-test.whl\n"
            "    fi\n"
            "else\n"
            '    exit "${FAKE_INSTALL_STATUS:-0}"\n'
            "fi",
        )

    @staticmethod
    def write_script(path: Path, body: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/bin/bash\n" + body + "\n", encoding="utf-8")
        path.chmod(0o755)

    def run_installer(
        self,
        *,
        enabled: bool = True,
        outside: bool = False,
        **settings: str,
    ) -> subprocess.CompletedProcess[str]:
        env = dict(os.environ)
        env.update(settings)
        # CI exports BASH_ENV=/root/.bashrc. Sourcing it in every Bash-based
        # fake command can recursively invoke the same fakes during startup.
        env.pop("BASH_ENV", None)
        env.pop("ENV", None)
        env["FAKE_COMMAND_LOG"] = str(self.log)
        env["PATH"] = str(self.bin_dir) + os.pathsep + env["PATH"]
        command = ["bash", str(self.root / "install_ascend.sh")]
        if enabled:
            command.append("--enable-shmem")
        result = subprocess.run(
            command,
            cwd=self.temp.name if outside else self.root,
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.commands = self.log.read_text(encoding="utf-8")
        return result

    def assert_failed_before_install(
        self,
        result: subprocess.CompletedProcess[str],
        error: str,
    ) -> None:
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(error, result.stdout + result.stderr)
        self.assertNotIn("pip:uninstall", self.commands)
        self.assertNotIn("python:-m pip install", self.commands)
        self.assertNotIn("pip3:install", self.commands)
        self.assertNotIn("source-env", self.commands)
        self.assertNotIn("Install shmem all success.", result.stdout)

    def test_native_failure_stops_before_wheel_or_uninstall(self):
        result = self.run_installer(FAKE_BUILD_STATUS="7")
        self.assert_failed_before_install(result, "shmem build failed")
        self.assertNotIn("python:setup.py", self.commands)

    def test_wheel_failure_does_not_install_stale_wheel(self):
        dist = self.python_dir / "dist"
        dist.mkdir()
        (dist / "shmem-stale.whl").touch()
        result = self.run_installer(FAKE_WHEEL_STATUS="9")
        self.assert_failed_before_install(result, "shmem wheel build failed")

    def test_missing_env_stops_before_package_changes(self):
        self.env_file.unlink()
        result = self.run_installer()
        self.assert_failed_before_install(result, "shmem environment script not found")

    def test_missing_wheel_stops_before_uninstall(self):
        result = self.run_installer(FAKE_NO_WHEEL="1")
        self.assert_failed_before_install(result, "No shmem wheel found")

    def test_pip_fallback_succeeds(self):
        result = self.run_installer(FAKE_INSTALL_STATUS="4")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("pip3:install", self.commands)
        self.assertIn("source-env", self.commands)
        self.assertIn("Install shmem all success.", result.stdout)

    def test_both_installers_fail(self):
        result = self.run_installer(FAKE_INSTALL_STATUS="4", FAKE_FALLBACK_STATUS="5")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("shmem-xxx.whl install failed", result.stdout + result.stderr)
        self.assertNotIn("source-env", self.commands)
        self.assertNotIn("Install shmem all success.", result.stdout)

    def test_environment_failure_is_reported(self):
        result = self.run_installer(FAKE_ENV_STATUS="6")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("set shmem env failed", result.stdout + result.stderr)
        self.assertNotIn("Install shmem all success.", result.stdout)

    def test_uninstall_failure_stops_before_install(self):
        result = self.run_installer(FAKE_UNINSTALL_STATUS="8")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Failed to uninstall old shmem package", result.stdout + result.stderr)
        self.assertNotIn("python:-m pip install", self.commands)
        self.assertNotIn("source-env", self.commands)

    def test_success_from_another_working_directory(self):
        result = self.run_installer(outside=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("native-build:" + str(self.shmem), self.commands)
        self.assertIn(str(self.python_dir / "dist/shmem-test.whl"), self.commands)
        self.assertLess(
            self.commands.index("python:setup.py"),
            self.commands.index("pip:uninstall"),
        )
        self.assertIn("source-env", self.commands)
        self.assertIn("source-env:" + str(self.shmem / "install"), self.commands)

    def test_disabled_shmem_does_not_run_shmem_commands(self):
        result = self.run_installer(enabled=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("native-build", self.commands)
        self.assertNotIn("python:", self.commands)
        self.assertNotIn("pip:uninstall", self.commands)
        self.assertNotIn("source-env", self.commands)

    def test_inherited_shell_startup_does_not_run_in_fake_commands(self):
        startup = self.root / "shell-startup.sh"
        self.write_script(
            startup,
            "depth=${FAKE_STARTUP_DEPTH:-0}\n"
            "if ((depth >= 3)); then\n"
            '    echo "unexpected recursive shell startup" >&2\n'
            "    exit 91\n"
            "fi\n"
            "export FAKE_STARTUP_DEPTH=$((depth + 1))\n"
            "grep ignored\n"
            "exit $?",
        )
        # Also cover PATH-based shell selection. The depth limit makes the
        # unfixed harness fail promptly without leaving an unbounded chain.
        self.write_script(self.bin_dir / "bash", 'exec /bin/bash "$@"')
        result = self.run_installer(BASH_ENV=str(startup), ENV=str(startup))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("recursive shell startup", result.stderr)
        self.assertIn("Install shmem all success.", result.stdout)


if __name__ == "__main__":
    unittest.main()
