import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import setup_python


class EnvironmentTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve() / "shared repo"
        (self.root / "tools").mkdir(parents=True)
        for name in ("runner.sh", "setup_python.py"):
            shutil.copyfile(REPO / "tools" / name, self.root / "tools" / name)
        (self.root / "tools" / "runner.sh").chmod(0o755)
        for entry in ("run_experiments.py", "runner_loop.py"):
            (self.root / "tools" / entry).write_text(
                "import json,sys\n"
                "print(json.dumps({'args':sys.argv[1:],'prefix':sys.prefix}))\n",
                encoding="utf-8",
            )
        self.env = {**os.environ, "GPU_NET_SIM_PYTHON": sys.executable, "PYTHONDONTWRITEBYTECODE": "1"}

    def run_wrapper(self, *args):
        return subprocess.run(
            [str(self.root / "tools" / "runner.sh"), *args],
            cwd=self.root.parent, env=self.env, text=True, capture_output=True, timeout=30,
        )

    def test_wrapper_outside_repo_uses_local_environment_and_forwards_arguments(self):
        result = self.run_wrapper("--once", "--worker-id", "node 01")
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["args"], ["--once", "--worker-id", "node 01"])
        prefix = Path(payload["prefix"])
        self.assertEqual(prefix.parent, self.root / ".venv")
        self.assertTrue(prefix.name.startswith(setup_python.platform_key() + "-py"))
        metadata = json.loads((prefix / ".runner-environment.json").read_text())
        self.assertEqual(metadata["provider"], "system-symlink")
        runtime = Path(metadata["python"])
        self.assertTrue(runtime.is_relative_to(self.root / ".python"))
        self.assertTrue(runtime.is_symlink())
        self.assertTrue((runtime.parent / "runtime.json").is_file())

    def test_venv_home_uses_the_canonical_base_interpreter(self):
        result = self.run_wrapper("--once")
        self.assertEqual(result.returncode, 0, result.stderr)
        prefix = Path(json.loads(result.stdout)["prefix"])
        metadata = json.loads((prefix / ".runner-environment.json").read_text())
        recorded_reference = Path(metadata["python"])
        self.assertTrue(recorded_reference.is_symlink())
        base = Path(metadata["runtime"]["executable"]).resolve()
        configuration = dict(line.split(" = ", 1) for line in
                             (prefix / "pyvenv.cfg").read_text().splitlines()
                             if " = " in line)
        self.assertEqual(Path(configuration["home"]).resolve(), base.parent)
        self.assertNotEqual(Path(configuration["home"]).resolve(), recorded_reference.parent)

    def test_existing_external_venv_uses_its_base_interpreter(self):
        external = self.root.parent / "external-environment"
        subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(external)],
                       check=True, capture_output=True, text=True, timeout=30)
        self.env["GPU_NET_SIM_PYTHON"] = str(external / "bin" / "python")
        result = self.run_wrapper("--once")
        self.assertEqual(result.returncode, 0, result.stderr)
        prefix = Path(json.loads(result.stdout)["prefix"])
        metadata = json.loads((prefix / ".runner-environment.json").read_text())
        self.assertNotEqual(Path(metadata["runtime"]["base_prefix"]), external)
        runtime = Path(metadata["runtime"]["executable"])
        self.assertFalse(runtime.is_relative_to(external))
        checked = subprocess.run(
            [str(prefix / "bin" / "python"), "-I", "-c",
             "import encodings,json,sys; print(json.dumps({'base':sys.base_prefix,'prefix':sys.prefix}))"],
            text=True, capture_output=True, timeout=30,
        )
        self.assertEqual(checked.returncode, 0, checked.stderr)
        self.assertEqual(Path(json.loads(checked.stdout)["prefix"]), prefix)

    def test_old_bootstrap_environment_is_preserved_when_replaced(self):
        info = setup_python.python_info(sys.executable)
        major, minor = info["version"][:2]
        old = self.root / ".venv" / (
            f"{setup_python.platform_key()}-py{major}.{minor}-{setup_python.runtime_fingerprint(info)}"
        )
        (old / "bin").mkdir(parents=True)
        broken_python = old / "bin" / "python"
        broken_python.write_text("#!/bin/sh\necho missing-encodings >&2\nexit 1\n")
        broken_python.chmod(0o755)
        marker = old / ".runner-environment.json"
        marker.write_text(json.dumps({"provider": "system-symlink", "runtime": info}))
        sentinel = old / "keep-for-debugging.txt"
        sentinel.write_text("old bootstrap evidence")
        result = self.run_wrapper("--once")
        self.assertEqual(result.returncode, 0, result.stderr)
        prefix = Path(json.loads(result.stdout)["prefix"])
        self.assertNotEqual(prefix, old)
        self.assertEqual(sentinel.read_text(), "old bootstrap evidence")
        self.assertEqual(broken_python.read_text(), "#!/bin/sh\necho missing-encodings >&2\nexit 1\n")

    def test_real_standalone_python_uses_canonical_home_and_loads_stdlib(self):
        override = os.environ.get("GPU_NET_SIM_TEST_STANDALONE")
        candidates = [Path(override)] if override else sorted(
            (Path.home() / ".local/share/uv/python").glob("cpython-3.*-*/bin/python3.*")
        )
        candidates = [path for path in candidates if path.is_file()
                      and (override or path.name.removeprefix("python3.").isdigit())]
        standalone = next((path for path in candidates if subprocess.run(
            [str(path), "-I", "-c", "import encodings,sys; assert sys.version_info >= (3,9)"],
            capture_output=True, text=True, timeout=15).returncode == 0), None)
        if standalone is None:
            self.skipTest("No local python-build-standalone runtime; set GPU_NET_SIM_TEST_STANDALONE to exercise it")
        self.env["GPU_NET_SIM_PYTHON"] = str(standalone)
        self.env.pop("PYTHONHOME", None)
        self.env.pop("PYTHONPATH", None)
        result = self.run_wrapper("--once")
        self.assertEqual(result.returncode, 0, result.stderr)
        prefix = Path(json.loads(result.stdout)["prefix"])
        configuration = dict(line.split(" = ", 1) for line in
                             (prefix / "pyvenv.cfg").read_text().splitlines()
                             if " = " in line)
        self.assertEqual(Path(configuration["home"]).resolve(), standalone.resolve().parent)
        checked = subprocess.run(
            [str(prefix / "bin/python"), "-I", "-c", "import encodings,json,ssl,venv; print('stdlib-ready')"],
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(checked.returncode, 0, checked.stderr)
        self.assertEqual(checked.stdout.strip(), "stdlib-ready")

    def test_help_does_not_create_environment_or_lock(self):
        result = self.run_wrapper("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["args"], ["--help"])
        for directory in (".python", ".venv", ".lock"):
            self.assertFalse((self.root / directory).exists())

    def test_two_bootstrappers_share_one_completed_environment(self):
        args = [str(self.root / "tools" / "runner.sh"), "--once"]
        children = [subprocess.Popen(args, env=self.env, text=True, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE) for _ in range(2)]
        prefixes = []
        try:
            for child in children:
                stdout, stderr = child.communicate(timeout=30)
                self.assertEqual(child.returncode, 0, stderr)
                prefixes.append(json.loads(stdout)["prefix"])
        finally:
            for child in children:
                if child.poll() is None:
                    child.kill()
                    child.wait()
        self.assertEqual(prefixes[0], prefixes[1])
        environments = list((self.root / ".venv").iterdir())
        self.assertEqual(len(environments), 1)
        self.assertTrue((environments[0] / ".runner-environment.json").is_file())
        self.assertEqual(list((self.root / ".lock").iterdir()), [])

    def test_incomplete_environment_is_preserved_for_inspection(self):
        first = self.run_wrapper("--once")
        self.assertEqual(first.returncode, 0, first.stderr)
        prefix = Path(json.loads(first.stdout)["prefix"])
        (prefix / ".runner-environment.json").unlink()
        retry = self.run_wrapper("--once")
        self.assertNotEqual(retry.returncode, 0)
        self.assertIn("Incomplete Python environment", retry.stderr)
        self.assertTrue((prefix / "bin" / "python").is_file())
        self.assertEqual(list((self.root / ".lock").iterdir()), [])

    def test_platform_and_runtime_identity_isolate_incompatible_nodes(self):
        with patch.object(setup_python.sys, "platform", "linux"):
            with patch.object(setup_python.platform, "machine", return_value="arm64"):
                arm_linux = setup_python.platform_key()
            with patch.object(setup_python.platform, "machine", return_value="aarch64"):
                self.assertEqual(setup_python.platform_key(), arm_linux)
            with patch.object(setup_python.platform, "machine", return_value="x86_64"):
                self.assertNotEqual(setup_python.platform_key(), arm_linux)
        with patch.object(setup_python.sys, "platform", "darwin"):
            with patch.object(setup_python.platform, "machine", return_value="arm64"):
                self.assertNotEqual(setup_python.platform_key(), arm_linux)
        info = setup_python.python_info(sys.executable)
        other_version = {**info, "version": [3, 12, 1] if info["version"][:2] != [3, 12] else [3, 11, 1]}
        other_location = {**info, "base_prefix": "/another/node/python"}
        self.assertNotEqual(setup_python.runtime_fingerprint(info), setup_python.runtime_fingerprint(other_version))
        self.assertNotEqual(setup_python.runtime_fingerprint(info), setup_python.runtime_fingerprint(other_location))

    def test_uv_manages_python_inside_requested_root(self):
        runtime_root = self.root / ".python" / setup_python.platform_key()
        runtime_root.mkdir(parents=True)
        python = runtime_root / "managed" / "cpython-3.12" / "bin" / "python"
        python.parent.mkdir(parents=True)
        python.touch()
        info = {"implementation": "cpython", "version": [3, 12, 1],
                "executable": str(python), "base_prefix": str(python.parent.parent),
                "prefix": str(python.parent.parent)}
        commands = []

        def fake_command(args, env):
            commands.append([str(arg) for arg in args])
            self.assertEqual(env["UV_PYTHON_INSTALL_DIR"], str(runtime_root / "managed"))
            if expected_native_tls is None:
                self.assertNotIn("UV_NATIVE_TLS", env)
            else:
                self.assertEqual(env["UV_NATIVE_TLS"], expected_native_tls)
            self.assertEqual(env.get("UV_SYSTEM_CERTS"), expected_system_certs)
            self.assertNotIn("--allow-insecure-host", [str(arg) for arg in args])
            if len(commands) == 1:
                raise RuntimeError("Python not installed")
            return str(python) if "find" in args else ""

        for configured in ({}, {"UV_NATIVE_TLS": "false"}, {"UV_SYSTEM_CERTS": "false"}):
            commands.clear()
            expected_system_certs = configured.get("UV_SYSTEM_CERTS")
            expected_native_tls = None if expected_system_certs else configured.get("UV_NATIVE_TLS", "true")
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("UV_NATIVE_TLS", None)
                os.environ.pop("UV_SYSTEM_CERTS", None)
                os.environ.update(configured)
                with patch.object(setup_python, "command", side_effect=fake_command):
                    with patch.object(setup_python, "python_info", return_value=info):
                        selected, _, metadata = setup_python.managed_runtime(runtime_root, "/bin/uv")
        self.assertEqual(selected, python)
        self.assertEqual(metadata["provider"], "uv")
        self.assertIn(["/bin/uv", "python", "install", "--install-dir", str(runtime_root / "managed"),
                       "--no-bin", setup_python.PYTHON_VERSION], commands)
        self.assertTrue((runtime_root / "managed-runtime.json").is_file())

    def test_working_current_python_avoids_managed_download(self):
        runtime_root = self.root / ".python" / setup_python.platform_key()
        runtime_root.mkdir(parents=True)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GPU_NET_SIM_PYTHON", None)
            with patch.object(setup_python.shutil, "which", return_value="/bin/uv"):
                with patch.object(setup_python, "managed_runtime", side_effect=AssertionError("unexpected download")) as managed:
                    python, _, metadata = setup_python.selected_runtime(runtime_root)
                managed.assert_not_called()
        self.assertEqual(metadata["provider"], "system-symlink")
        self.assertTrue(python.is_symlink())
        self.assertTrue(python.is_relative_to(runtime_root))

    def test_unusable_current_python_attempts_managed_runtime(self):
        runtime_root = self.root / ".python" / setup_python.platform_key()
        selected = runtime_root / "managed/bin/python"
        info = {"implementation": "cpython", "version": [3, 12, 1],
                "executable": str(selected), "base_prefix": str(selected.parent.parent),
                "prefix": str(selected.parent.parent)}
        expected = selected, info, {"provider": "uv", "python": str(selected), "runtime": info}
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GPU_NET_SIM_PYTHON", None)
            with patch.object(setup_python, "existing_runtime", side_effect=RuntimeError("missing standard library")):
                with patch.object(setup_python.shutil, "which", return_value="/bin/uv"):
                    with patch.object(setup_python, "managed_runtime", return_value=expected) as managed:
                        actual = setup_python.selected_runtime(runtime_root)
        self.assertEqual(actual, expected)
        managed.assert_called_once_with(runtime_root, "/bin/uv")

    def test_invalid_explicit_python_does_not_select_another_runtime(self):
        runtime_root = self.root / ".python" / setup_python.platform_key()
        with patch.dict(os.environ, {"GPU_NET_SIM_PYTHON": "/unavailable/python"}):
            with patch.object(setup_python, "existing_runtime", side_effect=RuntimeError("explicit runtime missing")):
                with patch.object(setup_python, "managed_runtime") as managed:
                    with self.assertRaisesRegex(RuntimeError, "explicit runtime missing"):
                        setup_python.selected_runtime(runtime_root)
        managed.assert_not_called()
